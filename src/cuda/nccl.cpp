#include "ms/cuda/nccl.hpp"
#include "ms/cuda/buffer.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
// ncclAvg landed in NCCL 2.10. NCCL_VERSION_CODE encodes 2.10.0 as 21000
// (X*10000 + Y*100 + Z above 2.8, X*1000 + Y*100 + Z at or below it), so the
// numeric comparison below is well-defined without invoking the function-like
// NCCL_VERSION() macro inside a #if.
#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 21000
#define MS_NCCL_HAS_AVG 1
#else
#define MS_NCCL_HAS_AVG 0
#endif
#endif

namespace ms::cuda {

namespace {

/// Size of an ncclUniqueId (NCCL_UNIQUE_ID_BYTES). Fixed by the NCCL ABI and
/// therefore checkable in the stub build too, where nccl.h is not included.
constexpr size_t kUniqueIdBytes = 128;

/// Element-wise reduction step shared by the host-staging paths.
double combine(ReduceOp op, double a, double b) {
    switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::Avg:
        return a + b;
    case ReduceOp::Prod:
        return a * b;
    case ReduceOp::Max:
        return a > b ? a : b;
    case ReduceOp::Min:
        return a < b ? a : b;
    }
    return a + b;
}

/// Rejects the degenerate buffer lists every `*_buffer` collective shares:
/// an empty list, a null rank or null payload, and a buffer too small to hold
/// `bytes`. O(ranks.size()).
Result<void> validate_ranks(std::span<DeviceBuffer* const> ranks, size_t bytes, const char* fn) {
    if (ranks.empty()) {
        return std::unexpected(DomainError{fn, "empty rank list"});
    }
    for (const DeviceBuffer* buf : ranks) {
        if (buf == nullptr || buf->data == nullptr) {
            return std::unexpected(DomainError{fn, "null rank buffer"});
        }
        if (buf->bytes < bytes) {
            return std::unexpected(DimensionMismatch{buf->bytes, bytes});
        }
    }
    return {};
}

/// Host-staged reduction across every rank buffer, producing the same result
/// the NCCL path would. Used whenever NCCL is off, unavailable, or the buffer
/// layout does not match this process's clique. O(count * ranks.size()).
void host_accumulate(std::span<DeviceBuffer* const> ranks, size_t count, ReduceOp op,
                     std::vector<double>& acc) {
    const size_t bytes = count * sizeof(double);
    acc.assign(count, 0.0);
    std::vector<double> tmp(count, 0.0);
    copy_device_to_host(*ranks[0], acc.data(), bytes);
    for (size_t r = 1; r < ranks.size(); ++r) {
        copy_device_to_host(*ranks[r], tmp.data(), bytes);
        for (size_t k = 0; k < count; ++k) {
            acc[k] = combine(op, acc[k], tmp[k]);
        }
    }
    if (op == ReduceOp::Avg) {
        const double inv = 1.0 / static_cast<double>(ranks.size());
        for (size_t k = 0; k < count; ++k) {
            acc[k] *= inv;
        }
    }
}

#if defined(MS_HAS_NCCL) && MS_HAS_NCCL

/// Multiplies `count` doubles of `buf` by `factor` through a host round-trip.
/// Only needed to finish an Avg reduction on an NCCL older than 2.10.
/// O(count).
void host_scale(DeviceBuffer& buf, size_t count, double factor) {
    std::vector<double> staging(count, 0.0);
    copy_device_to_host(buf, staging.data(), count * sizeof(double));
    for (size_t k = 0; k < count; ++k) {
        staging[k] *= factor;
    }
    copy_host_to_device(staging.data(), buf, count * sizeof(double));
}

/// Owns this process's NCCL communicators and their streams.
///
/// Local mode: `ensure()` calls ncclCommInitAll over every visible GPU, so this
///   process owns every rank of the communicator (`world_size_ == 0` marks it).
/// Rank mode: `adopt_rank()` calls ncclCommInitRank, so this process owns
///   exactly one rank of a `world_size_`-rank communicator.
///
/// Non-copyable and non-movable; the handles live in vectors rather than owning
/// raw pointer members, and the destructor releases them. Not thread-safe (same
/// contract as StreamPool). A failed lazy init is remembered in `attempted_` so
/// it is not retried on every collective.
class NcclClique {
public:
    NcclClique() = default;
    NcclClique(const NcclClique&) = delete;
    NcclClique& operator=(const NcclClique&) = delete;
    NcclClique(NcclClique&&) = delete;
    NcclClique& operator=(NcclClique&&) = delete;
    ~NcclClique() { release(); }

    /// Lazily brings up local mode over every visible GPU. Idempotent, and a
    /// no-op once an attempt has been made. O(device_count()).
    bool ensure() {
        if (attempted_) {
            return ready_;
        }
        attempted_ = true;
        const int devices = device_count();
        if (devices <= 0) {
            return false;
        }
        devices_.resize(static_cast<size_t>(devices));
        for (int i = 0; i < devices; ++i) {
            devices_[static_cast<size_t>(i)] = i;
        }
        comms_.assign(static_cast<size_t>(devices), nullptr);
        if (ncclCommInitAll(comms_.data(), devices, devices_.data()) != ncclSuccess) {
            comms_.clear();
            devices_.clear();
            return false;
        }
        streams_.assign(static_cast<size_t>(devices), nullptr);
        for (int i = 0; i < devices; ++i) {
            cudaStream_t stream = nullptr;
            if (cudaSetDevice(i) != cudaSuccess || cudaStreamCreate(&stream) != cudaSuccess) {
                release();
                return false;
            }
            streams_[static_cast<size_t>(i)] = stream;
        }
        ready_ = true;
        return true;
    }

    /// Joins an existing multi-process communicator, replacing anything this
    /// process already held. Blocks in the NCCL rendezvous.
    bool adopt_rank(const ncclUniqueId& id, int rank, int world_size, int device) {
        reset();
        attempted_ = true;
        if (cudaSetDevice(device) != cudaSuccess) {
            return false;
        }
        devices_.assign(1, device);
        comms_.assign(1, nullptr);
        if (ncclCommInitRank(comms_.data(), world_size, id, rank) != ncclSuccess) {
            comms_.clear();
            devices_.clear();
            return false;
        }
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            release();
            return false;
        }
        streams_.assign(1, stream);
        world_size_ = static_cast<size_t>(world_size);
        rank_ = rank;
        ready_ = true;
        return true;
    }

    /// Destroys every handle. Deliberately leaves `attempted_` set so a failed
    /// lazy init is not retried on every collective; `reset()` clears it.
    void release() {
        for (size_t i = 0; i < streams_.size(); ++i) {
            if (streams_[i] != nullptr) {
                if (i < devices_.size()) {
                    cudaSetDevice(devices_[i]);
                }
                cudaStreamDestroy(streams_[i]);
            }
        }
        streams_.clear();
        for (ncclComm_t comm : comms_) {
            if (comm != nullptr) {
                ncclCommDestroy(comm);
            }
        }
        comms_.clear();
        devices_.clear();
        world_size_ = 0;
        rank_ = 0;
        ready_ = false;
    }

    void reset() {
        release();
        attempted_ = false;
    }

    /// Waits for every stream this process drives. O(local rank count).
    bool sync() {
        bool ok = true;
        for (size_t i = 0; i < streams_.size(); ++i) {
            if (i < devices_.size() && cudaSetDevice(devices_[i]) != cudaSuccess) {
                ok = false;
                continue;
            }
            if (cudaStreamSynchronize(streams_[i]) != cudaSuccess) {
                ok = false;
            }
        }
        return ok;
    }

    size_t local_count() const { return comms_.size(); }
    size_t world_size() const { return world_size_; } // 0 unless rank mode
    int rank() const { return rank_; }
    ncclComm_t comm(size_t i) const { return comms_[i]; }
    cudaStream_t stream(size_t i) const { return streams_[i]; }
    int device(size_t i) const { return devices_[i]; }

private:
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
    std::vector<int> devices_;
    size_t world_size_ = 0;
    int rank_ = 0;
    bool attempted_ = false;
    bool ready_ = false;
};

/// Process-wide holder: constructed on first use, released at process teardown
/// (its destructor swallows NCCL/CUDA errors because the runtime may already be
/// shutting down). Call nccl_finalize() for deterministic teardown.
NcclClique& clique() {
    static NcclClique instance;
    return instance;
}

ncclRedOp_t to_nccl_op(ReduceOp op) {
    switch (op) {
    case ReduceOp::Sum:
        return ncclSum;
    case ReduceOp::Prod:
        return ncclProd;
    case ReduceOp::Max:
        return ncclMax;
    case ReduceOp::Min:
        return ncclMin;
    case ReduceOp::Avg:
#if MS_NCCL_HAS_AVG
        return ncclAvg;
#else
        return ncclSum;
#endif
    }
    return ncclSum;
}

/// True when the linked NCCL implements Avg natively; otherwise Avg is Sum
/// followed by an explicit 1/comm_size scale.
constexpr bool nccl_avg_is_native() {
#if MS_NCCL_HAS_AVG
    return true;
#else
    return false;
#endif
}

Result<void> nccl_allreduce_device(std::span<DeviceBuffer* const> ranks, size_t count,
                                   ReduceOp op) {
    NcclClique& c = clique();
    const ncclRedOp_t nccl_op = to_nccl_op(op);
    if (ncclGroupStart() != ncclSuccess) {
        return std::unexpected(DeviceError{42, "ncclGroupStart failed"});
    }
    ncclResult_t status = ncclSuccess;
    for (size_t i = 0; i < ranks.size(); ++i) {
        const ncclResult_t r = ncclAllReduce(ranks[i]->data, ranks[i]->data, count, ncclDouble,
                                             nccl_op, c.comm(i), c.stream(i));
        if (r != ncclSuccess) {
            status = r;
        }
    }
    if (ncclGroupEnd() != ncclSuccess || status != ncclSuccess) {
        return std::unexpected(DeviceError{43, "nccl collective failed"});
    }
    if (!c.sync()) {
        return std::unexpected(DeviceError{44, "stream synchronise failed"});
    }
    if (op == ReduceOp::Avg && !nccl_avg_is_native()) {
        const double factor = 1.0 / static_cast<double>(std::max<size_t>(1, nccl_comm_size()));
        for (DeviceBuffer* buf : ranks) {
            host_scale(*buf, count, factor);
        }
    }
    return {};
}

Result<void> nccl_broadcast_device(std::span<DeviceBuffer* const> ranks, size_t count, int root) {
    NcclClique& c = clique();
    if (ncclGroupStart() != ncclSuccess) {
        return std::unexpected(DeviceError{42, "ncclGroupStart failed"});
    }
    ncclResult_t status = ncclSuccess;
    for (size_t i = 0; i < ranks.size(); ++i) {
        const ncclResult_t r = ncclBroadcast(ranks[i]->data, ranks[i]->data, count, ncclDouble,
                                             root, c.comm(i), c.stream(i));
        if (r != ncclSuccess) {
            status = r;
        }
    }
    if (ncclGroupEnd() != ncclSuccess || status != ncclSuccess) {
        return std::unexpected(DeviceError{43, "nccl collective failed"});
    }
    if (!c.sync()) {
        return std::unexpected(DeviceError{44, "stream synchronise failed"});
    }
    return {};
}

/// ncclReduce writes only the root rank's receive buffer, so issuing the calls
/// in place leaves every non-root buffer untouched — matching the host-staging
/// guarantee of reduce_buffer().
Result<void> nccl_reduce_device(std::span<DeviceBuffer* const> ranks, size_t count, ReduceOp op,
                                int root) {
    NcclClique& c = clique();
    const ncclRedOp_t nccl_op = to_nccl_op(op);
    if (ncclGroupStart() != ncclSuccess) {
        return std::unexpected(DeviceError{42, "ncclGroupStart failed"});
    }
    ncclResult_t status = ncclSuccess;
    for (size_t i = 0; i < ranks.size(); ++i) {
        const ncclResult_t r = ncclReduce(ranks[i]->data, ranks[i]->data, count, ncclDouble,
                                          nccl_op, root, c.comm(i), c.stream(i));
        if (r != ncclSuccess) {
            status = r;
        }
    }
    if (ncclGroupEnd() != ncclSuccess || status != ncclSuccess) {
        return std::unexpected(DeviceError{43, "nccl collective failed"});
    }
    if (!c.sync()) {
        return std::unexpected(DeviceError{44, "stream synchronise failed"});
    }
    if (op == ReduceOp::Avg && !nccl_avg_is_native()) {
        const double factor = 1.0 / static_cast<double>(std::max<size_t>(1, nccl_comm_size()));
        const bool owns_root = (c.world_size() == 0) || (c.rank() == root);
        const size_t local_root = (c.world_size() > 0) ? 0 : static_cast<size_t>(root);
        if (owns_root && local_root < ranks.size()) {
            host_scale(*ranks[local_root], count, factor);
        }
    }
    return {};
}

Result<void> nccl_allgather_device(std::span<DeviceBuffer* const> send,
                                   std::span<DeviceBuffer* const> recv, size_t count) {
    NcclClique& c = clique();
    if (ncclGroupStart() != ncclSuccess) {
        return std::unexpected(DeviceError{42, "ncclGroupStart failed"});
    }
    ncclResult_t status = ncclSuccess;
    for (size_t i = 0; i < send.size(); ++i) {
        const ncclResult_t r = ncclAllGather(send[i]->data, recv[i]->data, count, ncclDouble,
                                             c.comm(i), c.stream(i));
        if (r != ncclSuccess) {
            status = r;
        }
    }
    if (ncclGroupEnd() != ncclSuccess || status != ncclSuccess) {
        return std::unexpected(DeviceError{43, "nccl collective failed"});
    }
    if (!c.sync()) {
        return std::unexpected(DeviceError{44, "stream synchronise failed"});
    }
    return {};
}

/// Allocates one staging buffer per local rank and replicates `values` onto
/// each. Returns the buffers; `out_ptrs` receives one pointer per buffer, in
/// clique-device order, ready for the device collectives above.
Result<void> stage_replicated(const std::vector<double>& values, std::vector<DeviceBuffer>& owned,
                              std::vector<DeviceBuffer*>& out_ptrs, size_t extra_recv_count) {
    NcclClique& c = clique();
    const size_t count = values.size() + extra_recv_count;
    const size_t bytes = count * sizeof(double);
    owned.clear();
    owned.reserve(c.local_count());
    for (size_t i = 0; i < c.local_count(); ++i) {
        owned.push_back(make_device_buffer(bytes, c.device(i)));
        if (owned.back().data == nullptr) {
            return std::unexpected(DeviceError{46, "device staging allocation failed"});
        }
        copy_host_to_device(values.data(), owned.back(), values.size() * sizeof(double));
    }
    out_ptrs.clear();
    out_ptrs.reserve(owned.size());
    for (DeviceBuffer& buf : owned) {
        out_ptrs.push_back(&buf);
    }
    return {};
}

#endif // MS_HAS_NCCL

/// True when the NCCL fast path can serve this exact buffer list: NCCL is up,
/// the caller supplied one buffer per local rank, and buffer i lives on clique
/// device i. Any mismatch (and every stub build) falls back to host staging,
/// which is slower but never wrong. O(ranks.size()).
bool fast_path_usable(std::span<DeviceBuffer* const> ranks) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        return false;
    }
    NcclClique& c = clique();
    if (!c.ensure()) {
        return false;
    }
    if (ranks.size() != c.local_count()) {
        return false;
    }
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (ranks[i]->device != c.device(i)) {
            return false;
        }
    }
    return true;
#else
    (void)ranks;
    return false;
#endif
}

/// Ranks a buffer collective spans: the whole communicator when the NCCL fast
/// path will be taken, otherwise just the buffers the caller handed over. This
/// is what a `root` index is validated against, so the host-staging path can
/// never be asked for a rank it does not hold.
size_t collective_world(size_t local_ranks, bool fast) {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (fast) {
        return std::max<size_t>(local_ranks, nccl_comm_size());
    }
#else
    (void)fast;
#endif
    return local_ranks;
}

/// Shared body of the scalar all-reduce entry points. The scalar is this
/// process's contribution for each rank it owns; a failed collective degrades
/// to the input value, the same shape as blas.cpp falling back to the CPU path.
double scalar_allreduce(double value, ReduceOp op) {
    if (nccl_comm_size() <= 1) {
        return value;
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    std::vector<double> staged{value};
    const auto status = allreduce_host(staged, op);
    if (!status.has_value() || staged.empty()) {
        return value;
    }
    return staged.front();
#else
    (void)op;
    return value;
#endif
}

} // namespace

bool nccl_available() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    return available();
#else
    return false;
#endif
}

int nccl_device_count() {
    return nccl_available() ? device_count() : 0;
}

size_t nccl_comm_size() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        return 1;
    }
    const size_t explicit_world = clique().world_size(); // 0 unless rank mode
    if (explicit_world > 0) {
        return explicit_world;
    }
    return static_cast<size_t>(device_count());
#else
    return 1;
#endif
}

int nccl_rank() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    return clique().rank();
#else
    return 0;
#endif
}

size_t nccl_local_rank_count() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        return 0;
    }
    if (clique().world_size() > 0) {
        return 1;
    }
    return static_cast<size_t>(device_count());
#else
    return 0;
#endif
}

std::string nccl_backend_name() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        return "stub";
    }
    return clique().world_size() > 0 ? "nccl-rank" : "nccl-local";
#else
    return "stub";
#endif
}

Result<std::vector<unsigned char>> nccl_make_unique_id() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        return std::unexpected(DeviceError{40, "nccl unavailable"});
    }
    ncclUniqueId id{};
    if (ncclGetUniqueId(&id) != ncclSuccess) {
        return std::unexpected(DeviceError{40, "ncclGetUniqueId failed"});
    }
    std::vector<unsigned char> out(kUniqueIdBytes, 0);
    for (size_t i = 0; i < kUniqueIdBytes; ++i) {
        out[i] = static_cast<unsigned char>(id.internal[i]);
    }
    return out;
#else
    return std::unexpected(DeviceError{40, "built without nccl"});
#endif
}

Result<void> nccl_init_rank(std::span<const unsigned char> unique_id, int rank, int world_size,
                            int device) {
    if (unique_id.size() != kUniqueIdBytes) {
        return std::unexpected(DimensionMismatch{unique_id.size(), kUniqueIdBytes});
    }
    if (world_size <= 0) {
        return std::unexpected(
            ValueOutOfRange{"world_size", static_cast<double>(world_size), 1.0, 1e9});
    }
    if (rank < 0 || rank >= world_size) {
        return std::unexpected(ValueOutOfRange{
            "rank", static_cast<double>(rank), 0.0, static_cast<double>(world_size - 1)});
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (!nccl_available()) {
        return std::unexpected(DeviceError{41, "nccl unavailable"});
    }
    ncclUniqueId id{};
    for (size_t i = 0; i < kUniqueIdBytes; ++i) {
        id.internal[i] = static_cast<char>(unique_id[i]);
    }
    if (!clique().adopt_rank(id, rank, world_size, device)) {
        return std::unexpected(DeviceError{41, "ncclCommInitRank failed"});
    }
    return {};
#else
    (void)device;
    return std::unexpected(DeviceError{41, "built without nccl"});
#endif
}

void nccl_finalize() {
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    clique().reset();
#endif
}

double allreduce_sum(double value) {
    return scalar_allreduce(value, ReduceOp::Sum);
}

double allreduce_max(double value) {
    return scalar_allreduce(value, ReduceOp::Max);
}

double allreduce_min(double value) {
    return scalar_allreduce(value, ReduceOp::Min);
}

double allreduce_prod(double value) {
    return scalar_allreduce(value, ReduceOp::Prod);
}

double allreduce_avg(double value) {
    const size_t size = nccl_comm_size();
    if (size <= 1) {
        return value;
    }
    return allreduce_sum(value) / static_cast<double>(size);
}

double broadcast(double value, int root) {
    const size_t size = nccl_comm_size();
    if (size <= 1 || root < 0 || static_cast<size_t>(root) >= size) {
        return value;
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    std::vector<double> staged{value};
    const auto status = broadcast_host(staged, root);
    if (!status.has_value() || staged.empty()) {
        return value;
    }
    return staged.front();
#else
    return value;
#endif
}

double reduce(double value, int root) {
    const size_t size = nccl_comm_size();
    if (size <= 1 || root < 0 || static_cast<size_t>(root) >= size) {
        return value;
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    std::vector<double> staged{value};
    const auto status = reduce_host(staged, ReduceOp::Sum, root);
    if (!status.has_value() || staged.empty()) {
        return value;
    }
    return staged.front();
#else
    return value;
#endif
}

double allgather(double value) {
    if (nccl_comm_size() <= 1) {
        return value;
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    const std::vector<double> staged{value};
    const auto gathered = allgather_host(staged);
    if (!gathered.has_value() || gathered->empty()) {
        return value;
    }
    return gathered->front();
#else
    return value;
#endif
}

Result<void> allreduce_host(std::vector<double>& values, ReduceOp op) {
    if (values.empty() || nccl_comm_size() <= 1) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    NcclClique& c = clique();
    if (!c.ensure()) {
        return std::unexpected(DeviceError{45, "nccl communicator init failed"});
    }
    const size_t count = values.size();
    std::vector<DeviceBuffer> owned;
    std::vector<DeviceBuffer*> ranks;
    const auto staged = stage_replicated(values, owned, ranks, 0);
    if (!staged.has_value()) {
        return std::unexpected(staged.error());
    }
    const auto status = nccl_allreduce_device(ranks, count, op);
    if (!status.has_value()) {
        return std::unexpected(status.error());
    }
    copy_device_to_host(owned.front(), values.data(), count * sizeof(double));
    return {};
#else
    (void)op;
    return {};
#endif
}

Result<void> broadcast_host(std::vector<double>& values, int root) {
    const size_t size = nccl_comm_size();
    if (root < 0 || static_cast<size_t>(root) >= size) {
        return std::unexpected(
            ValueOutOfRange{"root", static_cast<double>(root), 0.0, static_cast<double>(size - 1)});
    }
    if (values.empty() || size <= 1) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    NcclClique& c = clique();
    if (!c.ensure()) {
        return std::unexpected(DeviceError{45, "nccl communicator init failed"});
    }
    const size_t count = values.size();
    std::vector<DeviceBuffer> owned;
    std::vector<DeviceBuffer*> ranks;
    const auto staged = stage_replicated(values, owned, ranks, 0);
    if (!staged.has_value()) {
        return std::unexpected(staged.error());
    }
    const auto status = nccl_broadcast_device(ranks, count, root);
    if (!status.has_value()) {
        return std::unexpected(status.error());
    }
    copy_device_to_host(owned.front(), values.data(), count * sizeof(double));
    return {};
#else
    return {};
#endif
}

Result<void> reduce_host(std::vector<double>& values, ReduceOp op, int root) {
    const size_t size = nccl_comm_size();
    if (root < 0 || static_cast<size_t>(root) >= size) {
        return std::unexpected(
            ValueOutOfRange{"root", static_cast<double>(root), 0.0, static_cast<double>(size - 1)});
    }
    if (values.empty() || size <= 1) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    NcclClique& c = clique();
    if (!c.ensure()) {
        return std::unexpected(DeviceError{45, "nccl communicator init failed"});
    }
    const size_t count = values.size();
    std::vector<DeviceBuffer> owned;
    std::vector<DeviceBuffer*> ranks;
    const auto staged = stage_replicated(values, owned, ranks, 0);
    if (!staged.has_value()) {
        return std::unexpected(staged.error());
    }
    const auto status = nccl_reduce_device(ranks, count, op, root);
    if (!status.has_value()) {
        return std::unexpected(status.error());
    }
    const bool owns_root = (c.world_size() == 0) || (c.rank() == root);
    const size_t local_root = (c.world_size() > 0) ? 0 : static_cast<size_t>(root);
    if (owns_root && local_root < owned.size()) {
        copy_device_to_host(owned[local_root], values.data(), count * sizeof(double));
    }
    return {};
#else
    (void)op;
    return {};
#endif
}

Result<std::vector<double>> allgather_host(const std::vector<double>& values) {
    const size_t size = nccl_comm_size();
    if (values.empty() || size <= 1) {
        return values;
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    NcclClique& c = clique();
    if (!c.ensure()) {
        return std::unexpected(DeviceError{45, "nccl communicator init failed"});
    }
    const size_t count = values.size();
    std::vector<DeviceBuffer> send_owned;
    std::vector<DeviceBuffer*> send;
    const auto staged_send = stage_replicated(values, send_owned, send, 0);
    if (!staged_send.has_value()) {
        return std::unexpected(staged_send.error());
    }
    std::vector<DeviceBuffer> recv_owned;
    std::vector<DeviceBuffer*> recv;
    const auto staged_recv = stage_replicated(values, recv_owned, recv, count * (size - 1));
    if (!staged_recv.has_value()) {
        return std::unexpected(staged_recv.error());
    }
    const auto status = nccl_allgather_device(send, recv, count);
    if (!status.has_value()) {
        return std::unexpected(status.error());
    }
    std::vector<double> gathered(count * size, 0.0);
    copy_device_to_host(recv_owned.front(), gathered.data(), gathered.size() * sizeof(double));
    return gathered;
#else
    return values;
#endif
}

Result<void> allreduce_buffer(std::span<DeviceBuffer* const> ranks, size_t count, ReduceOp op) {
    const size_t bytes = count * sizeof(double);
    const auto valid = validate_ranks(ranks, bytes, "allreduce_buffer");
    if (!valid.has_value()) {
        return std::unexpected(valid.error());
    }
    if (count == 0 || (ranks.size() == 1 && nccl_comm_size() <= 1)) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (fast_path_usable(ranks)) {
        return nccl_allreduce_device(ranks, count, op);
    }
#endif
    std::vector<double> acc;
    host_accumulate(ranks, count, op, acc);
    for (DeviceBuffer* buf : ranks) {
        copy_host_to_device(acc.data(), *buf, bytes);
    }
    return {};
}

Result<void> broadcast_buffer(std::span<DeviceBuffer* const> ranks, size_t count, int root) {
    const size_t bytes = count * sizeof(double);
    const auto valid = validate_ranks(ranks, bytes, "broadcast_buffer");
    if (!valid.has_value()) {
        return std::unexpected(valid.error());
    }
    const bool fast = fast_path_usable(ranks);
    const size_t world = collective_world(ranks.size(), fast);
    if (root < 0 || static_cast<size_t>(root) >= world) {
        return std::unexpected(ValueOutOfRange{"root", static_cast<double>(root), 0.0,
                                               static_cast<double>(world - 1)});
    }
    if (count == 0 || (ranks.size() == 1 && nccl_comm_size() <= 1)) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (fast) {
        return nccl_broadcast_device(ranks, count, root);
    }
#endif
    std::vector<double> src(count, 0.0);
    copy_device_to_host(*ranks[static_cast<size_t>(root)], src.data(), bytes);
    for (size_t r = 0; r < ranks.size(); ++r) {
        if (r == static_cast<size_t>(root)) {
            continue;
        }
        copy_host_to_device(src.data(), *ranks[r], bytes);
    }
    return {};
}

Result<void> reduce_buffer(std::span<DeviceBuffer* const> ranks, size_t count, ReduceOp op,
                           int root) {
    const size_t bytes = count * sizeof(double);
    const auto valid = validate_ranks(ranks, bytes, "reduce_buffer");
    if (!valid.has_value()) {
        return std::unexpected(valid.error());
    }
    const bool fast = fast_path_usable(ranks);
    const size_t world = collective_world(ranks.size(), fast);
    if (root < 0 || static_cast<size_t>(root) >= world) {
        return std::unexpected(ValueOutOfRange{"root", static_cast<double>(root), 0.0,
                                               static_cast<double>(world - 1)});
    }
    if (count == 0 || (ranks.size() == 1 && nccl_comm_size() <= 1)) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (fast) {
        return nccl_reduce_device(ranks, count, op, root);
    }
#endif
    std::vector<double> acc;
    host_accumulate(ranks, count, op, acc);
    copy_host_to_device(acc.data(), *ranks[static_cast<size_t>(root)], bytes);
    return {};
}

Result<void> allgather_buffer(std::span<DeviceBuffer* const> send,
                              std::span<DeviceBuffer* const> recv, size_t count) {
    if (send.size() != recv.size()) {
        return std::unexpected(DimensionMismatch{send.size(), recv.size()});
    }
    const size_t send_bytes = count * sizeof(double);
    const auto send_valid = validate_ranks(send, send_bytes, "allgather_buffer");
    if (!send_valid.has_value()) {
        return std::unexpected(send_valid.error());
    }
    const bool fast = fast_path_usable(send) && fast_path_usable(recv);
    const size_t world = collective_world(send.size(), fast);
    const size_t recv_bytes = send_bytes * world;
    const auto recv_valid = validate_ranks(recv, recv_bytes, "allgather_buffer");
    if (!recv_valid.has_value()) {
        return std::unexpected(recv_valid.error());
    }
    if (count == 0) {
        return {};
    }
#if defined(MS_HAS_NCCL) && MS_HAS_NCCL
    if (fast) {
        return nccl_allgather_device(send, recv, count);
    }
#endif
    std::vector<double> gathered(count * send.size(), 0.0);
    for (size_t r = 0; r < send.size(); ++r) {
        copy_device_to_host(*send[r], &gathered[r * count], send_bytes);
    }
    for (size_t r = 0; r < recv.size(); ++r) {
        copy_host_to_device(gathered.data(), *recv[r], recv_bytes);
    }
    return {};
}

} // namespace ms::cuda
