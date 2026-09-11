// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions
#pragma once

#include "ms/cuda/buffer.hpp"
#include "ms/error/error_types.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

/// NCCL multi-GPU collectives (`ms/cuda/nccl.hpp`).
///
/// Build-time macro `MS_HAS_NCCL` (CMake: `-DMS_ENABLE_NCCL=ON` with CUDA and
/// an installed NCCL library):
///   - `MS_HAS_NCCL=0` (default stub build): `nccl_available()` is false,
///     `nccl_comm_size()` is 1, `nccl_local_rank_count()` is 0, and every
///     scalar collective returns its argument unchanged (single-rank /
///     host-safe identity, same as a 1-process MPI allreduce).
///   - `MS_HAS_NCCL=1`: NCCL is linked. The process then runs in one of two
///     modes.
///
/// Local mode (the default, entered lazily): the first collective calls
/// `ncclCommInitAll` over every visible GPU, so this process owns *every* rank
/// of the communicator. A scalar handed to a collective is the contribution of
/// each of those ranks (replicated-value semantics), which is why
/// `allreduce_max` / `allreduce_min` / `allreduce_avg` / `broadcast` /
/// `allgather` stay identities even with N GPUs, while `allreduce_sum` returns
/// `N*x` and `allreduce_prod` returns `x^N`.
///
/// Rank mode (explicit): `nccl_init_rank()` joins an existing multi-process
/// communicator with `ncclCommInitRank`; this process then owns exactly one of
/// `world_size` ranks and the collectives are genuine cross-process
/// reductions. The `ncclUniqueId` bytes must be produced once by
/// `nccl_make_unique_id()` on rank 0 and distributed out of band (MPI
/// broadcast, file, socket) — this header deliberately does not depend on MPI.
///
/// Communicators are owned by one process-wide RAII holder, created lazily and
/// released at process teardown or by `nccl_finalize()`. The module is not
/// thread-safe (same contract as `StreamPool` in `buffer.hpp`): drive the
/// collectives from a single thread.
///
/// Whenever the communicator size is 1 every collective is the identity, so a
/// stub build and a single-GPU build behave identically. Use
/// `nccl_available()` before assuming multi-GPU collectives are active.

namespace ms::cuda {

/// Reduction operator for the vector and device-buffer collectives.
/// `Avg` is `Sum` divided by the number of contributing ranks (`ncclAvg` when
/// the linked NCCL is >= 2.10, otherwise `ncclSum` followed by an explicit
/// scale, and a plain divide on the host-staging path).
enum class ReduceOp { Sum, Prod, Max, Min, Avg };

/// True when MathScript was built with NCCL and CUDA is usable.
bool nccl_available();

/// GPUs visible to NCCL (0 when unavailable).
int nccl_device_count();

/// Communicator size: the local GPU count in local mode, `world_size` in rank
/// mode, otherwise 1 (stub). Never 0, and never triggers communicator
/// creation. O(1) plus one `cudaGetDeviceCount`.
size_t nccl_comm_size();

/// This process's rank inside the communicator: 0 in the stub build and in
/// local mode (where this process owns every rank), the rank passed to
/// `nccl_init_rank()` in rank mode.
int nccl_rank();

/// Ranks this process itself drives: 0 when NCCL is inactive,
/// `nccl_device_count()` in local mode, 1 in rank mode. This is the number of
/// `DeviceBuffer`s the `*_buffer` collectives need for the NCCL fast path;
/// any other buffer count is served correctly by host staging instead.
size_t nccl_local_rank_count();

/// "stub", "nccl-local" or "nccl-rank"; mirrors `ms::distributed::backend_name`.
std::string nccl_backend_name();

/// A fresh `ncclUniqueId` as its 128 raw bytes, for distribution to the other
/// ranks of a multi-process communicator. Call once on rank 0 only.
/// Stub build, or an NCCL build with no usable GPU: `DeviceError{40}`.
/// O(1) plus the NCCL bootstrap-address setup.
Result<std::vector<unsigned char>> nccl_make_unique_id();

/// Joins the multi-process communicator identified by `unique_id` (exactly 128
/// bytes, produced by `nccl_make_unique_id()` on rank 0) as `rank` of
/// `world_size`, binding this process to CUDA device `device`. Replaces any
/// communicator this process already held.
///
/// Validation happens in *both* builds, before any NCCL call:
/// `unique_id.size() != 128` -> `DimensionMismatch{got, 128}`;
/// `world_size <= 0` -> `ValueOutOfRange{"world_size", ...}`;
/// `rank` outside `[0, world_size)` -> `ValueOutOfRange{"rank", ...}`.
/// Stub build: `DeviceError{41}`. O(1) plus the NCCL rendezvous, which blocks
/// until all `world_size` ranks have called in.
Result<void> nccl_init_rank(std::span<const unsigned char> unique_id, int rank, int world_size,
                            int device);

/// Destroys every communicator and stream this process owns. A no-op when NCCL
/// is unavailable and safe to call repeatedly. Afterwards local mode
/// re-initialises lazily on the next collective; rank mode does not come back
/// (call `nccl_init_rank()` again). O(local rank count).
void nccl_finalize();

/// All-reduce sum of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
/// Multi-rank path: `comm_size * value` in local mode (the scalar is the
/// contribution of every rank this process owns), the true cross-process sum
/// in rank mode.
double allreduce_sum(double value);

/// All-reduce max of a scalar across the NCCL communicator.
/// Stub / single-rank path — and all of local mode, where the scalar is
/// replicated onto every rank — returns `value` unchanged.
double allreduce_max(double value);

/// All-reduce min of a scalar across the NCCL communicator.
/// Stub / single-rank path (and all of local mode): returns `value` unchanged.
double allreduce_min(double value);

/// All-reduce product of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
/// Local mode: `value` raised to the power `nccl_comm_size()`.
double allreduce_prod(double value);

/// All-reduce average of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
/// Multi-rank path: `allreduce_sum(value) / nccl_comm_size()`.
double allreduce_avg(double value);

/// Broadcast a scalar from `root` across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged. A `root` outside
/// `[0, nccl_comm_size())` is a no-op and also returns `value` unchanged.
double broadcast(double value, int root = 0);

/// Reduce (sum-to-root) a scalar to `root` across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged; a `root` outside
/// `[0, nccl_comm_size())` likewise. Multi-rank path: the root rank returns the
/// sum, every non-root rank returns its own `value` (NCCL writes only the
/// root's receive buffer).
double reduce(double value, int root = 0);

/// All-gather a scalar across the NCCL communicator, returning the rank-0
/// element of the gathered vector. Stub / single-rank path and local mode
/// (where every rank contributes the same scalar): returns `value` unchanged.
/// Use `allgather_host` when the whole gathered vector is wanted.
double allgather(double value);

/// In-place all-reduce of a host vector, whose contents are the contribution of
/// every rank this process owns. `values` empty or `nccl_comm_size() <= 1`:
/// no-op success (identity), `values` is not touched. O(values.size()) host
/// traffic plus one NCCL collective.
Result<void> allreduce_host(std::vector<double>& values, ReduceOp op = ReduceOp::Sum);

/// In-place broadcast of a host vector from `root`. `root` outside
/// `[0, nccl_comm_size())` -> `ValueOutOfRange{"root", ...}` in every build,
/// checked before the size test. `values` empty or comm size 1: no-op success.
Result<void> broadcast_host(std::vector<double>& values, int root = 0);

/// Reduce a host vector to `root`, with the same root validation as
/// `broadcast_host`. On a process that does not own `root`, `values` is left
/// unchanged. `values` empty or comm size 1: no-op success.
Result<void> reduce_host(std::vector<double>& values, ReduceOp op = ReduceOp::Sum, int root = 0);

/// All-gather a host vector: returns `values.size() * nccl_comm_size()` doubles
/// in rank order. `values` empty or comm size 1: returns a copy of `values`.
Result<std::vector<double>> allgather_host(const std::vector<double>& values);

/// In-place all-reduce of `count` doubles across one buffer per participating
/// rank, where `ranks[i]` holds rank `i`'s contribution. Every buffer receives
/// the result.
///
/// Runs on NCCL when NCCL is active, `ranks.size() == nccl_local_rank_count()`
/// and `ranks[i]->device` is the i-th clique device; otherwise it stages
/// through the host and computes the identical result, so the operation is real
/// in every build (in a CUDA-less build the buffers are host memory anyway).
///
/// Degenerate cases: `ranks` empty -> `DomainError{"allreduce_buffer",
/// "empty rank list"}`; any null `ranks[i]` or null `ranks[i]->data` ->
/// `DomainError{..., "null rank buffer"}`; any `ranks[i]->bytes <
/// count*sizeof(double)` -> `DimensionMismatch{bytes, count*sizeof(double)}`;
/// `count == 0` -> success with no write; a single rank in a size-1
/// communicator -> success with no write (a single-rank all-reduce, `Avg`
/// included, is the identity). O(count * ranks.size()).
Result<void> allreduce_buffer(std::span<DeviceBuffer* const> ranks, size_t count,
                              ReduceOp op = ReduceOp::Sum);

/// In-place broadcast of `count` doubles from the root rank's buffer to every
/// other rank. On the NCCL fast path `root` is a global rank index in
/// `[0, nccl_comm_size())`; on the host-staging path it is the index into
/// `ranks`, so the valid range is `[0, ranks.size())`. Same buffer validation
/// as `allreduce_buffer`, plus `root` out of that range ->
/// `ValueOutOfRange{"root", ...}`. O(count * ranks.size()).
Result<void> broadcast_buffer(std::span<DeviceBuffer* const> ranks, size_t count, int root = 0);

/// Reduce `count` doubles to the root rank's buffer only; every other buffer is
/// left untouched. Same validation and same `root` convention as
/// `broadcast_buffer`. O(count * ranks.size()).
Result<void> reduce_buffer(std::span<DeviceBuffer* const> ranks, size_t count,
                           ReduceOp op = ReduceOp::Sum, int root = 0);

/// All-gather: concatenates `count` doubles from each `send[i]`, in rank order,
/// into every `recv[i]`, each of which must hold `count * world` doubles
/// (`world` is `nccl_comm_size()` on the NCCL fast path and `send.size()`
/// otherwise). `send.size() != recv.size()` -> `DimensionMismatch{send.size(),
/// recv.size()}`; the usual empty / null / short-buffer errors otherwise;
/// `count == 0` -> success with no write.
/// O(count * send.size() * recv.size()).
Result<void> allgather_buffer(std::span<DeviceBuffer* const> send,
                              std::span<DeviceBuffer* const> recv, size_t count);

} // namespace ms::cuda
