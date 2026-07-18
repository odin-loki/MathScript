#pragma once

#include <cstddef>

/// NCCL multi-GPU collectives (`ms/cuda/nccl.hpp`).
///
/// Build-time macro `MS_HAS_NCCL` (CMake: `-DMS_ENABLE_NCCL=ON` with CUDA and
/// an installed NCCL library):
///   - `MS_HAS_NCCL=0` (default stub build): `nccl_available()` is false,
///     `nccl_comm_size()` is 1, and `allreduce_sum` / `allreduce_max` /
///     `allreduce_min` / `allreduce_prod` / `allreduce_avg` return `x`
///     unchanged (single-rank /
///     host-safe identity,
///     same as a 1-process MPI allreduce).
///   - `MS_HAS_NCCL=1`: NCCL is linked; availability follows CUDA runtime
///     visibility and `nccl_comm_size()` reflects the local GPU count.
///     When the communicator size is 1, collectives remain identity.
///
/// Use `nccl_available()` before assuming multi-GPU collectives are active.

namespace ms::cuda {

/// True when MathScript was built with NCCL and CUDA is usable.
bool nccl_available();

/// GPUs visible to NCCL (0 when unavailable).
int nccl_device_count();

/// Communicator size: local GPU count when NCCL is active, otherwise 1 (stub).
size_t nccl_comm_size();

/// All-reduce sum of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
double allreduce_sum(double value);

/// All-reduce max of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
double allreduce_max(double value);

/// All-reduce min of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
double allreduce_min(double value);

/// All-reduce product of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
double allreduce_prod(double value);

/// All-reduce average of a scalar across the NCCL communicator.
/// Stub / single-rank path: returns `value` unchanged.
/// Multi-GPU path: `allreduce_sum(value) / nccl_comm_size()`.
double allreduce_avg(double value);

} // namespace ms::cuda
