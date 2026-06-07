// MathScript Dispatch Layer Header

#pragma once

#include "ms/error/error_types.hpp"

namespace ms::runtime {

enum class Backend { CPU, CUDA };

struct DispatchDecision {
    Backend backend;
    int n_threads;
    int cuda_device;
    int cuda_stream;
};

enum class OpClass { General, DenseMatmul, SparseMatmul, FFT, SpecialFunction };

enum class ExecPolicy { CPU, GPU, AUTO };

DispatchDecision decide(size_t n, OpClass op, ExecPolicy policy,
                        const SystemTopology& topo);

std::expected<void, Error> execute_decision(DispatchDecision& decision);

} // namespace ms::runtime