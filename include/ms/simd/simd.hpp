#pragma once

#include "ms/simd/isa.hpp"
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ms::simd {

enum class Kernel : uint8_t { Scalar, Avx2, Avx512 };

struct DispatchInfo {
    Kernel active = Kernel::Scalar;
    IsaFeatures isa;
};

DispatchInfo dispatch_info();

void add(std::span<const double> a, std::span<const double> b, std::span<double> out);
void mul(std::span<const double> a, std::span<const double> b, std::span<double> out);
void scale(double alpha, std::span<const double> x, std::span<double> out);
void axpy(double alpha, std::span<const double> x, std::span<double> y);
double dot(std::span<const double> a, std::span<const double> b);
double sum(std::span<const double> x);
double sum_squares(std::span<const double> x);
void exp_map(std::span<const double> x, std::span<double> out);

std::vector<double> gemv(
    std::span<const double> A,
    size_t rows,
    size_t cols,
    std::span<const double> x);

} // namespace ms::simd
