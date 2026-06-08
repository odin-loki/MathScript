#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

namespace {

double read_f64(const uint8_t* data, size_t size, size_t offset) {
    if (offset + sizeof(double) > size) {
        return 0.0;
    }
    double value = 0.0;
    std::memcpy(&value, data + offset, sizeof(double));
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return value;
}

size_t bounded_dim(uint8_t byte, size_t max_dim) {
    return static_cast<size_t>(byte % static_cast<uint8_t>(max_dim + 1));
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 24) {
        return 0;
    }

    const size_t rows = bounded_dim(data[0], 4) + 1;
    const size_t cols = bounded_dim(data[1], 4) + 1;

    ms::Matrix<double> a(rows, cols);
    ms::Matrix<double> b(cols, rows);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const size_t offset = 2 + ((r * cols + c) % 8) * sizeof(double);
            a(r, c) = read_f64(data, size, offset);
        }
    }
    for (size_t r = 0; r < cols; ++r) {
        for (size_t c = 0; c < rows; ++c) {
            const size_t offset = 10 + ((r * rows + c) % 8) * sizeof(double);
            b(r, c) = read_f64(data, size, offset);
        }
    }

    (void)ms::matmul(a, b);
    (void)ms::norm(a);
    (void)ms::trace(a);
    if (rows == cols) {
        (void)ms::det(a);
        (void)ms::lu(a);
    }
    return 0;
}
