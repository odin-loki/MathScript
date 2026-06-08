#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

#include "ms/poly/poly.hpp"

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
    return std::max(-16.0, std::min(16.0, value));
}

size_t bounded_count(uint8_t byte, size_t max_count) {
    return static_cast<size_t>(byte % static_cast<uint8_t>(max_count + 1)) + 1;
}

std::vector<double> read_coeffs(const uint8_t* data, size_t size, size_t count, size_t seed) {
    std::vector<double> coeffs;
    coeffs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t offset = (seed + i * 7) % size;
        coeffs.push_back(read_f64(data, size, offset));
    }
    return coeffs;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 24) {
        return 0;
    }

    const size_t count_a = bounded_count(data[0], 5);
    const size_t count_b = bounded_count(data[1], 5);
    const auto a = read_coeffs(data, size, count_a, 2);
    const auto b = read_coeffs(data, size, count_b, 9);
    const double x = read_f64(data, size, 16);

    (void)ms::poly_eval(a, x);
    (void)ms::poly_eval(b, x);
    (void)ms::poly_add(a, b);
    (void)ms::poly_mul(a, b);
    (void)ms::poly_deriv(a);
    (void)ms::poly_deriv(ms::poly_mul(a, b));
    return 0;
}
