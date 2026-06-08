#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>

#include "ms/core/scalar.hpp"

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
    return std::max(-1e6, std::min(1e6, value));
}

std::string pick_unit(uint8_t byte) {
    switch (byte % 4) {
    case 0:
        return "m";
    case 1:
        return "s";
    case 2:
        return "kg";
    default:
        return "";
    }
}

std::string pick_prefix(uint8_t byte) {
    switch (byte % 3) {
    case 0:
        return "k";
    case 1:
        return "m";
    default:
        return "";
    }
}

ms::Scalar make_scalar(const uint8_t* data, size_t size, size_t seed) {
    const size_t value_offset = seed % size;
    const size_t unit_offset = (seed + 3) % size;
    const size_t prefix_offset = (seed + 5) % size;
    return ms::Scalar(
        read_f64(data, size, value_offset),
        pick_unit(data[unit_offset]),
        pick_prefix(data[prefix_offset]));
}

double safe_divisor(double value) {
    if (std::abs(value) < 1e-6) {
        return value >= 0.0 ? 1.0 : -1.0;
    }
    return value;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) {
        return 0;
    }

    const auto a = make_scalar(data, size, 0);
    const auto b = make_scalar(data, size, 8);
    const auto c = make_scalar(data, size, 16);

    (void)(a + b);
    (void)(a - b);
    (void)(a * b);
    (void)(a / ms::Scalar(safe_divisor(b.value()), b.unit()));

    (void)(b + c);
    (void)(b * c);
    (void)(c - a);

    const ms::Scalar edge_cases[] = {
        ms::Scalar(0.0),
        ms::Scalar(1e-12, "m"),
        ms::Scalar(-1e-12, "s"),
        ms::Scalar(1e6, "kg", "k"),
        ms::Scalar(-1e6, "", "m"),
    };
    for (const auto& edge : edge_cases) {
        (void)(edge + a);
        (void)(edge * b);
        (void)(edge / ms::Scalar(safe_divisor(c.value())));
    }

    (void)a.value();
    (void)a.unit();
    (void)a.full_unit();
    return 0;
}
