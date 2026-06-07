#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

#include "ms/special/special.hpp"

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

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) {
        return 0;
    }

    const double x = read_f64(data, size, 0);
    const double s = std::abs(read_f64(data, size, 8)) + 1.5;
    const double q = std::min(std::abs(read_f64(data, size, 16)), 0.49);
    const double z = std::min(std::abs(read_f64(data, size, 24)), 0.49);

    (void)ms::erf(x);
    (void)ms::gamma_func(s);
    (void)ms::bessel_j0(x);
    (void)ms::zeta(s);
    (void)ms::polylog(2, std::max(-0.99, std::min(0.99, x)));
    (void)ms::theta3(z, q);
    (void)ms::heun_c(0.1, 0.2, 0.3, 0.4, 0.5, std::min(z, 0.15));
    (void)ms::painleve1(std::min(std::abs(x), 1.0), 0.0, 0.0);
    return 0;
}
