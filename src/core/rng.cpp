#include "ms/core/rng.hpp"

#include <cmath>
#include <limits>

namespace ms {

namespace {

RngUniformFn g_uniform = nullptr;
RngNormalFn g_normal = nullptr;

} // namespace

void set_session_rng(RngUniformFn uniform, RngNormalFn normal) {
    g_uniform = uniform;
    g_normal = normal;
}

void clear_session_rng() {
    g_uniform = nullptr;
    g_normal = nullptr;
}

bool session_rng_active() {
    return g_uniform != nullptr && g_normal != nullptr;
}

double session_uniform() {
    return g_uniform ? g_uniform() : 0.0;
}

double session_normal() {
    return g_normal ? g_normal() : 0.0;
}

double session_exponential(double rate) {
    if (rate <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // 1.0 - u lands in (0, 1] when session_uniform() is in its conventional
    // [0, 1) range, so log() never sees a non-positive argument.
    return -std::log(1.0 - session_uniform()) / rate;
}

} // namespace ms
