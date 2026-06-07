#include "ms/core/rng.hpp"

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

} // namespace ms
