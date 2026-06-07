#pragma once

namespace ms {

using RngNormalFn = double (*)();
using RngUniformFn = double (*)();

void set_session_rng(RngUniformFn uniform, RngNormalFn normal);
void clear_session_rng();
bool session_rng_active();
double session_uniform();
double session_normal();

} // namespace ms
