#pragma once

namespace ms {

using RngNormalFn = double (*)();
using RngUniformFn = double (*)();

void set_session_rng(RngUniformFn uniform, RngNormalFn normal);
void clear_session_rng();
bool session_rng_active();
double session_uniform();
double session_normal();

// Exponentially-distributed draw (mean 1/rate) via inverse-CDF transform on
// session_uniform(). Mirrors session_uniform()'s no-active-RNG fallback (i.e.
// if no session RNG is set, this returns -std::log(1.0) / rate == 0.0, same
// as session_uniform() falling back to 0.0). Returns NaN for rate <= 0.
double session_exponential(double rate = 1.0);

} // namespace ms
