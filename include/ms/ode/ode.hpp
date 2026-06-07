#pragma once

#include <functional>
#include <vector>

namespace ms {

using OdeFunc = std::function<double(double t, double y)>;

struct OdeResult {
    std::vector<double> t;
    std::vector<double> y;
};

OdeResult ode_euler(OdeFunc f, double t0, double y0, double t_end, size_t steps);
OdeResult ode_rk4(OdeFunc f, double t0, double y0, double t_end, size_t steps);

} // namespace ms
