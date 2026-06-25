#pragma once

#include <functional>
#include <vector>

namespace ms {

using OdeFunc    = std::function<double(double t, double y)>;
using OdeFuncVec = std::function<std::vector<double>(
    double t, const std::vector<double>& y)>;

struct OdeResult {
    std::vector<double> t;
    std::vector<double> y;
};

struct OdeResultVec {
    std::vector<double> t;
    std::vector<std::vector<double>> y;
};

// Fixed-step scalar solvers
OdeResult ode_euler(OdeFunc f, double t0, double y0, double t_end, size_t steps);
OdeResult ode_rk4(OdeFunc f, double t0, double y0, double t_end, size_t steps);
OdeResult ode_midpoint(OdeFunc f, double t0, double y0, double t_end, size_t steps);
OdeResult ode_rk2(OdeFunc f, double t0, double y0, double t_end, size_t steps);

// Adaptive scalar solvers (Dormand-Prince RK45)
OdeResult ode_rk45(OdeFunc f, double t0, double y0, double t_end,
                   double rtol = 1e-6, double atol = 1e-9);

// Adaptive scalar solver (Bogacki-Shampine RK23)
OdeResult ode_rk23(OdeFunc f, double t0, double y0, double t_end,
                   double rtol = 1e-4, double atol = 1e-7);

// Vector ODE fixed-step
OdeResultVec ode_euler_vec(OdeFuncVec f, double t0,
                           const std::vector<double>& y0,
                           double t_end, size_t steps);
OdeResultVec ode_rk4_vec(OdeFuncVec f, double t0,
                          const std::vector<double>& y0,
                          double t_end, size_t steps);

// Adams-Bashforth 2-step
OdeResult ode_adams_bashforth2(OdeFunc f, double t0, double y0,
                                double t_end, size_t steps);

} // namespace ms
