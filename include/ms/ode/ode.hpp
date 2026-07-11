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

// Implicit backward Euler (scalar stiff IVP)
OdeResult ode_backward_euler(OdeFunc f, double t0, double y0,
                              double t_end, size_t steps);

// Second-order BVP via shooting: y'' = f(t, y, y')
using OdeBvpFunc = std::function<double(double t, double y, double yp)>;

struct OdeBvpResult {
    std::vector<double> t;
    std::vector<double> y;
    std::vector<double> yp;
    bool converged = false;
    size_t iterations = 0;
};

OdeBvpResult ode_bvp_shooting(OdeBvpFunc f, double t0, double y_a,
                               double t_end, double y_b, size_t steps);

// Scalar delay differential equation (fixed-step RK4)
OdeResult ode_dde_fixed_step(std::function<double(double, double, double)> f,
                              std::function<double(double)> history,
                              double t0, double t_end, double tau,
                              size_t steps);

// Scalar IVP with root-crossing event detection (RK4 + bisection)
struct OdeEventResult {
    std::vector<double> t;
    std::vector<double> y;
    std::vector<double> event_times;
    std::vector<double> event_values;
};

OdeEventResult ode_event_detect(OdeFunc f, OdeFunc event_g,
                                 double t0, double y0, double t_end,
                                 size_t steps);

} // namespace ms
