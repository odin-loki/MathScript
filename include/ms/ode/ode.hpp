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

/// Cash-Karp embedded Runge-Kutta (4,5) adaptive-step scalar IVP solver.
/// A 6-stage explicit RK pair that produces a 5th-order solution advanced at
/// each accepted step, together with an embedded 4th-order estimate computed
/// from the same 6 stage derivatives; the difference between the two drives
/// step-size control. Butcher tableau coefficients are the canonical
/// published values (Cash & Karp 1990 / Numerical Recipes).
/// @param f Right-hand side dy/dt = f(t, y).
/// @param t0 Initial time.
/// @param y0 Initial value y(t0).
/// @param t_end Final time. If t_end <= t0, returns immediately with the
///        single point (t0, y0) and takes no steps (same convention as
///        ode_rk45).
/// @param rtol Relative tolerance for the per-step error control.
/// @param atol Absolute tolerance for the per-step error control.
/// @return OdeResult with the accepted (t, y) pairs, starting at (t0, y0)
///         and ending at (or numerically at) t_end.
/// @note Defensive: no exceptions; a hard cap on the number of steps
///       prevents runaway loops if the step size collapses.
/// @accuracy Local error per step is O(h^5); the embedded 4th-order estimate
///           is used only for error control, not returned. Step size is
///           rescaled by a safety factor times (tol/err)^(1/5), clamped to
///           [0.2x, 5x] growth per step, mirroring ode_rk45's control loop.
OdeResult ode_cashkarp(OdeFunc f, double t0, double y0, double t_end,
                       double rtol = 1e-6, double atol = 1e-9);

// Vector ODE fixed-step
OdeResultVec ode_euler_vec(OdeFuncVec f, double t0,
                           const std::vector<double>& y0,
                           double t_end, size_t steps);
OdeResultVec ode_rk4_vec(OdeFuncVec f, double t0,
                          const std::vector<double>& y0,
                          double t_end, size_t steps);

// Adaptive vector solver (Dormand-Prince RK45)
OdeResultVec ode_rk45_vec(OdeFuncVec f, double t0,
                           const std::vector<double>& y0,
                           double t_end,
                           double rtol = 1e-6, double atol = 1e-9);

// Symplectic velocity Verlet for second-order ODE q'' = a(t, q)
using OdeAccelFunc = std::function<double(double t, double q)>;
using OdeAccelFuncVec = std::function<std::vector<double>(
    double t, const std::vector<double>& q)>;

struct OdeVerletResult {
    std::vector<double> t;
    std::vector<double> q;
    std::vector<double> v;
};

struct OdeVerletResultVec {
    std::vector<double> t;
    std::vector<std::vector<double>> q;
    std::vector<std::vector<double>> v;
};

OdeVerletResult ode_verlet(OdeAccelFunc a, double t0, double q0, double v0,
                           double t_end, size_t steps);
OdeVerletResultVec ode_verlet_vec(OdeAccelFuncVec a, double t0,
                                   const std::vector<double>& q0,
                                   const std::vector<double>& v0,
                                   double t_end, size_t steps);

// Adams-Bashforth 2-step
OdeResult ode_adams_bashforth2(OdeFunc f, double t0, double y0,
                                double t_end, size_t steps);

// Implicit backward Euler (scalar stiff IVP)
OdeResult ode_backward_euler(OdeFunc f, double t0, double y0,
                              double t_end, size_t steps);

/// BDF2 (backward differentiation formula, order 2) implicit multistep solver for stiff ODEs.
/// Solves (3*y_{n+1} - 4*y_n + y_{n-1}) / (2*h) = f(t_{n+1}, y_{n+1}) via Newton iteration
/// with a numerically-differentiated df/dy Jacobian (same technique as ode_backward_euler).
/// A-stable for stiff problems; second-order accurate in time. The first step bootstraps with
/// one step of ode_backward_euler (BDF1) because y_{n-1} is unavailable at t_0.
OdeResult ode_bdf2(OdeFunc f, double t0, double y0, double t_end, size_t steps);

// Implicit backward Euler (vector stiff IVP, Picard iteration)
OdeResultVec ode_backward_euler_vec(OdeFuncVec f, double t0,
                                     const std::vector<double>& y0,
                                     double t_end, size_t steps);

// Semi-explicit index-1 DAE: dy/dt = f(t,y,z), 0 = g(t,y,z)
using DaeDiffFunc = std::function<std::vector<double>(
    double t, const std::vector<double>& y, const std::vector<double>& z)>;
using DaeAlgFunc = std::function<std::vector<double>(
    double t, const std::vector<double>& y, const std::vector<double>& z)>;

struct DaeResult {
    std::vector<double> t;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<double>> z;
    bool converged = false;
};

DaeResult ode_dae_index1(DaeDiffFunc f, DaeAlgFunc g, double t0,
                          const std::vector<double>& y0,
                          const std::vector<double>& z0,
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
