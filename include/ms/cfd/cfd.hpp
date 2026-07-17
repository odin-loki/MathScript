#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ms {
namespace cfd {

enum class BoundaryCondition {
    Periodic,
    ZeroFlux,
};

struct Grid1D {
    double x0 = 0.0;
    double x1 = 0.0;
    double dx = 0.0;
    std::size_t n = 0;
    std::vector<double> x;
};

struct AdvectionResult {
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

/// Uniform 1D finite-volume grid with @p n cell-centered nodes on [x0, x1].
/// @note Returns an empty grid when n < 2 or x1 <= x0.
Grid1D grid1d(double x0, double x1, std::size_t n);

/// Square pulse initial condition centered at @p xc with total width @p width.
std::vector<double> square_pulse(
    const Grid1D& grid,
    double xc,
    double width,
    double amplitude = 1.0);

/// Constant velocity field (length @p n, every entry @p v).
std::vector<double> constant_velocity(std::size_t n, double v);

/// One explicit Euler finite-volume upwind step for du/dt + v du/dx = 0.
/// @param u Cell-averaged field (length n).
/// @param v Velocity field: either one constant value or length n (cell-centered).
/// @param dt Time step.
/// @param dx Cell width.
/// @param bc Periodic or zero-flux (Neumann) boundaries.
/// @return Updated field, or empty on invalid input / CFL violation (|v|max*dt/dx > 1).
/// @note Complexity: O(n).
std::vector<double> upwind_fvm_advection(
    std::span<const double> u,
    std::span<const double> v,
    double dt,
    double dx,
    BoundaryCondition bc = BoundaryCondition::Periodic);

/// Time integration of du/dt + v du/dx = 0 from t = 0 to @p t_end using step @p dt.
/// Uses the last step size min(dt, t_end - t) when t_end is not a multiple of dt.
/// @note Returns empty result when inputs are invalid or any step fails CFL.
AdvectionResult run_advection(
    const std::vector<double>& u0,
    const std::vector<double>& v,
    double t_end,
    double dt,
    double dx,
    BoundaryCondition bc = BoundaryCondition::Periodic);

/// Discrete mass integral sum(u) * dx.
double integrated_mass(std::span<const double> u, double dx);

} // namespace cfd
} // namespace ms
