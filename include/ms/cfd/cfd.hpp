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

struct Grid2D {
    double x0 = 0.0;
    double x1 = 0.0;
    double y0 = 0.0;
    double y1 = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::vector<double> x;
    std::vector<double> y;
};

struct AdvectionResult {
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

struct Advection2DResult {
    std::vector<double> t;
    std::vector<std::vector<std::vector<double>>> u;
};

/// Uniform 1D finite-volume grid with @p n cell-centered nodes on [x0, x1].
/// @note Returns an empty grid when n < 2 or x1 <= x0.
Grid1D grid1d(double x0, double x1, std::size_t n);

/// Uniform 2D finite-volume grid with @p nx by @p ny cell-centered nodes on
/// [x0, x1] x [y0, y1].
/// @note Returns an empty grid when nx or ny < 2 or x1 <= x0 or y1 <= y0.
Grid2D grid2d(
    double x0,
    double x1,
    double y0,
    double y1,
    std::size_t nx,
    std::size_t ny);

/// Square pulse initial condition centered at @p xc with total width @p width.
std::vector<double> square_pulse(
    const Grid1D& grid,
    double xc,
    double width,
    double amplitude = 1.0);

/// Axis-aligned square pulse on a 2D grid centered at (@p xc, @p yc).
std::vector<std::vector<double>> square_pulse_2d(
    const Grid2D& grid,
    double xc,
    double yc,
    double width_x,
    double width_y,
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

/// One explicit Euler finite-volume upwind step for
/// du/dt + vx du/dx + vy du/dy = 0 on a structured grid.
/// @param u Cell-averaged field (ny rows, nx columns).
/// @param vx x-velocity: one constant value or length nx*ny (cell-centered, row-major).
/// @param vy y-velocity: one constant value or length nx*ny (cell-centered, row-major).
/// @return Updated field, or empty on invalid input / CFL violation
///         (|vx|max*dt/dx + |vy|max*dt/dy > 1).
/// @note Complexity: O(nx * ny).
std::vector<std::vector<double>> upwind_fvm_advection_2d(
    const std::vector<std::vector<double>>& u,
    std::span<const double> vx,
    std::span<const double> vy,
    double dt,
    double dx,
    double dy,
    BoundaryCondition bc_x = BoundaryCondition::Periodic,
    BoundaryCondition bc_y = BoundaryCondition::Periodic);

/// Time integration of du/dt + vx du/dx + vy du/dy = 0 from t = 0 to @p t_end.
/// Uses the last step size min(dt, t_end - t) when t_end is not a multiple of dt.
/// @note Returns empty result when inputs are invalid or any step fails CFL.
Advection2DResult run_advection_2d(
    const std::vector<std::vector<double>>& u0,
    std::span<const double> vx,
    std::span<const double> vy,
    double t_end,
    double dt,
    double dx,
    double dy,
    BoundaryCondition bc_x = BoundaryCondition::Periodic,
    BoundaryCondition bc_y = BoundaryCondition::Periodic);

/// Discrete mass integral sum(u) * dx.
double integrated_mass(std::span<const double> u, double dx);

/// Discrete mass integral sum(u) * dx * dy on a structured 2D field.
double integrated_mass_2d(
    const std::vector<std::vector<double>>& u,
    double dx,
    double dy);

} // namespace cfd
} // namespace ms
