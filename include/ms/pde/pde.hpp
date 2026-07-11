#pragma once

#include <cstddef>
#include <vector>

namespace ms {

struct Heat1DResult {
    std::vector<double> x;
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

struct Heat2DResult {
    std::vector<double> t;
    std::vector<std::vector<std::vector<double>>> u;
};

struct Wave1DResult {
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

struct Advection1DResult {
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

struct Poisson2DResult {
    std::vector<std::vector<double>> u;
    std::size_t iterations = 0;
    bool converged = false;
};

struct Burgers1DResult {
    std::vector<double> t;
    std::vector<std::vector<double>> u;
};

struct Poisson1DResult {
    std::vector<double> u;
};

struct Wave2DResult {
    std::vector<double> t;
    std::vector<std::vector<std::vector<double>>> u;
};

Heat1DResult pde_heat_1d(
    const std::vector<double>& x0,
    double alpha,
    double dx,
    double dt,
    std::size_t steps);

/// Crank-Nicolson (implicit) finite-difference solver for the 1D heat equation u_t = alpha u_xx
/// with fixed zero Dirichlet boundaries. Unconditionally stable; no restriction on r = alpha*dt/dx^2.
/// @note Returns empty result when inputs are invalid.
/// Complexity: O(steps * n).
Heat1DResult pde_heat_1d_cn(
    const std::vector<double>& x0,
    double alpha,
    double dx,
    double dt,
    std::size_t steps);

/// Explicit FTCS finite-difference solver for the 2D heat equation u_t = alpha Laplacian(u)
/// with Dirichlet zero boundaries. Stability requires alpha*dt*(1/dx^2+1/dy^2) <= 0.5.
/// @note Returns empty result when the stability condition is violated or inputs are invalid.
/// Complexity: O(steps * nx * ny).
Heat2DResult pde_heat_2d(
    const std::vector<std::vector<double>>& u0,
    double alpha,
    double dx,
    double dy,
    double dt,
    std::size_t steps);

/// Explicit leapfrog (central-difference) solver for the 1D wave equation u_tt = c^2 u_xx
/// with fixed zero Dirichlet boundaries. CFL stability requires c*dt/dx <= 1.
/// @note Returns empty result when CFL is violated or inputs are invalid.
/// Complexity: O(steps * n).
Wave1DResult pde_wave_1d(
    const std::vector<double>& u0,
    const std::vector<double>& v0,
    double c,
    double dx,
    double dt,
    std::size_t steps);

/// Explicit leapfrog (central-difference) solver for the 2D wave equation u_tt = c^2 Laplacian(u)
/// with fixed zero Dirichlet boundaries. CFL stability requires c^2*dt^2*(1/dx^2+1/dy^2) <= 1.
/// @note Returns empty result when CFL is violated or inputs are invalid.
/// Complexity: O(steps * nx * ny).
Wave2DResult pde_wave_2d(
    const std::vector<std::vector<double>>& u0,
    const std::vector<std::vector<double>>& v0,
    double c,
    double dx,
    double dy,
    double dt,
    std::size_t steps);

/// First-order upwind finite-difference solver for u_t + v u_x = 0 with periodic boundaries.
/// CFL stability requires |v|*dt/dx <= 1.
/// @note Returns empty result when CFL is violated or inputs are invalid.
/// Complexity: O(steps * n).
Advection1DResult pde_advection_1d(
    const std::vector<double>& u0,
    double v,
    double dx,
    double dt,
    std::size_t steps);

/// Direct tridiagonal finite-difference solve for the 1D Poisson equation u'' = f(x) on [0, L]
/// with Dirichlet boundaries u(0)=ua, u(L)=ub. f is sampled at all grid points (including boundaries).
/// @note Returns empty result when the grid is too small or inputs are invalid.
/// Complexity: O(n).
Poisson1DResult pde_poisson_1d(
    const std::vector<double>& f,
    double dx,
    double ua,
    double ub);

/// Gauss-Seidel iterative finite-difference solver for the 2D Poisson equation
/// Laplacian(u) = f with Dirichlet zero boundaries. Iterates until the maximum absolute
/// change per sweep falls below tolerance or max_iterations is reached.
/// @note Returns empty result when the grid is too small or inputs are invalid.
/// Complexity: O(max_iterations * nx * ny).
Poisson2DResult pde_poisson_2d(
    const std::vector<std::vector<double>>& f,
    double dx,
    double dy,
    std::size_t max_iterations,
    double tolerance);

/// Explicit finite-difference solver for the viscous Burgers equation
/// u_t + u u_x = nu u_xx with upwind advection and central diffusion, fixed boundaries.
/// @note Returns empty result when inputs are invalid.
/// Complexity: O(steps * n).
Burgers1DResult pde_burgers_1d(
    const std::vector<double>& u0,
    double nu,
    double dx,
    double dt,
    std::size_t steps);

} // namespace ms
