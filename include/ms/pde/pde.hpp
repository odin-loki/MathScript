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

struct Helmholtz2DResult {
    std::vector<std::vector<double>> u;
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

/// Lax-Wendroff (second-order predictor-corrector) finite-difference solver for
/// u_t + v u_x = 0 with periodic boundaries. Adds a central-difference advection term to a
/// numerical-diffusion correction term proportional to (v*dt/dx)^2, giving second-order
/// accuracy in both space and time versus the first-order upwind scheme of pde_advection_1d.
/// CFL stability requires |v|*dt/dx <= 1.
/// @param u0 Initial condition sampled on a periodic grid; requires at least 3 points.
/// @param v Constant advection velocity.
/// @param dx Grid spacing.
/// @param dt Time step.
/// @param steps Number of time steps to advance.
/// @return Time series of solution snapshots (t and u), one entry per step plus the initial
///     condition.
/// @accuracy O(dx^2, dt^2), compared to O(dx, dt) for pde_advection_1d. For smooth profiles
///     this yields substantially lower error at equal resolution; near sharp gradients it can
///     introduce small dispersive (non-monotone) oscillations that upwind's numerical
///     diffusion avoids.
/// @note Returns empty result when CFL is violated or inputs are invalid.
/// Complexity: O(steps * n).
Advection1DResult pde_advection_1d_lax_wendroff(
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

/// Direct second-order central finite-difference solve for the 2D Helmholtz equation
/// Laplacian(u) + k^2*u = f with Dirichlet boundaries prescribed by g on the grid
/// perimeter (interior entries of g are ignored). When g is empty, zero Dirichlet
/// boundaries are used. The discrete 5-point Laplacian stencil is assembled over
/// interior unknowns and solved as a dense linear system.
/// @note Returns empty result when the grid is too small or inputs are invalid.
/// @note Known limitation: for k^2 near an eigenvalue of the negative discrete
///     Laplacian on the same grid (resonance), the linear system becomes
///     ill-conditioned or singular and the solution may be unreliable.
/// Complexity: O((nx*ny)^3) for the dense solve.
Helmholtz2DResult pde_helmholtz_2d(
    const std::vector<std::vector<double>>& f,
    double k,
    double dx,
    double dy,
    const std::vector<std::vector<double>>& g = {});

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

/// Explicit FTCS + forward-Euler solver for the Fisher-KPP reaction-diffusion equation
/// u_t = D u_xx + r u (1 - u) on a 1D grid. Models population growth with logistic
/// saturation (reaction term) coupled to spatial spreading (diffusion). Zero-flux
/// Neumann boundaries (du/dx = 0) are enforced via ghost-point reflection.
/// Diffusion stability requires D*dt/dx^2 <= 0.5.
/// @note Returns empty result when the stability condition is violated or inputs are invalid.
/// Complexity: O(steps * n).
Heat1DResult pde_reaction_diffusion_1d(
    const std::vector<double>& u0,
    double D,
    double r,
    double dx,
    double dt,
    std::size_t steps);

/// Peaceman-Rachford ADI Crank-Nicolson solver for the 2D heat equation
/// u_t = alpha Laplacian(u) with zero Dirichlet boundaries. Each timestep splits into
/// two half-steps: implicit in x / explicit in y, then implicit in y / explicit in x,
/// solving tridiagonal systems along rows and columns via the Thomas algorithm.
/// Unconditionally stable (no CFL restriction on dt); second-order accurate in time and space.
/// @note Returns empty result when inputs are invalid.
/// Complexity: O(steps * nx * ny).
Heat2DResult pde_heat_2d_cn_adi(
    const std::vector<std::vector<double>>& u0,
    double alpha,
    double dx,
    double dy,
    double dt,
    std::size_t steps);

} // namespace ms
