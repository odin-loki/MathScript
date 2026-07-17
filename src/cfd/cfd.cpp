#include "ms/cfd/cfd.hpp"

#include <algorithm>
#include <cmath>

namespace ms {
namespace cfd {

namespace {

double face_velocity(std::span<const double> v, std::size_t left_cell, std::size_t n) {
    if (v.size() == 1) {
        return v[0];
    }
    if (v.size() != n) {
        return 0.0;
    }
    if (left_cell + 1 < n) {
        return 0.5 * (v[left_cell] + v[left_cell + 1]);
    }
    return v[n - 1];
}

double sample_cell(std::span<const double> u, int idx, BoundaryCondition bc) {
    const int n = static_cast<int>(u.size());
    if (n <= 0) {
        return 0.0;
    }
    if (idx >= 0 && idx < n) {
        return u[static_cast<std::size_t>(idx)];
    }
    if (bc == BoundaryCondition::Periodic) {
        const int wrapped = ((idx % n) + n) % n;
        return u[static_cast<std::size_t>(wrapped)];
    }
    if (idx < 0) {
        return u.front();
    }
    return u.back();
}

double upwind_face_flux(
    std::span<const double> u,
    std::span<const double> v,
    int face_idx,
    std::size_t n,
    BoundaryCondition bc) {
    if (bc == BoundaryCondition::ZeroFlux &&
        (face_idx <= 0 || face_idx >= static_cast<int>(n))) {
        return 0.0;
    }

    const std::size_t left_cell =
        face_idx <= 0 ? 0 : static_cast<std::size_t>(face_idx - 1);
    const double vf = face_velocity(v, left_cell, n);
    const double u_up =
        (vf >= 0.0) ? sample_cell(u, face_idx - 1, bc) : sample_cell(u, face_idx, bc);
    return vf * u_up;
}

double max_wave_speed(std::span<const double> v) {
    if (v.empty()) {
        return 0.0;
    }
    double max_speed = 0.0;
    for (double vi : v) {
        max_speed = std::max(max_speed, std::abs(vi));
    }
    return max_speed;
}

} // namespace

Grid1D grid1d(double x0, double x1, std::size_t n) {
    Grid1D grid;
    if (n < 2 || x1 <= x0) {
        return grid;
    }

    grid.x0 = x0;
    grid.x1 = x1;
    grid.n = n;
    grid.dx = (x1 - x0) / static_cast<double>(n);
    grid.x.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        grid.x[i] = x0 + (static_cast<double>(i) + 0.5) * grid.dx;
    }
    return grid;
}

std::vector<double> square_pulse(
    const Grid1D& grid,
    double xc,
    double width,
    double amplitude) {
    std::vector<double> u(grid.n, 0.0);
    if (grid.n == 0 || width <= 0.0) {
        return u;
    }

    const double half = 0.5 * width;
    for (std::size_t i = 0; i < grid.n; ++i) {
        const double x = grid.x[i];
        if (std::abs(x - xc) <= half) {
            u[i] = amplitude;
        }
    }
    return u;
}

std::vector<double> constant_velocity(std::size_t n, double v) {
    return std::vector<double>(n, v);
}

std::vector<double> upwind_fvm_advection(
    std::span<const double> u,
    std::span<const double> v,
    double dt,
    double dx,
    BoundaryCondition bc) {
    if (u.empty() || v.empty() || dt <= 0.0 || dx <= 0.0) {
        return {};
    }
    if (v.size() != 1 && v.size() != u.size()) {
        return {};
    }

    const std::size_t n = u.size();
    const double max_speed = max_wave_speed(v);
    if (max_speed * dt / dx > 1.0) {
        return {};
    }

    std::vector<double> next(n);
    const double scale = dt / dx;
    for (std::size_t i = 0; i < n; ++i) {
        const double flux_right = upwind_face_flux(u, v, static_cast<int>(i) + 1, n, bc);
        const double flux_left = upwind_face_flux(u, v, static_cast<int>(i), n, bc);
        next[i] = u[i] - scale * (flux_right - flux_left);
    }
    return next;
}

AdvectionResult run_advection(
    const std::vector<double>& u0,
    const std::vector<double>& v,
    double t_end,
    double dt,
    double dx,
    BoundaryCondition bc) {
    AdvectionResult result;
    if (u0.empty() || v.empty() || t_end <= 0.0 || dt <= 0.0 || dx <= 0.0) {
        return result;
    }
    if (v.size() != 1 && v.size() != u0.size()) {
        return result;
    }

    std::vector<double> u = u0;
    result.t.push_back(0.0);
    result.u.push_back(u);

    double t = 0.0;
    while (t < t_end - 1e-15) {
        const double step = std::min(dt, t_end - t);
        std::vector<double> next = upwind_fvm_advection(u, v, step, dx, bc);
        if (next.empty()) {
            return {};
        }
        u = std::move(next);
        t += step;
        result.t.push_back(t);
        result.u.push_back(u);
    }

    return result;
}

double integrated_mass(std::span<const double> u, double dx) {
    if (u.empty() || dx <= 0.0) {
        return 0.0;
    }
    double sum = 0.0;
    for (double ui : u) {
        sum += ui;
    }
    return sum * dx;
}

} // namespace cfd
} // namespace ms
