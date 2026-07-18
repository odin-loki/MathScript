#include "ms/cfd/cfd.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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

bool upwind_fvm_advection_into(
    std::span<const double> u,
    std::span<const double> v,
    double dt,
    double dx,
    BoundaryCondition bc,
    std::span<double> out) {
    if (u.empty() || v.empty() || dt <= 0.0 || dx <= 0.0 || out.size() != u.size()) {
        return false;
    }
    if (v.size() != 1 && v.size() != u.size()) {
        return false;
    }

    const std::size_t n = u.size();
    const double max_speed = max_wave_speed(v);
    if (max_speed * dt / dx > 1.0) {
        return false;
    }

    const double scale = dt / dx;
    for (std::size_t i = 0; i < n; ++i) {
        const double flux_right = upwind_face_flux(u, v, static_cast<int>(i) + 1, n, bc);
        const double flux_left = upwind_face_flux(u, v, static_cast<int>(i), n, bc);
        out[i] = u[i] - scale * (flux_right - flux_left);
    }
    return true;
}

bool is_rectangular_grid(const std::vector<std::vector<double>>& u) {
    if (u.empty()) {
        return false;
    }
    const std::size_t nx = u[0].size();
    if (nx == 0) {
        return false;
    }
    for (const auto& row : u) {
        if (row.size() != nx) {
            return false;
        }
    }
    return true;
}

double cell_component(
    std::span<const double> field,
    std::size_t nx,
    std::size_t ny,
    std::size_t i,
    std::size_t j) {
    if (field.size() == 1) {
        return field[0];
    }
    if (field.size() != nx * ny) {
        return 0.0;
    }
    return field[j * nx + i];
}

double face_velocity_x(
    std::span<const double> vx,
    std::size_t nx,
    std::size_t ny,
    std::size_t left_cell,
    std::size_t j) {
    if (vx.size() == 1) {
        return vx[0];
    }
    if (vx.size() != nx * ny) {
        return 0.0;
    }
    if (left_cell + 1 < nx) {
        return 0.5 * (cell_component(vx, nx, ny, left_cell, j)
                      + cell_component(vx, nx, ny, left_cell + 1, j));
    }
    return cell_component(vx, nx, ny, nx - 1, j);
}

double face_velocity_y(
    std::span<const double> vy,
    std::size_t nx,
    std::size_t ny,
    std::size_t i,
    std::size_t lower_cell) {
    if (vy.size() == 1) {
        return vy[0];
    }
    if (vy.size() != nx * ny) {
        return 0.0;
    }
    if (lower_cell + 1 < ny) {
        return 0.5 * (cell_component(vy, nx, ny, i, lower_cell)
                      + cell_component(vy, nx, ny, i, lower_cell + 1));
    }
    return cell_component(vy, nx, ny, i, ny - 1);
}

double sample_cell_2d(
    const std::vector<std::vector<double>>& u,
    int i,
    int j,
    BoundaryCondition bc_x,
    BoundaryCondition bc_y) {
    const int nx = static_cast<int>(u[0].size());
    const int ny = static_cast<int>(u.size());
    if (nx <= 0 || ny <= 0) {
        return 0.0;
    }

    if (i < 0 || i >= nx) {
        if (bc_x == BoundaryCondition::Periodic) {
            i = ((i % nx) + nx) % nx;
        } else if (i < 0) {
            i = 0;
        } else {
            i = nx - 1;
        }
    }
    if (j < 0 || j >= ny) {
        if (bc_y == BoundaryCondition::Periodic) {
            j = ((j % ny) + ny) % ny;
        } else if (j < 0) {
            j = 0;
        } else {
            j = ny - 1;
        }
    }

    return u[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
}

double upwind_x_face_flux(
    const std::vector<std::vector<double>>& u,
    std::span<const double> vx,
    int face_i,
    int face_j,
    std::size_t nx,
    std::size_t ny,
    BoundaryCondition bc_x,
    BoundaryCondition bc_y) {
    if (bc_x == BoundaryCondition::ZeroFlux &&
        (face_i <= 0 || face_i >= static_cast<int>(nx))) {
        return 0.0;
    }

    const std::size_t left_cell =
        face_i <= 0 ? 0 : static_cast<std::size_t>(face_i - 1);
    const double vxf = face_velocity_x(vx, nx, ny, left_cell, static_cast<std::size_t>(face_j));
    const double u_up = (vxf >= 0.0)
        ? sample_cell_2d(u, face_i - 1, face_j, bc_x, bc_y)
        : sample_cell_2d(u, face_i, face_j, bc_x, bc_y);
    return vxf * u_up;
}

double upwind_y_face_flux(
    const std::vector<std::vector<double>>& u,
    std::span<const double> vy,
    int face_i,
    int face_j,
    std::size_t nx,
    std::size_t ny,
    BoundaryCondition bc_x,
    BoundaryCondition bc_y) {
    if (bc_y == BoundaryCondition::ZeroFlux &&
        (face_j <= 0 || face_j >= static_cast<int>(ny))) {
        return 0.0;
    }

    const std::size_t lower_cell =
        face_j <= 0 ? 0 : static_cast<std::size_t>(face_j - 1);
    const double vyf = face_velocity_y(vy, nx, ny, static_cast<std::size_t>(face_i), lower_cell);
    const double u_up = (vyf >= 0.0)
        ? sample_cell_2d(u, face_i, face_j - 1, bc_x, bc_y)
        : sample_cell_2d(u, face_i, face_j, bc_x, bc_y);
    return vyf * u_up;
}

bool passes_cfl_2d(
    std::span<const double> vx,
    std::span<const double> vy,
    double dt,
    double dx,
    double dy) {
    return max_wave_speed(vx) * dt / dx + max_wave_speed(vy) * dt / dy <= 1.0;
}

bool upwind_fvm_advection_2d_into(
    const std::vector<std::vector<double>>& u,
    std::span<const double> vx,
    std::span<const double> vy,
    double dt,
    double dx,
    double dy,
    BoundaryCondition bc_x,
    BoundaryCondition bc_y,
    std::vector<std::vector<double>>& out) {
    if (!is_rectangular_grid(u) || vx.empty() || vy.empty() || dt <= 0.0 || dx <= 0.0
        || dy <= 0.0) {
        return false;
    }

    const std::size_t ny = u.size();
    const std::size_t nx = u[0].size();
    if (out.size() != ny || (ny > 0 && out[0].size() != nx)) {
        return false;
    }
    if (vx.size() != 1 && vx.size() != nx * ny) {
        return false;
    }
    if (vy.size() != 1 && vy.size() != nx * ny) {
        return false;
    }
    if (!passes_cfl_2d(vx, vy, dt, dx, dy)) {
        return false;
    }

    const double scale_x = dt / dx;
    const double scale_y = dt / dy;
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const double flux_right = upwind_x_face_flux(
                u, vx, static_cast<int>(i) + 1, static_cast<int>(j), nx, ny, bc_x, bc_y);
            const double flux_left = upwind_x_face_flux(
                u, vx, static_cast<int>(i), static_cast<int>(j), nx, ny, bc_x, bc_y);
            const double flux_top = upwind_y_face_flux(
                u, vy, static_cast<int>(i), static_cast<int>(j) + 1, nx, ny, bc_x, bc_y);
            const double flux_bottom = upwind_y_face_flux(
                u, vy, static_cast<int>(i), static_cast<int>(j), nx, ny, bc_x, bc_y);
            out[j][i] = u[j][i]
                - scale_x * (flux_right - flux_left)
                - scale_y * (flux_top - flux_bottom);
        }
    }
    return true;
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

Grid2D grid2d(
    double x0,
    double x1,
    double y0,
    double y1,
    std::size_t nx,
    std::size_t ny) {
    Grid2D grid;
    if (nx < 2 || ny < 2 || x1 <= x0 || y1 <= y0) {
        return grid;
    }

    grid.x0 = x0;
    grid.x1 = x1;
    grid.y0 = y0;
    grid.y1 = y1;
    grid.nx = nx;
    grid.ny = ny;
    grid.dx = (x1 - x0) / static_cast<double>(nx);
    grid.dy = (y1 - y0) / static_cast<double>(ny);
    grid.x.resize(nx);
    grid.y.resize(ny);
    for (std::size_t i = 0; i < nx; ++i) {
        grid.x[i] = x0 + (static_cast<double>(i) + 0.5) * grid.dx;
    }
    for (std::size_t j = 0; j < ny; ++j) {
        grid.y[j] = y0 + (static_cast<double>(j) + 0.5) * grid.dy;
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

std::vector<std::vector<double>> square_pulse_2d(
    const Grid2D& grid,
    double xc,
    double yc,
    double width_x,
    double width_y,
    double amplitude) {
    std::vector<std::vector<double>> u(
        grid.ny, std::vector<double>(grid.nx, 0.0));
    if (grid.nx == 0 || grid.ny == 0 || width_x <= 0.0 || width_y <= 0.0) {
        return u;
    }

    const double half_x = 0.5 * width_x;
    const double half_y = 0.5 * width_y;
    for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
            const double x = grid.x[i];
            const double y = grid.y[j];
            if (std::abs(x - xc) <= half_x && std::abs(y - yc) <= half_y) {
                u[j][i] = amplitude;
            }
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
    std::vector<double> next(n);
    if (!upwind_fvm_advection_into(u, v, dt, dx, bc, next)) {
        return {};
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
    std::vector<double> next(u0.size());
    result.t.push_back(0.0);
    result.u.push_back(u);

    double t = 0.0;
    while (t < t_end - 1e-15) {
        const double step = std::min(dt, t_end - t);
        if (!upwind_fvm_advection_into(u, v, step, dx, bc, next)) {
            return {};
        }
        u.swap(next);
        t += step;
        result.t.push_back(t);
        result.u.push_back(u);
    }

    return result;
}

std::vector<std::vector<double>> upwind_fvm_advection_2d(
    const std::vector<std::vector<double>>& u,
    std::span<const double> vx,
    std::span<const double> vy,
    double dt,
    double dx,
    double dy,
    BoundaryCondition bc_x,
    BoundaryCondition bc_y) {
    if (!is_rectangular_grid(u) || vx.empty() || vy.empty() || dt <= 0.0 || dx <= 0.0
        || dy <= 0.0) {
        return {};
    }

    const std::size_t ny = u.size();
    const std::size_t nx = u[0].size();
    std::vector<std::vector<double>> next(ny, std::vector<double>(nx));
    if (!upwind_fvm_advection_2d_into(u, vx, vy, dt, dx, dy, bc_x, bc_y, next)) {
        return {};
    }
    return next;
}

Advection2DResult run_advection_2d(
    const std::vector<std::vector<double>>& u0,
    std::span<const double> vx,
    std::span<const double> vy,
    double t_end,
    double dt,
    double dx,
    double dy,
    BoundaryCondition bc_x,
    BoundaryCondition bc_y) {
    Advection2DResult result;
    if (!is_rectangular_grid(u0) || vx.empty() || vy.empty() || t_end <= 0.0 || dt <= 0.0
        || dx <= 0.0 || dy <= 0.0) {
        return result;
    }

    const std::size_t ny = u0.size();
    const std::size_t nx = u0[0].size();
    if (vx.size() != 1 && vx.size() != nx * ny) {
        return result;
    }
    if (vy.size() != 1 && vy.size() != nx * ny) {
        return result;
    }

    std::vector<std::vector<double>> u = u0;
    std::vector<std::vector<double>> next(ny, std::vector<double>(nx));
    result.t.push_back(0.0);
    result.u.push_back(u);

    double t = 0.0;
    while (t < t_end - 1e-15) {
        const double step = std::min(dt, t_end - t);
        if (!upwind_fvm_advection_2d_into(u, vx, vy, step, dx, dy, bc_x, bc_y, next)) {
            return {};
        }
        u.swap(next);
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

double integrated_mass_2d(
    const std::vector<std::vector<double>>& u,
    double dx,
    double dy) {
    if (!is_rectangular_grid(u) || dx <= 0.0 || dy <= 0.0) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& row : u) {
        for (double ui : row) {
            sum += ui;
        }
    }
    return sum * dx * dy;
}

} // namespace cfd
} // namespace ms
