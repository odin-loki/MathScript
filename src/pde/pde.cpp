#include "ms/pde/pde.hpp"

#include <algorithm>
#include <cmath>

namespace ms {

namespace {

bool is_rectangular_grid(const std::vector<std::vector<double>>& grid) {
    if (grid.empty()) {
        return false;
    }
    const std::size_t cols = grid[0].size();
    if (cols < 3) {
        return false;
    }
    for (const auto& row : grid) {
        if (row.size() != cols) {
            return false;
        }
    }
    return grid.size() >= 3;
}

} // namespace

Heat1DResult pde_heat_1d(
    const std::vector<double>& x0,
    double alpha,
    double dx,
    double dt,
    std::size_t steps) {
    Heat1DResult result;
    if (x0.size() < 3 || steps == 0) {
        return result;
    }

    const std::size_t n = x0.size();
    const double r = alpha * dt / (dx * dx);
    if (r > 0.5) {
        return result;
    }

    result.x.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.x[i] = static_cast<double>(i) * dx;
    }

    std::vector<double> u = x0;
    std::vector<double> next(n);
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        }
        next[0] = u[0];
        next[n - 1] = u[n - 1];
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

Heat2DResult pde_heat_2d(
    const std::vector<std::vector<double>>& u0,
    double alpha,
    double dx,
    double dy,
    double dt,
    std::size_t steps) {
    Heat2DResult result;
    if (!is_rectangular_grid(u0) || steps == 0 || dx <= 0.0 || dy <= 0.0 || dt <= 0.0) {
        return result;
    }

    const double stability = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    if (stability > 0.5) {
        return result;
    }

    const std::size_t ny = u0.size();
    const std::size_t nx = u0[0].size();
    const double rx = alpha * dt / (dx * dx);
    const double ry = alpha * dt / (dy * dy);

    std::vector<std::vector<double>> u = u0;
    std::vector<std::vector<double>> next(ny, std::vector<double>(nx));
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                if (i == 0 || i + 1 == nx || j == 0 || j + 1 == ny) {
                    next[j][i] = 0.0;
                } else {
                    next[j][i] = u[j][i]
                        + rx * (u[j][i - 1] - 2.0 * u[j][i] + u[j][i + 1])
                        + ry * (u[j - 1][i] - 2.0 * u[j][i] + u[j + 1][i]);
                }
            }
        }
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

Wave1DResult pde_wave_1d(
    const std::vector<double>& u0,
    const std::vector<double>& v0,
    double c,
    double dx,
    double dt,
    std::size_t steps) {
    Wave1DResult result;
    if (u0.size() < 3 || u0.size() != v0.size() || steps == 0 || dx <= 0.0 || dt <= 0.0) {
        return result;
    }

    const double cfl = c * dt / dx;
    if (std::abs(cfl) > 1.0) {
        return result;
    }

    const std::size_t n = u0.size();
    const double r = cfl * cfl;

    std::vector<double> prev(n, 0.0);
    std::vector<double> curr = u0;
    std::vector<double> next(n);

    for (std::size_t i = 1; i + 1 < n; ++i) {
        next[i] = u0[i] + dt * v0[i]
            + 0.5 * r * (u0[i - 1] - 2.0 * u0[i] + u0[i + 1]);
    }
    next[0] = 0.0;
    next[n - 1] = 0.0;

    result.t.push_back(0.0);
    result.u.push_back(curr);

    prev = curr;
    curr = next;
    result.t.push_back(dt);
    result.u.push_back(curr);

    for (std::size_t step = 2; step <= steps; ++step) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            next[i] = 2.0 * curr[i] - prev[i]
                + r * (curr[i - 1] - 2.0 * curr[i] + curr[i + 1]);
        }
        next[0] = 0.0;
        next[n - 1] = 0.0;
        prev = curr;
        curr = next;
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(curr);
    }

    return result;
}

Advection1DResult pde_advection_1d(
    const std::vector<double>& u0,
    double v,
    double dx,
    double dt,
    std::size_t steps) {
    Advection1DResult result;
    if (u0.size() < 3 || steps == 0 || dx <= 0.0 || dt <= 0.0) {
        return result;
    }

    const double cfl = std::abs(v) * dt / dx;
    if (cfl > 1.0) {
        return result;
    }

    const std::size_t n = u0.size();
    const double coeff = v * dt / dx;

    std::vector<double> u = u0;
    std::vector<double> next(n);
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t i = 0; i < n; ++i) {
            if (v >= 0.0) {
                const std::size_t im = (i + n - 1) % n;
                next[i] = u[i] - coeff * (u[i] - u[im]);
            } else {
                const std::size_t ip = (i + 1) % n;
                next[i] = u[i] - coeff * (u[ip] - u[i]);
            }
        }
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

Poisson2DResult pde_poisson_2d(
    const std::vector<std::vector<double>>& f,
    double dx,
    double dy,
    std::size_t max_iterations,
    double tolerance) {
    Poisson2DResult result;
    if (!is_rectangular_grid(f) || max_iterations == 0 || dx <= 0.0 || dy <= 0.0 || tolerance <= 0.0) {
        return result;
    }

    const std::size_t ny = f.size();
    const std::size_t nx = f[0].size();
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * inv_dx2 + 2.0 * inv_dy2;

    std::vector<std::vector<double>> u(ny, std::vector<double>(nx, 0.0));

    for (std::size_t iter = 1; iter <= max_iterations; ++iter) {
        double max_change = 0.0;
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            for (std::size_t i = 1; i + 1 < nx; ++i) {
                const double old = u[j][i];
                u[j][i] = ((u[j][i - 1] + u[j][i + 1]) * inv_dx2
                    + (u[j - 1][i] + u[j + 1][i]) * inv_dy2 - f[j][i]) / denom;
                max_change = std::max(max_change, std::abs(u[j][i] - old));
            }
        }
        result.iterations = iter;
        if (max_change < tolerance) {
            result.converged = true;
            break;
        }
    }

    result.u = std::move(u);
    return result;
}

Burgers1DResult pde_burgers_1d(
    const std::vector<double>& u0,
    double nu,
    double dx,
    double dt,
    std::size_t steps) {
    Burgers1DResult result;
    if (u0.size() < 3 || steps == 0 || dx <= 0.0 || dt <= 0.0 || nu < 0.0) {
        return result;
    }

    const std::size_t n = u0.size();
    const double diff_coeff = nu * dt / (dx * dx);

    std::vector<double> u = u0;
    std::vector<double> next(n);
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            double adv = 0.0;
            if (u[i] >= 0.0) {
                adv = u[i] * (u[i] - u[i - 1]) / dx;
            } else {
                adv = u[i] * (u[i + 1] - u[i]) / dx;
            }
            next[i] = u[i] - dt * adv
                + diff_coeff * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        }
        next[0] = u0[0];
        next[n - 1] = u0[n - 1];
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

} // namespace ms
