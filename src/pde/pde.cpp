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

/// Solve a tridiagonal system with lower diag a, main diag b, upper diag c, rhs d.
/// Overwrites d with the solution. Returns false if the system is singular.
bool thomas_solve(
    std::vector<double>& a,
    std::vector<double>& b,
    std::vector<double>& c,
    std::vector<double>& d) {
    const std::size_t n = d.size();
    if (n == 0) {
        return false;
    }

    for (std::size_t i = 1; i < n; ++i) {
        const double w = a[i - 1] / b[i - 1];
        b[i] -= w * c[i - 1];
        d[i] -= w * d[i - 1];
    }

    if (std::abs(b[n - 1]) < 1e-300) {
        return false;
    }
    d[n - 1] /= b[n - 1];

    for (std::size_t i = n - 1; i-- > 0;) {
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
    }

    return true;
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

Heat1DResult pde_heat_1d_cn(
    const std::vector<double>& x0,
    double alpha,
    double dx,
    double dt,
    std::size_t steps) {
    Heat1DResult result;
    if (x0.size() < 3 || steps == 0 || dx <= 0.0 || dt <= 0.0 || alpha <= 0.0) {
        return result;
    }

    const std::size_t n = x0.size();
    const double r = alpha * dt / (dx * dx);
    const std::size_t m = n - 2;

    result.x.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.x[i] = static_cast<double>(i) * dx;
    }

    std::vector<double> u = x0;
    result.t.push_back(0.0);
    result.u.push_back(u);

    std::vector<double> a(m, -0.5 * r);
    std::vector<double> b(m, 1.0 + r);
    std::vector<double> c(m, -0.5 * r);
    std::vector<double> rhs(m);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t i = k + 1;
            rhs[k] = (1.0 - r) * u[i] + 0.5 * r * (u[i - 1] + u[i + 1]);
        }

        std::vector<double> a_copy = a;
        std::vector<double> b_copy = b;
        std::vector<double> c_copy = c;
        if (!thomas_solve(a_copy, b_copy, c_copy, rhs)) {
            return Heat1DResult{};
        }

        u[0] = 0.0;
        u[n - 1] = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            u[k + 1] = rhs[k];
        }

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

Wave2DResult pde_wave_2d(
    const std::vector<std::vector<double>>& u0,
    const std::vector<std::vector<double>>& v0,
    double c,
    double dx,
    double dy,
    double dt,
    std::size_t steps) {
    Wave2DResult result;
    if (!is_rectangular_grid(u0) || !is_rectangular_grid(v0) || steps == 0
        || dx <= 0.0 || dy <= 0.0 || dt <= 0.0) {
        return result;
    }

    const std::size_t ny = u0.size();
    const std::size_t nx = u0[0].size();
    if (v0.size() != ny || v0[0].size() != nx) {
        return result;
    }

    const double cfl = c * c * dt * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
    if (cfl > 1.0) {
        return result;
    }

    const double sig_x = c * c * dt * dt / (dx * dx);
    const double sig_y = c * c * dt * dt / (dy * dy);

    std::vector<std::vector<double>> prev(ny, std::vector<double>(nx, 0.0));
    std::vector<std::vector<double>> curr = u0;
    std::vector<std::vector<double>> next(ny, std::vector<double>(nx, 0.0));

    for (std::size_t j = 1; j + 1 < ny; ++j) {
        for (std::size_t i = 1; i + 1 < nx; ++i) {
            next[j][i] = u0[j][i] + dt * v0[j][i]
                + 0.5 * sig_x * (u0[j][i - 1] - 2.0 * u0[j][i] + u0[j][i + 1])
                + 0.5 * sig_y * (u0[j - 1][i] - 2.0 * u0[j][i] + u0[j + 1][i]);
        }
    }

    result.t.push_back(0.0);
    result.u.push_back(curr);

    prev = curr;
    curr = next;
    result.t.push_back(dt);
    result.u.push_back(curr);

    for (std::size_t step = 2; step <= steps; ++step) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                if (i == 0 || i + 1 == nx || j == 0 || j + 1 == ny) {
                    next[j][i] = 0.0;
                } else {
                    next[j][i] = 2.0 * curr[j][i] - prev[j][i]
                        + sig_x * (curr[j][i - 1] - 2.0 * curr[j][i] + curr[j][i + 1])
                        + sig_y * (curr[j - 1][i] - 2.0 * curr[j][i] + curr[j + 1][i]);
                }
            }
        }
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

Advection1DResult pde_advection_1d_lax_wendroff(
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
    const double c1 = v * dt / (2.0 * dx);
    const double c2 = 0.5 * cfl * cfl;

    std::vector<double> u = u0;
    std::vector<double> next(n);
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t ip = (i + 1) % n;
            const std::size_t im = (i + n - 1) % n;
            next[i] = u[i] - c1 * (u[ip] - u[im]) + c2 * (u[ip] - 2.0 * u[i] + u[im]);
        }
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

Poisson1DResult pde_poisson_1d(
    const std::vector<double>& f,
    double dx,
    double ua,
    double ub) {
    Poisson1DResult result;
    if (f.size() < 3 || dx <= 0.0) {
        return result;
    }

    const std::size_t n = f.size();
    const std::size_t m = n - 2;
    const double inv_dx2 = 1.0 / (dx * dx);

    result.u.resize(n);
    result.u[0] = ua;
    result.u[n - 1] = ub;

    if (m == 0) {
        return result;
    }

    std::vector<double> a(m, inv_dx2);
    std::vector<double> b(m, -2.0 * inv_dx2);
    std::vector<double> c(m, inv_dx2);
    std::vector<double> rhs(m);

    for (std::size_t k = 0; k < m; ++k) {
        const std::size_t i = k + 1;
        rhs[k] = f[i];
        if (k == 0) {
            rhs[k] -= ua * inv_dx2;
        }
        if (k + 1 == m) {
            rhs[k] -= ub * inv_dx2;
        }
    }

    if (!thomas_solve(a, b, c, rhs)) {
        return Poisson1DResult{};
    }

    for (std::size_t k = 0; k < m; ++k) {
        result.u[k + 1] = rhs[k];
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

Heat1DResult pde_reaction_diffusion_1d(
    const std::vector<double>& u0,
    double D,
    double r,
    double dx,
    double dt,
    std::size_t steps) {
    Heat1DResult result;
    if (u0.size() < 3 || steps == 0 || dx <= 0.0 || dt <= 0.0 || D < 0.0) {
        return result;
    }

    const std::size_t n = u0.size();
    const double diff_r = D * dt / (dx * dx);
    if (diff_r > 0.5) {
        return result;
    }

    result.x.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        result.x[i] = static_cast<double>(i) * dx;
    }

    std::vector<double> u = u0;
    std::vector<double> next(n);
    result.t.push_back(0.0);
    result.u.push_back(u);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t i = 1; i + 1 < n; ++i) {
            const double diffusion = diff_r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
            const double reaction = dt * r * u[i] * (1.0 - u[i]);
            next[i] = u[i] + diffusion + reaction;
        }
        {
            const double diffusion = 2.0 * diff_r * (u[1] - u[0]);
            const double reaction = dt * r * u[0] * (1.0 - u[0]);
            next[0] = u[0] + diffusion + reaction;
        }
        {
            const double diffusion = 2.0 * diff_r * (u[n - 2] - u[n - 1]);
            const double reaction = dt * r * u[n - 1] * (1.0 - u[n - 1]);
            next[n - 1] = u[n - 1] + diffusion + reaction;
        }
        u.swap(next);
        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

Heat2DResult pde_heat_2d_cn_adi(
    const std::vector<std::vector<double>>& u0,
    double alpha,
    double dx,
    double dy,
    double dt,
    std::size_t steps) {
    Heat2DResult result;
    if (!is_rectangular_grid(u0) || steps == 0 || dx <= 0.0 || dy <= 0.0 || dt <= 0.0
        || alpha <= 0.0) {
        return result;
    }

    const std::size_t ny = u0.size();
    const std::size_t nx = u0[0].size();
    const double lam_x = alpha * dt / (2.0 * dx * dx);
    const double lam_y = alpha * dt / (2.0 * dy * dy);

    std::vector<std::vector<double>> u = u0;
    std::vector<std::vector<double>> half(ny, std::vector<double>(nx));
    result.t.push_back(0.0);
    result.u.push_back(u);

    const std::size_t m_x = nx - 2;
    const std::size_t m_y = ny - 2;
    std::vector<double> a_x(m_x, -lam_x);
    std::vector<double> b_x(m_x, 1.0 + 2.0 * lam_x);
    std::vector<double> c_x(m_x, -lam_x);
    std::vector<double> a_y(m_y, -lam_y);
    std::vector<double> b_y(m_y, 1.0 + 2.0 * lam_y);
    std::vector<double> c_y(m_y, -lam_y);

    for (std::size_t step = 1; step <= steps; ++step) {
        for (std::size_t j = 1; j + 1 < ny; ++j) {
            std::vector<double> rhs(m_x);
            for (std::size_t k = 0; k < m_x; ++k) {
                const std::size_t i = k + 1;
                rhs[k] = u[j][i]
                    + lam_y * (u[j - 1][i] - 2.0 * u[j][i] + u[j + 1][i]);
            }

            std::vector<double> a_copy = a_x;
            std::vector<double> b_copy = b_x;
            std::vector<double> c_copy = c_x;
            if (!thomas_solve(a_copy, b_copy, c_copy, rhs)) {
                return Heat2DResult{};
            }

            half[j][0] = 0.0;
            half[j][nx - 1] = 0.0;
            for (std::size_t k = 0; k < m_x; ++k) {
                half[j][k + 1] = rhs[k];
            }
        }

        for (std::size_t j = 0; j < ny; ++j) {
            half[j][0] = 0.0;
            half[j][nx - 1] = 0.0;
        }

        for (std::size_t i = 1; i + 1 < nx; ++i) {
            std::vector<double> rhs(m_y);
            for (std::size_t k = 0; k < m_y; ++k) {
                const std::size_t j = k + 1;
                rhs[k] = half[j][i]
                    + lam_x * (half[j][i - 1] - 2.0 * half[j][i] + half[j][i + 1]);
            }

            std::vector<double> a_copy = a_y;
            std::vector<double> b_copy = b_y;
            std::vector<double> c_copy = c_y;
            if (!thomas_solve(a_copy, b_copy, c_copy, rhs)) {
                return Heat2DResult{};
            }

            u[0][i] = 0.0;
            u[ny - 1][i] = 0.0;
            for (std::size_t k = 0; k < m_y; ++k) {
                u[k + 1][i] = rhs[k];
            }
        }

        for (std::size_t j = 0; j < ny; ++j) {
            u[j][0] = 0.0;
            u[j][nx - 1] = 0.0;
        }

        result.t.push_back(static_cast<double>(step) * dt);
        result.u.push_back(u);
    }

    return result;
}

} // namespace ms
