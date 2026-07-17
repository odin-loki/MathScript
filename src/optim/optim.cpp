#include "ms/optim/optim.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace ms {

namespace {

double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// Solves A*x = b for a small dense n x n system via Gaussian elimination
// with partial pivoting. Returns x, or a zero vector if A is (numerically)
// singular.
std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                         std::vector<double> b) {
    const int n = static_cast<int>(b.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::abs(A[static_cast<size_t>(col)][static_cast<size_t>(col)]);
        for (int row = col + 1; row < n; ++row) {
            double v = std::abs(A[static_cast<size_t>(row)][static_cast<size_t>(col)]);
            if (v > best) { best = v; pivot = row; }
        }
        if (best < 1e-300) {
            return std::vector<double>(static_cast<size_t>(n), 0.0);
        }
        if (pivot != col) {
            std::swap(A[static_cast<size_t>(pivot)], A[static_cast<size_t>(col)]);
            std::swap(b[static_cast<size_t>(pivot)], b[static_cast<size_t>(col)]);
        }
        for (int row = col + 1; row < n; ++row) {
            double factor = A[static_cast<size_t>(row)][static_cast<size_t>(col)] /
                             A[static_cast<size_t>(col)][static_cast<size_t>(col)];
            if (factor == 0.0) continue;
            for (int k = col; k < n; ++k) {
                A[static_cast<size_t>(row)][static_cast<size_t>(k)] -=
                    factor * A[static_cast<size_t>(col)][static_cast<size_t>(k)];
            }
            b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
        }
    }

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int row = n - 1; row >= 0; --row) {
        double sum = b[static_cast<size_t>(row)];
        for (int col = row + 1; col < n; ++col) {
            sum -= A[static_cast<size_t>(row)][static_cast<size_t>(col)] *
                   x[static_cast<size_t>(col)];
        }
        x[static_cast<size_t>(row)] = sum / A[static_cast<size_t>(row)][static_cast<size_t>(row)];
    }
    return x;
}

} // namespace

GradientDescentResult gradient_descent(
    Func2D f, double x0, double y0, double learning_rate, int max_iterations) {
    GradientDescentResult result{};
    result.x = x0;
    result.y = y0;
    result.iterations = 0;
    result.converged = false;

    for (int i = 0; i < max_iterations; ++i) {
        constexpr double h = 1e-7;
        constexpr double inv_2h = 1.0 / (2.0 * h);
        const double grad_x = (f(x0 + h, y0) - f(x0 - h, y0)) * inv_2h;
        const double grad_y = (f(x0, y0 + h) - f(x0, y0 - h)) * inv_2h;

        x0 -= learning_rate * grad_x;
        y0 -= learning_rate * grad_y;

        result.x = x0;
        result.y = y0;
        result.iterations = static_cast<size_t>(i + 1);

        if (std::abs(grad_x) < 1e-7 && std::abs(grad_y) < 1e-7) {
            result.converged = true;
            break;
        }
    }

    result.f_val = f(x0, y0);
    return result;
}

std::pair<double, double> newton_raphson(
    Func2D f, double x0, double y0, double tol, int max_iter) {
    constexpr double h = 1e-8;
    const int limit = max_iter < 50 ? max_iter : 50;
    double x = x0;

    for (int i = 0; i < limit; ++i) {
        const double fx = f(x, y0);
        if (std::abs(fx) < tol) {
            break;
        }
        const double dfx = (f(x + h, y0) - f(x - h, y0)) / (2.0 * h);
        if (std::abs(dfx) < 1e-14) {
            break;
        }
        const double step = fx / dfx;
        x -= step;
        if (std::abs(step) < tol) {
            break;
        }
    }

    return {x, y0};
}

std::pair<double, double> broyden(Func2D f, double x0, double y0) {
    constexpr double tol = 1e-10;
    constexpr int max_iter = 50;
    constexpr double h = 1e-4;

    double x_prev = x0;
    double x = x0 + h;
    double f_prev = f(x_prev, y0);
    double f_curr = f(x, y0);

    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(f_curr) < tol) {
            break;
        }
        const double denom = f_curr - f_prev;
        if (std::abs(denom) < 1e-14) {
            break;
        }
        const double x_next = x - f_curr * (x - x_prev) / denom;
        x_prev = x;
        f_prev = f_curr;
        x = x_next;
        f_curr = f(x, y0);
        if (std::abs(x - x_prev) < tol) {
            break;
        }
    }

    return {x, y0};
}

std::tuple<double, double, double> minimize_with_constraints(
    Func2D f, double x0, double y0, std::vector<double> constraints) {
    double x_lo = -1e300;
    double x_hi = 1e300;
    double y_lo = -1e300;
    double y_hi = 1e300;
    if (constraints.size() >= 2) {
        x_lo = y_lo = constraints[0];
        x_hi = y_hi = constraints[1];
    }
    if (constraints.size() >= 4) {
        x_lo = constraints[0];
        x_hi = constraints[1];
        y_lo = constraints[2];
        y_hi = constraints[3];
    }

    double x = clip(x0, x_lo, x_hi);
    double y = clip(y0, y_lo, y_hi);

    for (int iter = 0; iter < 20; ++iter) {
        const Func1D fx = [&](double tx) { return f(tx, y); };
        x = golden_section(fx, x_lo, x_hi, 1e-6);
        x = clip(x, x_lo, x_hi);

        const Func1D fy = [&](double ty) { return f(x, ty); };
        y = golden_section(fy, y_lo, y_hi, 1e-6);
        y = clip(y, y_lo, y_hi);
    }

    return {x, y, f(x, y)};
}

std::tuple<std::vector<double>, double> simplex_solver(
    std::vector<double> c, std::vector<std::vector<double>> A, std::vector<double> b) {
    if (c.size() != 2) {
        return std::tuple<std::vector<double>, double>{std::vector<double>{}, 0.0};
    }

    const auto eval = [&](double x, double y) {
        double val = c[0] * x + c[1] * y;
        for (size_t i = 0; i < A.size(); ++i) {
            double lhs = 0.0;
            for (size_t j = 0; j < A[i].size() && j < 2; ++j) {
                lhs += A[i][j] * (j == 0 ? x : y);
            }
            if (i < b.size() && lhs > b[i]) {
                const double viol = lhs - b[i];
                val += 1e6 * viol * viol;
            }
        }
        if (x < 0.0) {
            val += 1e6 * x * x;
        }
        if (y < 0.0) {
            val += 1e6 * y * y;
        }
        return val;
    };

    std::array<std::array<double, 2>, 3> simplex{};
    simplex[0] = {0.0, 0.0};
    simplex[1] = {b.size() > 0 ? std::max(b[0], 0.1) : 0.1, 0.0};
    simplex[2] = {0.0, b.size() > 1 ? std::max(b[1], 0.1) : 0.1};

    const auto value = [&](const std::array<double, 2>& p) { return eval(p[0], p[1]); };

    for (int iter = 0; iter < 100; ++iter) {
        std::array<size_t, 3> order{0, 1, 2};
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b_idx) {
            return value(simplex[a]) < value(simplex[b_idx]);
        });

        const auto& best = simplex[order[0]];
        const auto& worst = simplex[order[2]];
        const auto& second = simplex[order[1]];

        const std::array<double, 2> centroid{
            0.5 * (best[0] + second[0]),
            0.5 * (best[1] + second[1]),
        };

        std::array<double, 2> reflected{
            centroid[0] + (centroid[0] - worst[0]),
            centroid[1] + (centroid[1] - worst[1]),
        };
        const double f_ref = value(reflected);

        if (f_ref < value(second) && f_ref >= value(best)) {
            simplex[order[2]] = reflected;
            continue;
        }
        if (f_ref < value(best)) {
            std::array<double, 2> expanded{
                centroid[0] + 2.0 * (reflected[0] - centroid[0]),
                centroid[1] + 2.0 * (reflected[1] - centroid[1]),
            };
            simplex[order[2]] = value(expanded) < f_ref ? expanded : reflected;
            continue;
        }

        std::array<double, 2> contracted{
            centroid[0] + 0.5 * (worst[0] - centroid[0]),
            centroid[1] + 0.5 * (worst[1] - centroid[1]),
        };
        if (value(contracted) < value(worst)) {
            simplex[order[2]] = contracted;
            continue;
        }

        simplex[order[1]] = {
            best[0] + 0.5 * (second[0] - best[0]),
            best[1] + 0.5 * (second[1] - best[1]),
        };
        simplex[order[2]] = {
            best[0] + 0.5 * (worst[0] - best[0]),
            best[1] + 0.5 * (worst[1] - best[1]),
        };
    }

    std::array<double, 2> best_pt = simplex[0];
    double best_val = value(best_pt);
    for (size_t i = 1; i < simplex.size(); ++i) {
        const double v = value(simplex[i]);
        if (v < best_val) {
            best_val = v;
            best_pt = simplex[i];
        }
    }

    return {{best_pt[0], best_pt[1]}, c[0] * best_pt[0] + c[1] * best_pt[1]};
}

double golden_section(Func1D f, double a, double b, double tol) {
    const double inv_phi = 2.0 / (1.0 + std::sqrt(5.0));
    double c = b - (b - a) * inv_phi;
    double d = a + (b - a) * inv_phi;
    double fc = f(c);
    double fd = f(d);
    while (std::abs(b - a) > tol) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * inv_phi;
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * inv_phi;
            fd = f(d);
        }
    }
    return (a + b) / 2.0;
}

double newton_1d(Func1D f, double x0, double tol, int max_iter) {
    constexpr double h = 1e-7;
    double x = x0;
    for (int i = 0; i < max_iter; ++i) {
        const double dfx = (f(x + h) - f(x - h)) / (2.0 * h);
        const double d2fx = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
        if (std::abs(d2fx) < 1e-14) {
            break;
        }
        const double step = dfx / d2fx;
        x -= step;
        if (std::abs(step) < tol) {
            break;
        }
    }
    return x;
}

// ----------------------------------------------------------------
// Nelder-Mead (derivative-free N-D minimizer)
// ----------------------------------------------------------------
OptimResult nelder_mead(FuncND f, std::vector<double> x0,
                        double tol, int max_iter) {
    const int n = static_cast<int>(x0.size());
    const double alpha = 1.0, gamma_exp = 2.0, rho = 0.5, sigma = 0.5;

    // Build initial simplex
    std::vector<std::vector<double>> simplex(static_cast<size_t>(n + 1), x0);
    for (int i = 0; i < n; ++i) {
        double step = (x0[static_cast<size_t>(i)] != 0.0)
                          ? 0.05 * x0[static_cast<size_t>(i)]
                          : 0.00025;
        simplex[static_cast<size_t>(i + 1)][static_cast<size_t>(i)] += step;
    }

    std::vector<double> fvals(static_cast<size_t>(n + 1));
    for (int i = 0; i <= n; ++i) {
        fvals[static_cast<size_t>(i)] = f(simplex[static_cast<size_t>(i)]);
    }

    const auto centroid = [&](size_t worst) {
        std::vector<double> c(static_cast<size_t>(n), 0.0);
        for (int i = 0; i <= n; ++i) {
            if (static_cast<size_t>(i) == worst) continue;
            for (int j = 0; j < n; ++j) {
                c[static_cast<size_t>(j)] +=
                    simplex[static_cast<size_t>(i)][static_cast<size_t>(j)];
            }
        }
        for (auto& v : c) v /= n;
        return c;
    };

    const auto reflect = [&](const std::vector<double>& cen,
                              const std::vector<double>& worst,
                              double coef) {
        std::vector<double> r(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) {
            r[static_cast<size_t>(j)] =
                cen[static_cast<size_t>(j)] +
                coef * (cen[static_cast<size_t>(j)] -
                        worst[static_cast<size_t>(j)]);
        }
        return r;
    };

    std::vector<size_t> idx(static_cast<size_t>(n + 1));
    for (int iter = 0; iter < max_iter; ++iter) {
        // Sort
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return fvals[a] < fvals[b]; });

        if (fvals[idx.back()] - fvals[idx.front()] < tol) break;

        auto cen = centroid(idx.back());
        auto xr  = reflect(cen, simplex[idx.back()], alpha);
        double fr = f(xr);

        if (fr < fvals[idx[0]]) {
            auto xe = reflect(cen, simplex[idx.back()], -gamma_exp);
            double fe = f(xe);
            if (fe < fr) { simplex[idx.back()] = xe; fvals[idx.back()] = fe; }
            else          { simplex[idx.back()] = xr; fvals[idx.back()] = fr; }
        } else if (fr < fvals[idx[static_cast<size_t>(n - 1)]]) {
            simplex[idx.back()] = xr;
            fvals[idx.back()]   = fr;
        } else {
            auto xc = reflect(cen, simplex[idx.back()], -rho);
            double fc = f(xc);
            if (fc < fvals[idx.back()]) {
                simplex[idx.back()] = xc;
                fvals[idx.back()]   = fc;
            } else {
                for (int i = 1; i <= n; ++i) {
                    auto& s = simplex[static_cast<size_t>(i)];
                    for (int j = 0; j < n; ++j) {
                        s[static_cast<size_t>(j)] =
                            simplex[idx[0]][static_cast<size_t>(j)] +
                            sigma * (s[static_cast<size_t>(j)] -
                                     simplex[idx[0]][static_cast<size_t>(j)]);
                    }
                    fvals[static_cast<size_t>(i)] = f(s);
                }
            }
        }
    }

    // Find best
    size_t best = 0;
    for (size_t i = 1; i < static_cast<size_t>(n + 1); ++i) {
        if (fvals[i] < fvals[best]) best = i;
    }
    return OptimResult{simplex[best], fvals[best],
                       static_cast<size_t>(max_iter), true};
}

// ----------------------------------------------------------------
// BFGS
// ----------------------------------------------------------------
OptimResult bfgs(FuncND f, std::vector<double> x0, double tol, int max_iter) {
    const int n = static_cast<int>(x0.size());
    constexpr double h = 1e-7;

    const auto grad = [&](const std::vector<double>& x) {
        std::vector<double> g(static_cast<size_t>(n));
        std::vector<double> xp = x;
        std::vector<double> xm = x;
        for (int i = 0; i < n; ++i) {
            xp[static_cast<size_t>(i)] += h;
            xm[static_cast<size_t>(i)] -= h;
            g[static_cast<size_t>(i)] = (f(xp) - f(xm)) / (2.0 * h);
            xp[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
            xm[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
        }
        return g;
    };

    // H = I (inverse Hessian approx)
    std::vector<std::vector<double>> H(
        static_cast<size_t>(n),
        std::vector<double>(static_cast<size_t>(n), 0.0));
    for (int i = 0; i < n; ++i) H[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;

    auto x = x0;
    auto g = grad(x);
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        double gnorm = 0.0;
        for (auto v : g) gnorm += v * v;
        if (std::sqrt(gnorm) < tol) { converged = true; break; }

        // direction p = -H*g
        std::vector<double> p(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                p[static_cast<size_t>(i)] -=
                    H[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                    g[static_cast<size_t>(j)];
            }
        }

        // Line search (Armijo backtracking)
        double step = 1.0;
        double fx = f(x);
        double ddf = 0.0;
        for (int i = 0; i < n; ++i) ddf += g[static_cast<size_t>(i)] * p[static_cast<size_t>(i)];
        std::vector<double> xnew = x;
        for (int ls = 0; ls < 30; ++ls) {
            for (int i = 0; i < n; ++i)
                xnew[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(i)] + step * p[static_cast<size_t>(i)];
            if (f(xnew) <= fx + 1e-4 * step * ddf) break;
            step *= 0.5;
        }

        std::vector<double> s(static_cast<size_t>(n)), y(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            s[static_cast<size_t>(i)] = step * p[static_cast<size_t>(i)];
        }
        auto gnew = grad(xnew);
        for (int i = 0; i < n; ++i)
            y[static_cast<size_t>(i)] = gnew[static_cast<size_t>(i)] - g[static_cast<size_t>(i)];

        double sy = 0.0;
        for (int i = 0; i < n; ++i) sy += s[static_cast<size_t>(i)] * y[static_cast<size_t>(i)];

        if (std::abs(sy) > 1e-14) {
            double rho_val = 1.0 / sy;
            // H = (I - rho*s*y^T)*H*(I - rho*y*s^T) + rho*s*s^T
            // Simplified rank-2 update
            std::vector<std::vector<double>> Hnew(
                static_cast<size_t>(n),
                std::vector<double>(static_cast<size_t>(n), 0.0));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    double Hy_i = 0.0, Hy_j = 0.0;
                    for (int k = 0; k < n; ++k) {
                        Hy_i += H[static_cast<size_t>(i)][static_cast<size_t>(k)] *
                                y[static_cast<size_t>(k)];
                        Hy_j += H[static_cast<size_t>(j)][static_cast<size_t>(k)] *
                                y[static_cast<size_t>(k)];
                    }
                    double yHy = 0.0;
                    for (int k = 0; k < n; ++k) {
                        double Hyk = 0.0;
                        for (int l = 0; l < n; ++l)
                            Hyk += H[static_cast<size_t>(k)][static_cast<size_t>(l)] *
                                   y[static_cast<size_t>(l)];
                        yHy += y[static_cast<size_t>(k)] * Hyk;
                    }
                    Hnew[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                        H[static_cast<size_t>(i)][static_cast<size_t>(j)] +
                        rho_val * (1.0 + rho_val * yHy) *
                            s[static_cast<size_t>(i)] * s[static_cast<size_t>(j)] -
                        rho_val * (s[static_cast<size_t>(i)] * Hy_j +
                                   s[static_cast<size_t>(j)] * Hy_i);
                }
            }
            H = Hnew;
        }
        x = xnew;
        g = gnew;
    }
    return OptimResult{x, f(x), static_cast<size_t>(max_iter), converged};
}

// ----------------------------------------------------------------
// L-BFGS (two-loop recursion)
// ----------------------------------------------------------------
OptimResult lbfgs(FuncND f, std::vector<double> x0,
                  int m, double tol, int max_iter) {
    const int n = static_cast<int>(x0.size());
    constexpr double h = 1e-7;

    const auto grad = [&](const std::vector<double>& x) {
        std::vector<double> g(static_cast<size_t>(n));
        std::vector<double> xp = x;
        std::vector<double> xm = x;
        for (int i = 0; i < n; ++i) {
            xp[static_cast<size_t>(i)] += h;
            xm[static_cast<size_t>(i)] -= h;
            g[static_cast<size_t>(i)] = (f(xp) - f(xm)) / (2.0 * h);
            xp[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
            xm[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
        }
        return g;
    };

    auto x = x0;
    auto g = grad(x);
    std::vector<std::vector<double>> S, Y;
    std::vector<double> rho_list;
    std::vector<double> alpha_buf;
    alpha_buf.reserve(static_cast<size_t>(m));
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        double gnorm = 0.0;
        for (auto v : g) gnorm += v * v;
        if (std::sqrt(gnorm) < tol) { converged = true; break; }

        // Two-loop L-BFGS direction
        std::vector<double> q = g;
        const int k = static_cast<int>(S.size());
        alpha_buf.resize(static_cast<size_t>(k));
        for (int i = k - 1; i >= 0; --i) {
            double si_q = 0.0;
            for (int j = 0; j < n; ++j)
                si_q += S[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                        q[static_cast<size_t>(j)];
            alpha_buf[static_cast<size_t>(i)] = rho_list[static_cast<size_t>(i)] * si_q;
            for (int j = 0; j < n; ++j)
                q[static_cast<size_t>(j)] -=
                    alpha_buf[static_cast<size_t>(i)] *
                    Y[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        double scale = 1.0;
        if (k > 0) {
            double yy = 0.0, sy = 0.0;
            for (int j = 0; j < n; ++j) {
                yy += Y.back()[static_cast<size_t>(j)] * Y.back()[static_cast<size_t>(j)];
                sy += S.back()[static_cast<size_t>(j)] * Y.back()[static_cast<size_t>(j)];
            }
            if (std::abs(yy) > 1e-14) scale = sy / yy;
        }
        std::vector<double> r(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) r[static_cast<size_t>(j)] = scale * q[static_cast<size_t>(j)];
        for (int i = 0; i < k; ++i) {
            double yi_r = 0.0;
            for (int j = 0; j < n; ++j)
                yi_r += Y[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                        r[static_cast<size_t>(j)];
            double beta = rho_list[static_cast<size_t>(i)] * yi_r;
            for (int j = 0; j < n; ++j)
                r[static_cast<size_t>(j)] +=
                    S[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                    (alpha_buf[static_cast<size_t>(i)] - beta);
        }

        // p = -r, line search
        double step = 1.0;
        double fx = f(x);
        double ddf = 0.0;
        for (int j = 0; j < n; ++j) ddf -= g[static_cast<size_t>(j)] * r[static_cast<size_t>(j)];
        std::vector<double> xnew = x;
        for (int ls = 0; ls < 30; ++ls) {
            for (int j = 0; j < n; ++j)
                xnew[static_cast<size_t>(j)] =
                    x[static_cast<size_t>(j)] - step * r[static_cast<size_t>(j)];
            if (f(xnew) <= fx + 1e-4 * step * ddf) break;
            step *= 0.5;
        }

        std::vector<double> s(static_cast<size_t>(n)), y(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) {
            s[static_cast<size_t>(j)] = xnew[static_cast<size_t>(j)] -
                                         x[static_cast<size_t>(j)];
        }
        auto gnew = grad(xnew);
        for (int j = 0; j < n; ++j)
            y[static_cast<size_t>(j)] = gnew[static_cast<size_t>(j)] -
                                         g[static_cast<size_t>(j)];
        double sy = 0.0;
        for (int j = 0; j < n; ++j) sy += s[static_cast<size_t>(j)] * y[static_cast<size_t>(j)];

        if (std::abs(sy) > 1e-14) {
            if (static_cast<int>(S.size()) >= m) {
                S.erase(S.begin()); Y.erase(Y.begin());
                rho_list.erase(rho_list.begin());
            }
            S.push_back(s); Y.push_back(y);
            rho_list.push_back(1.0 / sy);
        }
        x = xnew;
        g = gnew;
    }
    return OptimResult{x, f(x), static_cast<size_t>(max_iter), converged};
}

// ----------------------------------------------------------------
// Nonlinear Conjugate Gradient (Polak-Ribière+ with Armijo line search)
// ----------------------------------------------------------------
OptimResult conjugate_gradient(FuncND f, GradND grad, std::vector<double> x0,
                               double tol, int max_iter) {
    const int n = static_cast<int>(x0.size());
    if (n <= 0) {
        return OptimResult{{}, 0.0, 0, false};
    }

    auto x = x0;
    auto g = grad(x);
    std::vector<double> d(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        d[static_cast<size_t>(i)] = -g[static_cast<size_t>(i)];
    }

    bool converged = false;
    size_t iterations = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        iterations = static_cast<size_t>(iter + 1);

        double gnorm_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            gnorm_sq += g[static_cast<size_t>(i)] * g[static_cast<size_t>(i)];
        }
        if (std::sqrt(gnorm_sq) < tol) {
            converged = true;
            break;
        }

        // Armijo backtracking along d (descent direction).
        double step = 1.0;
        const double fx = f(x);
        double ddf = 0.0;
        for (int i = 0; i < n; ++i) {
            ddf += g[static_cast<size_t>(i)] * d[static_cast<size_t>(i)];
        }
        if (ddf >= 0.0) {
            for (int i = 0; i < n; ++i) {
                d[static_cast<size_t>(i)] = -g[static_cast<size_t>(i)];
            }
            ddf = -gnorm_sq;
        }

        std::vector<double> xnew = x;
        for (int ls = 0; ls < 30; ++ls) {
            for (int i = 0; i < n; ++i) {
                xnew[static_cast<size_t>(i)] =
                    x[static_cast<size_t>(i)] + step * d[static_cast<size_t>(i)];
            }
            if (f(xnew) <= fx + 1e-4 * step * ddf) {
                break;
            }
            step *= 0.5;
        }

        auto gnew = grad(xnew);

        // Polak-Ribière+ beta with automatic restart when beta < 0.
        double beta = 0.0;
        if (gnorm_sq > 1e-300) {
            double num = 0.0;
            for (int i = 0; i < n; ++i) {
                num += gnew[static_cast<size_t>(i)] *
                       (gnew[static_cast<size_t>(i)] -
                        g[static_cast<size_t>(i)]);
            }
            beta = num / gnorm_sq;
            if (beta < 0.0) {
                beta = 0.0;
            }
        }

        for (int i = 0; i < n; ++i) {
            d[static_cast<size_t>(i)] =
                -gnew[static_cast<size_t>(i)] +
                beta * d[static_cast<size_t>(i)];
        }

        x = xnew;
        g = gnew;
    }

    return OptimResult{x, f(x), iterations, converged};
}

// ----------------------------------------------------------------
// Adam
// ----------------------------------------------------------------
OptimResult adam(FuncND f, std::vector<double> x0,
                 double alpha, double beta1, double beta2, int max_iter) {
    const int n = static_cast<int>(x0.size());
    constexpr double eps = 1e-8;
    constexpr double h   = 1e-7;

    const auto grad = [&](const std::vector<double>& x) {
        std::vector<double> g(static_cast<size_t>(n));
        std::vector<double> xp = x;
        std::vector<double> xm = x;
        for (int i = 0; i < n; ++i) {
            xp[static_cast<size_t>(i)] += h;
            xm[static_cast<size_t>(i)] -= h;
            g[static_cast<size_t>(i)] = (f(xp) - f(xm)) / (2.0 * h);
            xp[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
            xm[static_cast<size_t>(i)] = x[static_cast<size_t>(i)];
        }
        return g;
    };

    auto x = x0;
    std::vector<double> m(static_cast<size_t>(n), 0.0);
    std::vector<double> v(static_cast<size_t>(n), 0.0);

    for (int t = 1; t <= max_iter; ++t) {
        auto g = grad(x);
        double gnorm = 0.0;
        for (auto gv : g) gnorm += gv * gv;
        if (std::sqrt(gnorm) < 1e-8) break;

        double b1t = std::pow(beta1, t);
        double b2t = std::pow(beta2, t);
        for (int i = 0; i < n; ++i) {
            m[static_cast<size_t>(i)] = beta1 * m[static_cast<size_t>(i)] +
                                         (1.0 - beta1) * g[static_cast<size_t>(i)];
            v[static_cast<size_t>(i)] = beta2 * v[static_cast<size_t>(i)] +
                                         (1.0 - beta2) * g[static_cast<size_t>(i)] *
                                         g[static_cast<size_t>(i)];
            double mhat = m[static_cast<size_t>(i)] / (1.0 - b1t);
            double vhat = v[static_cast<size_t>(i)] / (1.0 - b2t);
            x[static_cast<size_t>(i)] -= alpha * mhat /
                                          (std::sqrt(vhat) + eps);
        }
    }
    return OptimResult{x, f(x), static_cast<size_t>(max_iter), true};
}

// ----------------------------------------------------------------
// RMSprop (adaptive gradient with running average of squared gradients)
// ----------------------------------------------------------------
OptimResult rmsprop(FuncND f, GradND grad, std::vector<double> x0,
                    double alpha, double rho, double eps, int max_iter) {
    const int n = static_cast<int>(x0.size());
    if (n <= 0) {
        return OptimResult{{}, 0.0, 0, false};
    }

    auto x = x0;
    std::vector<double> sq_avg(static_cast<size_t>(n), 0.0);

    bool converged = false;
    size_t iterations = 0;

    for (int t = 1; t <= max_iter; ++t) {
        iterations = static_cast<size_t>(t);
        auto g = grad(x);

        double gnorm_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            gnorm_sq += g[static_cast<size_t>(i)] * g[static_cast<size_t>(i)];
        }
        if (std::sqrt(gnorm_sq) < 1e-8) {
            converged = true;
            break;
        }

        for (int i = 0; i < n; ++i) {
            const double gi = g[static_cast<size_t>(i)];
            sq_avg[static_cast<size_t>(i)] =
                rho * sq_avg[static_cast<size_t>(i)] + (1.0 - rho) * gi * gi;
            x[static_cast<size_t>(i)] -=
                alpha * gi / (std::sqrt(sq_avg[static_cast<size_t>(i)]) + eps);
        }
    }

    return OptimResult{x, f(x), iterations, converged};
}

// ----------------------------------------------------------------
// Adadelta (adaptive learning rate via running averages of gradients
//           and parameter updates)
// ----------------------------------------------------------------
OptimResult adadelta(FuncND f, GradND grad, std::vector<double> x0,
                     double lr, double rho, double eps, int max_iter) {
    const int n = static_cast<int>(x0.size());
    if (n <= 0) {
        return OptimResult{{}, 0.0, 0, false};
    }

    auto x = x0;
    std::vector<double> sq_grad_avg(static_cast<size_t>(n), 0.0);
    std::vector<double> sq_update_avg(static_cast<size_t>(n), 0.0);

    bool converged = false;
    size_t iterations = 0;

    for (int t = 1; t <= max_iter; ++t) {
        iterations = static_cast<size_t>(t);
        auto g = grad(x);

        double gnorm_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            gnorm_sq += g[static_cast<size_t>(i)] * g[static_cast<size_t>(i)];
        }
        if (std::sqrt(gnorm_sq) < 1e-8) {
            converged = true;
            break;
        }

        for (int i = 0; i < n; ++i) {
            const double gi = g[static_cast<size_t>(i)];
            sq_grad_avg[static_cast<size_t>(i)] =
                rho * sq_grad_avg[static_cast<size_t>(i)] + (1.0 - rho) * gi * gi;

            const double rms_update =
                std::sqrt(sq_update_avg[static_cast<size_t>(i)] + eps);
            const double rms_grad =
                std::sqrt(sq_grad_avg[static_cast<size_t>(i)] + eps);
            const double dx = -lr * (rms_update / rms_grad) * gi;

            sq_update_avg[static_cast<size_t>(i)] =
                rho * sq_update_avg[static_cast<size_t>(i)] + (1.0 - rho) * dx * dx;
            x[static_cast<size_t>(i)] += dx;
        }
    }

    return OptimResult{x, f(x), iterations, converged};
}

// ----------------------------------------------------------------
// Levenberg-Marquardt (damped Gauss-Newton for nonlinear least squares)
// ----------------------------------------------------------------
OptimResult levenberg_marquardt(ResidualFunc residuals, std::vector<double> x0,
                                 int max_iter, double tol, double lambda0) {
    const int n = static_cast<int>(x0.size());
    constexpr double h = 1e-6;

    const auto sum_sq = [](const std::vector<double>& r) {
        double s = 0.0;
        for (double v : r) s += v * v;
        return s;
    };

    // Central-difference Jacobian: J[i][j] = d(r_i)/d(x_j)
    const auto jacobian = [&](const std::vector<double>& x,
                               const std::vector<double>& r0) {
        const int m = static_cast<int>(r0.size());
        std::vector<std::vector<double>> J(
            static_cast<size_t>(m), std::vector<double>(static_cast<size_t>(n), 0.0));
        std::vector<double> xp = x;
        std::vector<double> xm = x;
        for (int j = 0; j < n; ++j) {
            xp[static_cast<size_t>(j)] += h;
            xm[static_cast<size_t>(j)] -= h;
            auto rp = residuals(xp);
            auto rm = residuals(xm);
            for (int i = 0; i < m; ++i) {
                J[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    (rp[static_cast<size_t>(i)] - rm[static_cast<size_t>(i)]) / (2.0 * h);
            }
            xp[static_cast<size_t>(j)] = x[static_cast<size_t>(j)];
            xm[static_cast<size_t>(j)] = x[static_cast<size_t>(j)];
        }
        return J;
    };

    auto x = x0;
    auto r = residuals(x);
    double cost = sum_sq(r);
    double lambda = lambda0;
    size_t iterations = 0;
    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter) {
        ++iterations;

        auto J = jacobian(x, r);
        const int m = static_cast<int>(r.size());

        // JtJ = J^T*J, Jtr = J^T*r
        std::vector<std::vector<double>> JtJ(
            static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
        std::vector<double> Jtr(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int k = 0; k < m; ++k) {
                    s += J[static_cast<size_t>(k)][static_cast<size_t>(i)] *
                         J[static_cast<size_t>(k)][static_cast<size_t>(j)];
                }
                JtJ[static_cast<size_t>(i)][static_cast<size_t>(j)] = s;
            }
            double s = 0.0;
            for (int k = 0; k < m; ++k) {
                s += J[static_cast<size_t>(k)][static_cast<size_t>(i)] * r[static_cast<size_t>(k)];
            }
            Jtr[static_cast<size_t>(i)] = s;
        }

        // Try increasing damping until a step improves the cost (or we give up).
        const double cost_before = cost;
        bool step_accepted = false;
        std::vector<double> delta(static_cast<size_t>(n), 0.0);
        for (int trial = 0; trial < 50; ++trial) {
            auto A = JtJ;
            for (int i = 0; i < n; ++i) {
                A[static_cast<size_t>(i)][static_cast<size_t>(i)] +=
                    lambda * JtJ[static_cast<size_t>(i)][static_cast<size_t>(i)];
            }
            std::vector<double> negJtr(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) negJtr[static_cast<size_t>(i)] = -Jtr[static_cast<size_t>(i)];

            delta = solve_linear_system(A, negJtr);

            auto x_new = x;
            for (int i = 0; i < n; ++i) x_new[static_cast<size_t>(i)] += delta[static_cast<size_t>(i)];
            auto r_new = residuals(x_new);
            double cost_new = sum_sq(r_new);

            if (std::isfinite(cost_new) && cost_new < cost) {
                x = x_new;
                r = r_new;
                cost = cost_new;
                lambda = std::max(lambda / 10.0, 1e-15);
                step_accepted = true;
                break;
            }
            lambda *= 10.0;
        }

        double delta_norm = 0.0;
        for (double v : delta) delta_norm += v * v;
        delta_norm = std::sqrt(delta_norm);

        // Converged if no improving step could be found (we're stuck at/near
        // a stationary point even with heavy damping), the step size is
        // negligible, or the cost improvement between iterations is negligible.
        if (!step_accepted || delta_norm < tol ||
            (cost_before - cost) < tol * (1.0 + cost_before)) {
            converged = true;
            break;
        }
    }

    return OptimResult{x, cost, iterations, converged};
}

// ----------------------------------------------------------------
// Simulated Annealing
// ----------------------------------------------------------------
OptimResult simulated_annealing(FuncND f, std::vector<double> x0,
                                double T0, double cooling,
                                int max_iter, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> step_dist(0.0, 0.1);

    const int n = static_cast<int>(x0.size());
    auto x = x0;
    auto x_best = x;
    double fx = f(x);
    double f_best = fx;
    double T = T0;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto x_new = x;
        for (int i = 0; i < n; ++i)
            x_new[static_cast<size_t>(i)] += T * step_dist(rng);

        double fx_new = f(x_new);
        double dE = fx_new - fx;
        if (dE < 0.0 || unif(rng) < std::exp(-dE / (T + 1e-14))) {
            x  = x_new;
            fx = fx_new;
        }
        if (fx < f_best) { f_best = fx; x_best = x; }
        T *= cooling;
    }
    return OptimResult{x_best, f_best, static_cast<size_t>(max_iter), true};
}

// ----------------------------------------------------------------
// Differential Evolution
// ----------------------------------------------------------------
OptimResult differential_evolution(FuncND f,
                                   std::vector<std::pair<double,double>> bounds,
                                   int pop, double F_scale, double CR,
                                   int max_iter, unsigned seed) {
    const int n = static_cast<int>(bounds.size());
    std::mt19937 rng(seed);

    // Initialize population
    std::vector<std::vector<double>> population(
        static_cast<size_t>(pop), std::vector<double>(static_cast<size_t>(n)));
    for (auto& ind : population) {
        for (int j = 0; j < n; ++j) {
            std::uniform_real_distribution<double> d(
                bounds[static_cast<size_t>(j)].first,
                bounds[static_cast<size_t>(j)].second);
            ind[static_cast<size_t>(j)] = d(rng);
        }
    }

    std::vector<double> fitness(static_cast<size_t>(pop));
    for (int i = 0; i < pop; ++i)
        fitness[static_cast<size_t>(i)] = f(population[static_cast<size_t>(i)]);

    std::uniform_int_distribution<int> pop_dist(0, pop - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int> dim_dist(0, n - 1);

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < pop; ++i) {
            int a = i, b = i, c = i;
            while (a == i) a = pop_dist(rng);
            while (b == i || b == a) b = pop_dist(rng);
            while (c == i || c == a || c == b) c = pop_dist(rng);

            std::vector<double> mutant(static_cast<size_t>(n));
            for (int j = 0; j < n; ++j) {
                mutant[static_cast<size_t>(j)] =
                    population[static_cast<size_t>(a)][static_cast<size_t>(j)] +
                    F_scale * (population[static_cast<size_t>(b)][static_cast<size_t>(j)] -
                               population[static_cast<size_t>(c)][static_cast<size_t>(j)]);
                mutant[static_cast<size_t>(j)] = std::max(
                    bounds[static_cast<size_t>(j)].first,
                    std::min(bounds[static_cast<size_t>(j)].second,
                             mutant[static_cast<size_t>(j)]));
            }

            int jrand = dim_dist(rng);
            std::vector<double> trial = population[static_cast<size_t>(i)];
            for (int j = 0; j < n; ++j) {
                if (unif(rng) < CR || j == jrand)
                    trial[static_cast<size_t>(j)] = mutant[static_cast<size_t>(j)];
            }

            double ft = f(trial);
            if (ft <= fitness[static_cast<size_t>(i)]) {
                population[static_cast<size_t>(i)] = trial;
                fitness[static_cast<size_t>(i)]    = ft;
            }
        }
    }

    size_t best = 0;
    for (size_t i = 1; i < static_cast<size_t>(pop); ++i) {
        if (fitness[i] < fitness[best]) best = i;
    }
    return OptimResult{population[best], fitness[best],
                       static_cast<size_t>(max_iter), true};
}

// ----------------------------------------------------------------
// Particle Swarm Optimization
// ----------------------------------------------------------------
OptimResult particle_swarm(FuncND f,
                           std::vector<std::pair<double,double>> bounds,
                           int n_particles, int max_iter, unsigned seed) {
    const int n = static_cast<int>(bounds.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const double w = 0.7, c1 = 1.5, c2 = 1.5;

    std::vector<std::vector<double>> pos(
        static_cast<size_t>(n_particles),
        std::vector<double>(static_cast<size_t>(n)));
    std::vector<std::vector<double>> vel(
        static_cast<size_t>(n_particles),
        std::vector<double>(static_cast<size_t>(n), 0.0));
    std::vector<std::vector<double>> pbest(
        static_cast<size_t>(n_particles),
        std::vector<double>(static_cast<size_t>(n)));
    std::vector<double> pbest_f(static_cast<size_t>(n_particles));

    for (int i = 0; i < n_particles; ++i) {
        for (int j = 0; j < n; ++j) {
            double lo = bounds[static_cast<size_t>(j)].first;
            double hi = bounds[static_cast<size_t>(j)].second;
            pos[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                lo + unif(rng) * (hi - lo);
        }
        pbest[static_cast<size_t>(i)] = pos[static_cast<size_t>(i)];
        pbest_f[static_cast<size_t>(i)] =
            f(pos[static_cast<size_t>(i)]);
    }

    size_t gbest_idx = 0;
    for (size_t i = 1; i < static_cast<size_t>(n_particles); ++i) {
        if (pbest_f[i] < pbest_f[gbest_idx]) gbest_idx = i;
    }
    auto gbest = pbest[gbest_idx];
    double gbest_f = pbest_f[gbest_idx];

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n_particles; ++i) {
            for (int j = 0; j < n; ++j) {
                double r1 = unif(rng), r2 = unif(rng);
                vel[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    w * vel[static_cast<size_t>(i)][static_cast<size_t>(j)] +
                    c1 * r1 * (pbest[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                               pos[static_cast<size_t>(i)][static_cast<size_t>(j)]) +
                    c2 * r2 * (gbest[static_cast<size_t>(j)] -
                               pos[static_cast<size_t>(i)][static_cast<size_t>(j)]);
                pos[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
                    vel[static_cast<size_t>(i)][static_cast<size_t>(j)];
                pos[static_cast<size_t>(i)][static_cast<size_t>(j)] = std::max(
                    bounds[static_cast<size_t>(j)].first,
                    std::min(bounds[static_cast<size_t>(j)].second,
                             pos[static_cast<size_t>(i)][static_cast<size_t>(j)]));
            }
            double fi = f(pos[static_cast<size_t>(i)]);
            if (fi < pbest_f[static_cast<size_t>(i)]) {
                pbest[static_cast<size_t>(i)] = pos[static_cast<size_t>(i)];
                pbest_f[static_cast<size_t>(i)] = fi;
                if (fi < gbest_f) { gbest_f = fi; gbest = pos[static_cast<size_t>(i)]; }
            }
        }
    }
    return OptimResult{gbest, gbest_f, static_cast<size_t>(max_iter), true};
}

// ----------------------------------------------------------------
// CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
// ----------------------------------------------------------------
namespace {

// Symmetric Jacobi eigendecomposition: A = V diag(d) V^T (V orthonormal).
void jacobi_eig_sym(std::vector<std::vector<double>> A,
                    std::vector<double>& evals,
                    std::vector<std::vector<double>>& evecs) {
    const int n = static_cast<int>(A.size());
    evals.assign(static_cast<size_t>(n), 0.0);
    evecs.assign(static_cast<size_t>(n),
                 std::vector<double>(static_cast<size_t>(n), 0.0));
    for (int i = 0; i < n; ++i) {
        evecs[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;
    }

    constexpr int max_sweeps = 100;
    constexpr double eps = 1e-15;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                off += A[static_cast<size_t>(p)][static_cast<size_t>(q)] *
                       A[static_cast<size_t>(p)][static_cast<size_t>(q)];
            }
        }
        if (off < eps) {
            break;
        }

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[static_cast<size_t>(p)][static_cast<size_t>(q)];
                if (std::abs(apq) < eps) {
                    continue;
                }
                const double app = A[static_cast<size_t>(p)][static_cast<size_t>(p)];
                const double aqq = A[static_cast<size_t>(q)][static_cast<size_t>(q)];
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                for (int k = 0; k < n; ++k) {
                    const double akp = A[static_cast<size_t>(k)][static_cast<size_t>(p)];
                    const double akq = A[static_cast<size_t>(k)][static_cast<size_t>(q)];
                    A[static_cast<size_t>(k)][static_cast<size_t>(p)] = c * akp - s * akq;
                    A[static_cast<size_t>(p)][static_cast<size_t>(k)] =
                        A[static_cast<size_t>(k)][static_cast<size_t>(p)];
                    A[static_cast<size_t>(k)][static_cast<size_t>(q)] = s * akp + c * akq;
                    A[static_cast<size_t>(q)][static_cast<size_t>(k)] =
                        A[static_cast<size_t>(k)][static_cast<size_t>(q)];
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = evecs[static_cast<size_t>(k)][static_cast<size_t>(p)];
                    const double vkq = evecs[static_cast<size_t>(k)][static_cast<size_t>(q)];
                    evecs[static_cast<size_t>(k)][static_cast<size_t>(p)] = c * vkp - s * vkq;
                    evecs[static_cast<size_t>(k)][static_cast<size_t>(q)] = s * vkp + c * vkq;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        evals[static_cast<size_t>(i)] =
            A[static_cast<size_t>(i)][static_cast<size_t>(i)];
    }
}

double chi_n(int n) {
    if (n <= 0) {
        return 1.0;
    }
    const double nd = static_cast<double>(n);
    return std::sqrt(nd) * (1.0 - 1.0 / (4.0 * nd) + 1.0 / (21.0 * nd * nd));
}

std::vector<double> mat_vec(const std::vector<std::vector<double>>& M,
                            const std::vector<double>& v) {
    const int n = static_cast<int>(v.size());
    std::vector<double> r(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            r[static_cast<size_t>(i)] +=
                M[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                v[static_cast<size_t>(j)];
        }
    }
    return r;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double norm2(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

std::vector<double> outer_apply(
    const std::vector<std::vector<double>>& V,
    const std::vector<double>& d,
    const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<double> tmp(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            s += V[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                 x[static_cast<size_t>(i)];
        }
        tmp[static_cast<size_t>(j)] = d[static_cast<size_t>(j)] * s;
    }
    return mat_vec(V, tmp);
}

} // namespace

OptimResult cmaes(FuncND f, std::vector<double> x0, double sigma0,
                  int max_iter, unsigned seed) {
    constexpr double tol = 1e-8;

    const int n = static_cast<int>(x0.size());
    if (n <= 0) {
        return OptimResult{{}, 0.0, 0, false};
    }

    const int lambda = 4 + static_cast<int>(std::floor(3.0 * std::log(static_cast<double>(n))));
    const int mu = lambda / 2;

    std::vector<double> weights(static_cast<size_t>(mu));
    double w_sum = 0.0;
    for (int i = 0; i < mu; ++i) {
        weights[static_cast<size_t>(i)] =
            std::log(static_cast<double>(mu) + 0.5) -
            std::log(static_cast<double>(i + 1));
        w_sum += weights[static_cast<size_t>(i)];
    }
    for (int i = 0; i < mu; ++i) {
        weights[static_cast<size_t>(i)] /= w_sum;
    }

    double w_sq_sum = 0.0;
    for (int i = 0; i < mu; ++i) {
        w_sq_sum += weights[static_cast<size_t>(i)] * weights[static_cast<size_t>(i)];
    }
    const double mu_eff = 1.0 / w_sq_sum;

    const double c_sigma = (mu_eff + 2.0) / (static_cast<double>(n) + mu_eff + 5.0);
    const double d_sigma =
        1.0 + c_sigma +
        2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (static_cast<double>(n) + 1.0)) - 1.0);
    const double c_c =
        (4.0 + mu_eff / static_cast<double>(n)) /
        (static_cast<double>(n) + 4.0 + 2.0 * mu_eff / static_cast<double>(n));
    const double c_1 =
        2.0 / ((static_cast<double>(n) + 1.3) * (static_cast<double>(n) + 1.3) + mu_eff);
    const double c_mu = std::min(
        1.0 - c_1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) /
            ((static_cast<double>(n) + 2.0) * (static_cast<double>(n) + 2.0) + mu_eff));

    const double chiN = chi_n(n);
    const double sigma_thresh =
        (1.4 + 2.0 / (static_cast<double>(n) + 1.0)) * chiN;

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    auto mean = x0;
    double sigma = sigma0;

    std::vector<std::vector<double>> C(
        static_cast<size_t>(n),
        std::vector<double>(static_cast<size_t>(n), 0.0));
    for (int i = 0; i < n; ++i) {
        C[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;
    }

    std::vector<double> p_sigma(static_cast<size_t>(n), 0.0);
    std::vector<double> p_c(static_cast<size_t>(n), 0.0);

    std::vector<std::vector<double>> pop(
        static_cast<size_t>(lambda),
        std::vector<double>(static_cast<size_t>(n)));
    std::vector<double> fitness(static_cast<size_t>(lambda));

    double f_best = std::numeric_limits<double>::infinity();
    auto x_best = mean;
    bool converged = false;
    int generations = 0;

    for (int gen = 0; gen < max_iter; ++gen) {
        generations = gen + 1;

        std::vector<double> evals;
        std::vector<std::vector<double>> evecs;
        jacobi_eig_sym(C, evals, evecs);

        std::vector<double> sqrt_evals(static_cast<size_t>(n));
        std::vector<double> inv_sqrt_evals(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const double ev = std::max(evals[static_cast<size_t>(i)], 0.0);
            sqrt_evals[static_cast<size_t>(i)] = std::sqrt(ev);
            inv_sqrt_evals[static_cast<size_t>(i)] =
                ev > 1e-14 ? 1.0 / std::sqrt(ev) : 0.0;
        }

        for (int k = 0; k < lambda; ++k) {
            std::vector<double> z(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                z[static_cast<size_t>(i)] = normal(rng);
            }
            auto step = outer_apply(evecs, sqrt_evals, z);
            for (int i = 0; i < n; ++i) {
                pop[static_cast<size_t>(k)][static_cast<size_t>(i)] =
                    mean[static_cast<size_t>(i)] + sigma * step[static_cast<size_t>(i)];
            }
            fitness[static_cast<size_t>(k)] = f(pop[static_cast<size_t>(k)]);
        }

        std::vector<size_t> idx(static_cast<size_t>(lambda));
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return fitness[a] < fitness[b]; });

        if (fitness[idx[0]] < f_best) {
            f_best = fitness[idx[0]];
            x_best = pop[idx[0]];
        }

        const double f_spread =
            fitness[idx[static_cast<size_t>(mu - 1)]] - fitness[idx[0]];
        if (f_best < tol || sigma < tol || f_spread < tol) {
            converged = true;
            break;
        }

        const auto mean_old = mean;
        std::fill(mean.begin(), mean.end(), 0.0);
        for (int i = 0; i < mu; ++i) {
            const auto& xi = pop[idx[static_cast<size_t>(i)]];
            for (int j = 0; j < n; ++j) {
                mean[static_cast<size_t>(j)] +=
                    weights[static_cast<size_t>(i)] * xi[static_cast<size_t>(j)];
            }
        }

        std::vector<std::vector<double>> y_mu(static_cast<size_t>(mu));
        std::vector<double> y_w(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < mu; ++i) {
            y_mu[static_cast<size_t>(i)].resize(static_cast<size_t>(n));
            for (int j = 0; j < n; ++j) {
                y_mu[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    (pop[idx[static_cast<size_t>(i)]][static_cast<size_t>(j)] -
                     mean_old[static_cast<size_t>(j)]) /
                    sigma;
                y_w[static_cast<size_t>(j)] +=
                    weights[static_cast<size_t>(i)] *
                    y_mu[static_cast<size_t>(i)][static_cast<size_t>(j)];
            }
        }

        const auto invsqrt_yw = outer_apply(evecs, inv_sqrt_evals, y_w);
        const double ps_factor =
            std::sqrt(c_sigma * (2.0 - c_sigma) * mu_eff);
        for (int j = 0; j < n; ++j) {
            p_sigma[static_cast<size_t>(j)] =
                (1.0 - c_sigma) * p_sigma[static_cast<size_t>(j)] +
                ps_factor * invsqrt_yw[static_cast<size_t>(j)];
        }

        const double ps_norm = norm2(p_sigma);
        const double gen_factor =
            std::sqrt(1.0 - std::pow(1.0 - c_sigma, 2.0 * static_cast<double>(gen + 1)));
        const double h_sig =
            (ps_norm / (gen_factor + 1e-14) < sigma_thresh) ? 1.0 : 0.0;

        const double pc_factor = h_sig * std::sqrt(c_c * (2.0 - c_c) * mu_eff);
        for (int j = 0; j < n; ++j) {
            p_c[static_cast<size_t>(j)] =
                (1.0 - c_c) * p_c[static_cast<size_t>(j)] +
                pc_factor * y_w[static_cast<size_t>(j)];
        }

        std::vector<std::vector<double>> C_new(
            static_cast<size_t>(n),
            std::vector<double>(static_cast<size_t>(n), 0.0));
        const double cc_factor = (1.0 - h_sig) * c_c * (2.0 - c_c);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                double rank1 = p_c[static_cast<size_t>(i)] * p_c[static_cast<size_t>(j)];
                double rank_mu = 0.0;
                for (int k = 0; k < mu; ++k) {
                    rank_mu += weights[static_cast<size_t>(k)] *
                               y_mu[static_cast<size_t>(k)][static_cast<size_t>(i)] *
                               y_mu[static_cast<size_t>(k)][static_cast<size_t>(j)];
                }
                C_new[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    (1.0 - c_1 - c_mu) * C[static_cast<size_t>(i)][static_cast<size_t>(j)] +
                    c_1 * (rank1 + cc_factor * C[static_cast<size_t>(i)][static_cast<size_t>(j)]) +
                    c_mu * rank_mu;
                C_new[static_cast<size_t>(j)][static_cast<size_t>(i)] =
                    C_new[static_cast<size_t>(i)][static_cast<size_t>(j)];
            }
        }
        C = C_new;

        sigma *= std::exp((c_sigma / d_sigma) * (ps_norm / chiN - 1.0));
    }

    return OptimResult{x_best, f_best, static_cast<size_t>(generations), converged};
}

// ----------------------------------------------------------------
// Scalar root finders
// ----------------------------------------------------------------
double bisection(Func1D f, double a, double b, double tol, int max_iter) {
    double fa = f(a);
    for (int i = 0; i < max_iter; ++i) {
        double mid = 0.5 * (a + b);
        if ((b - a) < tol) break;
        double fmid = f(mid);
        if (fa * fmid <= 0.0) { b = mid; }
        else { a = mid; fa = fmid; }
    }
    return 0.5 * (a + b);
}

double brentq(Func1D f, double a, double b, double tol, int max_iter) {
    double fa = f(a), fb = f(b);
    if (fa * fb > 0.0) return 0.5 * (a + b);
    double c = a, fc = fa, d = b - a, e = d;
    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(fb) < tol) break;
        if ((fb > 0.0) == (fc > 0.0)) { c = a; fc = fa; d = e = b - a; }
        if (std::abs(fc) < std::abs(fb)) {
            a = b; fa = fb; b = c; fb = fc; c = a; fc = fa;
        }
        double tol1 = 2.0 * 2.2e-16 * std::abs(b) + 0.5 * tol;
        double xm = 0.5 * (c - b);
        if (std::abs(xm) <= tol1 || std::abs(fb) < 1e-15) break;
        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc; double r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q; else p = -p;
            if (2.0 * p < std::min(3.0 * xm * q - std::abs(tol1 * q),
                                    std::abs(e * q))) {
                e = d; d = p / q;
            } else { d = xm; e = d; }
        } else { d = xm; e = d; }
        a = b; fa = fb;
        b += (std::abs(d) > tol1) ? d : (xm > 0.0 ? tol1 : -tol1);
        fb = f(b);
    }
    return b;
}

double illinois(Func1D f, double a, double b, double tol, int max_iter) {
    double fa = f(a), fb = f(b);
    if (fa * fb > 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double c = a;
    // -1 => a retained on the previous iteration, +1 => b retained, 0 => none yet.
    int retained = 0;
    for (int i = 0; i < max_iter; ++i) {
        const double denom = fb - fa;
        if (std::abs(denom) < 1e-300) break;
        c = b - fb * (b - a) / denom;
        const double fc = f(c);
        if (std::abs(fc) < tol || std::abs(b - a) < tol) {
            return c;
        }
        if (fa * fc < 0.0) {
            // Root lies in [a, c]: b is replaced, a is retained.
            b = c; fb = fc;
            if (retained == -1) { fa *= 0.5; }
            retained = -1;
        } else {
            // Root lies in [c, b]: a is replaced, b is retained.
            a = c; fa = fc;
            if (retained == 1) { fb *= 0.5; }
            retained = 1;
        }
    }
    return c;
}

double secant(Func1D f, double x0, double x1, double tol, int max_iter) {
    for (int i = 0; i < max_iter; ++i) {
        double f0 = f(x0), f1 = f(x1);
        if (std::abs(f1) < tol) break;
        double denom = f1 - f0;
        if (std::abs(denom) < 1e-14) break;
        double x2 = x1 - f1 * (x1 - x0) / denom;
        if (std::abs(x2 - x1) < tol) { x1 = x2; break; }
        x0 = x1; x1 = x2;
    }
    return x1;
}

double halley(Func1D f, Func1D df, Func1D d2f, double x0,
              double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; ++i) {
        double fx = f(x), dfx = df(x), d2fx = d2f(x);
        double denom = 2.0 * dfx * dfx - fx * d2fx;
        if (std::abs(denom) < 1e-14) break;
        double step = 2.0 * fx * dfx / denom;
        x -= step;
        if (std::abs(step) < tol) break;
    }
    return x;
}

double fixed_point(Func1D g, double x0, double tol, int max_iter) {
    double x = x0;
    for (int i = 0; i < max_iter; ++i) {
        double xnew = g(x);
        if (std::abs(xnew - x) < tol) { x = xnew; break; }
        x = xnew;
    }
    return x;
}

} // namespace ms
