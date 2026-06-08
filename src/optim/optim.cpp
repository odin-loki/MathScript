#include "ms/optim/optim.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace ms {

namespace {

double clip(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
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
        const double grad_x = (f(x0 + h, y0) - f(x0 - h, y0)) / (2.0 * h);
        const double grad_y = (f(x0, y0 + h) - f(x0, y0 - h)) / (2.0 * h);

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
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    double c = b - (b - a) / phi;
    double d = a + (b - a) / phi;
    while (std::abs(b - a) > tol) {
        if (f(c) < f(d)) {
            b = d;
        } else {
            a = c;
        }
        c = b - (b - a) / phi;
        d = a + (b - a) / phi;
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

} // namespace ms
