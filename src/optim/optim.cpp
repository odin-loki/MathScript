#include "ms/optim/optim.hpp"
#include <cmath>

namespace ms {

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
    Func2D f, double x0, double y0, double, int) {
    return {x0, y0};
}

std::pair<double, double> broyden(Func2D f, double x0, double y0) {
    return {x0, y0};
}

std::tuple<double, double, double> minimize_with_constraints(
    Func2D f, double x0, double y0, std::vector<double>) {
    return {x0, y0, f(x0, y0)};
}

std::tuple<std::vector<double>, double> simplex_solver(
    std::vector<double>, std::vector<std::vector<double>>, std::vector<double>) {
    return std::tuple<std::vector<double>, double>{std::vector<double>{}, 0.0};
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
