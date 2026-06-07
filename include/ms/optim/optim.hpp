#pragma once

#include <functional>
#include <tuple>
#include <vector>

namespace ms {

using Func1D = std::function<double(double)>;
using Func2D = std::function<double(double, double)>;

struct GradientDescentResult {
    double x;
    double y;
    double f_val;
    size_t iterations;
    bool converged;
};

GradientDescentResult gradient_descent(
    Func2D f, double x0, double y0, double learning_rate, int max_iterations);
std::pair<double, double> newton_raphson(
    Func2D f, double x0, double y0, double tol = 1e-10, int max_iter = 100);
std::pair<double, double> broyden(Func2D f, double x0, double y0);
std::tuple<double, double, double> minimize_with_constraints(
    Func2D f, double x0, double y0, std::vector<double> constraints);
std::tuple<std::vector<double>, double> simplex_solver(
    std::vector<double> c, std::vector<std::vector<double>> A, std::vector<double> b);

double golden_section(Func1D f, double a, double b, double tol = 1e-8);
double newton_1d(Func1D f, double x0, double tol = 1e-10, int max_iter = 100);

} // namespace ms
