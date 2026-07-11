#pragma once

#include <functional>
#include <tuple>
#include <vector>

namespace ms {

using Func1D = std::function<double(double)>;
using Func2D = std::function<double(double, double)>;
using FuncND = std::function<double(const std::vector<double>&)>;

struct GradientDescentResult {
    double x;
    double y;
    double f_val;
    size_t iterations;
    bool converged;
};

struct OptimResult {
    std::vector<double> x;
    double f_val;
    size_t iterations;
    bool converged;
};

// --- existing ---
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

// --- N-dimensional unconstrained ---
OptimResult nelder_mead(FuncND f, std::vector<double> x0,
                        double tol = 1e-8, int max_iter = 1000);
OptimResult bfgs(FuncND f, std::vector<double> x0,
                 double tol = 1e-8, int max_iter = 500);
OptimResult lbfgs(FuncND f, std::vector<double> x0,
                  int m = 5, double tol = 1e-8, int max_iter = 500);
OptimResult adam(FuncND f, std::vector<double> x0,
                 double alpha = 0.001, double beta1 = 0.9,
                 double beta2 = 0.999, int max_iter = 1000);

// --- global optimisers ---
OptimResult simulated_annealing(FuncND f, std::vector<double> x0,
                                double T0 = 1.0, double cooling = 0.995,
                                int max_iter = 10000, unsigned seed = 42);
OptimResult differential_evolution(FuncND f,
                                   std::vector<std::pair<double,double>> bounds,
                                   int pop = 15, double F = 0.8, double CR = 0.9,
                                   int max_iter = 1000, unsigned seed = 42);
OptimResult particle_swarm(FuncND f,
                           std::vector<std::pair<double,double>> bounds,
                           int n_particles = 30, int max_iter = 500,
                           unsigned seed = 42);
OptimResult cmaes(FuncND f, std::vector<double> x0, double sigma0 = 0.5,
                  int max_iter = 1000, unsigned seed = 42);

// --- nonlinear equation solvers ---
double bisection(Func1D f, double a, double b, double tol = 1e-10,
                 int max_iter = 200);
double brentq(Func1D f, double a, double b, double tol = 1e-10,
              int max_iter = 200);
double secant(Func1D f, double x0, double x1, double tol = 1e-10,
              int max_iter = 100);
double halley(Func1D f, Func1D df, Func1D d2f, double x0,
              double tol = 1e-12, int max_iter = 50);
double fixed_point(Func1D g, double x0, double tol = 1e-10,
                   int max_iter = 200);

} // namespace ms
