#pragma once

#include <functional>
#include <tuple>
#include <vector>

namespace ms {

using Func1D = std::function<double(double)>;
using Func2D = std::function<double(double, double)>;
using FuncND = std::function<double(const std::vector<double>&)>;

// Residual function type: maps parameters to a vector of residuals r(x) (e.g.
// r_i = model(x, t_i) - y_i for a curve-fitting problem with data points
// (t_i, y_i)). Used by levenberg_marquardt, which minimizes sum(r_i(x)^2),
// i.e. nonlinear least squares.
using ResidualFunc = std::function<std::vector<double>(const std::vector<double>&)>;

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

/// @brief Levenberg-Marquardt algorithm for nonlinear least squares: a damped
///        Gauss-Newton method that interpolates between Gauss-Newton (fast
///        convergence near the optimum, using the Jacobian-transpose-Jacobian
///        approximation to the Hessian) and gradient descent (robust but
///        slow, used when far from the optimum or when the local quadratic
///        approximation is poor).
///
/// At each iteration:
///   1. Compute the Jacobian J via central finite differences of the
///      residual function: J[i][j] = d(r_i)/d(x_j).
///   2. Solve the damped normal equations
///      (J^T*J + lambda*diag(J^T*J)) * delta = -J^T*r for the step `delta`.
///   3. Evaluate the new sum-of-squares at x+delta. If it's lower than the
///      current sum-of-squares, accept the step (x = x+delta) and decrease
///      lambda (move toward Gauss-Newton). If not, reject the step (keep x
///      unchanged), increase lambda (move toward gradient descent), and
///      retry step 2 with the new lambda (the Jacobian, still valid at the
///      unchanged x, is not recomputed).
///   4. Repeat until convergence (step size or cost improvement below `tol`)
///      or `max_iter` reached.
///
/// @param residuals the residual function r(x) (see ResidualFunc above).
/// @param x0 initial parameter guess.
/// @param max_iter maximum outer iterations.
/// @param tol convergence tolerance on step size / cost improvement.
/// @param lambda0 initial damping parameter.
/// @return OptimResult{x, f_val, iterations, converged} where f_val is the
///         final sum of squared residuals (sum(r_i(x)^2)) at the returned x.
OptimResult levenberg_marquardt(ResidualFunc residuals, std::vector<double> x0,
                                 int max_iter = 200, double tol = 1e-10,
                                 double lambda0 = 1e-3);

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

/// @brief Illinois algorithm: an anti-stagnation modification of regula falsi
///        (false position) for bracketed scalar root-finding.
///
/// Standard regula falsi picks the next estimate as the x-intercept of the
/// secant line through (a, f(a)) and (b, f(b)): c = b - f(b)*(b-a)/(f(b)-f(a)).
/// For strongly convex or concave functions this can stagnate badly: one
/// endpoint is retained for many consecutive iterations while the other
/// crawls toward the root, giving convergence that is barely linear. The
/// Illinois modification tracks which endpoint was retained on the previous
/// step; if the same endpoint is retained again on the current step, its
/// stored function value is halved before the next false-position update.
/// This shrinks that endpoint's influence on the secant estimate, forcing
/// the stagnant side to move and restoring superlinear convergence, while
/// still guaranteeing the root stays bracketed at every step.
///
/// @param f        Continuous scalar function to find a root of.
/// @param a        Left bracket endpoint.
/// @param b        Right bracket endpoint.
/// @param tol      Convergence tolerance, applied to both |f(c)| and the
///                 bracket width |b-a|.
/// @param max_iter Maximum number of iterations.
/// @return The estimated root. Returns NaN if the interval does not bracket
///         a root, i.e. f(a) and f(b) have the same sign (f(a)*f(b) > 0).
/// @note Converges at least as fast as plain regula falsi, and typically
///       much faster on functions where regula falsi stagnates (e.g.
///       strongly convex/concave functions on asymmetric brackets), since a
///       stagnating endpoint's contribution is repeatedly halved instead of
///       being left fixed indefinitely.
/// @accuracy Superlinear convergence (order ~1.4-1.8) in the typical case;
///           never worse than linear, unlike plain regula falsi which can
///           stagnate to sub-linear effective progress on adversarial
///           brackets.
double illinois(Func1D f, double a, double b, double tol = 1e-10,
                int max_iter = 200);

double secant(Func1D f, double x0, double x1, double tol = 1e-10,
              int max_iter = 100);
double halley(Func1D f, Func1D df, Func1D d2f, double x0,
              double tol = 1e-12, int max_iter = 50);
double fixed_point(Func1D g, double x0, double tol = 1e-10,
                   int max_iter = 200);

} // namespace ms
