// Wave 56: Global optimization and new scalar root finder tests
#define _USE_MATH_DEFINES
#include "ms/optim/optim.hpp"
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------------------------------------
// Rosenbrock function (classic test)
// -----------------------------------------------------------------------
static double rosenbrock(const std::vector<double>& x) {
    return 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) +
           (1.0 - x[0]) * (1.0 - x[0]);
}

// Simple sphere
static double sphere(const std::vector<double>& x) {
    double s = 0.0;
    for (auto v : x) s += v * v;
    return s;
}

// -----------------------------------------------------------------------
// Nelder-Mead
// -----------------------------------------------------------------------
TEST(OptimNelderMead, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::nelder_mead(sphere, {2.0, 3.0});
    EXPECT_LT(r.f_val, 0.01);
}

TEST(OptimNelderMead, Rosenbrock_FvalNearZero) {
    ms::OptimResult r = ms::nelder_mead(rosenbrock, {0.0, 0.0}, 1e-8, 5000);
    EXPECT_LT(r.f_val, 1.0);
}

TEST(OptimNelderMead, ResultIsFinite) {
    ms::OptimResult r = ms::nelder_mead(sphere, {1.0, 1.0, 1.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    for (auto v : r.x) EXPECT_TRUE(std::isfinite(v));
}

TEST(OptimNelderMead, CorrectDimension) {
    ms::OptimResult r = ms::nelder_mead(sphere, {1.0, 2.0, 3.0});
    EXPECT_EQ(r.x.size(), 3u);
}

// -----------------------------------------------------------------------
// BFGS
// -----------------------------------------------------------------------
TEST(OptimBFGS, Sphere2D_ConvergesNearZero) {
    ms::OptimResult r = ms::bfgs(sphere, {2.0, 3.0});
    EXPECT_LT(r.f_val, 1e-6);
}

TEST(OptimBFGS, Sphere5D_Converges) {
    ms::OptimResult r = ms::bfgs(sphere, {1.0, 1.0, 1.0, 1.0, 1.0});
    EXPECT_LT(r.f_val, 1e-4);
}

TEST(OptimBFGS, ResultIsFinite) {
    ms::OptimResult r = ms::bfgs(sphere, {3.0, -2.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
}

// -----------------------------------------------------------------------
// L-BFGS
// -----------------------------------------------------------------------
TEST(OptimLBFGS, Sphere2D_Converges) {
    ms::OptimResult r = ms::lbfgs(sphere, {5.0, 5.0});
    EXPECT_LT(r.f_val, 1e-4);
}

TEST(OptimLBFGS, ResultIsFinite) {
    ms::OptimResult r = ms::lbfgs(sphere, {1.0, 2.0, 3.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
    EXPECT_EQ(r.x.size(), 3u);
}

// -----------------------------------------------------------------------
// Adam
// -----------------------------------------------------------------------
TEST(OptimAdam, Sphere2D_ReducesObjective) {
    ms::OptimResult r = ms::adam(sphere, {2.0, 2.0});
    EXPECT_LT(r.f_val, 4.0);  // Should reduce from initial 8.0
}

TEST(OptimAdam, ResultIsFinite) {
    ms::OptimResult r = ms::adam(sphere, {1.0, 1.0});
    EXPECT_TRUE(std::isfinite(r.f_val));
}

// -----------------------------------------------------------------------
// Simulated Annealing
// -----------------------------------------------------------------------
TEST(OptimSimulatedAnnealing, Sphere2D_FindsNearOptimum) {
    ms::OptimResult r = ms::simulated_annealing(
        sphere, {2.0, 2.0}, 1.0, 0.99, 5000, 42);
    EXPECT_LT(r.f_val, 2.0);
}

TEST(OptimSimulatedAnnealing, ResultIsFinite) {
    ms::OptimResult r = ms::simulated_annealing(
        sphere, {3.0, -3.0}, 2.0, 0.995, 3000, 99);
    EXPECT_TRUE(std::isfinite(r.f_val));
}

TEST(OptimSimulatedAnnealing, CorrectDimension) {
    ms::OptimResult r = ms::simulated_annealing(
        sphere, {1.0, 2.0, 3.0}, 1.0, 0.99, 1000, 1);
    EXPECT_EQ(r.x.size(), 3u);
}

// -----------------------------------------------------------------------
// Differential Evolution
// -----------------------------------------------------------------------
TEST(OptimDifferentialEvolution, Sphere2D_FindsNearOptimum) {
    std::vector<std::pair<double,double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    ms::OptimResult r = ms::differential_evolution(sphere, bounds, 20, 0.8, 0.9, 500, 42);
    EXPECT_LT(r.f_val, 0.5);
}

TEST(OptimDifferentialEvolution, ResultIsFinite) {
    std::vector<std::pair<double,double>> bounds = {{-10.0, 10.0}, {-10.0, 10.0}};
    ms::OptimResult r = ms::differential_evolution(sphere, bounds, 10, 0.8, 0.9, 200, 7);
    EXPECT_TRUE(std::isfinite(r.f_val));
    EXPECT_EQ(r.x.size(), 2u);
}

TEST(OptimDifferentialEvolution, RespectsRespectsBounds) {
    std::vector<std::pair<double,double>> bounds = {{-2.0, 2.0}, {-2.0, 2.0}};
    ms::OptimResult r = ms::differential_evolution(sphere, bounds, 15, 0.8, 0.9, 300, 3);
    EXPECT_GE(r.x[0], -2.0);
    EXPECT_LE(r.x[0],  2.0);
    EXPECT_GE(r.x[1], -2.0);
    EXPECT_LE(r.x[1],  2.0);
}

// -----------------------------------------------------------------------
// Particle Swarm
// -----------------------------------------------------------------------
TEST(OptimParticleSwarm, Sphere2D_ReducesObjective) {
    std::vector<std::pair<double,double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    ms::OptimResult r = ms::particle_swarm(sphere, bounds, 20, 200, 42);
    EXPECT_LT(r.f_val, 1.0);
}

TEST(OptimParticleSwarm, ResultIsFinite) {
    std::vector<std::pair<double,double>> bounds = {{-10.0, 10.0}, {-10.0, 10.0}};
    ms::OptimResult r = ms::particle_swarm(sphere, bounds, 15, 100, 99);
    EXPECT_TRUE(std::isfinite(r.f_val));
    EXPECT_EQ(r.x.size(), 2u);
}

TEST(OptimParticleSwarm, BestNotWorse) {
    std::vector<std::pair<double,double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    ms::OptimResult r = ms::particle_swarm(sphere, bounds, 20, 300, 42);
    EXPECT_LE(r.f_val, sphere(r.x));  // should be the best value
}

// -----------------------------------------------------------------------
// Bisection
// -----------------------------------------------------------------------
TEST(OptimBisection, FindsRootOfLinear) {
    // f(x) = x - 3  => root at 3
    ms::Func1D f = [](double x) { return x - 3.0; };
    double r = ms::bisection(f, 0.0, 10.0);
    EXPECT_NEAR(r, 3.0, 1e-8);
}

TEST(OptimBisection, FindsRootOfSine) {
    // sin(x) = 0 in [1, 4] => root at pi
    double r = ms::bisection([](double x){ return std::sin(x); }, 1.0, 4.0);
    EXPECT_NEAR(r, M_PI, 1e-8);
}

// -----------------------------------------------------------------------
// Brent's method
// -----------------------------------------------------------------------
TEST(OptimBrentq, FindsRootOfCubic) {
    // f(x) = x^3 - 2x - 5 => root near 2.0946
    ms::Func1D f = [](double x) { return x * x * x - 2.0 * x - 5.0; };
    double r = ms::brentq(f, 1.0, 3.0);
    EXPECT_NEAR(f(r), 0.0, 1e-8);
}

TEST(OptimBrentq, FindsRootOfCos) {
    double r = ms::brentq([](double x){ return std::cos(x); }, 0.0, 2.0);
    EXPECT_NEAR(r, M_PI / 2.0, 1e-8);
}

// -----------------------------------------------------------------------
// Secant method
// -----------------------------------------------------------------------
TEST(OptimSecant, FindsRootOfQuadratic) {
    ms::Func1D f = [](double x) { return x * x - 4.0; };
    double r = ms::secant(f, 1.0, 3.0);
    EXPECT_NEAR(std::abs(r), 2.0, 1e-6);
}

// -----------------------------------------------------------------------
// Halley's method
// -----------------------------------------------------------------------
TEST(OptimHalley, FindsRootOfCubic) {
    ms::Func1D f  = [](double x) { return x * x * x - x - 1.0; };
    ms::Func1D df = [](double x) { return 3.0 * x * x - 1.0; };
    ms::Func1D d2f = [](double x) { return 6.0 * x; };
    double r = ms::halley(f, df, d2f, 1.5);
    EXPECT_NEAR(f(r), 0.0, 1e-10);
}

// -----------------------------------------------------------------------
// Fixed point
// -----------------------------------------------------------------------
TEST(OptimFixedPoint, SqrtViaFixedPoint) {
    // g(x) = (x + 2/x)/2 => fixed point is sqrt(2)
    ms::Func1D g = [](double x) { return (x + 2.0 / x) / 2.0; };
    double r = ms::fixed_point(g, 1.5);
    EXPECT_NEAR(r, std::sqrt(2.0), 1e-8);
}
