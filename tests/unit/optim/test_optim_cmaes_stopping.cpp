// Regression test for cmaes's stopping criterion.
//
// The test was `f_best < tol || sigma < tol || f_spread < tol` with tol = 1e-8.
// The first clause compares the raw OBJECTIVE VALUE against a tolerance, as
// though the optimum were known to be zero. For any objective whose optimum is
// negative it is true almost everywhere, so the search stopped on iteration 1
// and reported success: minimising sum (x_i - 3)^2 - 100 from (0, 0) returned
// (0.474, 1.368) with converged = true.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/optim/optim.hpp"

namespace {

// Sphere with its optimum shifted to an arbitrary value, so that the objective
// value at the optimum carries no information about proximity to it.
ms::FuncND shifted_sphere(double centre, double offset) {
    return [centre, offset](const std::vector<double>& x) {
        double s = offset;
        for (double v : x) {
            s += (v - centre) * (v - centre);
        }
        return s;
    };
}

} // namespace

TEST(CmaesStopping, ConvergesWhenTheOptimumValueIsNegative) {
    // Optimum at (3, 3) with value -100.
    const auto r = ms::cmaes(shifted_sphere(3.0, -100.0), {0.0, 0.0}, 1.0, 500, 42u, {});
    ASSERT_EQ(r.x.size(), 2u);
    EXPECT_NEAR(r.x[0], 3.0, 1e-3);
    EXPECT_NEAR(r.x[1], 3.0, 1e-3);
    EXPECT_NEAR(r.f_val, -100.0, 1e-6);
    EXPECT_GT(r.iterations, 1u) << "stopped immediately on the objective value";
}

TEST(CmaesStopping, ConvergesWhenTheOptimumValueIsLargeAndPositive) {
    // Optimum at (-2, -2) with value +50: the old `f_best < tol` clause could
    // never fire here, so this path always worked and must keep working.
    const auto r = ms::cmaes(shifted_sphere(-2.0, 50.0), {0.0, 0.0}, 1.0, 500, 42u, {});
    ASSERT_EQ(r.x.size(), 2u);
    EXPECT_NEAR(r.x[0], -2.0, 1e-3);
    EXPECT_NEAR(r.x[1], -2.0, 1e-3);
    EXPECT_NEAR(r.f_val, 50.0, 1e-6);
}

TEST(CmaesStopping, ConvergesOnAPlainSphereAtTheOrigin) {
    const auto r = ms::cmaes(shifted_sphere(0.0, 0.0), {1.5, -1.5}, 1.0, 500, 7u, {});
    ASSERT_EQ(r.x.size(), 2u);
    EXPECT_NEAR(r.x[0], 0.0, 1e-3);
    EXPECT_NEAR(r.x[1], 0.0, 1e-3);
}

TEST(CmaesStopping, HandlesAStronglyNegativeOptimum) {
    // Scale the offset far below the tolerance so nothing accidental rescues it.
    const auto r = ms::cmaes(shifted_sphere(1.0, -1e6), {5.0, 5.0}, 2.0, 800, 11u, {});
    ASSERT_EQ(r.x.size(), 2u);
    EXPECT_NEAR(r.x[0], 1.0, 1e-2);
    EXPECT_NEAR(r.x[1], 1.0, 1e-2);
}

TEST(CmaesStopping, FindsTheRosenbrockValley) {
    // Optimum (1, 1) with value 0 -- the classic hard case, and one where the
    // objective value at the optimum IS zero, so the old clause was harmless.
    const ms::FuncND rosen = [](const std::vector<double>& x) {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    };
    const auto r = ms::cmaes(rosen, {-1.2, 1.0}, 0.5, 2000, 3u, {});
    ASSERT_EQ(r.x.size(), 2u);
    EXPECT_NEAR(r.x[0], 1.0, 1e-2);
    EXPECT_NEAR(r.x[1], 1.0, 2e-2);
}

TEST(CmaesStopping, DegenerateInputIsRejected) {
    const auto r = ms::cmaes(shifted_sphere(0.0, 0.0), {}, 1.0, 10, 1u, {});
    EXPECT_TRUE(r.x.empty());
    EXPECT_FALSE(r.converged);
}
