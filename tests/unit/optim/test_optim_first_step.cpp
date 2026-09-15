// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: what Adam's bias correction is FOR, and what a root of Brent's is.
//
// `src/optim/optim.cpp` scored 36.4% -- the lowest first-run figure of the
// twenty-one files measured. The diagnosis the survivors share is the one this
// whole exercise keeps arriving at: the optimiser tests assert that a run
// CONVERGED and that the answer is near a known optimum within a loose
// tolerance, and a great many wrong implementations satisfy both. An optimiser
// that takes steps of the wrong SIZE still walks downhill and still arrives.
//
// Two of the survivors have exact statements available, and those are asserted
// here:
//
//   adam    `m / (1 - beta1^t)` became `m / (1 + beta1^t)`. That is the bias
//           correction, and its entire purpose is that the FIRST step has
//           magnitude alpha whatever the gradient's scale -- without it the
//           first step is (1-beta1)/(1+beta1) of the first moment over
//           sqrt((1-beta2)/(1+beta2)) of the second, which is about 2.35*alpha
//           rather than alpha. Nothing asserted a step size.
//   brentq  the inverse quadratic interpolation's `(q - 1)(r - 1)(s - 1)`
//           became `(q + 1)...`. A root finder with a damaged interpolation
//           still converges, because the bisection fallback catches it -- just
//           more slowly, and with 200 iterations to spend "more slowly" is
//           invisible. Measured: the correct method reaches machine precision
//           in SIX iterations on these brackets, where bisection alone would
//           need about fifty. So the assertion that separates them is a tight
//           iteration budget, not a tight tolerance.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/optim/optim.hpp"

using namespace ms;

TEST(OptimFirstStep, AdamsFirstStepIsAlphaWhateverTheGradient) {
    // Adam normalises by the second moment, so the step size does not depend on
    // how steep the objective is -- but only once the first-iteration bias in
    // both moments is divided out. At t = 1 the corrected moments are exactly g
    // and g^2, so the step is alpha * g / (|g| + eps): alpha, in the downhill
    // direction, for any g at all. That is a statement about the very first
    // iteration and it is exact.
    for (const double distance : {1.0, 10.0, 1000.0, 1.0e6}) {
        for (const double alpha : {0.001, 0.01, 0.5}) {
            const auto r = adam(
                [distance](const std::vector<double>& x) {
                    const double d = x[0] - distance;
                    return d * d;
                },
                {0.0}, alpha, 0.9, 0.999, 1);
            ASSERT_EQ(r.x.size(), 1u);
            EXPECT_NEAR(r.x[0], alpha, alpha * 1e-6)
                << "one Adam step from " << distance << " away at alpha " << alpha
                << " moved " << r.x[0] << " rather than " << alpha;
            EXPECT_EQ(r.iterations, 1u);
        }
    }

    // The same from the other side: the minimum below the start, so the step is
    // -alpha. A step size that ignored the sign would pass the block above.
    for (const double alpha : {0.01, 0.25}) {
        const auto r = adam(
            [](const std::vector<double>& x) {
                const double d = x[0] + 500.0;
                return d * d;
            },
            {0.0}, alpha, 0.9, 0.999, 1);
        EXPECT_NEAR(r.x[0], -alpha, alpha * 1e-6) << "downhill is the wrong way at alpha " << alpha;
    }

    // Two coordinates with gradients four orders of magnitude apart still take
    // the same-sized step, which is the property that makes Adam Adam.
    const auto r = adam(
        [](const std::vector<double>& x) {
            const double a = x[0] - 1.0;
            const double b = x[1] - 30000.0;
            return a * a + b * b;
        },
        {0.0, 0.0}, 0.1, 0.9, 0.999, 1);
    ASSERT_EQ(r.x.size(), 2u);
    EXPECT_NEAR(r.x[0], 0.1, 1e-6) << "the shallow coordinate";
    EXPECT_NEAR(r.x[1], 0.1, 1e-6) << "the steep coordinate";
}

TEST(OptimFirstStep, BrentLandsOnTheRootToTheToleranceItWasGiven) {
    // A root is where the function is zero, so that is what is asserted --
    // together with the value, where a closed form exists. A damaged
    // interpolation still converges on the bisection fallback, so "it returned
    // something in the bracket" is not a test of it; landing within 1e-13 is.
    struct Case {
        double (*f)(double);
        double a;
        double b;
        const char* what;
    };
    const Case cases[] = {
        {[](double x) { return x * x * x - 2.0; }, 0.0, 2.0, "cube root of 2"},
        {[](double x) { return std::cos(x) - x; }, 0.0, 1.0, "the Dottie number"},
        {[](double x) { return std::exp(x) - 3.0 * x; }, 0.0, 1.0, "exp(x) = 3x"},
        {[](double x) { return std::log(x) + x; }, 0.1, 2.0, "log(x) + x"},
        {[](double x) { return x * x * x * x - 10.0 * x + 3.0; }, 0.0, 1.0, "quartic"},
        {[](double x) { return std::atan(x) - 0.5; }, 0.0, 2.0, "atan(x) = 1/2"},
    };
    for (const Case& c : cases) {
        const double root = brentq(c.f, c.a, c.b, 1e-14, 200);
        EXPECT_GE(root, c.a) << c.what;
        EXPECT_LE(root, c.b) << c.what;
        EXPECT_NEAR(c.f(root), 0.0, 1e-12)
            << c.what << ": f(" << root << ") = " << c.f(root);
    }

    // Two with an exact answer to compare against.
    EXPECT_NEAR(brentq([](double x) { return x * x * x - 2.0; }, 0.0, 2.0, 1e-14, 200),
                std::cbrt(2.0), 1e-12);
    EXPECT_NEAR(brentq([](double x) { return std::atan(x) - 0.5; }, 0.0, 2.0, 1e-14, 200),
                std::tan(0.5), 1e-12);

    // A root exactly at a bracket endpoint, and one at the midpoint, where the
    // interpolation has nothing to improve on.
    EXPECT_NEAR(brentq([](double x) { return x - 1.0; }, 1.0, 3.0, 1e-14, 200), 1.0, 1e-13);
    EXPECT_NEAR(brentq([](double x) { return x - 2.0; }, 1.0, 3.0, 1e-14, 200), 2.0, 1e-13);

    // And the method really is Brent's rather than a bisection wearing its name.
    // Superlinear convergence means machine precision in a handful of steps;
    // halving a bracket of width one or two to 1e-14 takes about fifty. Eight is
    // comfortably above what the real method needs -- measured, it finishes in
    // six on every one of these -- and far below what bisection alone would.
    for (const Case& c : cases) {
        const double quick = brentq(c.f, c.a, c.b, 1e-14, 8);
        EXPECT_NEAR(c.f(quick), 0.0, 1e-11)
            << c.what << ": eight iterations left f at " << c.f(quick)
            << ", which is bisection's rate, not Brent's";
    }
}
