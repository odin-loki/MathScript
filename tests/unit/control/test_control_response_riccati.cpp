// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for two audited control defects.
//
//  * step_response / impulse_response sat under a section heading reading
//    "Time response via matrix exponential (trapezoidal integration)" but were
//    forward Euler with a fixed dt = t_end/(n_pts-1) and no stability guard:
//    unconditionally unstable whenever |1 + dt*lambda| > 1, with no error
//    return. They now propagate exactly with the zero-order-hold discretisation
//    the heading always described.
//  * riccati, dare, lqr and lqe built R^{-1} as the element-wise reciprocal of
//    R's diagonal, which equals the inverse only when R is diagonal. Every
//    off-diagonal entry was discarded silently, so a cross-weighted control cost
//    produced a wrong gain with no indication.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/control/control.hpp"

namespace {

using Mat = std::vector<std::vector<double>>;

Mat mat_mul(const Mat& a, const Mat& b) {
    Mat c(a.size(), std::vector<double>(b[0].size(), 0.0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t k = 0; k < b.size(); ++k) {
            for (std::size_t j = 0; j < b[0].size(); ++j) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

Mat mat_transpose(const Mat& a) {
    Mat t(a[0].size(), std::vector<double>(a.size(), 0.0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[0].size(); ++j) {
            t[j][i] = a[i][j];
        }
    }
    return t;
}

} // namespace

TEST(ControlTimeResponse, StepMatchesTheClosedFormForAFirstOrderLag) {
    // G(s) = 1/(s+1) has step response 1 - exp(-t) exactly.
    const ms::control::TransferFunction g{{1.0}, {1.0, 1.0}};
    const auto r = ms::control::step_response(g, 5.0, 200);
    ASSERT_EQ(r.t.size(), 200u);
    double worst = 0.0;
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        worst = std::max(worst, std::abs(r.y[i] - (1.0 - std::exp(-r.t[i]))));
    }
    EXPECT_LT(worst, 1e-12);
}

TEST(ControlTimeResponse, ImpulseMatchesTheClosedFormForAFirstOrderLag) {
    // Impulse response of 1/(s+1) is exp(-t).
    const ms::control::TransferFunction g{{1.0}, {1.0, 1.0}};
    const auto r = ms::control::impulse_response(g, 5.0, 200);
    double worst = 0.0;
    for (std::size_t i = 0; i < r.t.size(); ++i) {
        worst = std::max(worst, std::abs(r.y[i] - std::exp(-r.t[i])));
    }
    EXPECT_LT(worst, 1e-12);
}

TEST(ControlTimeResponse, StiffPlantDoesNotDiverge) {
    // G(s) = 100/(s+100), simulated over 1 s with 50 points, so
    // dt = 0.0204 and 1 + dt*(-100) = -1.04. Forward Euler amplifies by 1.04
    // every step and the trace blows up; the exact propagator settles to the
    // DC gain of 1 regardless of dt.
    const ms::control::TransferFunction g{{100.0}, {1.0, 100.0}};
    const auto r = ms::control::step_response(g, 1.0, 50);
    ASSERT_FALSE(r.y.empty());
    for (double v : r.y) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LT(std::abs(v), 10.0) << "response diverged";
    }
    EXPECT_NEAR(r.y.back(), 1.0, 1e-6);
}

TEST(ControlTimeResponse, SettlesToTheDcGain) {
    // A lightly damped second-order plant: y(inf) must equal dcgain.
    const ms::control::TransferFunction g{{1.0}, {1.0, 0.4, 1.0}};
    const auto r = ms::control::step_response(g, 40.0, 400);
    EXPECT_NEAR(r.y.back(), ms::control::dcgain(g), 1e-3);
}

TEST(ControlRiccati, SolvesTheEquationWithANonDiagonalR) {
    // R is symmetric positive definite but NOT diagonal, so the element-wise
    // reciprocal of its diagonal is not its inverse. The returned X must satisfy
    // A'X + XA - X B R^-1 B' X + Q = 0 with the TRUE inverse
    //   [[2, 0.5], [0.5, 1]]^-1 = (1/1.75) [[1, -0.5], [-0.5, 2]].
    const Mat A{{0.0, 1.0}, {-2.0, -3.0}};
    const Mat B{{0.0, 0.0}, {1.0, 1.0}};
    const Mat Q{{1.0, 0.0}, {0.0, 1.0}};
    const Mat R{{2.0, 0.5}, {0.5, 1.0}};
    const Mat Rinv{{1.0 / 1.75, -0.5 / 1.75}, {-0.5 / 1.75, 2.0 / 1.75}};

    const auto X = ms::control::riccati(A, B, Q, R);
    ASSERT_TRUE(X.has_value());

    const Mat quad = mat_mul(mat_mul(mat_mul(mat_mul(*X, B), Rinv), mat_transpose(B)), *X);
    const Mat ATX = mat_mul(mat_transpose(A), *X);
    const Mat XA = mat_mul(*X, A);
    double worst = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            worst = std::max(worst, std::abs(ATX[i][j] + XA[i][j] - quad[i][j] + Q[i][j]));
        }
    }
    EXPECT_LT(worst, 1e-8);

    // X must be symmetric positive definite.
    EXPECT_NEAR((*X)[0][1], (*X)[1][0], 1e-9);
    EXPECT_GT((*X)[0][0], 0.0);
    EXPECT_GT((*X)[0][0] * (*X)[1][1] - (*X)[0][1] * (*X)[1][0], 0.0);
}

TEST(ControlLqr, GainUsesTheTrueInverseOfR) {
    // K = R^-1 B' X. With a non-diagonal R the diagonal-reciprocal shortcut
    // gives a different gain, so compare against the hand-computed inverse.
    const Mat A{{0.0, 1.0}, {-2.0, -3.0}};
    const Mat B{{0.0, 0.0}, {1.0, 1.0}};
    const Mat Q{{1.0, 0.0}, {0.0, 1.0}};
    const Mat R{{2.0, 0.5}, {0.5, 1.0}};
    const Mat Rinv{{1.0 / 1.75, -0.5 / 1.75}, {-0.5 / 1.75, 2.0 / 1.75}};

    const auto X = ms::control::riccati(A, B, Q, R);
    ASSERT_TRUE(X.has_value());
    const auto K = ms::control::lqr(A, B, Q, R);
    ASSERT_TRUE(K.has_value());

    const Mat expected = mat_mul(mat_mul(Rinv, mat_transpose(B)), *X);
    ASSERT_EQ(K->size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (std::size_t j = 0; j < expected[i].size(); ++j) {
            EXPECT_NEAR((*K)[i][j], expected[i][j], 1e-8) << "K(" << i << "," << j << ")";
        }
    }
}

TEST(ControlRiccati, SingularRIsReported) {
    // A singular R has no inverse; the diagonal shortcut would have produced
    // infinities or silently wrong numbers instead of an error.
    const Mat A{{0.0, 1.0}, {-2.0, -3.0}};
    const Mat B{{0.0, 0.0}, {1.0, 1.0}};
    const Mat Q{{1.0, 0.0}, {0.0, 1.0}};
    const Mat R{{1.0, 1.0}, {1.0, 1.0}};  // rank 1
    EXPECT_FALSE(ms::control::riccati(A, B, Q, R).has_value());
    EXPECT_FALSE(ms::control::lqr(A, B, Q, R).has_value());
}

TEST(ControlRiccati, DiagonalRStillAgreesWithTheOldShortcut) {
    // For a diagonal R the reciprocal-of-diagonal happens to be correct, so this
    // case must be unchanged by the fix.
    const Mat A{{0.0, 1.0}, {-2.0, -3.0}};
    const Mat B{{0.0}, {1.0}};
    const Mat Q{{1.0, 0.0}, {0.0, 1.0}};
    const Mat R{{2.0}};
    const auto X = ms::control::riccati(A, B, Q, R);
    ASSERT_TRUE(X.has_value());

    const Mat Rinv{{0.5}};
    const Mat quad = mat_mul(mat_mul(mat_mul(mat_mul(*X, B), Rinv), mat_transpose(B)), *X);
    const Mat ATX = mat_mul(mat_transpose(A), *X);
    const Mat XA = mat_mul(*X, A);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(ATX[i][j] + XA[i][j] - quad[i][j] + Q[i][j], 0.0, 1e-8);
        }
    }
}

TEST(ControlTf2ss, LeadingZeroDenominatorDoesNotProduceAnInfiniteRealisation) {
    // Coefficients are in descending order, so den[0] is the LEADING one and the whole
    // realisation divides by it. A leading zero does not change the polynomial -- [0, 1]
    // is the constant 1 -- but dividing by it gave A = -inf, and the matrix exponential's
    // scaling loop then halved an infinite norm forever: control_impulse_final([1],[0,1])
    // never returned. Found by fuzzing the REPL entry point.
    const ms::control::TransferFunction lead_zero{{1.0}, {0.0, 1.0}};
    const auto ss = ms::control::tf2ss(lead_zero);
    for (const auto& row : ss.A) {
        for (const double v : row) EXPECT_TRUE(std::isfinite(v));
    }
    for (const auto& row : ss.B) {
        for (const double v : row) EXPECT_TRUE(std::isfinite(v));
    }
    // 1/(0*s + 1) is the unit gain, so it is all feedthrough.
    ASSERT_FALSE(ss.D.empty());
    EXPECT_NEAR(ss.D[0][0], 1.0, 1e-12);
    EXPECT_NEAR(ms::control::dcgain(lead_zero), 1.0, 1e-12);

    // Several leading zeros, and zeros in the numerator too.
    const ms::control::TransferFunction many{{0.0, 0.0, 2.0}, {0.0, 0.0, 1.0, 1.0}};
    const auto ss2 = ms::control::tf2ss(many);
    for (const auto& row : ss2.A) {
        for (const double v : row) EXPECT_TRUE(std::isfinite(v));
    }
    // 2/(s + 1): DC gain 2.
    EXPECT_NEAR(ms::control::dcgain(many), 2.0, 1e-9);

    // An identically zero denominator is not a transfer function; the realisation is the
    // zero system rather than a division by zero.
    const ms::control::TransferFunction degenerate{{1.0}, {0.0, 0.0}};
    const auto ss3 = ms::control::tf2ss(degenerate);
    for (const auto& row : ss3.A) {
        for (const double v : row) EXPECT_TRUE(std::isfinite(v));
    }

    // The responses now return instead of spinning.
    const auto imp = ms::control::impulse_response(lead_zero);
    EXPECT_FALSE(imp.y.empty());
    for (const double v : imp.y) EXPECT_TRUE(std::isfinite(v));
    const auto step = ms::control::step_response(lead_zero);
    ASSERT_FALSE(step.y.empty());
    EXPECT_NEAR(step.y.back(), 1.0, 1e-9);

    // An ordinary system is unaffected: 1/(s+1) still has DC gain 1 and settles there.
    const ms::control::TransferFunction lag{{1.0}, {1.0, 1.0}};
    EXPECT_NEAR(ms::control::dcgain(lag), 1.0, 1e-12);
    EXPECT_NEAR(ms::control::step_response(lag).y.back(), 1.0, 1e-3);
}
