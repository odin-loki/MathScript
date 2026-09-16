// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4: the closed forms in finance.cpp, asserted as values rather than signs.
//
// `src/finance/finance.cpp` has 286 tests over it, and a great many of them say
// only that a quantity is positive, or that two quantities differ, or that one
// model is within half a point of another. Every function below has a formula
// in a textbook, so the reference values here are computed from that formula in
// double precision and pinned. A sign test passes for an infinity of wrong
// implementations; a value test passes for one.
//
// The references were produced independently of the code under test, from
// `math.erfc` and `math.exp` in the same precision, and are reproduced in the
// comment above each block so a reader can check them without running anything.

#define _USE_MATH_DEFINES
#include "ms/finance/finance.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <variant>
#include <vector>

using namespace ms::finance;

namespace {
// S = K = 100, T = 1, r = 0.05, sigma = 0.2 -> d1 = 0.35, d2 = 0.15 exactly.
constexpr double kS = 100.0, kK = 100.0, kT = 1.0, kR = 0.05, kSigma = 0.2;
}  // namespace

TEST(FinanceClosedForms, TheGreeksAtTheCanonicalPointAreTheirFormulas) {
    // phi(0.35) = 0.3752403469169379, N(0.35) = 0.6368306511756191,
    // N(0.15)  = 0.5596176923702425, e^-0.05 = 0.951229424500714.
    EXPECT_NEAR(bs_call(kS, kK, kT, kR, kSigma), 10.450583572185565, 1e-12);
    EXPECT_NEAR(bs_put(kS, kK, kT, kR, kSigma), 5.573526022256971, 1e-12);

    EXPECT_NEAR(bs_delta(kS, kK, kT, kR, kSigma, true), 0.6368306511756191, 1e-12);
    EXPECT_NEAR(bs_delta(kS, kK, kT, kR, kSigma, false), -0.3631693488243809, 1e-12);

    // gamma = phi(d1) / (S*sigma*sqrt(T)); the suite asserted only that it is
    // positive, which every scale factor and every sign-preserving typo also is.
    EXPECT_NEAR(bs_gamma(kS, kK, kT, kR, kSigma), 0.018762017345846895, 1e-15);

    // vega = S*phi(d1)*sqrt(T), per unit of volatility (not per percentage
    // point): the header names no divisor, so the raw derivative is the
    // contract. Same story -- the only assertion was "positive".
    EXPECT_NEAR(bs_vega(kS, kK, kT, kR, kSigma), 37.52403469169379, 1e-12);

    EXPECT_NEAR(bs_theta(kS, kK, kT, kR, kSigma, true), -6.414027546438197, 1e-12);
    EXPECT_NEAR(bs_theta(kS, kK, kT, kR, kSigma, false), -1.657880423934626, 1e-12);
    EXPECT_NEAR(bs_rho(kS, kK, kT, kR, kSigma, true), 53.232481545376345, 1e-12);
    EXPECT_NEAR(bs_rho(kS, kK, kT, kR, kSigma, false), -41.89046090469506, 1e-12);
}

TEST(FinanceClosedForms, TheGreeksAwayFromTheMoneyToo) {
    // S = 110, K = 95, T = 0.5, r = 0.03, sigma = 0.35: nothing here is
    // symmetric, so a formula that happens to be right at the money has no
    // second chance.
    const double S = 110.0, K = 95.0, T = 0.5, r = 0.03, sg = 0.35;
    EXPECT_NEAR(bs_call(S, K, T, r, sg), 20.280204205989804, 1e-12);
    EXPECT_NEAR(bs_gamma(S, K, T, r, sg), 0.010838285938371753, 1e-15);
    EXPECT_NEAR(bs_vega(S, K, T, r, sg), 22.95007047450219, 1e-12);
    EXPECT_NEAR(bs_theta(S, K, T, r, sg, true), -10.002534286196386, 1e-12);
    EXPECT_NEAR(bs_rho(S, K, T, r, sg, true), 32.83349366867701, 1e-12);

    // Vega and gamma are related: vega = gamma * S^2 * sigma * T.
    EXPECT_NEAR(bs_vega(S, K, T, r, sg),
                bs_gamma(S, K, T, r, sg) * S * S * sg * T, 1e-10);
}

TEST(FinanceClosedForms, Black76AndBachelierAgainstTheirFormulas) {
    // F = K = 100, T = 1, r = 0.05, sigma = 0.2. At the money forward the
    // Black-76 call and put are equal, and both discount by e^-rT.
    EXPECT_NEAR(black76(100.0, 100.0, 1.0, 0.05, 0.2, true), 7.57708214642728, 1e-12);
    EXPECT_NEAR(black76(100.0, 100.0, 1.0, 0.05, 0.2, false), 7.57708214642728, 1e-12);

    // Bachelier's sigma is an ABSOLUTE volatility, in price units, so an
    // at-the-money call is worth disc * sigma * sqrt(T) * phi(0) -- two orders
    // of magnitude below the lognormal answer for the same number.
    EXPECT_NEAR(bachelier_call(100.0, 100.0, 1.0, 0.05, 0.2), 0.07589712715905148, 1e-14);
    EXPECT_NEAR(bachelier_put(100.0, 100.0, 1.0, 0.05, 0.2), 0.07589712715905148, 1e-14);
}

TEST(FinanceClosedForms, ShortRateBondPricesAreTheirAffineFormulas) {
    // r = 0.03, a = 0.5, b = 0.05, sigma = 0.02, tau = 5.
    // Vasicek: B = (1-e^-a tau)/a = 1.8358300027522023,
    //          log A = (b - s^2/2a^2)(B - tau) - (s^2/4a) B^2.
    EXPECT_NEAR(vasicek_bond_price(0.03, 0.5, 0.05, 0.02, 5.0), 0.8094290808345329, 1e-14);
    // CIR with h = sqrt(a^2 + 2 s^2): B = 1.8348981850258554.
    EXPECT_NEAR(cir_bond_price(0.03, 0.5, 0.05, 0.02, 5.0), 0.8079870622025103, 1e-14);

    // The two models agree as sigma goes to zero, where both reduce to the
    // deterministic mean-reverting path.
    EXPECT_NEAR(vasicek_bond_price(0.03, 0.5, 0.05, 0.0, 5.0),
                cir_bond_price(0.03, 0.5, 0.05, 0.0, 5.0), 1e-12);
    // A zero horizon is a unit discount factor in both.
    EXPECT_EQ(vasicek_bond_price(0.03, 0.5, 0.05, 0.02, 0.0), 1.0);
    EXPECT_EQ(cir_bond_price(0.03, 0.5, 0.05, 0.02, 0.0), 1.0);
}

TEST(FinanceClosedForms, SharpeAndTheOtherRatiosAreQuotientsNotSigns) {
    // {0.10, 0.12, 0.08, 0.11, 0.09}: mean 0.10, sample stddev
    // sqrt(0.001/4) = 0.015811388300841896, so Sharpe at rf = 0.05 is
    // 0.05/0.015811388300841896 = 3.1622776601683795 = 2*sqrt(2.5).
    const std::vector<double> returns{0.10, 0.12, 0.08, 0.11, 0.09};
    EXPECT_NEAR(sharpe_ratio(returns, 0.05), 3.1622776601683795, 1e-12);

    // The sample standard deviation divides by n-1. Dividing by n instead would
    // give 0.01414..., and a Sharpe of 3.5355 -- still positive, which is all
    // the suite had been asking for.
    EXPECT_GT(sharpe_ratio(returns, 0.05), 3.16);
    EXPECT_LT(sharpe_ratio(returns, 0.05), 3.17);

    // Sortino uses only the returns BELOW the threshold, and divides by their
    // count rather than by n-1. At rf = 0.095 the below-threshold set is
    // {0.08, 0.09}, so the downside deviation is
    // sqrt(((0.015)^2 + (0.005)^2)/2) = 0.011180339887498949.
    const double downside = std::sqrt((0.015 * 0.015 + 0.005 * 0.005) / 2.0);
    EXPECT_NEAR(sortino_ratio(returns, 0.095), (0.10 - 0.095) / downside, 1e-12);

    // With nothing below the threshold the function reports its sentinel.
    EXPECT_NEAR(sortino_ratio(returns, 0.0), 1e9, 1e-6);

    // Treynor is the excess return over beta, nothing more.
    EXPECT_NEAR(treynor_ratio(returns, 0.05, 1.25), 0.05 / 1.25, 1e-15);
}

TEST(FinanceClosedForms, HestonAgainstAnIndependentlyConvergedQuadrature) {
    // heston_call integrates the two Heston probabilities with a fixed
    // trapezoid: 100,000 points of step 0.001 over [1e-5, 100]. Nothing had
    // asserted what that grid actually converges TO, only that the price moves
    // the right way with the parameters.
    //
    // The references below come from re-integrating the same characteristic
    // function with composite Simpson -- a fourth-order rule rather than a
    // second-order one -- at 40,000 and 60,000 nodes over [1e-8, 200] and
    // [1e-8, 300], which agree to 1e-13, so the truncation at 100 and the step
    // of 0.001 are both accounted for. The residual measured against the
    // implementation's own grid is 8.2e-6 for the first point and 2.8e-5 for
    // the second; 1e-4 leaves an order of magnitude.
    EXPECT_NEAR(heston_call(100.0, 100.0, 0.5, 0.03, 0.05, 1.2, 0.06, 0.3, -0.7),
                7.048094796742028, 1e-4);
    EXPECT_NEAR(heston_call(90.0, 105.0, 1.5, 0.02, 0.09, 2.5, 0.04, 0.5, -0.5),
                4.764474997740045, 1e-4);

    // Put-call parity holds exactly, by construction of heston_put.
    const double c = heston_call(100.0, 100.0, 0.5, 0.03, 0.05, 1.2, 0.06, 0.3, -0.7);
    const double p = heston_put(100.0, 100.0, 0.5, 0.03, 0.05, 1.2, 0.06, 0.3, -0.7);
    EXPECT_NEAR(c - p, 100.0 - 100.0 * std::exp(-0.03 * 0.5), 1e-12);

    // As the vol-of-vol goes to zero the variance process becomes deterministic
    // and the price must approach Black-Scholes at sqrt(v0). The approach is
    // QUADRATIC in sigma_v, which a "they are close" assertion cannot see.
    //
    // It is quadratic only until the quadrature's own error takes over, and
    // measuring where that happens is the point. Against
    // bs_call(100,100,1,0.05,0.2) = 10.450583572185565 the gaps are
    //
    //   sigma_v  0.3    1.75966e-1
    //   sigma_v  0.1    2.09842e-2    ratio 8.39 against a theoretical 9.00
    //   sigma_v  0.03   1.91543e-3    ratio 10.96 against a theoretical 11.11
    //   sigma_v  0.01   2.24702e-4    ratio 8.52 against a theoretical 9.00
    //   sigma_v  0.003  3.22350e-5
    //   sigma_v  0.001  1.53133e-5    <- no longer falling
    //
    // Below about 0.003 the fixed trapezoid's own discretisation error, ~1.5e-5
    // at these parameters, is larger than the model's remaining sigma_v^2 term,
    // so the sequence flattens out. That floor is a property of THIS integrator
    // and not of the Heston model, and it is recorded rather than asserted away.
    const double bs = bs_call(100.0, 100.0, 1.0, 0.05, 0.2);
    double gaps[4];
    const double svs[4] = {0.3, 0.1, 0.03, 0.01};
    for (int i = 0; i < 4; ++i) {
        gaps[i] = std::abs(heston_call(100.0, 100.0, 1.0, 0.05, 0.04, 2.0, 0.04,
                                       svs[i], 0.0) - bs);
        if (i > 0) {
            EXPECT_LT(gaps[i], gaps[i - 1]) << "sigma_v " << svs[i];
            const double want = (svs[i - 1] / svs[i]) * (svs[i - 1] / svs[i]);
            const double got = gaps[i - 1] / gaps[i];
            EXPECT_GT(got, 0.7 * want) << "sigma_v " << svs[i] << ": ratio " << got;
            EXPECT_LT(got, 1.4 * want) << "sigma_v " << svs[i] << ": ratio " << got;
        }
    }
    EXPECT_LT(gaps[3], 3e-4) << "gap at sigma_v = 0.01 is " << gaps[3];
    // And the flattening itself: a further factor of ten in sigma_v does NOT
    // buy another factor of a hundred, because the quadrature floor is reached.
    const double gap_1e3 = std::abs(
        heston_call(100.0, 100.0, 1.0, 0.05, 0.04, 2.0, 0.04, 0.001, 0.0) - bs);
    EXPECT_LT(gap_1e3, 5e-5) << "gap at sigma_v = 0.001 is " << gap_1e3;
}

TEST(FinanceClosedForms, ABondWhosePriceCannotBeMatchedExactlyStillHasAYield) {
    // bond_ytm bisects for 200 iterations and returns as soon as the price is
    // matched to 1e-10. When it never is, a second block computes one last
    // midpoint and accepts it if the residual is under 1e-6. Nothing had ever
    // reached that block, so `0.5 * (lo + hi)` written as `0.5 * (lo - hi)` --
    // which turns a 5% yield into -3e-18 -- went unnoticed.
    //
    // Reaching it needs a price that no double-precision yield reproduces to
    // 1e-10, which is simply a bond quoted in large units: at a face value of
    // 1e7 the spacing between representable prices is about 2e-9, so the best
    // achievable residual is ~1.5e-8 -- too coarse for the loop's 1e-10 exit
    // and comfortably inside the fallback's 1e-6. The yield is still 5%.
    constexpr double kEps = 2.220446049250313e-16;
    for (const double fv : {1.0e6, 1.0e7}) {
        const double base = bond_price(0.05, 0.05, 10, fv);
        for (const int k : {0, 1, 2, 5, 20}) {
            const double price = base * (1.0 + static_cast<double>(k) * kEps);
            const auto y = bond_ytm(price, 0.05, 10, fv);
            ASSERT_TRUE(y.has_value()) << "fv " << fv << " k " << k;
            EXPECT_NEAR(y.value(), 0.05, 1e-10) << "fv " << fv << " k " << k;
        }
    }
    // The ordinary scale still goes through the loop and lands on the same yield.
    const auto par = bond_ytm(bond_price(0.05, 0.05, 10, 100.0), 0.05, 10, 100.0);
    ASSERT_TRUE(par.has_value());
    EXPECT_NEAR(par.value(), 0.05, 1e-12);
}

TEST(FinanceClosedForms, TheMonteCarloGuardsRejectOnlyWhatIsActuallyDegenerate) {
    // Each of the four Monte-Carlo pricers opens with one guard line:
    //
    //   if (n_paths <= 0 || n_steps <= 0 || S <= 0.0 || K <= 0.0 ||
    //       T <= 0.0 || sigma <= 0.0) return 0.0;
    //
    // Six comparisons on one line, and every test in the suite passed values
    // far from all of them, so moving any bound from 0 to 1 -- which turns a
    // one-path run, a one-step run, or a unit spot, strike, horizon or
    // volatility into a silent zero -- changed nothing anyone could see.
    //
    // The assertions below are not statistical. For a fixed-strike lookback
    // call the path maximum is at least the spot, so the payoff is at least
    // S - K whenever S > K, whatever the draws are; the mirror holds for the
    // put. Each call puts exactly ONE parameter at the boundary value the
    // mutant would reject.
    //
    // A FLOATING-strike lookback is the one payoff here that is not bounded
    // below by intrinsic value: its call pays s_T - min(path), which is zero
    // exactly when the path ends at its own minimum, and with a single path
    // that is an ordinary outcome rather than a rare one (seed 20260915 is such
    // a path). The seed is therefore fixed at 1, where the single path does not
    // end at its minimum, and the one-step case uses the PUT -- whose payoff,
    // max(path) - s_T, is positive whenever the single step goes down.
    const unsigned seed = 1u;

    // n_paths = 1, then n_steps = 1.
    EXPECT_GT(mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 1, 50, seed), 0.0);
    EXPECT_GT(mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 64, 1, seed), 0.0);
    EXPECT_GT(mc_asian_call(100.0, 50.0, 1.0, 0.05, 0.2, 1, 50, seed), 0.0);
    EXPECT_GT(mc_asian_call(100.0, 50.0, 1.0, 0.05, 0.2, 64, 1, seed), 0.0);
    EXPECT_GT(mc_european_call(100.0, 1.0, 1.0, 0.05, 0.2, 1, seed), 0.0);
    EXPECT_GT(mc_lookback_floating_call(100.0, 1.0, 0.05, 0.2, 1, 50, seed), 0.0);
    EXPECT_GT(mc_lookback_floating_put(100.0, 1.0, 0.05, 0.2, 1, 1, seed), 0.0);
    EXPECT_GT(mc_lookback_floating_call(100.0, 1.0, 0.05, 0.2, 64, 1, seed), 0.0);

    // S = 1 exactly, with a strike below it so the payoff cannot be zero.
    EXPECT_GT(mc_lookback_fixed_call(1.0, 0.5, 1.0, 0.05, 0.2, 64, 20, seed), 0.0);
    EXPECT_GT(mc_european_call(1.0, 0.5, 1.0, 0.05, 0.2, 64, seed), 0.0);
    EXPECT_GT(mc_lookback_floating_call(1.0, 1.0, 0.05, 0.2, 64, 20, seed), 0.0);

    // K = 1 exactly, with a spot far above it.
    EXPECT_GT(mc_lookback_fixed_call(100.0, 1.0, 1.0, 0.05, 0.2, 64, 20, seed), 0.0);
    EXPECT_GT(mc_asian_call(100.0, 1.0, 1.0, 0.05, 0.2, 64, 20, seed), 0.0);
    EXPECT_GT(mc_european_call(100.0, 1.0, 1.0, 0.05, 0.2, 64, seed), 0.0);

    // T = 1 exactly, and sigma = 1 exactly.
    EXPECT_GT(mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 64, 20, seed), 0.0);
    EXPECT_GT(mc_lookback_fixed_call(100.0, 90.0, 0.5, 0.05, 1.0, 64, 20, seed), 0.0);
    EXPECT_GT(mc_asian_call(100.0, 50.0, 1.0, 0.05, 1.0, 64, 20, seed), 0.0);
    EXPECT_GT(mc_european_call(100.0, 50.0, 1.0, 0.05, 1.0, 64, seed), 0.0);

    // And what the guard IS for still returns zero.
    EXPECT_EQ(mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 0, 50, seed), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 64, 0, seed), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(0.0, 90.0, 1.0, 0.05, 0.2, 64, 50, seed), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(100.0, 0.0, 1.0, 0.05, 0.2, 64, 50, seed), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(100.0, 90.0, 0.0, 0.05, 0.2, 64, 50, seed), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.0, 64, 50, seed), 0.0);
    EXPECT_EQ(mc_european_call(100.0, 90.0, 1.0, 0.05, 0.2, 0, seed), 0.0);
    EXPECT_EQ(mc_asian_call(100.0, 90.0, 1.0, 0.05, 0.2, 64, 0, seed), 0.0);
    EXPECT_EQ(mc_lookback_floating_call(100.0, 1.0, 0.05, 0.2, 0, 50, seed), 0.0);
}

TEST(FinanceClosedForms, ACashFlowStreamWithNoSignChangeHasNoInternalRate) {
    // irr() is Newton on the NPV. Its convergence test is `|f| < 1e-10`, and
    // nothing had ever handed it a stream that does not converge, so writing
    // that test as `<=` -- which accepts an NPV of exactly the tolerance and
    // returns the GUESS as though it were a rate -- was invisible.
    //
    // A stream with a single non-zero entry is the cleanest case: the NPV is
    // that entry at every rate, so there is no root, and the derivative is
    // identically zero. Whenever the entry is at least the tolerance the
    // function must report failure rather than answer.
    for (const double c : {1e-10, 1e-9, 1.0, 1000.0, -1.0}) {
        const std::vector<double> single{c};
        const auto rate = irr(single, 0.1, 100);
        EXPECT_FALSE(rate.has_value())
            << "single cashflow " << c << " has no internal rate, but irr returned "
            << (rate.has_value() ? rate.value() : 0.0);
    }
    // Below the tolerance the NPV is zero for practical purposes, and the
    // function says so by returning the rate it was given.
    const std::vector<double> negligible{1e-12};
    const auto tiny = irr(negligible, 0.1, 100);
    ASSERT_TRUE(tiny.has_value());
    EXPECT_NEAR(tiny.value(), 0.1, 1e-15);

    // All-positive and all-negative multi-period streams have no root either.
    const std::vector<double> all_positive{100.0, 50.0, 25.0};
    EXPECT_FALSE(irr(all_positive, 0.1, 100).has_value());

    // And the ordinary case still converges to the rate that zeroes the NPV.
    const std::vector<double> project{-1000.0, 400.0, 400.0, 400.0};
    const auto ok = irr(project, 0.1, 100);
    ASSERT_TRUE(ok.has_value());
    EXPECT_NEAR(npv(ok.value(), project), 0.0, 1e-9);
}

TEST(FinanceClosedForms, SabrsHaganVolIsAFormulaWithAValue) {
    // Every SABR assertion in the suite was a shape: positive, below spot, above
    // intrinsic, decreasing in strike, or "within 0.5" of a Black-Scholes price.
    // Hagan's expansion is a closed form, so here are its values.
    //
    // S = 100, T = 1, r = 0.03, alpha = 0.25, beta = 0.5, rho = -0.3, nu = 0.4.
    // The forward is F = S*e^(rT) = 103.0454533953517, which is also where the
    // implementation switches to its at-the-money-FORWARD branch.
    const double S = 100.0, T = 1.0, r = 0.03;
    const double alpha = 0.25, beta = 0.5, rho = -0.3, nu = 0.4;

    EXPECT_NEAR(sabr_call(S, 80.0, T, r, alpha, beta, rho, nu), 22.364359125881123, 1e-9);
    EXPECT_NEAR(sabr_call(S, 100.0, T, r, alpha, beta, rho, nu), 3.146625128028333, 1e-9);
    // Six significant figures on a price of 7.8e-7: the assertion is relative
    // in effect, since a deeply out-of-the-money price is an erfc tail.
    EXPECT_NEAR(sabr_call(S, 120.0, T, r, alpha, beta, rho, nu), 7.79888132221182e-07, 1e-12);
    EXPECT_NEAR(sabr_put(S, 120.0, T, r, alpha, beta, rho, nu), 16.453464805709118, 1e-9);
    EXPECT_NEAR(sabr_put(S, 100.0, T, r, alpha, beta, rho, nu), 0.19117848287915074, 1e-11);

    // The at-the-money-forward branch is a different expression in the source
    // (the log(F/K) series collapses), and it is reached only at K = F exactly.
    const double F = S * std::exp(r * T);
    EXPECT_NEAR(sabr_call(S, F, T, r, alpha, beta, rho, nu), 0.993456171905926, 1e-9);
    // At the forward a call and a put are worth the same thing, which is the
    // statement put-call parity makes about that one strike.
    EXPECT_NEAR(sabr_call(S, F, T, r, alpha, beta, rho, nu),
                sabr_put(S, F, T, r, alpha, beta, rho, nu), 1e-9);

    // beta = 1 with nu = 0 collapses Hagan's expansion to the constant alpha --
    // every correction term carries a factor of (1-beta) or of nu. So SABR is
    // Black-Scholes at alpha EXACTLY, not within half a point.
    for (const double rho_any : {-0.9, 0.0, 0.7}) {
        EXPECT_NEAR(sabr_call(S, 100.0, T, r, 0.25, 1.0, rho_any, 0.0),
                    bs_call(S, 100.0, T, r, 0.25), 1e-12)
            << "rho " << rho_any;
        EXPECT_NEAR(sabr_put(S, 100.0, T, r, 0.25, 1.0, rho_any, 0.0),
                    bs_put(S, 100.0, T, r, 0.25), 1e-12)
            << "rho " << rho_any;
    }

    // Put-call parity, asserted between the two functions rather than between a
    // call and an expression built from it.
    const double c = sabr_call(S, 105.0, T, r, alpha, beta, rho, nu);
    const double p = sabr_put(S, 105.0, T, r, alpha, beta, rho, nu);
    EXPECT_NEAR(c - p, S - 105.0 * std::exp(-r * T), 1e-10);
}

TEST(FinanceClosedForms, MertonExpandsItsBracketOnlyWhenTheRateIsNegative) {
    // merton_solve_asset_value brackets the asset value in [E, E+D] and then
    // widens the upper end while the two endpoint residuals share a sign:
    //
    //   for (int expand = 0; expand < 20 && f_lo * f_hi > 0.0; ++expand)
    //
    // That loop had never run. f(V) = call(V, D) - E is negative at V = E,
    // because a call is worth less than its underlying; and at V = E + D the
    // call is worth at least (E + D) - D*e^(-rT), which is at least E for any
    // NON-NEGATIVE rate. The bracket therefore always straddles, and the whole
    // expansion path -- and the `> 0.0` in its condition -- was dead code to
    // every test in the suite.
    //
    // A negative rate breaks the inequality: e^(-rT) exceeds 1, the lower bound
    // falls below E, and deep-out-of-the-money equity over a long horizon lands
    // both endpoints on the same side. Reported negative policy rates over a
    // long-dated liability are the case.
    //
    // Reference values below are from an independent implementation of the same
    // fixed-point iteration in double precision.
    {
        // 40-year horizon at -2%: three expansions, fourteen outer iterations.
        const MertonResult m = merton_implied_asset_params(1.0, 1.0, 100.0, -0.02, 40.0,
                                                            100, 1e-8);
        EXPECT_TRUE(m.converged) << "V " << m.implied_asset_value;
        EXPECT_NEAR(m.implied_asset_value, 1.018565938808024, 1e-6);
        EXPECT_NEAR(m.implied_asset_volatility, 0.9929334110035845, 1e-6);
        EXPECT_TRUE(std::isfinite(m.distance_to_default));
    }
    {
        // 30-year horizon at -5%: eight expansions.
        const MertonResult m = merton_implied_asset_params(1.0, 1.0, 100.0, -0.05, 30.0,
                                                            100, 1e-8);
        EXPECT_TRUE(m.converged) << "V " << m.implied_asset_value;
        EXPECT_NEAR(m.implied_asset_value, 1.1075084088370204, 1e-6);
        EXPECT_NEAR(m.implied_asset_volatility, 0.9660764272953354, 1e-6);
    }
    {
        // The ordinary non-negative-rate case still converges without ever
        // entering the loop, so the two paths are both exercised here.
        const MertonResult m = merton_implied_asset_params(30.0, 0.5, 100.0, 0.03, 1.0,
                                                            100, 1e-8);
        EXPECT_TRUE(m.converged);
        EXPECT_GT(m.implied_asset_value, 30.0);
        EXPECT_TRUE(std::isfinite(m.probability_of_default));
        EXPECT_GE(m.probability_of_default, 0.0);
        EXPECT_LE(m.probability_of_default, 1.0);
    }
}

TEST(FinanceClosedForms, InPlusOutIsTheVanillaOptionForAllEightBarriers) {
    // barrier_option assembles four Reiner-Rubinstein coefficients into eight
    // combinations -- {call, put} x {up, down} x {in, out} -- chosen by a chain
    // of ternaries, and the branch taken also depends on whether the strike is
    // above or below the barrier. That is sixteen expressions.
    //
    // In-out parity WAS tested, in `FinanceBarrier.KnockInOutParityCall` and
    // `...Put` -- for one combination each, and the two chosen combinations
    // (down call with K >= B, up put with K < B) happen to select the SAME pair
    // of expressions, `c.C` and `c.A - c.C`. Two of sixteen. The other fourteen
    // were never compared to anything, which is how `c.A - c.B + c.D` could
    // become `c.A - c.B - c.D` in the down-and-in call with a strike below the
    // barrier -- an error of 7.84 on a 4.44 option -- without a test noticing.
    //
    // The contract is model-independent and exact, and it costs nothing to
    // state for all eight: a knock-in and a knock-out with the same barrier pay,
    // between them, exactly what the vanilla option pays, in every state of the
    // world. So in + out == vanilla, to floating point, as long as the spot
    // starts on the live side of the barrier.
    const double T = 1.0, r = 0.05, sigma = 0.25;
    struct Case { double S, K, B; bool call, up; const char* name; };
    const Case cases[] = {
        {100.0,  90.0, 80.0, true,  false, "down call, K > B"},
        {100.0,  70.0, 80.0, true,  false, "down call, K < B"},
        {100.0, 130.0, 120.0, true, true,  "up call, K > B"},
        {100.0, 110.0, 120.0, true, true,  "up call, K < B"},
        {100.0,  90.0, 80.0, false, false, "down put, K > B"},
        {100.0,  70.0, 80.0, false, false, "down put, K < B"},
        {100.0, 130.0, 120.0, false, true, "up put, K > B"},
        {100.0, 110.0, 120.0, false, true, "up put, K < B"},
    };
    for (const Case& c : cases) {
        const double in = barrier_option(c.S, c.K, c.B, T, r, sigma, c.call, true, c.up);
        const double out = barrier_option(c.S, c.K, c.B, T, r, sigma, c.call, false, c.up);
        const double vanilla = c.call ? bs_call(c.S, c.K, T, r, sigma)
                                      : bs_put(c.S, c.K, T, r, sigma);
        ASSERT_TRUE(std::isfinite(in)) << c.name;
        ASSERT_TRUE(std::isfinite(out)) << c.name;
        EXPECT_NEAR(in + out, vanilla, 1e-9)
            << c.name << ": in " << in << " + out " << out << " != vanilla " << vanilla;
        // Neither leg may be negative, and neither may exceed the vanilla: a
        // barrier only ever removes payoff.
        EXPECT_GE(in, -1e-9) << c.name;
        EXPECT_GE(out, -1e-9) << c.name;
        EXPECT_LE(in, vanilla + 1e-9) << c.name;
        EXPECT_LE(out, vanilla + 1e-9) << c.name;
    }

    // Already through the barrier: a knock-in is the vanilla option and a
    // knock-out is worthless, on both sides.
    EXPECT_NEAR(barrier_option(70.0, 90.0, 80.0, T, r, sigma, true, true, false),
                bs_call(70.0, 90.0, T, r, sigma), 1e-12);
    EXPECT_EQ(barrier_option(70.0, 90.0, 80.0, T, r, sigma, true, false, false), 0.0);
    EXPECT_NEAR(barrier_option(130.0, 110.0, 120.0, T, r, sigma, false, true, true),
                bs_put(130.0, 110.0, T, r, sigma), 1e-12);
    EXPECT_EQ(barrier_option(130.0, 110.0, 120.0, T, r, sigma, false, false, true), 0.0);
}

TEST(FinanceClosedForms, EveryNonFiniteArgumentToHestonAndSabrGivesNaN) {
    // Both models open with a fifteen-term validation chain, and the suite had
    // asserted exactly two of the fifteen (a zero kappa and a negative spot).
    // Every argument position is checked here, with +infinity and with a NaN.
    //
    // T is the one exception, and deliberately so: -infinity satisfies the
    // `T <= 0` expiry test that runs BEFORE the validation chain, so it returns
    // intrinsic value rather than NaN. That is the documented expiry path, not
    // a gap, so T is driven with +infinity and NaN only.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (const double bad : {inf, nan}) {
        // heston_call(S, K, T, r, v0, kappa, theta, sigma_v, rho)
        EXPECT_TRUE(std::isnan(heston_call(bad, 100, 1, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5))) << "S";
        EXPECT_TRUE(std::isnan(heston_call(100, bad, 1, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5))) << "K";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, bad, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5))) << "T";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, bad, 0.04, 2.0, 0.04, 0.3, -0.5))) << "r";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, bad, 2.0, 0.04, 0.3, -0.5))) << "v0";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, bad, 0.04, 0.3, -0.5))) << "kappa";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, 2.0, bad, 0.3, -0.5))) << "theta";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, 2.0, 0.04, bad, -0.5))) << "sigma_v";
        EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, 2.0, 0.04, 0.3, bad))) << "rho";
        EXPECT_TRUE(std::isnan(heston_put(100, 100, 1, 0.05, 0.04, 2.0, bad, 0.3, -0.5))) << "put theta";

        // sabr_call(S, K, T, r, alpha, beta, rho, nu)
        EXPECT_TRUE(std::isnan(sabr_call(bad, 100, 1, 0.03, 0.25, 0.5, -0.3, 0.4))) << "S";
        EXPECT_TRUE(std::isnan(sabr_call(100, bad, 1, 0.03, 0.25, 0.5, -0.3, 0.4))) << "K";
        EXPECT_TRUE(std::isnan(sabr_call(100, 100, bad, 0.03, 0.25, 0.5, -0.3, 0.4))) << "T";
        EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, bad, 0.25, 0.5, -0.3, 0.4))) << "r";
        EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, bad, 0.5, -0.3, 0.4))) << "alpha";
        EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, bad, -0.3, 0.4))) << "beta";
        EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, 0.5, bad, 0.4))) << "rho";
        EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, 0.5, -0.3, bad))) << "nu";
        EXPECT_TRUE(std::isnan(sabr_put(100, 100, 1, 0.03, 0.25, bad, -0.3, 0.4))) << "put beta";
    }

    // The out-of-range checks that are not about finiteness: beta outside [0,1],
    // rho outside [-1,1], a negative alpha, a negative vol-of-vol.
    EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, -0.1, 0.5, -0.3, 0.4)));
    EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, 1.5, -0.3, 0.4)));
    EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, -0.5, -0.3, 0.4)));
    EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, 0.5, -1.5, 0.4)));
    EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, 0.5, 1.5, 0.4)));
    EXPECT_TRUE(std::isnan(sabr_call(100, 100, 1, 0.03, 0.25, 0.5, -0.3, -0.4)));
    EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, -0.04, 2.0, 0.04, 0.3, -0.5)));
    EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, -2.0, 0.04, 0.3, -0.5)));
    EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, 2.0, -0.04, 0.3, -0.5)));
    EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, 2.0, 0.04, -0.3, -0.5)));
    EXPECT_TRUE(std::isnan(heston_call(100, 100, 1, 0.05, 0.04, 2.0, 0.04, 0.3, -1.5)));

    // And the expiry path, which runs first and is not a validation failure.
    EXPECT_EQ(sabr_call(110, 100, 0.0, 0.03, 0.25, 0.5, -0.3, 0.4), 10.0);
    EXPECT_EQ(heston_call(110, 100, 0.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5), 10.0);
    EXPECT_EQ(sabr_put(90, 100, 0.0, 0.03, 0.25, 0.5, -0.3, 0.4), 10.0);
    EXPECT_EQ(heston_put(90, 100, 0.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5), 10.0);
}

TEST(FinanceClosedForms, TheMonteCarloEstimatorsUseEveryPathTheyAreAskedFor) {
    // All four pricers draw in ANTITHETIC pairs: `n_draws = (n_paths + 1) / 2`
    // normal draws, each used twice with opposite sign, stopping early on the
    // odd path. Written as `(n_paths - 1) / 2` the estimator quietly drops a
    // pair -- 62 of 64 paths, with the divisor still 64 -- and for a single path
    // it draws nothing at all and returns zero. Nothing had asserted a
    // Monte-Carlo value tightly enough to see a 3% scale error.
    //
    // The structure is assertable without any statistics. With a fixed seed,
    // one path is the first draw's payoff and two paths are that payoff
    // averaged with its antithetic twin, so 2*price(2) - price(1) is the twin's
    // payoff and cannot be negative. Three paths add a second draw's payoff.
    const unsigned seed = 4242u;
    const double p1 = mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 1, 40, seed);
    const double p2 = mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 2, 40, seed);
    const double p3 = mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 3, 40, seed);
    const double p4 = mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, 4, 40, seed);
    for (const double p : {p1, p2, p3, p4}) {
        EXPECT_GT(p, 0.0) << "a lookback call with spot above the strike is worth something";
    }
    EXPECT_GE(2.0 * p2 - p1, -1e-9) << "the antithetic twin's payoff is not negative";
    EXPECT_GE(3.0 * p3 - 2.0 * p2, -1e-9) << "the second draw's payoff is not negative";
    EXPECT_GE(4.0 * p4 - 3.0 * p3, -1e-9) << "the second twin's payoff is not negative";

    // Every path is at least the intrinsic S - K, discounted, because the path
    // maximum is at least the spot -- so the average over however many paths is
    // too, and the bound holds for any count.
    const double floor_value = (100.0 - 90.0) * std::exp(-0.05 * 1.0);
    for (const int n : {1, 2, 3, 4, 7, 64, 129}) {
        const double p = mc_lookback_fixed_call(100.0, 90.0, 1.0, 0.05, 0.2, n, 40, seed);
        EXPECT_GE(p, floor_value - 1e-9) << "n_paths " << n << " gave " << p;
    }
}

TEST(FinanceClosedForms, BondYtmReportsHowFarItGotWhenNoYieldFits) {
    // The yield search starts with the bracket [-0.9999, 1] and doubles the
    // upper end up to sixty times. If the bond is still worth more than the
    // asking price at a yield of 2^60, no yield in the search's reach
    // reproduces it, and the function returns ConvergenceFail carrying the gap
    // it could not close: `ConvergenceFail{60, p_hi - price}`.
    //
    // That line had never run, because closing the gap needs a bond whose cash
    // flows survive a yield of 1.15e18 -- which means a face value around 1e20.
    // Written as `p_hi + price` the reported residual is off by twice the price
    // and nothing would say so, since the payload was never read.
    const double fv = 1.1e20, c = 0.05;
    const int n = 1;
    const double price = 100.0;
    const double ceiling_price = bond_price(c, std::pow(2.0, 60), n, fv);
    ASSERT_GT(ceiling_price, price) << "the setup must actually exhaust the doublings";

    const auto y = bond_ytm(price, c, n, fv);
    ASSERT_FALSE(y.has_value()) << "yield " << y.value_or(0.0);
    const auto* fail = std::get_if<ms::ConvergenceFail>(&y.error());
    ASSERT_NE(fail, nullptr) << "the failure should be a ConvergenceFail";
    EXPECT_EQ(fail->iterations, 60u) << "sixty doublings is the ceiling";
    EXPECT_NEAR(fail->residual, ceiling_price - price, 1e-9)
        << "the residual is the gap that was left, not the sum of the two sides";
    EXPECT_LT(fail->residual, price) << "residual " << fail->residual;

    // The other failure the function reports is on the opposite side: a price
    // that not even the lowest admissible yield reaches. The floor is -0.9999,
    // and bond_price(0.05, -0.9999, 10, 100) is 1.05e42, so the asking price has
    // to exceed that. The payload carries zero iterations, because the bracket
    // was rejected before any bisection, and the gap on that side.
    const double lowest = bond_price(0.05, -0.9999, 10, 100.0);
    ASSERT_GT(lowest, 1e40) << "lowest admissible price is " << lowest;
    const auto too_dear = bond_ytm(1e300, 0.05, 10, 100.0);
    ASSERT_FALSE(too_dear.has_value());
    const auto* dear_fail = std::get_if<ms::ConvergenceFail>(&too_dear.error());
    ASSERT_NE(dear_fail, nullptr);
    EXPECT_EQ(dear_fail->iterations, 0u);
    EXPECT_NEAR(dear_fail->residual, 1e300 - lowest, 1e285);

    // A price the bisection can bracket but not match to 1e-10, and cannot get
    // inside 1e-6 of either, fails with the iteration count it spent: a bond
    // asked at 1e9 against a face value of 100 has a yield near the -0.9999
    // floor, where the price moves by enormous amounts per unit of yield.
    const auto coarse = bond_ytm(1e9, 0.05, 10, 100.0);
    ASSERT_FALSE(coarse.has_value());
    const auto* coarse_fail = std::get_if<ms::ConvergenceFail>(&coarse.error());
    ASSERT_NE(coarse_fail, nullptr);
    EXPECT_EQ(coarse_fail->iterations, 200u) << "the bisection ran to its limit";
    EXPECT_GT(coarse_fail->residual, 1e-10);

    // And the ordinary bond still returns its yield.
    const auto ok = bond_ytm(bond_price(0.05, 0.07, 10, 100.0), 0.05, 10, 100.0);
    ASSERT_TRUE(ok.has_value());
    EXPECT_NEAR(ok.value(), 0.07, 1e-9);
}

TEST(FinanceClosedForms, ATreeWithNoStepsIsNotAPrice) {
    // `binomial_tree` computes dt = T/steps, so zero steps divides by zero: u is
    // exp(sigma*sqrt(inf)), d is 0, and the risk-neutral probability is inf/inf.
    // None of that reached the answer, because the single terminal node is
    // S*pow(u, 0)*pow(d, 0) and `pow(anything, 0)` is 1 -- so the function
    // returned the UNDISCOUNTED intrinsic value, a number carrying neither the
    // rate nor the volatility, and called it a price. A refusal is the honest
    // answer, and in a library that does not throw, a refusal is NaN.
    EXPECT_TRUE(std::isnan(binomial_call(100.0, 90.0, 1.0, 0.05, 0.2, 0)));
    EXPECT_TRUE(std::isnan(binomial_put(100.0, 110.0, 1.0, 0.05, 0.2, 0)));
    EXPECT_TRUE(std::isnan(american_option(100.0, 110.0, 1.0, 0.05, 0.2, false, 0)));
    EXPECT_TRUE(std::isnan(trinomial_option(100.0, 110.0, 1.0, 0.05, 0.2, 0, true, false)));
    EXPECT_TRUE(std::isnan(binomial_call(100.0, 90.0, 1.0, 0.05, 0.2, -3)));

    // One step is the smallest tree there is, and it is a closed form: with
    // dt = 1, u = e^0.2, d = 1/u and p = (e^0.05 - d)/(u - d), only the down
    // node of the put and the up node of the call are in the money.
    const double u = std::exp(0.2);
    const double d = 1.0 / u;
    const double p = (std::exp(0.05) - d) / (u - d);
    const double disc = std::exp(-0.05);
    EXPECT_NEAR(binomial_put(100.0, 110.0, 1.0, 0.05, 0.2, 1),
                disc * (1.0 - p) * (110.0 - 100.0 * d), 1e-12);
    // The call's down node, at 100/u = 81.87, is out of the money against a
    // strike of 90, so only the up node pays.
    EXPECT_NEAR(binomial_call(100.0, 90.0, 1.0, 0.05, 0.2, 1),
                disc * p * (100.0 * u - 90.0), 1e-12);
}
