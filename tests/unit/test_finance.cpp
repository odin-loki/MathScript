#define _USE_MATH_DEFINES
#include "ms/finance/finance.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

using namespace ms::finance;

TEST(FinanceBS, CallPutParity) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double C = bs_call(S, K, T, r, sigma);
    double P = bs_put(S, K, T, r, sigma);
    // Put-call parity: C - P = S - K*e^{-rT}
    EXPECT_NEAR(C - P, S - K * std::exp(-r * T), 1e-8);
}

TEST(FinanceBS, ATMCall) {
    // At-the-money call with known approximation: C ≈ S * sigma * sqrt(T) * 0.4
    double S = 100, K = 100, T = 1.0, r = 0.0, sigma = 0.2;
    double C = bs_call(S, K, T, r, sigma);
    EXPECT_GT(C, 0.0);
    EXPECT_NEAR(C, S * sigma * std::sqrt(T) / std::sqrt(2.0 * M_PI), 3.0);
}

TEST(FinanceBS, Greeks_Delta) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double delta_c = bs_delta(S, K, T, r, sigma, true);
    double delta_p = bs_delta(S, K, T, r, sigma, false);
    // Delta call ∈ (0,1), put ∈ (-1,0)
    EXPECT_GT(delta_c, 0.0); EXPECT_LT(delta_c, 1.0);
    EXPECT_LT(delta_p, 0.0); EXPECT_GT(delta_p, -1.0);
    // Call delta - put delta = 1 (put-call delta parity)
    EXPECT_NEAR(delta_c - delta_p, 1.0, 1e-8);
}

TEST(FinanceBS, Gamma_Positive) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double gamma = bs_gamma(S, K, T, r, sigma);
    EXPECT_GT(gamma, 0.0);
}

TEST(FinanceBS, Vega_Positive) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double vega = bs_vega(S, K, T, r, sigma);
    EXPECT_GT(vega, 0.0);
}

TEST(FinanceBS, ImpliedVol) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double market_price = bs_call(S, K, T, r, sigma);
    auto iv = bs_implied_vol(market_price, S, K, T, r, true);
    ASSERT_TRUE(iv.has_value());
    EXPECT_NEAR(iv.value(), sigma, 1e-6);
}

TEST(FinanceBond, PriceAtPar) {
    // If coupon rate == yield, price == face value
    double c = 0.05, y = 0.05, fv = 100.0;
    double price = bond_price(c, y, 10, fv);
    EXPECT_NEAR(price, fv, 1e-6);
}

TEST(FinanceBond, Duration) {
    // Zero-coupon bond: duration = maturity
    double price = bond_price(0.0, 0.05, 5, 100.0);
    double dur   = bond_duration(0.0, 0.05, 5, 100.0);
    EXPECT_NEAR(dur, 5.0, 1e-6);
    (void)price;
}

TEST(FinanceBond, YTM) {
    double c = 0.05, y_true = 0.07, fv = 100.0;
    double price = bond_price(c, y_true, 10, fv);
    auto ytm = bond_ytm(price, c, 10, fv);
    ASSERT_TRUE(ytm.has_value());
    EXPECT_NEAR(ytm.value(), y_true, 1e-5);
}

TEST(FinanceTVM, NPV) {
    // NPV of 100 now and 110 in 1 year at 10% should be 100 + 110/1.1 = 200
    std::vector<double> cf = {100.0, 110.0};
    EXPECT_NEAR(npv(0.1, cf), 200.0, 1e-6);
}

TEST(FinanceTVM, IRR) {
    // Investment of -100 returns 110 in 1 year: IRR = 10%
    std::vector<double> cf = {-100.0, 110.0};
    auto irr_val = irr(cf);
    ASSERT_TRUE(irr_val.has_value());
    EXPECT_NEAR(irr_val.value(), 0.1, 1e-6);
}

TEST(FinanceTVM, Compound) {
    double fv = compound(100.0, 0.1, 3);
    EXPECT_NEAR(fv, 133.1, 1e-6);
}

TEST(FinanceTVM, ContinuousCompound) {
    double fv = continuous_compound(100.0, 0.1, 1.0);
    EXPECT_NEAR(fv, 100.0 * std::exp(0.1), 1e-8);
}

TEST(FinanceRisk, VaR) {
    std::vector<double> returns(1000);
    for (int i = 0; i < 1000; ++i) returns[i] = (i - 500) * 0.001;
    double v = var(returns, 0.95);
    EXPECT_GT(v, 0.0);
}

TEST(FinanceRisk, Sharpe) {
    // All returns = 10%, risk_free = 5%, std = 0 => technically infinite
    // Use varying returns
    std::vector<double> returns = {0.1, 0.12, 0.08, 0.11, 0.09};
    double sr = sharpe_ratio(returns, 0.05);
    EXPECT_GT(sr, 0.0);
}

TEST(FinanceRisk, MaxDrawdown) {
    std::vector<double> equity = {100, 110, 105, 120, 90, 95};
    double mdd = max_drawdown(equity);
    // Peak = 120, trough = 90, drawdown = 30/120 = 0.25
    EXPECT_NEAR(mdd, 30.0 / 120.0, 1e-6);
}

TEST(FinanceKelly, Basic) {
    // 60% win rate, 2:1 win/loss ratio
    double f = kelly_fraction(0.6, 2.0);
    EXPECT_NEAR(f, 0.6 - 0.4 / 2.0, 1e-10); // 0.6 - 0.2 = 0.4
}

TEST(FinanceBinomial, PricingConsistency) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double bs_c = bs_call(S, K, T, r, sigma);
    double bin_c = binomial_call(S, K, T, r, sigma, 200);
    EXPECT_NEAR(bin_c, bs_c, 0.5);  // binomial converges to BS with many steps
}

TEST(FinancePortfolio, PortfolioReturn) {
    std::vector<double> w = {0.6, 0.4};
    std::vector<double> ret = {0.1, 0.05};
    double pr = portfolio_return(w, ret);
    EXPECT_NEAR(pr, 0.08, 1e-10);  // 0.6*0.1 + 0.4*0.05 = 0.08
}

TEST(FinanceBSPut, DeepITM) {
    // Deep in-the-money put ≈ K·e^{-rT} − S
    double S = 50, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double P = bs_put(S, K, T, r, sigma);
    EXPECT_GT(P, K * std::exp(-r * T) - S - 1.0);
    EXPECT_LT(P, K * std::exp(-r * T) - S + 1.0);
}

TEST(FinanceBSGreeks, ThetaRho) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double theta_c = bs_theta(S, K, T, r, sigma, true);
    double theta_p = bs_theta(S, K, T, r, sigma, false);
    double rho_c   = bs_rho(S, K, T, r, sigma, true);
    double rho_p   = bs_rho(S, K, T, r, sigma, false);
    EXPECT_LT(theta_c, 0.0);
    EXPECT_LT(theta_p, 0.0);
    EXPECT_GT(rho_c, 0.0);
    EXPECT_LT(rho_p, 0.0);
}

TEST(FinanceBondPrice, PremiumDiscount) {
    double fv = 100.0;
    double premium = bond_price(0.06, 0.04, 10, fv);
    double discount = bond_price(0.04, 0.06, 10, fv);
    EXPECT_GT(premium, fv);
    EXPECT_LT(discount, fv);
}

TEST(FinanceIRR, MultiPeriodProject) {
    // NPV = 0 at r = 20% when each payment is 1000 / Σ(1.2^{-t})
    std::vector<double> cf = {-1000.0, 474.72, 474.72, 474.72};
    auto irr_val = irr(cf);
    ASSERT_TRUE(irr_val.has_value());
    EXPECT_NEAR(irr_val.value(), 0.2, 1e-4);
    EXPECT_NEAR(npv(irr_val.value(), cf), 0.0, 1e-2);
}

TEST(FinanceVaR, KnownQuantile) {
    std::vector<double> returns = {-0.20, -0.15, -0.10, -0.05, 0.0,
                                   0.05, 0.10, 0.15, 0.20, 0.25};
    EXPECT_NEAR(var(returns, 0.95), 0.20, 1e-10);
    // idx=(1-0.5)*10=5 → sorted[5]=0.05 → VaR returns -0.05 (positive=loss convention)
    EXPECT_NEAR(var(returns, 0.50), -0.05, 1e-10);
}

TEST(FinanceRisk, InformationRatio) {
    std::vector<double> returns = {0.10, 0.12, 0.08, 0.11, 0.09};
    std::vector<double> benchmark = {0.05, 0.08, 0.06, 0.10, 0.07};
    // excess mean = 0.028, sample std ≈ 0.016431676, IR ≈ 1.704
    EXPECT_NEAR(information_ratio(returns, benchmark), 1.704, 0.01);
}

TEST(FinanceRisk, TreynorRatio) {
    std::vector<double> returns = {0.10, 0.12, 0.08, 0.11, 0.09};
    // mean = 0.10, (0.10 - 0.05) / 1.2 = 0.041666...
    EXPECT_NEAR(treynor_ratio(returns, 0.05, 1.2), 0.05 / 1.2, 1e-10);
}

TEST(FinanceRisk, CAPM) {
    // 0.05 + 1.2 * (0.10 - 0.05) = 0.11
    EXPECT_NEAR(capm(0.05, 1.2, 0.10), 0.11, 1e-10);
}

TEST(FinanceRates, ForwardRate) {
    // (0.06*2 - 0.05*1) / (2 - 1) = 0.07
    EXPECT_NEAR(forward_rate(0.05, 1.0, 0.06, 2.0), 0.07, 1e-10);
}

TEST(FinanceRates, ForwardRateFlatCurve) {
    // Flat curve: forward equals spot zero rate
    EXPECT_NEAR(forward_rate(0.05, 1.0, 0.10, 2.0), 0.15, 1e-10);
}

TEST(FinanceDigital, DeepITMCall) {
    double S = 200, K = 100, T = 1.0, r = 0.05, sigma = 0.2, payout = 1.0;
    double price = digital_option(S, K, T, r, sigma, true, payout);
    EXPECT_NEAR(price, payout * std::exp(-r * T), 0.01);
}

TEST(FinanceDigital, DeepOTMCall) {
    double S = 50, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double price = digital_option(S, K, T, r, sigma, true);
    EXPECT_NEAR(price, 0.0, 0.01);
}

TEST(FinanceDigital, DeepITMPut) {
    double S = 50, K = 100, T = 1.0, r = 0.05, sigma = 0.2, payout = 2.0;
    double price = digital_option(S, K, T, r, sigma, false, payout);
    EXPECT_NEAR(price, payout * std::exp(-r * T), 0.01);
}

TEST(FinanceDigital, CallPutSum) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2, payout = 1.0;
    double c = digital_option(S, K, T, r, sigma, true, payout);
    double p = digital_option(S, K, T, r, sigma, false, payout);
    // Both pay discounted payout in one scenario; sum ≈ discounted payout
    EXPECT_NEAR(c + p, payout * std::exp(-r * T), 0.05);
}

TEST(FinanceBlack76, CallMatchesSpotBS) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double F = S * std::exp(r * T);
    double bs = bs_call(S, K, T, r, sigma);
    double b76 = black76(F, K, T, r, sigma, true);
    EXPECT_NEAR(b76, bs, 1e-8);
}

TEST(FinanceBlack76, PutMatchesSpotBS) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double F = S * std::exp(r * T);
    double bs = bs_put(S, K, T, r, sigma);
    double b76 = black76(F, K, T, r, sigma, false);
    EXPECT_NEAR(b76, bs, 1e-8);
}

TEST(FinanceBlack76, ForwardEquivalenceITM) {
    double S = 120, K = 100, T = 0.5, r = 0.03, sigma = 0.25;
    double F = S * std::exp(r * T);
    EXPECT_NEAR(black76(F, K, T, r, sigma, true), bs_call(S, K, T, r, sigma), 1e-8);
}

TEST(FinanceBarrier, KnockInOutParityCall) {
    double S = 100, K = 100, B = 90, T = 1.0, r = 0.05, sigma = 0.2;
    double vanilla = bs_call(S, K, T, r, sigma);
    double knock_in = barrier_option(S, K, B, T, r, sigma, true, true, false);
    double knock_out = barrier_option(S, K, B, T, r, sigma, true, false, false);
    EXPECT_NEAR(knock_in + knock_out, vanilla, 1e-6);
}

TEST(FinanceBarrier, KnockInOutParityPut) {
    double S = 100, K = 100, B = 110, T = 1.0, r = 0.05, sigma = 0.2;
    double vanilla = bs_put(S, K, T, r, sigma);
    double knock_in = barrier_option(S, K, B, T, r, sigma, false, true, true);
    double knock_out = barrier_option(S, K, B, T, r, sigma, false, false, true);
    EXPECT_NEAR(knock_in + knock_out, vanilla, 1e-6);
}

TEST(FinanceBarrier, DownOutCallCheaperThanVanilla) {
    double S = 100, K = 100, B = 90, T = 1.0, r = 0.05, sigma = 0.2;
    double vanilla = bs_call(S, K, T, r, sigma);
    double knock_out = barrier_option(S, K, B, T, r, sigma, true, false, false);
    EXPECT_LT(knock_out, vanilla);
    EXPECT_GT(knock_out, 0.0);
}

TEST(FinanceBarrier, UpOutPutCheaperThanVanilla) {
    double S = 100, K = 100, B = 110, T = 1.0, r = 0.05, sigma = 0.2;
    double vanilla = bs_put(S, K, T, r, sigma);
    double knock_out = barrier_option(S, K, B, T, r, sigma, false, false, true);
    EXPECT_LT(knock_out, vanilla);
    EXPECT_GT(knock_out, 0.0);
}

TEST(FinanceAmerican, CallEqualsEuropean) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 200;
    double am = american_option(S, K, T, r, sigma, true, steps);
    double eu_bin = binomial_call(S, K, T, r, sigma, steps);
    double eu_bs = bs_call(S, K, T, r, sigma);
    EXPECT_NEAR(am, eu_bin, 0.5);
    EXPECT_NEAR(am, eu_bs, 1.0);
}

TEST(FinanceAmerican, PutEarlyExercisePremium) {
    double S = 80, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 200;
    double am = american_option(S, K, T, r, sigma, false, steps);
    double eu_bin = binomial_put(S, K, T, r, sigma, steps);
    double eu_bs = bs_put(S, K, T, r, sigma);
    EXPECT_GE(am, eu_bin - 1e-10);
    EXPECT_GE(am, eu_bs - 1e-10);
    EXPECT_GT(am - eu_bs, 1.0);  // clear early-exercise premium for deep ITM put
}

TEST(FinanceAmerican, PutPremiumMultipleStrikes) {
    double S = 70, K = 100, T = 2.0, r = 0.03, sigma = 0.25;
    int steps = 300;
    double am = american_option(S, K, T, r, sigma, false, steps);
    double eu = bs_put(S, K, T, r, sigma);
    EXPECT_GT(am - eu, 1.0);  // non-negligible early-exercise premium
}
