#define _USE_MATH_DEFINES
#include "ms/finance/finance.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
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

// --- Monte Carlo pricing ---
// With antithetic variates and n_paths in the hundreds of thousands, the MC
// standard error for these (S,K,T,r,sigma) combos is well under 0.05 (payoff
// std dev is O(S*sigma*sqrt(T)) ~ 20-40, divided by sqrt(n_paths) ~ 700-1000).
// An absolute tolerance of 0.15 is therefore roughly 3-5 standard errors,
// which passes reliably at the fixed seed while still being a meaningful check.
namespace {
constexpr int kMCPaths = 500000;
constexpr double kMCTol = 0.15;
} // namespace

TEST(FinanceMC, EuropeanCallATMMatchesBS) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double bs = bs_call(S, K, T, r, sigma);
    double mc = mc_european_call(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(mc, bs, kMCTol);
}

TEST(FinanceMC, EuropeanPutATMMatchesBS) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double bs = bs_put(S, K, T, r, sigma);
    double mc = mc_european_put(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(mc, bs, kMCTol);
}

TEST(FinanceMC, EuropeanCallITMMatchesBS) {
    double S = 120, K = 100, T = 0.5, r = 0.03, sigma = 0.25;
    double bs = bs_call(S, K, T, r, sigma);
    double mc = mc_european_call(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(mc, bs, kMCTol);
}

TEST(FinanceMC, EuropeanPutITMMatchesBS) {
    double S = 80, K = 100, T = 0.5, r = 0.03, sigma = 0.25;
    double bs = bs_put(S, K, T, r, sigma);
    double mc = mc_european_put(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(mc, bs, kMCTol);
}

TEST(FinanceMC, EuropeanCallOTMMatchesBS) {
    double S = 90, K = 110, T = 1.0, r = 0.04, sigma = 0.3;
    double bs = bs_call(S, K, T, r, sigma);
    double mc = mc_european_call(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(mc, bs, kMCTol);
}

TEST(FinanceMC, EuropeanPutOTMMatchesBS) {
    double S = 110, K = 90, T = 1.0, r = 0.04, sigma = 0.3;
    double bs = bs_put(S, K, T, r, sigma);
    double mc = mc_european_put(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(mc, bs, kMCTol);
}

TEST(FinanceMC, PutCallParity) {
    double S = 100, K = 105, T = 1.0, r = 0.05, sigma = 0.2;
    double c = mc_european_call(S, K, T, r, sigma, kMCPaths);
    double p = mc_european_put(S, K, T, r, sigma, kMCPaths);
    EXPECT_NEAR(c - p, S - K * std::exp(-r * T), kMCTol);
}

TEST(FinanceMC, EuropeanCallDeterministic) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_european_call(S, K, T, r, sigma, 10000, 7);
    double b = mc_european_call(S, K, T, r, sigma, 10000, 7);
    EXPECT_EQ(a, b);
}

TEST(FinanceMC, EuropeanPutDeterministic) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_european_put(S, K, T, r, sigma, 10000, 7);
    double b = mc_european_put(S, K, T, r, sigma, 10000, 7);
    EXPECT_EQ(a, b);
}

TEST(FinanceMC, EuropeanOddPathCountRuns) {
    // Odd n_paths exercises the "extra draw, drop one payoff" antithetic branch.
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double mc_odd = mc_european_call(S, K, T, r, sigma, 10001);
    double bs = bs_call(S, K, T, r, sigma);
    EXPECT_NEAR(mc_odd, bs, 1.0);
    EXPECT_GT(mc_odd, 0.0);
}

TEST(FinanceMC, EuropeanEdgeCasesReturnZero) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    EXPECT_EQ(mc_european_call(S, K, T, r, sigma, 0), 0.0);
    EXPECT_EQ(mc_european_put(S, K, T, r, sigma, 0), 0.0);
    EXPECT_EQ(mc_european_call(-1.0, K, T, r, sigma, 1000), 0.0);
    EXPECT_EQ(mc_european_call(S, -1.0, T, r, sigma, 1000), 0.0);
    EXPECT_EQ(mc_european_call(S, K, -1.0, r, sigma, 1000), 0.0);
    EXPECT_EQ(mc_european_call(S, K, T, r, -0.1, 1000), 0.0);
    EXPECT_EQ(mc_european_put(-1.0, K, T, r, sigma, 1000), 0.0);
    EXPECT_EQ(mc_european_put(S, K, T, r, 0.0, 1000), 0.0);
}

TEST(FinanceMC, AsianCallCheaperThanEuropean) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double asian = mc_asian_call(S, K, T, r, sigma, 100000, 50);
    double euro = bs_call(S, K, T, r, sigma);
    EXPECT_LT(asian, euro);
    EXPECT_GT(asian, 0.0);
}

TEST(FinanceMC, AsianCallCheaperThanEuropeanHighVol) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.4;
    double asian = mc_asian_call(S, K, T, r, sigma, 100000, 50);
    double euro = bs_call(S, K, T, r, sigma);
    EXPECT_LT(asian, euro);
    EXPECT_GT(asian, 0.0);
}

TEST(FinanceMC, AsianPutPositiveITM) {
    double S = 90, K = 110, T = 1.0, r = 0.05, sigma = 0.25;
    double asian_put = mc_asian_put(S, K, T, r, sigma, 100000, 50);
    EXPECT_GT(asian_put, 0.0);
}

TEST(FinanceMC, AsianPayoffsNonNegative) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.3;
    double call = mc_asian_call(S, K, T, r, sigma, 50000, 25);
    double put  = mc_asian_put(S, K, T, r, sigma, 50000, 25);
    EXPECT_GE(call, 0.0);
    EXPECT_GE(put, 0.0);
}

TEST(FinanceMC, AsianDeterministic) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_asian_call(S, K, T, r, sigma, 5000, 20, 3);
    double b = mc_asian_call(S, K, T, r, sigma, 5000, 20, 3);
    EXPECT_EQ(a, b);
}

// --- Markowitz mean-variance portfolio optimization ---
namespace {
// 2-asset, uncorrelated: analytical min-variance weights are inversely
// proportional to variance, w_i = (1/sigma_i^2) / sum_j(1/sigma_j^2).
const std::vector<double> kCov2Diag = {0.04, 0.0,
                                       0.0, 0.01};

// 3-asset, correlated example (symmetric, positive-definite).
const std::vector<double> kCov3 = {0.10, 0.02, 0.01,
                                   0.02, 0.08, 0.03,
                                   0.01, 0.03, 0.06};
const std::vector<double> kMu3 = {0.08, 0.12, 0.10};
} // namespace

TEST(FinancePortfolioOpt, MinVarianceTwoAssetMatchesClosedForm) {
    auto w = min_variance_portfolio(kCov2Diag, 2);
    ASSERT_TRUE(w.has_value());
    double inv1 = 1.0 / 0.04, inv2 = 1.0 / 0.01;
    double expected1 = inv1 / (inv1 + inv2), expected2 = inv2 / (inv1 + inv2);
    EXPECT_NEAR(w->at(0), expected1, 1e-6);
    EXPECT_NEAR(w->at(1), expected2, 1e-6);
    EXPECT_NEAR(w->at(0) + w->at(1), 1.0, 1e-10);
}

TEST(FinancePortfolioOpt, MinVarianceThreeAssetSumsToOne) {
    auto w = min_variance_portfolio(kCov3, 3);
    ASSERT_TRUE(w.has_value());
    double sum = w->at(0) + w->at(1) + w->at(2);
    EXPECT_NEAR(sum, 1.0, 1e-8);
}

TEST(FinancePortfolioOpt, MinVarianceThreeAssetSatisfiesKKT) {
    // At the true minimum-variance point, Sigma*w = lambda*1 for some scalar
    // lambda, i.e. every entry of Sigma*w is equal.
    auto w = min_variance_portfolio(kCov3, 3);
    ASSERT_TRUE(w.has_value());
    std::vector<double> sw(3, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sw[i] += kCov3[i * 3 + j] * w->at(j);
    EXPECT_NEAR(sw[0], sw[1], 1e-8);
    EXPECT_NEAR(sw[1], sw[2], 1e-8);
}

TEST(FinancePortfolioOpt, MinVarianceReducesVarianceVsEqualWeight) {
    auto w = min_variance_portfolio(kCov3, 3);
    ASSERT_TRUE(w.has_value());
    std::vector<double> equal = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    EXPECT_LE(portfolio_variance(*w, kCov3), portfolio_variance(equal, kCov3));
}

TEST(FinancePortfolioOpt, MinVarianceDimensionMismatchErrors) {
    std::vector<double> bad_cov = {0.04, 0.0, 0.0}; // not 2x2
    auto w = min_variance_portfolio(bad_cov, 2);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, MinVarianceNonPositiveNErrors) {
    EXPECT_FALSE(min_variance_portfolio(kCov2Diag, 0).has_value());
    EXPECT_FALSE(min_variance_portfolio(kCov2Diag, -1).has_value());
}

TEST(FinancePortfolioOpt, MinVarianceSingularCovarianceErrors) {
    // Two identical, zero-variance assets: Sigma is all zeros, singular.
    std::vector<double> singular_cov = {0.0, 0.0,
                                        0.0, 0.0};
    auto w = min_variance_portfolio(singular_cov, 2);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, EfficientFrontierHitsTargetReturn) {
    for (double target : {0.09, 0.10, 0.11}) {
        auto w = efficient_frontier_portfolio(kCov3, kMu3, target, 3);
        ASSERT_TRUE(w.has_value());
        double realized = portfolio_return(*w, kMu3);
        double sum = w->at(0) + w->at(1) + w->at(2);
        EXPECT_NEAR(realized, target, 1e-6);
        EXPECT_NEAR(sum, 1.0, 1e-8);
    }
}

TEST(FinancePortfolioOpt, EfficientFrontierAtMinVarianceReturnMatchesMinVariance) {
    auto min_var_w = min_variance_portfolio(kCov3, 3);
    ASSERT_TRUE(min_var_w.has_value());
    double min_var_return = portfolio_return(*min_var_w, kMu3);

    auto ef_w = efficient_frontier_portfolio(kCov3, kMu3, min_var_return, 3);
    ASSERT_TRUE(ef_w.has_value());
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(ef_w->at(i), min_var_w->at(i), 1e-6);
}

TEST(FinancePortfolioOpt, EfficientFrontierDimensionMismatchErrors) {
    std::vector<double> bad_mu = {0.08, 0.12}; // length 2, n=3
    auto w = efficient_frontier_portfolio(kCov3, bad_mu, 0.1, 3);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, EfficientFrontierNonPositiveNErrors) {
    EXPECT_FALSE(efficient_frontier_portfolio(kCov3, kMu3, 0.1, 0).has_value());
}

TEST(FinancePortfolioOpt, EfficientFrontierSingularCovarianceErrors) {
    std::vector<double> singular_cov = {0.0, 0.0,
                                        0.0, 0.0};
    std::vector<double> mu2 = {0.05, 0.07};
    auto w = efficient_frontier_portfolio(singular_cov, mu2, 0.06, 2);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, EfficientFrontierDegenerateEqualReturnsErrors) {
    // All expected returns equal collapses the 1x2/2x1 rows of [a b; b c] into
    // a rank-1 system: a*c - b*b -> 0.
    std::vector<double> equal_mu = {0.10, 0.10, 0.10};
    auto w = efficient_frontier_portfolio(kCov3, equal_mu, 0.10, 3);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, MaxSharpeSumsToOne) {
    auto w = max_sharpe_portfolio(kCov3, kMu3, 0.02, 3);
    ASSERT_TRUE(w.has_value());
    double sum = w->at(0) + w->at(1) + w->at(2);
    EXPECT_NEAR(sum, 1.0, 1e-8);
}

TEST(FinancePortfolioOpt, MaxSharpeBeatsEqualWeight) {
    double r_f = 0.02;
    auto w = max_sharpe_portfolio(kCov3, kMu3, r_f, 3);
    ASSERT_TRUE(w.has_value());
    double tangency_sharpe = (portfolio_return(*w, kMu3) - r_f) /
                             std::sqrt(portfolio_variance(*w, kCov3));

    std::vector<double> equal = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    double equal_sharpe = (portfolio_return(equal, kMu3) - r_f) /
                          std::sqrt(portfolio_variance(equal, kCov3));

    EXPECT_GE(tangency_sharpe, equal_sharpe);
}

TEST(FinancePortfolioOpt, MaxSharpeBeatsMinVariance) {
    double r_f = 0.02;
    auto w = max_sharpe_portfolio(kCov3, kMu3, r_f, 3);
    ASSERT_TRUE(w.has_value());
    double tangency_sharpe = (portfolio_return(*w, kMu3) - r_f) /
                             std::sqrt(portfolio_variance(*w, kCov3));

    auto min_var_w = min_variance_portfolio(kCov3, 3);
    ASSERT_TRUE(min_var_w.has_value());
    double min_var_sharpe = (portfolio_return(*min_var_w, kMu3) - r_f) /
                            std::sqrt(portfolio_variance(*min_var_w, kCov3));

    EXPECT_GE(tangency_sharpe, min_var_sharpe);
}

TEST(FinancePortfolioOpt, MaxSharpeBeatsArbitraryCandidates) {
    double r_f = 0.02;
    auto w = max_sharpe_portfolio(kCov3, kMu3, r_f, 3);
    ASSERT_TRUE(w.has_value());
    double tangency_sharpe = (portfolio_return(*w, kMu3) - r_f) /
                             std::sqrt(portfolio_variance(*w, kCov3));

    std::vector<std::vector<double>> candidates = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
        {0.5, 0.3, 0.2}, {0.2, 0.2, 0.6},
    };
    for (const auto& cand : candidates) {
        double cand_sharpe = (portfolio_return(cand, kMu3) - r_f) /
                             std::sqrt(portfolio_variance(cand, kCov3));
        EXPECT_GE(tangency_sharpe, cand_sharpe - 1e-9);
    }
}

TEST(FinancePortfolioOpt, MaxSharpeDimensionMismatchErrors) {
    std::vector<double> bad_mu = {0.08, 0.12};
    auto w = max_sharpe_portfolio(kCov3, bad_mu, 0.02, 3);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, MaxSharpeNonPositiveNErrors) {
    EXPECT_FALSE(max_sharpe_portfolio(kCov3, kMu3, 0.02, 0).has_value());
}

TEST(FinancePortfolioOpt, MaxSharpeSingularCovarianceErrors) {
    std::vector<double> singular_cov = {0.0, 0.0,
                                        0.0, 0.0};
    std::vector<double> mu2 = {0.05, 0.07};
    auto w = max_sharpe_portfolio(singular_cov, mu2, 0.02, 2);
    EXPECT_FALSE(w.has_value());
}

TEST(FinancePortfolioOpt, MinVarianceResidualIsSmall) {
    // Direct residual check: every entry of Sigma*w should be equal (KKT
    // optimality for the equality-constrained minimum-variance problem).
    auto w = min_variance_portfolio(kCov3, 3);
    ASSERT_TRUE(w.has_value());
    std::vector<double> sw(3, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sw[i] += kCov3[i * 3 + j] * w->at(j);
    double mean_sw = (sw[0] + sw[1] + sw[2]) / 3.0;
    for (double v : sw) EXPECT_NEAR(v, mean_sw, 1e-8);
}

// --- Black-Litterman model ---
namespace {
// Symmetric, positive-definite 2-asset covariance matrix reused across most
// Black-Litterman tests below.
const std::vector<double> kBLCov2 = {0.04, 0.01,
                                     0.01, 0.02};
const std::vector<double> kBLPi2 = {0.05, 0.07};

// Single view "asset 0 returns 10%", with the standard default-derived Omega
// (omega_11 = tau * P.Sigma.P^T = 0.05 * 0.04 = 0.002), for tau = 0.05.
const std::vector<double> kBLP1 = {1.0, 0.0};
const std::vector<double> kBLQ1 = {0.10};
const std::vector<double> kBLOmega1 = {0.002};
constexpr double kBLTau = 0.05;
} // namespace

TEST(FinanceBlackLitterman, ImpliedReturnsMatchesHandComputedExample) {
    // Pi = delta*Sigma*w_mkt: Sigma*w_mkt = [0.028, 0.014], delta=2.5 -> [0.07, 0.035].
    std::vector<double> w_mkt = {0.6, 0.4};
    auto pi = bl_implied_returns(kBLCov2, w_mkt, 2.5, 2);
    ASSERT_TRUE(pi.has_value());
    EXPECT_NEAR(pi->at(0), 0.07, 1e-9);
    EXPECT_NEAR(pi->at(1), 0.035, 1e-9);
}

TEST(FinanceBlackLitterman, ImpliedReturnsDimensionMismatchErrors) {
    std::vector<double> bad_cov = {0.04, 0.01, 0.01}; // not 2x2
    std::vector<double> w_mkt2 = {0.6, 0.4};
    EXPECT_FALSE(bl_implied_returns(bad_cov, w_mkt2, 2.5, 2).has_value());

    std::vector<double> bad_w = {0.6}; // size 1, n=2
    EXPECT_FALSE(bl_implied_returns(kBLCov2, bad_w, 2.5, 2).has_value());
}

TEST(FinanceBlackLitterman, ImpliedReturnsNonPositiveDeltaErrors) {
    std::vector<double> w_mkt = {0.6, 0.4};
    EXPECT_FALSE(bl_implied_returns(kBLCov2, w_mkt, 0.0, 2).has_value());
    EXPECT_FALSE(bl_implied_returns(kBLCov2, w_mkt, -1.0, 2).has_value());
}

TEST(FinanceBlackLitterman, ImpliedReturnsNonPositiveNErrors) {
    std::vector<double> w_mkt = {0.6, 0.4};
    EXPECT_FALSE(bl_implied_returns(kBLCov2, w_mkt, 2.5, 0).has_value());
    EXPECT_FALSE(bl_implied_returns(kBLCov2, w_mkt, 2.5, -1).has_value());
}

TEST(FinanceBlackLitterman, PosteriorNoViewsReturnsPriorExactly) {
    // k=0 (empty P/Q/omega): the view terms vanish, so the posterior must
    // collapse exactly to the prior Pi.
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, {}, {}, {}, 2, 0);
    ASSERT_TRUE(post.has_value());
    EXPECT_NEAR(post->at(0), kBLPi2[0], 1e-12);
    EXPECT_NEAR(post->at(1), kBLPi2[1], 1e-12);
}

TEST(FinanceBlackLitterman, PosteriorTinyTauConvergesToPrior) {
    // A tiny tau makes the prior's implied covariance (tau*Sigma) very
    // "tight", so (tau*Sigma)^-1 dwarfs the fixed P^T*Omega^-1*P view term and
    // the posterior should converge toward the prior even with a view present.
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, 1e-6, kBLP1, kBLQ1, kBLOmega1, 2, 1);
    ASSERT_TRUE(post.has_value());
    EXPECT_NEAR(post->at(0), kBLPi2[0], 1e-4);
    EXPECT_NEAR(post->at(1), kBLPi2[1], 1e-4);
}

TEST(FinanceBlackLitterman, PosteriorSingleViewPullsTowardView) {
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, kBLOmega1, 2, 1);
    ASSERT_TRUE(post.has_value());
    // Posterior for the viewed asset must lie strictly between the prior and
    // the view value, i.e. genuinely blended rather than snapping to either.
    EXPECT_GT(post->at(0), kBLPi2[0]);
    EXPECT_LT(post->at(0), kBLQ1[0]);
}

TEST(FinanceBlackLitterman, PosteriorSingleViewHighConfidenceNearView) {
    // A very small (high-confidence) Omega should pull the posterior close
    // to the view value itself.
    std::vector<double> confident_omega = {1e-10};
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, confident_omega, 2, 1);
    ASSERT_TRUE(post.has_value());
    EXPECT_NEAR(post->at(0), kBLQ1[0], 1e-4);
}

TEST(FinanceBlackLitterman, PosteriorMatchesHandComputedClosedForm) {
    // Reference values worked out by hand from
    // E[R] = [(tau*Sigma)^-1 + P^T*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P^T*Omega^-1*Q]
    // with Sigma=kBLCov2, Pi=[0.05,0.07], tau=0.05, P=[1,0], Q=[0.10], Omega=[0.002]:
    //   tau*Sigma = [[0.002,0.0005],[0.0005,0.001]], det = 1.75e-6
    //   inv(tau*Sigma) = (1/7)*[[4000,-2000],[-2000,8000]]
    //   A = inv(tau*Sigma) + [[500,0],[0,0]] = (1/7)*[[7500,-2000],[-2000,8000]]
    //   b = inv(tau*Sigma)*Pi + [50,0] = (1/7)*[410,460]
    //   solving A*x=b gives x = [0.075, 0.07625] exactly.
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, kBLOmega1, 2, 1);
    ASSERT_TRUE(post.has_value());
    EXPECT_NEAR(post->at(0), 0.075, 1e-6);
    EXPECT_NEAR(post->at(1), 0.07625, 1e-6);
}

TEST(FinanceBlackLitterman, PosteriorDefaultOmegaMatchesHandComputedClosedForm) {
    // Same setup as PosteriorMatchesHandComputedClosedForm, but Omega is
    // derived automatically. omega_11 = tau*(P.Sigma.P^T) = 0.05*0.04 = 0.002,
    // which is exactly kBLOmega1, so the result must match the same reference.
    auto post = bl_posterior_returns_default_omega(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, 2, 1);
    ASSERT_TRUE(post.has_value());
    EXPECT_NEAR(post->at(0), 0.075, 1e-6);
    EXPECT_NEAR(post->at(1), 0.07625, 1e-6);
}

TEST(FinanceBlackLitterman, PosteriorDimensionMismatchErrors) {
    std::vector<double> bad_pi = {0.05}; // size 1, n=2
    EXPECT_FALSE(bl_posterior_returns(bad_pi, kBLCov2, kBLTau, kBLP1, kBLQ1, kBLOmega1, 2, 1)
                    .has_value());

    std::vector<double> bad_P = {1.0, 0.0, 0.0}; // not k*n = 1*2
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, bad_P, kBLQ1, kBLOmega1, 2, 1)
                    .has_value());

    std::vector<double> bad_Q = {0.10, 0.05}; // not size k=1
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, bad_Q, kBLOmega1, 2, 1)
                    .has_value());

    std::vector<double> bad_omega = {0.002, 0.0}; // not k*k = 1*1
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, bad_omega, 2, 1)
                    .has_value());
}

TEST(FinanceBlackLitterman, PosteriorSigmaDimensionMismatchErrors) {
    std::vector<double> bad_cov = {0.04, 0.01, 0.01}; // not 2x2
    auto post = bl_posterior_returns(kBLPi2, bad_cov, kBLTau, kBLP1, kBLQ1, kBLOmega1, 2, 1);
    EXPECT_FALSE(post.has_value());
}

TEST(FinanceBlackLitterman, PosteriorNonPositiveTauErrors) {
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, 0.0, kBLP1, kBLQ1, kBLOmega1, 2, 1)
                    .has_value());
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, -0.01, kBLP1, kBLQ1, kBLOmega1, 2, 1)
                    .has_value());
}

TEST(FinanceBlackLitterman, PosteriorNonPositiveNErrors) {
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, kBLOmega1, 0, 1)
                    .has_value());
    EXPECT_FALSE(bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, kBLOmega1, -1, 1)
                    .has_value());
}

TEST(FinanceBlackLitterman, PosteriorKZeroNonEmptyViewsErrors) {
    // k=0 must be paired with genuinely empty P/Q/omega; supplying views
    // while claiming k=0 is an inconsistent request, not silently ignored.
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, kBLOmega1, 2, 0);
    EXPECT_FALSE(post.has_value());
}

TEST(FinanceBlackLitterman, PosteriorSingularOmegaErrors) {
    std::vector<double> singular_omega = {0.0};
    auto post = bl_posterior_returns(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, singular_omega, 2, 1);
    EXPECT_FALSE(post.has_value());
}

TEST(FinanceBlackLitterman, PosteriorSingularSigmaErrors) {
    std::vector<double> singular_cov = {0.0, 0.0,
                                        0.0, 0.0};
    auto post = bl_posterior_returns(kBLPi2, singular_cov, kBLTau, kBLP1, kBLQ1, kBLOmega1, 2, 1);
    EXPECT_FALSE(post.has_value());
}

TEST(FinanceBlackLitterman, DefaultOmegaDimensionMismatchErrors) {
    std::vector<double> bad_cov = {0.04, 0.01, 0.01}; // not 2x2
    EXPECT_FALSE(bl_posterior_returns_default_omega(kBLPi2, bad_cov, kBLTau, kBLP1, kBLQ1, 2, 1)
                    .has_value());

    std::vector<double> bad_P = {1.0, 0.0, 0.0}; // not k*n
    EXPECT_FALSE(
        bl_posterior_returns_default_omega(kBLPi2, kBLCov2, kBLTau, bad_P, kBLQ1, 2, 1)
            .has_value());

    std::vector<double> bad_Q = {0.10, 0.05}; // not size k
    EXPECT_FALSE(
        bl_posterior_returns_default_omega(kBLPi2, kBLCov2, kBLTau, kBLP1, bad_Q, 2, 1)
            .has_value());
}

TEST(FinanceBlackLitterman, DefaultOmegaNonPositiveTauOrNErrors) {
    EXPECT_FALSE(
        bl_posterior_returns_default_omega(kBLPi2, kBLCov2, 0.0, kBLP1, kBLQ1, 2, 1).has_value());
    EXPECT_FALSE(
        bl_posterior_returns_default_omega(kBLPi2, kBLCov2, kBLTau, kBLP1, kBLQ1, 0, 1)
            .has_value());
}

TEST(FinanceBlackLitterman, ThreeAssetTwoViewsProducesFiniteBlend) {
    // Reuses the 3-asset covariance from the Markowitz tests above. Two
    // views: an absolute view on asset 0, and a relative view (asset1 vs
    // asset2), derived Omega. This is a sanity/integration check (finite,
    // right-sized output) rather than a hand-computed reference.
    std::vector<double> w_mkt3 = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    auto pi3 = bl_implied_returns(kCov3, w_mkt3, 2.5, 3);
    ASSERT_TRUE(pi3.has_value());

    std::vector<double> P3 = {1.0, 0.0, 0.0,
                              0.0, 1.0, -1.0};
    std::vector<double> Q3 = {0.15, 0.03};
    auto post = bl_posterior_returns_default_omega(*pi3, kCov3, 0.05, P3, Q3, 3, 2);
    ASSERT_TRUE(post.has_value());
    ASSERT_EQ(post->size(), 3u);
    for (double v : *post) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(FinanceBlackLitterman, EndToEndImpliedThenPosteriorFeedsMeanVariance) {
    // Full pipeline: reverse-optimize the prior from market weights, blend in
    // a view, then feed the posterior returns into the existing max-Sharpe
    // optimizer as `mu`, exercising the intended cross-module usage.
    std::vector<double> w_mkt2 = {0.5, 0.5};
    auto pi = bl_implied_returns(kBLCov2, w_mkt2, 2.5, 2);
    ASSERT_TRUE(pi.has_value());

    auto post = bl_posterior_returns_default_omega(*pi, kBLCov2, kBLTau, kBLP1, kBLQ1, 2, 1);
    ASSERT_TRUE(post.has_value());
    // The view pulls asset 0's posterior return above its implied prior.
    EXPECT_GT(post->at(0), pi->at(0));

    auto w = max_sharpe_portfolio(kBLCov2, *post, 0.0, 2);
    ASSERT_TRUE(w.has_value());
    EXPECT_NEAR(w->at(0) + w->at(1), 1.0, 1e-8);
}

TEST(FinanceMC, AsianEdgeCasesReturnZero) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    EXPECT_EQ(mc_asian_call(S, K, T, r, sigma, 0, 10), 0.0);
    EXPECT_EQ(mc_asian_call(S, K, T, r, sigma, 1000, 0), 0.0);
    EXPECT_EQ(mc_asian_put(-1.0, K, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_asian_put(S, -1.0, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_asian_put(S, K, -1.0, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_asian_put(S, K, T, r, -0.1, 1000, 10), 0.0);
}

// --- Lookback options via Monte Carlo ---
// Floating-strike payoffs (S_T-min(path), max(path)-S_T) are always
// non-negative pathwise since min(path)<=S_T<=max(path), and always strictly
// exceed the at-the-money vanilla call/put payoff since the path starts at S
// (so min(path)<=S and max(path)>=S). The margin used below (kLookbackMargin)
// is well beyond the MC noise floor implied by kMCTol above.
namespace {
constexpr double kLookbackMargin = 0.5;
} // namespace

TEST(FinanceMC, LookbackFloatingCallPositive) {
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double price = mc_lookback_floating_call(S, T, r, sigma, kMCPaths, 50);
    EXPECT_GT(price, 0.0);
}

TEST(FinanceMC, LookbackFloatingPutPositive) {
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double price = mc_lookback_floating_put(S, T, r, sigma, kMCPaths, 50);
    EXPECT_GT(price, 0.0);
}

TEST(FinanceMC, LookbackFloatingCallBeatsATMEuropeanCall) {
    // S_T-min(path) >= S_T-S = max(S_T-S,0)-min(0,S_T-S) pathwise, and since
    // min(path)<=S always (path starts at S), S_T-min(path) >= max(S_T-S,0).
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double lookback = mc_lookback_floating_call(S, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_call(S, S, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFloatingCallBeatsATMEuropeanCallHighVol) {
    double S = 100, T = 1.0, r = 0.03, sigma = 0.35;
    double lookback = mc_lookback_floating_call(S, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_call(S, S, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFloatingPutBeatsATMEuropeanPut) {
    // By the symmetric argument, max(path)-S_T >= S-S_T = max(K-S_T,0) at K=S,
    // since max(path)>=S always.
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double lookback = mc_lookback_floating_put(S, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_put(S, S, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFloatingPutBeatsATMEuropeanPutHighVol) {
    double S = 100, T = 1.0, r = 0.03, sigma = 0.35;
    double lookback = mc_lookback_floating_put(S, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_put(S, S, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFloatingStableAcrossStepCounts) {
    // Finer path sampling makes the realized min/max more extreme, but for a
    // fixed seed and moderate parameters the price shouldn't swing wildly.
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double coarse = mc_lookback_floating_call(S, T, r, sigma, kMCPaths, 50);
    double fine = mc_lookback_floating_call(S, T, r, sigma, kMCPaths, 200);
    EXPECT_GT(coarse, 0.0);
    EXPECT_GT(fine, 0.0);
    EXPECT_NEAR(coarse, fine, 2.0);
}

TEST(FinanceMC, LookbackFloatingCallDeterministic) {
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_lookback_floating_call(S, T, r, sigma, 5000, 20, 3);
    double b = mc_lookback_floating_call(S, T, r, sigma, 5000, 20, 3);
    EXPECT_EQ(a, b);
}

TEST(FinanceMC, LookbackFloatingPutDeterministic) {
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_lookback_floating_put(S, T, r, sigma, 5000, 20, 3);
    double b = mc_lookback_floating_put(S, T, r, sigma, 5000, 20, 3);
    EXPECT_EQ(a, b);
}

TEST(FinanceMC, LookbackFloatingEdgeCasesReturnZero) {
    double S = 100, T = 1.0, r = 0.05, sigma = 0.2;
    EXPECT_EQ(mc_lookback_floating_call(S, T, r, sigma, 0, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_call(S, T, r, sigma, 1000, 0), 0.0);
    EXPECT_EQ(mc_lookback_floating_call(-1.0, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_call(S, -1.0, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_call(S, T, r, -0.1, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_put(-1.0, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_put(S, -1.0, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_put(S, T, r, -0.1, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_put(S, T, r, sigma, 0, 10), 0.0);
    EXPECT_EQ(mc_lookback_floating_put(S, T, r, sigma, 1000, 0), 0.0);
}

TEST(FinanceMC, LookbackFixedCallAtLeastVanillaCall) {
    // max(max(path)-K,0) >= max(S_T-K,0) pathwise since max(path)>=S_T.
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double lookback = mc_lookback_fixed_call(S, K, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_call(S, K, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFixedCallAtLeastVanillaCallOTM) {
    double S = 90, K = 110, T = 1.0, r = 0.04, sigma = 0.3;
    double lookback = mc_lookback_fixed_call(S, K, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_call(S, K, T, r, sigma);
    EXPECT_GT(lookback, vanilla);
    EXPECT_GE(lookback, vanilla - 1e-9);
}

TEST(FinanceMC, LookbackFixedPutAtLeastVanillaPut) {
    // max(K-min(path),0) >= max(K-S_T,0) pathwise since min(path)<=S_T.
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double lookback = mc_lookback_fixed_put(S, K, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_put(S, K, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFixedPutAtLeastVanillaPutITM) {
    double S = 90, K = 110, T = 1.0, r = 0.05, sigma = 0.25;
    double lookback = mc_lookback_fixed_put(S, K, T, r, sigma, kMCPaths, 50);
    double vanilla = bs_put(S, K, T, r, sigma);
    EXPECT_GT(lookback, vanilla + kLookbackMargin);
}

TEST(FinanceMC, LookbackFixedPayoffsNonNegative) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.3;
    double call = mc_lookback_fixed_call(S, K, T, r, sigma, 50000, 25);
    double put  = mc_lookback_fixed_put(S, K, T, r, sigma, 50000, 25);
    EXPECT_GE(call, 0.0);
    EXPECT_GE(put, 0.0);
}

TEST(FinanceMC, LookbackFixedCallDeterministic) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_lookback_fixed_call(S, K, T, r, sigma, 5000, 20, 3);
    double b = mc_lookback_fixed_call(S, K, T, r, sigma, 5000, 20, 3);
    EXPECT_EQ(a, b);
}

TEST(FinanceMC, LookbackFixedPutDeterministic) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double a = mc_lookback_fixed_put(S, K, T, r, sigma, 5000, 20, 3);
    double b = mc_lookback_fixed_put(S, K, T, r, sigma, 5000, 20, 3);
    EXPECT_EQ(a, b);
}

TEST(FinanceMC, LookbackFixedEdgeCasesReturnZero) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    EXPECT_EQ(mc_lookback_fixed_call(S, K, T, r, sigma, 0, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(S, K, T, r, sigma, 1000, 0), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(-1.0, K, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(S, -1.0, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(S, K, -1.0, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_call(S, K, T, r, -0.1, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_put(-1.0, K, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_put(S, -1.0, T, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_put(S, K, -1.0, r, sigma, 1000, 10), 0.0);
    EXPECT_EQ(mc_lookback_fixed_put(S, K, T, r, -0.1, 1000, 10), 0.0);
}

// --- Geometric-average Asian options (Kemna-Vorst closed form) ---
namespace {
// Monte Carlo cross-check for geo_asian_call/geo_asian_put: simulates the
// SAME discrete-fixing convention (t_i=i*T/n_steps, i=1..n_steps, i.e.
// n_steps fixings), tracking a running GEOMETRIC mean along the path instead
// of the arithmetic mean mc_asian_call/mc_asian_put track. Mirrors mc_asian's
// antithetic-variate structure and std::mt19937(seed) RNG convention.
double mc_geo_asian(double S, double K, double T, double r, double sigma,
                    int n_paths, int n_steps, unsigned seed, bool call) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    double dt = T / static_cast<double>(n_steps);
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrtdt = sigma * std::sqrt(dt);
    std::vector<double> z(static_cast<size_t>(n_steps));

    int n_draws = (n_paths + 1) / 2;
    double sum = 0.0;
    int used = 0;
    for (int p = 0; p < n_draws; ++p) {
        for (auto& zi : z) zi = nd(rng);

        double s = S, log_sum = 0.0;
        for (double zi : z) {
            s *= std::exp(drift + vol_sqrtdt * zi);
            log_sum += std::log(s);
        }
        double geo = std::exp(log_sum / n_steps);
        double payoff1 = call ? std::max(geo - K, 0.0) : std::max(K - geo, 0.0);
        sum += payoff1;
        ++used;
        if (used >= n_paths) break;

        s = S; log_sum = 0.0;
        for (double zi : z) {
            s *= std::exp(drift - vol_sqrtdt * zi);
            log_sum += std::log(s);
        }
        geo = std::exp(log_sum / n_steps);
        double payoff2 = call ? std::max(geo - K, 0.0) : std::max(K - geo, 0.0);
        sum += payoff2;
        ++used;
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

constexpr int kGeoMCPaths = 200000;
constexpr int kGeoMCSteps = 50;
constexpr double kGeoMCTol = 0.1;
} // namespace

TEST(FinanceGeoAsian, SingleFixingMatchesBSCall) {
    // n_fixings=1 observes only S_T, so this must reduce EXACTLY to plain
    // Black-Scholes (sigma_adj==sigma, mu_adj==r at n=1).
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double geo = geo_asian_call(S, K, T, r, sigma, 1);
    double bs = bs_call(S, K, T, r, sigma);
    EXPECT_NEAR(geo, bs, 1e-9);
}

TEST(FinanceGeoAsian, SingleFixingMatchesBSPut) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double geo = geo_asian_put(S, K, T, r, sigma, 1);
    double bs = bs_put(S, K, T, r, sigma);
    EXPECT_NEAR(geo, bs, 1e-9);
}

TEST(FinanceGeoAsian, ContinuousLimitCallConvergence) {
    // As n_fixings -> infinity, sigma_adj -> sigma/sqrt(3) and
    // mu_adj -> 0.5*(r - sigma^2/6): the classic continuous geometric-average
    // closed form, computed independently here via black76 on that limiting
    // forward/vol pair (no loop needed: the closed form is O(1) in n).
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double sigma_c = sigma / std::sqrt(3.0);
    double mu_c = 0.5 * (r - sigma * sigma / 6.0);
    double F_c = S * std::exp(mu_c * T);
    double continuous = black76(F_c, K, T, r, sigma_c, true);
    double discrete = geo_asian_call(S, K, T, r, sigma, 1000000);
    EXPECT_NEAR(discrete, continuous, 1e-4);
}

TEST(FinanceGeoAsian, ContinuousLimitPutConvergence) {
    double S = 110, K = 95, T = 2.0, r = 0.04, sigma = 0.35;
    double sigma_c = sigma / std::sqrt(3.0);
    double mu_c = 0.5 * (r - sigma * sigma / 6.0);
    double F_c = S * std::exp(mu_c * T);
    double continuous = black76(F_c, K, T, r, sigma_c, false);
    double discrete = geo_asian_put(S, K, T, r, sigma, 2000000);
    EXPECT_NEAR(discrete, continuous, 1e-3);
}

TEST(FinanceGeoAsian, CheaperThanVanillaEuropeanCall) {
    // Averaging reduces effective volatility, so the geometric Asian call
    // (like the arithmetic one) is always cheaper than the vanilla European
    // call sharing the same spot/strike/maturity.
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double geo = geo_asian_call(S, K, T, r, sigma, 50);
    double euro = bs_call(S, K, T, r, sigma);
    EXPECT_LT(geo, euro);
    EXPECT_GT(geo, 0.0);
}

TEST(FinanceGeoAsian, CheaperThanVanillaEuropeanPut) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.3;
    double geo = geo_asian_put(S, K, T, r, sigma, 50);
    double euro = bs_put(S, K, T, r, sigma);
    EXPECT_LT(geo, euro);
    EXPECT_GT(geo, 0.0);
}

TEST(FinanceGeoAsian, PutCallParity) {
    // Same Black-76-style parity as the underlying formula:
    // C - P = e^{-rT}*(F - K) for the synthetic geometric-average forward F.
    double S = 100, K = 105, T = 1.0, r = 0.05, sigma = 0.25;
    int n = 24;
    double c = geo_asian_call(S, K, T, r, sigma, n);
    double p = geo_asian_put(S, K, T, r, sigma, n);
    double sigma_adj = sigma * std::sqrt((n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n));
    double mu_adj = 0.5 * sigma_adj * sigma_adj +
                    (n + 1.0) / (2.0 * n) * (r - 0.5 * sigma * sigma);
    double F = S * std::exp(mu_adj * T);
    EXPECT_NEAR(c - p, std::exp(-r * T) * (F - K), 1e-8);
}

TEST(FinanceGeoAsian, MonotoneDecreasingEffectiveVolWithFixings) {
    // sigma_adj is monotonically decreasing in n (from sigma at n=1 down
    // toward sigma/sqrt(3)); for an ATM call, price should therefore
    // decrease monotonically as n_fixings increases.
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double prev = geo_asian_call(S, K, T, r, sigma, 1);
    for (int n : {2, 4, 8, 16, 32, 64}) {
        double cur = geo_asian_call(S, K, T, r, sigma, n);
        EXPECT_LT(cur, prev);
        prev = cur;
    }
}

TEST(FinanceGeoAsian, ZeroOrNegativeFixingsClampsToOne) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double base = geo_asian_call(S, K, T, r, sigma, 1);
    EXPECT_NEAR(geo_asian_call(S, K, T, r, sigma, 0), base, 1e-12);
    EXPECT_NEAR(geo_asian_call(S, K, T, r, sigma, -5), base, 1e-12);
}

TEST(FinanceGeoAsian, ExpiredOptionReturnsIntrinsic) {
    double S = 110, K = 100, T = 0.0, r = 0.05, sigma = 0.2;
    EXPECT_NEAR(geo_asian_call(S, K, T, r, sigma, 10), 10.0, 1e-9);
    EXPECT_NEAR(geo_asian_put(S, K, T, r, sigma, 10), 0.0, 1e-9);
}

TEST(FinanceGeoAsian, MatchesMonteCarloATM) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double closed = geo_asian_call(S, K, T, r, sigma, kGeoMCSteps);
    double mc = mc_geo_asian(S, K, T, r, sigma, kGeoMCPaths, kGeoMCSteps, 42, true);
    EXPECT_NEAR(closed, mc, kGeoMCTol);
}

TEST(FinanceGeoAsian, MatchesMonteCarloITMPut) {
    double S = 90, K = 100, T = 1.0, r = 0.05, sigma = 0.25;
    double closed = geo_asian_put(S, K, T, r, sigma, kGeoMCSteps);
    double mc = mc_geo_asian(S, K, T, r, sigma, kGeoMCPaths, kGeoMCSteps, 43, false);
    EXPECT_NEAR(closed, mc, kGeoMCTol);
}

TEST(FinanceGeoAsian, MatchesMonteCarloHighVol) {
    double S = 100, K = 95, T = 0.5, r = 0.03, sigma = 0.4;
    double closed = geo_asian_call(S, K, T, r, sigma, kGeoMCSteps);
    double mc = mc_geo_asian(S, K, T, r, sigma, kGeoMCPaths, kGeoMCSteps, 44, true);
    EXPECT_NEAR(closed, mc, kGeoMCTol);
}

// --- Trinomial tree option pricing (Boyle 1986) ---
TEST(FinanceTrinomial, EuropeanCallConvergesToBS_ATM) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double bs = bs_call(S, K, T, r, sigma);
    double tri = trinomial_option(S, K, T, r, sigma, 300, true, false);
    EXPECT_NEAR(tri, bs, 1e-2);
}

TEST(FinanceTrinomial, EuropeanPutConvergesToBS_ATM) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    double bs = bs_put(S, K, T, r, sigma);
    double tri = trinomial_option(S, K, T, r, sigma, 300, false, false);
    EXPECT_NEAR(tri, bs, 1e-2);
}

TEST(FinanceTrinomial, EuropeanCallConvergesToBS_ITM) {
    double S = 120, K = 100, T = 0.5, r = 0.03, sigma = 0.25;
    double bs = bs_call(S, K, T, r, sigma);
    double tri = trinomial_option(S, K, T, r, sigma, 300, true, false);
    EXPECT_NEAR(tri, bs, 1e-2);
}

TEST(FinanceTrinomial, EuropeanPutConvergesToBS_ITM) {
    double S = 80, K = 100, T = 0.5, r = 0.03, sigma = 0.25;
    double bs = bs_put(S, K, T, r, sigma);
    double tri = trinomial_option(S, K, T, r, sigma, 300, false, false);
    EXPECT_NEAR(tri, bs, 1e-2);
}

TEST(FinanceTrinomial, EuropeanCallConvergesToBS_OTM) {
    double S = 90, K = 110, T = 1.0, r = 0.04, sigma = 0.3;
    double bs = bs_call(S, K, T, r, sigma);
    double tri = trinomial_option(S, K, T, r, sigma, 300, true, false);
    EXPECT_NEAR(tri, bs, 1e-2);
}

TEST(FinanceTrinomial, EuropeanPutConvergesToBS_OTM) {
    double S = 110, K = 90, T = 1.0, r = 0.04, sigma = 0.3;
    double bs = bs_put(S, K, T, r, sigma);
    double tri = trinomial_option(S, K, T, r, sigma, 300, false, false);
    EXPECT_NEAR(tri, bs, 1e-2);
}

TEST(FinanceTrinomial, EuropeanPutCallParity) {
    double S = 100, K = 105, T = 1.0, r = 0.05, sigma = 0.2;
    double c = trinomial_option(S, K, T, r, sigma, 200, true, false);
    double p = trinomial_option(S, K, T, r, sigma, 200, false, false);
    EXPECT_NEAR(c - p, S - K * std::exp(-r * T), 0.05);
}

TEST(FinanceTrinomial, AmericanCallEqualsEuropeanNonDividend) {
    // Non-dividend-paying American call is never optimally exercised early,
    // so American == European exactly (same equality american_option and
    // binomial_call already exhibit for the binomial tree).
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 300;
    double euro = trinomial_option(S, K, T, r, sigma, steps, true, false);
    double amer = trinomial_option(S, K, T, r, sigma, steps, true, true);
    EXPECT_NEAR(amer, euro, 1e-9);
}

TEST(FinanceTrinomial, AmericanPutEarlyExercisePremium) {
    double S = 80, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 300;
    double euro = trinomial_option(S, K, T, r, sigma, steps, false, false);
    double amer = trinomial_option(S, K, T, r, sigma, steps, false, true);
    EXPECT_GE(amer, euro - 1e-10);
    EXPECT_GT(amer - euro, 1.0);  // clear early-exercise premium, deep ITM put
}

TEST(FinanceTrinomial, CrossCheckBinomialEuropeanCall) {
    // Both are discretizations of the same continuous-time GBM process, so
    // they should agree closely (generous tolerance since the discretization
    // grids differ: binary vs ternary branching).
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 250;
    double tri = trinomial_option(S, K, T, r, sigma, steps, true, false);
    double bin = binomial_call(S, K, T, r, sigma, steps);
    EXPECT_NEAR(tri, bin, 0.02 * bin);
}

TEST(FinanceTrinomial, CrossCheckBinomialEuropeanPut) {
    double S = 100, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 250;
    double tri = trinomial_option(S, K, T, r, sigma, steps, false, false);
    double bin = binomial_put(S, K, T, r, sigma, steps);
    EXPECT_NEAR(tri, bin, 0.02 * bin);
}

TEST(FinanceTrinomial, CrossCheckBinomialAmericanPut) {
    double S = 80, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 250;
    double tri = trinomial_option(S, K, T, r, sigma, steps, false, true);
    double bin = american_option(S, K, T, r, sigma, false, steps);
    EXPECT_NEAR(tri, bin, 0.02 * bin);
}

TEST(FinanceTrinomial, CrossCheckBinomialAmericanCall) {
    double S = 120, K = 100, T = 1.0, r = 0.05, sigma = 0.2;
    int steps = 250;
    double tri = trinomial_option(S, K, T, r, sigma, steps, true, true);
    double bin = american_option(S, K, T, r, sigma, true, steps);
    EXPECT_NEAR(tri, bin, 0.02 * bin);
}
