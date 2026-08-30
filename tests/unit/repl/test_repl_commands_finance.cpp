#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, finance_info_bindings) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_call(S,K,T,r,sigma)");
    expect_contains(interp, "help", "finance_npv(rate,cf)");
    expect_contains(interp, "help", "finance_sharpe(r)");
    expect_contains(interp, "help", "info_entropy(p)");

    expect_contains(interp, "finance_npv(0.1, [-100, 50, 60])", "-4.958");

    expect_ok(interp, "cf = [100, 50, 60]");
    expect_ok(interp, "v = finance_npv(0.1, cf)");
    EXPECT_NEAR(interp.state().scalars.at("v"), 100.0 + 50.0 / 1.1 + 60.0 / 1.21, 1e-6);

    expect_ok(interp, "h = info_entropy([0.5; 0.5])");
    EXPECT_NEAR(interp.state().scalars.at("h"), 1.0, 1e-9);

    expect_ok(interp, "sh = finance_sharpe([0.1; 0.2; -0.05])");
    EXPECT_GT(interp.state().scalars.at("sh"), 0.0);

    expect_contains(interp, "info_entropy([0.5, 0.5])", "1");
}

TEST(ReplCommandsTest, finance_irr) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_irr(cf)");

    expect_contains(interp, "finance_irr([-100, 110])", "0.1");

    expect_ok(interp, "cf = [-100; 110]");
    expect_ok(interp, "r = finance_irr(cf)");
    EXPECT_NEAR(interp.state().scalars.at("r"), 0.1, 1e-6);

    expect_ok(interp, "r2 = finance_irr([-100, 110])");
    EXPECT_NEAR(interp.state().scalars.at("r2"), 0.1, 1e-6);
}

TEST(ReplCommandsTest, finance_var) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_var(r)");

    expect_ok(interp, "r = [0.1; -0.05; 0.2]");
    expect_ok(interp, "v = finance_var(r)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("v")));

    expect_contains(interp, "finance_var([0.1; -0.05; 0.2])", "0.");
}

TEST(ReplCommandsTest, finance_cvar) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_cvar(r)");

    expect_ok(interp, "r = [0.1; -0.05; 0.2]");
    expect_ok(interp, "cv = finance_cvar(r)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("cv")));

    expect_contains(interp, "finance_cvar([0.1; -0.05; 0.2])", "0.");
}

TEST(ReplCommandsTest, finance_sortino) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_sortino(r)");

    expect_ok(interp, "ret = [0.1; -0.05; 0.2]");
    expect_ok(interp, "sortino = finance_sortino(ret)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("sortino")));
    EXPECT_GT(interp.state().scalars.at("sortino"), 0.0);

    expect_contains(interp, "finance_sortino([0.1; -0.05; 0.2])", "1.666");
}

TEST(ReplCommandsTest, finance_max_drawdown) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_max_drawdown(equity)");

    expect_ok(interp, "equity = [100; 110; 105; 120; 90; 95]");
    expect_ok(interp, "mdd = finance_max_drawdown(equity)");
    EXPECT_NEAR(interp.state().scalars.at("mdd"), 0.25, 1e-6);

    expect_contains(interp, "finance_max_drawdown([100; 110; 105; 120; 90; 95])", "0.25");
}

TEST(ReplCommandsTest, finance_kelly_fraction) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_kelly_fraction(p,b)");

    expect_ok(interp, "kelly = finance_kelly_fraction(0.6, 2.0)");
    EXPECT_NEAR(interp.state().scalars.at("kelly"), 0.4, 1e-9);

    expect_contains(interp, "finance_kelly_fraction(0.6, 2.0)", "0.4");
}

TEST(ReplCommandsTest, finance_compound) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_compound(principal,rate,n_periods,compounds_per_period)");

    expect_ok(interp, "fv = finance_compound(100, 0.1, 3)");
    EXPECT_NEAR(interp.state().scalars.at("fv"), 133.1, 1e-6);

    expect_contains(interp, "finance_compound(100, 0.1, 3)", "133.1");
}

TEST(ReplCommandsTest, finance_continuous_compound) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_continuous_compound(principal,rate,t)");

    expect_ok(interp, "cc = finance_continuous_compound(100, 0.1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), 100.0 * std::exp(0.1), 1e-6);

    expect_contains(interp, "finance_continuous_compound(100, 0.1, 1)", "110.517");
}

TEST(ReplCommandsTest, finance_pv) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_pv(rate,n,pmt,fv)");

    expect_ok(interp, "pv0 = finance_pv(0, 5, -10, 0)");
    EXPECT_NEAR(interp.state().scalars.at("pv0"), 50.0, 1e-9);

    expect_contains(interp, "finance_pv(0, 5, -10, 0)", "50");
}

TEST(ReplCommandsTest, finance_fv_annuity) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_fv_annuity(rate,n,pmt,pv0)");

    expect_ok(interp, "fv0 = finance_fv_annuity(0, 5, -10, 0)");
    EXPECT_NEAR(interp.state().scalars.at("fv0"), 50.0, 1e-9);

    expect_contains(interp, "finance_fv_annuity(0, 5, -10, 0)", "50");
}

TEST(ReplCommandsTest, finance_pmt_annuity) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_pmt_annuity(rate,n,pv0,fv)");

    expect_ok(interp, "pmt = finance_pmt_annuity(0, 5, -50, 0)");
    EXPECT_NEAR(interp.state().scalars.at("pmt"), 10.0, 1e-9);

    expect_contains(interp, "finance_pmt_annuity(0, 5, -50, 0)", "10");
}

TEST(ReplCommandsTest, finance_binomial_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_binomial_call(S,K,T,r,sigma,steps)");

    expect_ok(interp, "bin_c = finance_binomial_call(100, 100, 1, 0.05, 0.2, 200)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bin_c")));
    EXPECT_GT(interp.state().scalars.at("bin_c"), 0.0);

    expect_contains(interp, "finance_binomial_call(100, 100, 1, 0.05, 0.2, 200)", "10.");
}

TEST(ReplCommandsTest, finance_binomial_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_binomial_put(S,K,T,r,sigma,steps)");

    expect_ok(interp, "bin_p = finance_binomial_put(100, 100, 1, 0.05, 0.2, 200)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bin_p")));
    EXPECT_GT(interp.state().scalars.at("bin_p"), 0.0);

    expect_contains(interp, "finance_binomial_put(100, 100, 1, 0.05, 0.2, 200)", "5.56");
}

TEST(ReplCommandsTest, finance_bs_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_put(S,K,T,r,sigma)");

    expect_ok(interp, "bs_p = finance_bs_put(100, 100, 1, 0.05, 0.2)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("bs_p")));
    EXPECT_GT(interp.state().scalars.at("bs_p"), 0.0);

    expect_contains(interp, "finance_bs_put(100, 100, 1, 0.05, 0.2)", "5.57");
}

TEST(ReplCommandsTest, finance_bs_gamma) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_gamma(S,K,T,r,sigma)");

    expect_ok(interp, "g = finance_bs_gamma(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("g"), 0.0);

    expect_contains(interp, "finance_bs_gamma(100, 100, 1, 0.05, 0.2)", "0.018762");
}

TEST(ReplCommandsTest, finance_bond_duration) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_duration(c,y,n)");

    expect_ok(interp, "dur = finance_bond_duration(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("dur"), 5.0, 1e-6);

    expect_contains(interp, "finance_bond_duration(0, 0.05, 5)", "5");
}

TEST(ReplCommandsTest, finance_bs_delta) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_delta(S,K,T,r,sigma,call)");

    expect_ok(interp, "d = finance_bs_delta(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_GT(interp.state().scalars.at("d"), 0.0);
    EXPECT_LT(interp.state().scalars.at("d"), 1.0);

    expect_contains(interp, "finance_bs_delta(100, 100, 1, 0.05, 0.2, 1)", "0.636831");
}

TEST(ReplCommandsTest, finance_bs_vega) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_vega(S,K,T,r,sigma)");

    expect_ok(interp, "v = finance_bs_vega(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("v"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("v")));

    expect_contains(interp, "finance_bs_vega(100, 100, 1, 0.05, 0.2)", "37.524035");
}

TEST(ReplCommandsTest, finance_bond_modified_duration) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_modified_duration(c,y,n)");

    expect_ok(interp, "mdur = finance_bond_modified_duration(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("mdur"), 5.0 / 1.05, 1e-6);

    expect_contains(interp, "finance_bond_modified_duration(0, 0.05, 5)", "4.761905");
}

TEST(ReplCommandsTest, finance_bs_theta) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_theta(S,K,T,r,sigma,call)");

    expect_ok(interp, "th = finance_bs_theta(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_LT(interp.state().scalars.at("th"), 0.0);

    expect_contains(interp, "finance_bs_theta(100, 100, 1, 0.05, 0.2, 1)", "-6.414028");
}

TEST(ReplCommandsTest, finance_bs_rho) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_rho(S,K,T,r,sigma,call)");

    expect_ok(interp, "rh = finance_bs_rho(100, 100, 1, 0.05, 0.2, 1)");
    EXPECT_GT(interp.state().scalars.at("rh"), 0.0);

    expect_contains(interp, "finance_bs_rho(100, 100, 1, 0.05, 0.2, 1)", "53.232482");
}

TEST(ReplCommandsTest, finance_bond_convexity) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_convexity(c,y,n)");

    expect_ok(interp, "conv = finance_bond_convexity(0, 0.05, 5)");
    EXPECT_NEAR(interp.state().scalars.at("conv"), 30.0 / (1.05 * 1.05), 1e-6);

    expect_contains(interp, "finance_bond_convexity(0, 0.05, 5)", "27.210884");
}

TEST(ReplCommandsTest, finance_bond_ytm) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_ytm(price,c,n)");

    expect_ok(interp, "bp = finance_bond_price(0.05, 0.07, 10)");
    expect_ok(interp, "ytm = finance_bond_ytm(bp, 0.05, 10)");
    EXPECT_NEAR(interp.state().scalars.at("ytm"), 0.07, 1e-5);

    expect_contains(interp, "finance_bond_ytm(85.952837, 0.05, 10)", "0.07");
}

TEST(ReplCommandsTest, finance_bs_implied_vol) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bs_implied_vol(price,S,K,T,r,call)");

    expect_ok(interp, "mp = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    expect_ok(interp, "iv = finance_bs_implied_vol(mp, 100, 100, 1, 0.05, 1)");
    EXPECT_NEAR(interp.state().scalars.at("iv"), 0.2, 1e-5);

    expect_contains(interp, "finance_bs_implied_vol(10.450583572185565, 100, 100, 1, 0.05, 1)",
                    "0.2");
}

TEST(ReplCommandsTest, finance_portfolio_return) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_portfolio_return(weights,returns)");

    expect_ok(interp, "w = [0.6; 0.4]");
    expect_ok(interp, "ret = [0.1; 0.05]");
    expect_ok(interp, "pr = finance_portfolio_return(w, ret)");
    EXPECT_NEAR(interp.state().scalars.at("pr"), 0.08, 1e-9);

    expect_contains(interp, "finance_portfolio_return([0.6; 0.4], [0.1; 0.05])", "0.08");
}

TEST(ReplCommandsTest, finance_portfolio_variance) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_portfolio_variance(weights,cov)");

    expect_ok(interp, "w = [0.5; 0.5]");
    expect_ok(interp, "cov = [0.04, 0.01; 0.01, 0.09]");
    expect_ok(interp, "pvar = finance_portfolio_variance(w, cov)");
    EXPECT_NEAR(interp.state().scalars.at("pvar"), 0.0375, 1e-9);

    expect_contains(interp, "finance_portfolio_variance([0.5; 0.5], [0.04, 0.01; 0.01, 0.09])",
                    "0.0375");
}

TEST(ReplCommandsTest, finance_bond_price_fv) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bond_price(c,y,n,fv)");

    expect_ok(interp, "bp100 = finance_bond_price(0.05, 0.05, 10, 100)");
    EXPECT_NEAR(interp.state().scalars.at("bp100"), 100.0, 0.5);
}

TEST(ReplCommandsTest, finance_compound_cpp) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_compound(principal,rate,n_periods,compounds_per_period)");

    expect_ok(interp, "fv4 = finance_compound(100, 0.1, 1, 4)");
    EXPECT_NEAR(interp.state().scalars.at("fv4"), 110.381, 0.05);
}

TEST(ReplCommandsTest, finance_capm) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_capm(risk_free,beta,market_return)");

    // 0.05 + 1.2 * (0.10 - 0.05) = 0.11.
    expect_ok(interp, "capm = finance_capm(0.05, 1.2, 0.10)");
    EXPECT_NEAR(interp.state().scalars.at("capm"), 0.11, 1e-9);

    expect_contains(interp, "finance_capm(0.05, 1.2, 0.10)", "0.11");
}

TEST(ReplCommandsTest, finance_forward_rate) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_forward_rate(r1,t1,r2,t2)");

    // (0.06*2 - 0.05*1) / (2 - 1) = 0.07.
    expect_ok(interp, "fr = finance_forward_rate(0.05, 1.0, 0.06, 2.0)");
    EXPECT_NEAR(interp.state().scalars.at("fr"), 0.07, 1e-9);

    expect_contains(interp, "finance_forward_rate(0.05, 1.0, 0.06, 2.0)", "0.07");
}

TEST(ReplCommandsTest, finance_black76) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_black76(F,K,T,r,sigma,call)");

    // With F = S*exp(r*T) (S=100,r=0.05,T=1), black76 call == bs_call(100,100,1,0.05,0.2)
    // == 10.450583572185565 (reused from the finance_bs_implied_vol precedent).
    expect_ok(interp, "b76 = finance_black76(105.12710963760241, 100, 1, 0.05, 0.2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("b76"), 10.450583572185565, 1e-6);

    expect_contains(interp, "finance_black76(105.12710963760241, 100, 1, 0.05, 0.2, 1)", "10.45");
}

TEST(ReplCommandsTest, finance_digital_option) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_digital_option(S,K,T,r,sigma,call,payout)");

    // Deep ITM call: price ~ payout * exp(-r*T) = exp(-0.05).
    expect_ok(interp, "d = finance_digital_option(200, 100, 1, 0.05, 0.2, 1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("d"), std::exp(-0.05), 0.01);

    expect_contains(interp, "finance_digital_option(200, 100, 1, 0.05, 0.2, 1, 1)", "0.9");
}

TEST(ReplCommandsTest, finance_heston) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_heston_call(S,K,T,r,v0,kappa,theta,sigma_v,rho)");
    expect_contains(interp, "help", "finance_heston_put(S,K,T,r,v0,kappa,theta,sigma_v,rho)");

    expect_contains(interp, "finance_heston_call(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, -0.5)", ".");
    expect_contains(interp, "finance_heston_put(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, -0.5)", ".");

    expect_error_contains(
        interp, "finance_heston_call(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, missing)",
        "finance_heston_call");
    expect_error_contains(
        interp, "finance_heston_put(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, missing)",
        "finance_heston_put");
}

TEST(ReplCommandsTest, finance_sabr) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_sabr_call(S,K,T,r,alpha,beta,rho,nu)");
    expect_contains(interp, "help", "finance_sabr_put");

    expect_ok(interp, "c = finance_sabr_call(100, 100, 1, 0.05, 0.2, 1, 0, 0.3)");
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("c")));
    expect_ok(interp, "p = finance_sabr_put(100, 100, 1, 0.05, 0.2, 1, 0, 0.3)");
    EXPECT_GT(interp.state().scalars.at("p"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("p")));
    expect_contains(interp, "finance_sabr_call(100, 100, 1, 0.05, 0.2, 1, 0, 0.3)", "\n");

    expect_error_contains(interp, "finance_sabr_call(100, 100, 1, 0.05, 0.2, 1, 0, missing)",
                         "finance_sabr_call");
}

TEST(ReplCommandsTest, finance_geo_asian) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_geo_asian_call(S,K,T,r,sigma,n_fixings)");
    expect_contains(interp, "help", "finance_geo_asian_put(S,K,T,r,sigma,n_fixings)");

    expect_ok(interp, "c = finance_geo_asian_call(100, 100, 1, 0.05, 0.2, 12)");
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("c")));
    expect_ok(interp, "p = finance_geo_asian_put(100, 100, 1, 0.05, 0.2, 12)");
    EXPECT_GT(interp.state().scalars.at("p"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("p")));
    expect_contains(interp, "finance_geo_asian_call(100, 100, 1, 0.05, 0.2, 12)", "\n");

    expect_error_contains(interp, "finance_geo_asian_call(100, 100, 1, 0.05, 0.2, 1.5)",
                         "n_fixings");
    expect_error_contains(interp, "finance_geo_asian_put(100, 100, 1, 0.05, 0.2, missing)",
                         "finance_geo_asian_put");
}

TEST(ReplCommandsTest, finance_american_option) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_american_option(S,K,T,r,sigma,call,steps)");

    // American call without dividends should be close to the European (BS) call price.
    expect_ok(interp, "am = finance_american_option(100, 100, 1, 0.05, 0.2, 1, 200)");
    EXPECT_NEAR(interp.state().scalars.at("am"), 10.450583572185565, 1.0);

    expect_contains(interp, "finance_american_option(100, 100, 1, 0.05, 0.2, 1, 200)", "\n");
}

TEST(ReplCommandsTest, finance_mc_european_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_european_call(S,K,T,r,sigma,n_paths,seed)");

    // Loose tolerance vs. the analytic BS call price: this checks REPL plumbing
    // (right function/args/result), not Monte Carlo statistical accuracy.
    expect_ok(interp, "mc = finance_mc_european_call(100, 100, 1, 0.05, 0.2, 20000, 7)");
    EXPECT_GT(interp.state().scalars.at("mc"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("mc"), 10.450583572185565, 3.0);

    expect_contains(interp, "finance_mc_european_call(100, 100, 1, 0.05, 0.2, 20000, 7)", "\n");
}

TEST(ReplCommandsTest, finance_mc_european_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_european_put(S,K,T,r,sigma,n_paths,seed)");

    // bs_put(100,100,1,0.05,0.2) via put-call parity: bs_call - S + K*exp(-r*T).
    const double bs_put_ref = 10.450583572185565 - 100.0 + 100.0 * std::exp(-0.05);
    expect_ok(interp, "mc = finance_mc_european_put(100, 100, 1, 0.05, 0.2, 20000, 7)");
    EXPECT_GT(interp.state().scalars.at("mc"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("mc"), bs_put_ref, 3.0);

    expect_contains(interp, "finance_mc_european_put(100, 100, 1, 0.05, 0.2, 20000, 7)", "\n");
}

TEST(ReplCommandsTest, finance_mc_asian_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_asian_call(S,K,T,r,sigma,n_paths,n_steps,seed)");

    // Arithmetic-average Asian call should be cheaper than the corresponding European call.
    expect_ok(interp, "ac = finance_mc_asian_call(100, 100, 1, 0.05, 0.2, 20000, 50, 3)");
    EXPECT_GT(interp.state().scalars.at("ac"), 0.0);
    EXPECT_LT(interp.state().scalars.at("ac"), 10.450583572185565);

    expect_contains(interp, "finance_mc_asian_call(100, 100, 1, 0.05, 0.2, 20000, 50, 3)", "\n");
}

TEST(ReplCommandsTest, finance_mc_asian_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_mc_asian_put(S,K,T,r,sigma,n_paths,n_steps,seed)");

    expect_ok(interp, "ap = finance_mc_asian_put(90, 110, 1, 0.05, 0.25, 20000, 50, 3)");
    EXPECT_GT(interp.state().scalars.at("ap"), 0.0);

    expect_contains(interp, "finance_mc_asian_put(90, 110, 1, 0.05, 0.25, 20000, 50, 3)", "\n");
}

TEST(ReplCommandsTest, finance_barrier_option) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_barrier_option(S,K,B,T,r,sigma,call,knock_in,up)");

    // Down-and-in + down-and-out call must reconstruct the vanilla call price.
    expect_ok(interp, "ki = finance_barrier_option(100, 100, 90, 1, 0.05, 0.2, 1, 1, 0)");
    expect_ok(interp, "ko = finance_barrier_option(100, 100, 90, 1, 0.05, 0.2, 1, 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("ki") + interp.state().scalars.at("ko"),
                10.450583572185565, 1e-4);

    // This also exercises the new 9-arg (nonary) regex literal-dispatch path added
    // specifically for finance_barrier_option.
    expect_contains(interp, "finance_barrier_option(100, 100, 90, 1, 0.05, 0.2, 1, 1, 0)", "\n");
}

TEST(ReplCommandsTest, finance_mc_lookback_floating_call) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_floating_call(S,T,r,sigma,n_paths,n_steps,seed)");

    // Compare REPL plumbing against the same direct library call with a fixed seed.
    const double ref = ms::finance::mc_lookback_floating_call(100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_floating_call(100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_floating_call(100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, finance_mc_lookback_floating_put) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_floating_put(S,T,r,sigma,n_paths,n_steps,seed)");

    const double ref = ms::finance::mc_lookback_floating_put(100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_floating_put(100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_floating_put(100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, finance_mc_lookback_fixed_call) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_fixed_call(S,K,T,r,sigma,n_paths,n_steps,seed)");

    const double ref =
        ms::finance::mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, finance_mc_lookback_fixed_put) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_mc_lookback_fixed_put(S,K,T,r,sigma,n_paths,n_steps,seed)");

    const double ref =
        ms::finance::mc_lookback_fixed_put(100, 100, 1, 0.05, 0.2, 2000, 50, 7);
    expect_ok(interp, "y = finance_mc_lookback_fixed_put(100, 100, 1, 0.05, 0.2, 2000, 50, 7)");
    EXPECT_NEAR(interp.state().scalars.at("y"), ref, 1e-9);
    EXPECT_GT(ref, 0.0);

    expect_contains(interp, "finance_mc_lookback_fixed_put(100, 100, 1, 0.05, 0.2, 2000, 50, 7)",
                    "\n");
}

TEST(ReplCommandsTest, finance_bachelier_call) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bachelier_call(F,K,T,r,sigma)");

    // ATM closed form: e^{-rT} * sigma*sqrt(T) / sqrt(2*pi) at F=K=100, r=0.04, sigma=0.25, T=1.5
    expect_ok(interp, "bc = finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("bc"), 0.11503712918258642, 1e-6);

    expect_contains(interp, "finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)", "0.115");
}

TEST(ReplCommandsTest, finance_bachelier_put) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bachelier_put(F,K,T,r,sigma)");

    expect_ok(interp, "bp = finance_bachelier_put(100, 100, 1.5, 0.04, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 0.11503712918258642, 1e-6);

    expect_contains(interp, "finance_bachelier_put(100, 100, 1.5, 0.04, 0.25)", "0.115");
}

TEST(ReplCommandsTest, finance_vasicek_bond_price) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_vasicek_bond_price(r,a,b,sigma,tau)");

    expect_ok(interp, "vz = finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("vz"), 0.9512737476480422, 1e-6);

    expect_contains(interp, "finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, 1.0)", "0.951");
}

TEST(ReplCommandsTest, finance_cir_bond_price) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_cir_bond_price(r,a,b,sigma,tau)");

    expect_ok(interp, "cr = finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), 0.951243269844748, 1e-6);

    expect_contains(interp, "finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, 1.0)", "0.951");
}

TEST(ReplCommandsTest, finance_trinomial_option) {
    Interpreter interp;
    expect_contains(interp, "help",
                    "finance_trinomial_option(S,K,T,r,sigma,n_steps,is_call,is_american)");

    // European call should be close to the analytic BS call price.
    expect_ok(interp, "tri = finance_trinomial_option(100, 100, 1, 0.05, 0.2, 200, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("tri"), 10.450583572185565, 0.5);

    expect_contains(interp, "finance_trinomial_option(100, 100, 1, 0.05, 0.2, 200, 1, 0)", "\n");
}

TEST(ReplCommandsTest, finance_bl_portfolio) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_bl_implied_returns(cov,w_mkt,delta)");
    expect_contains(interp, "help", "finance_bl_posterior_returns(pi,cov,P,Q,tau)");

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pi_bl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pi_bl").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(1, 0), 0.035, 1e-6);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("post").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("post")(0, 0), 0.075, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("post")(1, 0), 0.07625, 1e-6);
}

TEST(ReplCommandsTest, finance_merton_historical) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_merton_distance_to_default(V,sigma_v,D,r,T)");
    expect_contains(interp, "help", "finance_historical_var(returns,confidence)");
    expect_contains(interp, "help", "finance_historical_cvar(returns,confidence)");

    expect_ok(interp, "dd = finance_merton_distance_to_default(150, 0.20, 100, 0.05, 1)");
    EXPECT_NEAR(interp.state().scalars.at("dd"), 2.177325543255, 1e-6);
    expect_contains(interp, "finance_merton_distance_to_default(150, 0.20, 100, 0.05, 1)", "2.177");

    expect_ok(interp, "ret = [-0.20; -0.15; -0.10; -0.05; 0.0; 0.05; 0.10; 0.15; 0.20; 0.25]");
    expect_ok(interp, "hv = finance_historical_var(ret, 0.95)");
    EXPECT_NEAR(interp.state().scalars.at("hv"), 0.20, 1e-6);
    expect_contains(interp, "finance_historical_var(ret, 0.95)", "0.2");

    expect_ok(interp, "hc = finance_historical_cvar(ret, 0.95)");
    EXPECT_NEAR(interp.state().scalars.at("hc"), 0.20, 1e-6);
    expect_contains(interp, "finance_historical_cvar(ret, 0.95)", "0.2");
}

TEST(ReplCommandsTest, finance_ratio_metrics) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_treynor(returns,risk_free,beta)");
    expect_contains(interp, "help", "finance_information_ratio(returns,benchmark)");

    expect_ok(interp, "ret = [0.10; 0.12; 0.08; 0.11; 0.09]");
    expect_ok(interp, "bench = [0.05; 0.08; 0.06; 0.10; 0.07]");

    expect_ok(interp, "tr = finance_treynor(ret, 0.05, 1.2)");
    EXPECT_NEAR(interp.state().scalars.at("tr"), 0.05 / 1.2, 1e-10);
    expect_contains(interp, "finance_treynor(ret, 0.05, 1.2)", "0.04166");

    expect_ok(interp, "ir = finance_information_ratio(ret, bench)");
    EXPECT_NEAR(interp.state().scalars.at("ir"), 1.704, 0.01);
    expect_contains(interp, "finance_information_ratio(ret, bench)", "1.70");
}

TEST(ReplCommandsTest, finance_merton_bl) {
    Interpreter interp;
    expect_contains(interp, "help", "finance_merton_implied_asset_params(E,sigma_E,D,r,T)");
    expect_contains(interp, "help", "finance_bl_posterior_returns_default_omega(pi,cov,P,Q,tau)");

    const double V_true = 180.0;
    const double sigma_V_true = 0.22;
    const double D = 100.0;
    const double r = 0.05;
    const double T = 1.0;
    const double E = ms::finance::bs_call(V_true, D, T, r, sigma_V_true);
    const double d1 = (std::log(V_true / D) + (r + 0.5 * sigma_V_true * sigma_V_true) * T) /
                      (sigma_V_true * std::sqrt(T));
    const double nd1 = 0.5 * std::erfc(-d1 / std::sqrt(2.0));
    const double sigma_E = nd1 * sigma_V_true * V_true / E;
    const ms::finance::MertonResult merton_ref =
        ms::finance::merton_implied_asset_params(E, sigma_E, D, r, T);
    ASSERT_TRUE(merton_ref.converged);

    std::ostringstream merton_cmd;
    merton_cmd << "merton = finance_merton_implied_asset_params(" << E << ", " << sigma_E << ", "
               << D << ", " << r << ", " << T << ")";
    expect_ok(interp, merton_cmd.str());
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);
    EXPECT_NEAR(interp.state().matrices.at("merton")(0, 0), merton_ref.distance_to_default, 1e-4);
    EXPECT_NEAR(interp.state().matrices.at("merton")(0, 1), merton_ref.probability_of_default,
                1e-4);
    EXPECT_NEAR(interp.state().matrices.at("merton")(0, 2), merton_ref.implied_asset_value, 1e-4);
    EXPECT_NEAR(interp.state().matrices.at("merton")(0, 3), merton_ref.implied_asset_volatility,
                1e-4);
    EXPECT_NEAR(interp.state().matrices.at("merton")(0, 4), 1.0, 1e-12);
    EXPECT_GT(interp.state().matrices.at("merton")(0, 5), 0.0);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("post").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("post")(0, 0), 0.075, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("post")(1, 0), 0.07625, 1e-6);
}

TEST(ReplCommandsTest, finance) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);

    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);

    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-6);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, quantum_mpc_gbm_backtest) {
    Interpreter interp;

    expect_ok(interp, "psi = [0.5; 0.5; 0.5; 0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_EQ(interp.state().matrices.at("sh").cols(), 3u);

    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100; 101; 102; 101]");
    expect_ok(interp, "pos = [1; 1; 1; 0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, log_minvar) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, frontier_sharpe) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, bl_implied_post) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, merton_blomega) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, gbm_backtest) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, log_minvar_2) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, frontier_sharpe_2) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, bl_implied_post_2) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, merton_blomega_2) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, gbm_backtest_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_2) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_2) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_2) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_2) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_3) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_3) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_3) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_3) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_3) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_4) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_4) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_4) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_4) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_4) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_5) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_5) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_5) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_5) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_5) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_6) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_6) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_6) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_6) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_6) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_7) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_7) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_7) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_7) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_7) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_8) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_8) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_8) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_8) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_8) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_9) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_9) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_9) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_9) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_9) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_10) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_10) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_10) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_10) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_10) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_11) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_11) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_11) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_11) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_11) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_12) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_12) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_12) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_12) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_12) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_13) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_13) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_13) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_13) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_13) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_14) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_14) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_14) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_14) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_14) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_15) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_15) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_15) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_15) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_15) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_16) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_16) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_16) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_16) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_16) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_17) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_17) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_17) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_17) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_17) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_18) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_18) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_18) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_18) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_18) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_19) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_19) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_19) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_19) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_19) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_20) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_20) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_20) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_20) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_20) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_21) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_21) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_21) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_21) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_21) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_22) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_22) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_22) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_22) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_22) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_23) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_23) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_23) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_23) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_23) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_24) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_24) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_24) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_24) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_24) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_25) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_25) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_25) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_25) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_25) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_26) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_26) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_26) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_26) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_26) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_27) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_27) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_27) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_27) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_27) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_28) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_28) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_28) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_28) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_28) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_29) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_29) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_29) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_29) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_29) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, laplacian_of_gaussian_finance_min_variance_portfolio_30) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    ASSERT_GT(interp.state().matrices.count("w_min"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0,
                1e-8);
}

TEST(ReplCommandsTest, finance_efficient_frontier_finance_max_sharpe_30) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "mu2 = [0.07; 0.035]");
    expect_ok(interp, "w_ef = finance_efficient_frontier(cov2, mu2, 0.07)");
    ASSERT_GT(interp.state().matrices.count("w_ef"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w_ef").rows(), 2u);

    expect_ok(interp, "w_ms = finance_max_sharpe(cov2, mu2, 0.02)");
    ASSERT_GT(interp.state().matrices.count("w_ms"), 0u);
}

TEST(ReplCommandsTest, finance_bl_implied_returns_finance_bl_posterior_returns_30) {
    Interpreter interp;

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_mkt = [0.6; 0.4]");
    expect_ok(interp, "pi_bl = finance_bl_implied_returns(cov2, w_mkt, 2.5)");
    ASSERT_GT(interp.state().matrices.count("pi_bl"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pi_bl")(0, 0), 0.07, 1e-8);

    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
    EXPECT_EQ(interp.state().matrices.at("post").rows(), 2u);
}

TEST(ReplCommandsTest, finance_merton_implied_asset_params_finance_bl_posterior_returns_default_omega_30) {
    Interpreter interp;

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    ASSERT_GT(interp.state().matrices.count("merton"), 0u);
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);
}

TEST(ReplCommandsTest, simulate_gbm_path_run_backtest_30) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    ASSERT_GT(interp.state().matrices.count("path"), 0u);
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100;101;102;101]");
    expect_ok(interp, "pos = [1;1;1;0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bt").cols(), 4u);
}

TEST(ReplCommandsTest, finance_binomial_call_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "finance_binomial_call(100, 100, 1, 0.05, 0.2, 50)", ".");
    expect_error_contains(interp, "finance_binomial_call(100, 100, 1, 0.05, 0.2, missing)",
                          "finance_binomial_call");
}

TEST(ReplCommandsTest, finance_binomial_put_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "finance_binomial_put(100, 100, 1, 0.05, 0.2, 50)", ".");
    expect_error_contains(interp, "finance_binomial_put(100, 100, 1, 0.05, 0.2, 1.5)",
                          "non-negative integer steps");
}

TEST(ReplCommandsTest, finance_bs_rho_execute_no_assign) {
    Interpreter interp;
    expect_contains(interp, "finance_bs_rho(100, 100, 1, 0.05, 0.2, 1)", "53.");
    expect_error_contains(interp, "finance_bs_rho(100, 100, 1, 0.05, 0.2, 0.5)",
                          "integer call");
}

TEST(ReplCommandsTest, finance_mc_lookback_fixed_call_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "finance_mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 80, 10, 7)");
    expect_error_contains(
        interp, "finance_mc_lookback_fixed_call(100, 100, 1, 0.05, 0.2, 1.5, 10, 7)",
        "non-negative integer n_paths");
}

TEST(ReplCommandsTest, finance_heston_call_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "finance_heston_call(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, -0.5)");
    expect_error(interp, "finance_heston_call(100, 100)");
    expect_error_contains(
        interp, "finance_heston_call(100, 100, 1, 0.05, 0.04, 2, 0.04, 0.3, missing)",
        "finance_heston_call");
}

TEST(ReplCommandsTest, finance_min_variance_portfolio_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_min_variance_portfolio([2, 0.1; 0.1, 1])",
                    "min_variance_portfolio");
    expect_error_contains(interp, "finance_min_variance_portfolio([1, 2])",
                          "square matrix");
}

TEST(ReplCommandsTest, izaac_gaussian_noise_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "izaac_gaussian_noise(5, 1, 1e-5, 1)");
    expect_error_contains(interp, "izaac_gaussian_noise(5, 1, 1e-5)",
                          "izaac_gaussian_noise");
}

TEST(ReplCommandsTest, finance_sortino_noassign) {
    Interpreter interp;
    expect_ok(interp, "finance_sortino([1; 2; 3])");
    expect_error_contains(interp, "finance_sortino([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, finance_irr_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_irr([-100, 110])", "0.1");
    expect_error_contains(interp, "finance_irr(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, finance_sharpe_noassign) {
    Interpreter interp;
    expect_ok(interp, "finance_sharpe([0.1; 0.2; -0.05])");
    expect_error_contains(interp, "finance_sharpe(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, finance_max_drawdown_noassign) {
    Interpreter interp;
    expect_ok(interp, "finance_max_drawdown([1; 1.1; 0.9; 1.2])");
    expect_error_contains(interp, "finance_max_drawdown(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, finance_var_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_var([0.1; -0.05; 0.2])", "0.");
    expect_error_contains(interp, "finance_var(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, finance_cvar_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_cvar([0.1; -0.05; 0.2])", "0.");
    expect_error_contains(interp, "finance_cvar(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, finance_historical_var_noassign) {
    Interpreter interp;
    expect_ok(interp, "ret = [-0.20; -0.15; -0.10; -0.05; 0.0; 0.05; 0.10; 0.15; 0.20; 0.25]");
    expect_contains(interp, "finance_historical_var(ret, 0.95)", "0.2");
    expect_error_contains(interp, "finance_historical_var(no_such_matrix, 0.95)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, finance_historical_cvar_noassign) {
    Interpreter interp;
    expect_ok(interp, "ret = [-0.20; -0.15; -0.10; -0.05; 0.0; 0.05; 0.10; 0.15; 0.20; 0.25]");
    expect_contains(interp, "finance_historical_cvar(ret, 0.95)", "0.2");
    expect_error_contains(interp, "finance_historical_cvar(no_such_matrix, 0.95)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, finance_bachelier_put_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bachelier_put(100, 100, 1.5, 0.04, 0.25)", "0.115");
    expect_error_contains(interp, "finance_bachelier_put(100, 100, 1.5, 0.04, missing)",
                          "finance_bachelier_put");
}

TEST(ReplCommandsTest, finance_vasicek_bond_price_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, 1.0)", "0.951");
    expect_error_contains(interp, "finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, missing)",
                          "finance_vasicek_bond_price");
}

TEST(ReplCommandsTest, finance_cir_bond_price_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, 1.0)", "0.951");
    expect_error_contains(interp, "finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, missing)",
                          "finance_cir_bond_price");
}

TEST(ReplCommandsTest, finance_bs_put_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bs_put(100, 100, 1, 0.05, 0.2)", "5.57");
    expect_error_contains(interp, "finance_bs_put(100, 100, 1, 0.05, missing)", "finance_bs_put");
}

TEST(ReplCommandsTest, finance_bs_gamma_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bs_gamma(100, 100, 1, 0.05, 0.2)", "0.018762");
    expect_error_contains(interp, "finance_bs_gamma(100, 100, 1, 0.05, missing)",
                          "finance_bs_gamma");
}

TEST(ReplCommandsTest, finance_bs_vega_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bs_vega(100, 100, 1, 0.05, 0.2)", "37.524");
    expect_error_contains(interp, "finance_bs_vega(100, 100, 1, 0.05, missing)",
                          "finance_bs_vega");
}

TEST(ReplCommandsTest, finance_bachelier_call_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)", "0.115");
    expect_error_contains(interp, "finance_bachelier_call(100, 100, 1.5, 0.04, missing)",
                          "finance_bachelier_call");
}

TEST(ReplCommandsTest, finance_sabr_put_noassign) {
    Interpreter interp;
    expect_ok(interp, "finance_sabr_put(100, 100, 1, 0.05, 0.2, 1, 0, 0.3)");
    expect_error_contains(interp, "finance_sabr_put(100, 100, 1, 0.05, 0.2, 1, 0, missing)",
                          "finance_sabr_put");
}

TEST(ReplCommandsTest, finance_geo_asian_put_noassign) {
    Interpreter interp;
    expect_ok(interp, "finance_geo_asian_put(100, 100, 1, 0.05, 0.2, 12)");
    expect_error_contains(interp, "finance_geo_asian_put(100, 100, 1, 0.05, 0.2, missing)",
                          "finance_geo_asian_put");
}

TEST(ReplCommandsTest, finance_portfolio_return_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_portfolio_return([0.6; 0.4], [0.1; 0.05])", "0.08");
    expect_error_contains(interp, "finance_portfolio_return(no_such_matrix, [0.1; 0.05])",
                          "unknown matrix");
}

TEST(ReplCommandsTest, finance_portfolio_variance_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_portfolio_variance([0.5; 0.5], [0.04, 0.01; 0.01, 0.09])",
                    "0.0375");
    expect_error_contains(interp,
                          "finance_portfolio_variance(no_such_matrix, [0.04, 0.01; 0.01, 0.09])",
                          "unknown matrix");
}

TEST(ReplCommandsTest, finance_information_ratio_noassign) {
    Interpreter interp;
    expect_ok(interp, "ret = [0.10; 0.12; 0.08; 0.11; 0.09]");
    expect_ok(interp, "bench = [0.05; 0.08; 0.06; 0.10; 0.07]");
    expect_contains(interp, "finance_information_ratio(ret, bench)", "1.70");
    expect_error_contains(interp, "finance_information_ratio(missing, bench)", "unknown matrix");
}

TEST(ReplCommandsTest, finance_max_sharpe_portfolio_noassign) {
    Interpreter interp;
    expect_ok(interp, "cov3 = [0.10, 0.02, 0.01; 0.02, 0.08, 0.03; 0.01, 0.03, 0.06]");
    expect_ok(interp, "mu3 = [0.08; 0.12; 0.10]");
    expect_contains(interp, "finance_max_sharpe_portfolio(cov3, mu3, 0.02)",
                    "max_sharpe_portfolio =");
    expect_error_contains(interp, "finance_max_sharpe_portfolio([1, 2], mu3, 0.02)",
                          "expected square matrix");
}

TEST(ReplCommandsTest, finance_bs_call_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bs_call(100, 100, 1, 0.05, 0.2)", "10.45");
    expect_error_contains(interp, "finance_bs_call(100, 100, 1, 0.05, missing)",
                          "finance_bs_call");
}

TEST(ReplCommandsTest, finance_bond_price_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bond_price(0.05, 0.05, 10, 100)", "100");
    expect_error_contains(interp, "finance_bond_price(0.05, 0.05, missing)",
                          "finance_bond_price");
}

TEST(ReplCommandsTest, finance_compound_4arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_compound(100, 0.1, 1, 4)", "110.38");
    expect_error_contains(interp, "finance_compound(100, 0.1, 1, missing)",
                          "expected finance_compound(principal,rate,n_periods,compounds_per_period)");
    expect_error_contains(interp, "finance_compound(100, 0.1, 1.5, 4)",
                          "expected non-negative integer periods n_periods");
    expect_error_contains(interp, "finance_compound(100, 0.1, 1, 0)",
                          "expected positive integer compounds_per_period");
}

TEST(ReplCommandsTest, finance_compound_3arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_compound(100, 0.1, 3)", "133.1");
    expect_error_contains(interp, "finance_compound(100, 0.1, missing)",
                          "expected finance_compound(principal,rate,n_periods)");
    expect_error_contains(interp, "finance_compound(100, 0.1, 1.5)",
                          "expected non-negative integer periods n_periods");
}

TEST(ReplCommandsTest, finance_pv_4arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_pv(0, 5, -10, 0)", "50");
    expect_error_contains(interp, "finance_pv(0, 5, -10, missing)",
                          "expected finance_pv(rate,n,pmt,fv)");
    expect_error_contains(interp, "finance_pv(0, 1.5, -10, 0)",
                          "expected non-negative integer n");
}

TEST(ReplCommandsTest, finance_pv_3arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_pv(0, 5, -10)", "50");
    expect_error_contains(interp, "finance_pv(0, 5, missing)",
                          "expected finance_pv(rate,n,pmt)");
    expect_error_contains(interp, "finance_pv(0, 1.5, -10)",
                          "expected non-negative integer n");
}

TEST(ReplCommandsTest, finance_fv_annuity_4arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_fv_annuity(0, 5, -10, 0)", "50");
    expect_error_contains(interp, "finance_fv_annuity(0, 5, -10, missing)",
                          "expected finance_fv_annuity(rate,n,pmt,pv0)");
    expect_error_contains(interp, "finance_fv_annuity(0, 1.5, -10, 0)",
                          "expected non-negative integer n");
}

TEST(ReplCommandsTest, finance_fv_annuity_3arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_fv_annuity(0, 5, -10)", "50");
    expect_error_contains(interp, "finance_fv_annuity(0, 5, missing)",
                          "expected finance_fv_annuity(rate,n,pmt)");
    expect_error_contains(interp, "finance_fv_annuity(0, 1.5, -10)",
                          "expected non-negative integer n");
}

TEST(ReplCommandsTest, finance_pmt_annuity_4arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_pmt_annuity(0, 5, -50, 0)", "10");
    expect_error_contains(interp, "finance_pmt_annuity(0, 5, -50, missing)",
                          "expected finance_pmt_annuity(rate,n,pv0,fv)");
    expect_error_contains(interp, "finance_pmt_annuity(0, 1.5, -50, 0)",
                          "expected non-negative integer n");
}

TEST(ReplCommandsTest, finance_pmt_annuity_3arg_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_pmt_annuity(0, 5, -50)", "10");
    expect_error_contains(interp, "finance_pmt_annuity(0, 5, missing)",
                          "expected finance_pmt_annuity(rate,n,pv0)");
    expect_error_contains(interp, "finance_pmt_annuity(0, 1.5, -50)",
                          "expected non-negative integer n");
}

TEST(ReplCommandsTest, finance_bond_convexity_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bond_convexity(0, 0.05, 5)", "27.210884");
    expect_error_contains(interp, "finance_bond_convexity(0, 0.05, missing)",
                          "expected finance_bond_convexity(c,y,n)");
    expect_error_contains(interp, "finance_bond_convexity(0, 0.05, 1.5)",
                          "expected non-negative integer periods n");
}

TEST(ReplCommandsTest, finance_bond_ytm_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bond_ytm(85.952837, 0.05, 10)", "0.07");
    expect_error_contains(interp, "finance_bond_ytm(85.952837, 0.05, missing)",
                          "expected finance_bond_ytm(price,c,n)");
    expect_error_contains(interp, "finance_bond_ytm(85.952837, 0.05, 1.5)",
                          "expected non-negative integer periods n");
}

TEST(ReplCommandsTest, finance_bond_modified_duration_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_bond_modified_duration(0, 0.05, 5)", "4.761905");
    expect_error_contains(interp, "finance_bond_modified_duration(0, 0.05, missing)",
                          "expected finance_bond_modified_duration(c,y,n)");
    expect_error_contains(interp, "finance_bond_modified_duration(0, 0.05, 1.5)",
                          "expected non-negative integer periods n");
}

TEST(ReplCommandsTest, finance_continuous_compound_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_continuous_compound(100, 0.1, 1)", "110.517");
    expect_error_contains(interp, "finance_continuous_compound(100, 0.1, missing)",
                          "expected finance_continuous_compound(principal,rate,t)");
}

TEST(ReplCommandsTest, finance_capm_noassign) {
    Interpreter interp;
    expect_contains(interp, "finance_capm(0.05, 1.2, 0.10)", "0.11");
    expect_error_contains(interp, "finance_capm(0.05, 1.2, missing)",
                          "expected finance_capm(risk_free,beta,market_return)");
}
