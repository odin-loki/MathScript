#pragma once
#include "ms/error/error_types.hpp"
#include <span>
#include <vector>

namespace ms {
namespace finance {

// --- Black-Scholes option pricing ---
// S=spot, K=strike, T=time_to_expiry(years), r=risk_free_rate, sigma=volatility
double bs_call(double S, double K, double T, double r, double sigma);
double bs_put(double S, double K, double T, double r, double sigma);

// Greeks
double bs_delta(double S, double K, double T, double r, double sigma, bool call);
double bs_gamma(double S, double K, double T, double r, double sigma);
double bs_vega(double S, double K, double T, double r, double sigma);
double bs_theta(double S, double K, double T, double r, double sigma, bool call);
double bs_rho(double S, double K, double T, double r, double sigma, bool call);

// Implied volatility (Newton-Raphson)
Result<double> bs_implied_vol(double price, double S, double K, double T,
                               double r, bool call);

// --- Bond pricing ---
// c=coupon rate (annual), y=yield, n=periods, fv=face value
double bond_price(double c, double y, int n, double fv = 100.0);
double bond_duration(double c, double y, int n, double fv = 100.0);
double bond_modified_duration(double c, double y, int n, double fv = 100.0);
double bond_convexity(double c, double y, int n, double fv = 100.0);
// Yield to maturity (bisection)
Result<double> bond_ytm(double price, double c, int n, double fv = 100.0);

// --- Time value of money ---
double npv(double rate, std::span<const double> cashflows);
Result<double> irr(std::span<const double> cashflows,
                   double guess = 0.1, int max_iter = 100);
double pv(double rate, int n, double pmt, double fv = 0.0);
double fv_annuity(double rate, int n, double pmt, double pv0 = 0.0);
double pmt_annuity(double rate, int n, double pv0, double fv = 0.0);

// --- Risk metrics ---
// Returns sorted losses (positive = loss). VaR at confidence level alpha.
double var(std::span<const double> returns, double alpha = 0.95);
double cvar(std::span<const double> returns, double alpha = 0.95); // ES

// Sharpe ratio: (mean_return - risk_free) / std_dev
double sharpe_ratio(std::span<const double> returns, double risk_free = 0.0);
double sortino_ratio(std::span<const double> returns, double risk_free = 0.0);

// Information ratio: mean(returns - benchmark) / stddev(returns - benchmark)
double information_ratio(std::span<const double> returns,
                         std::span<const double> benchmark);

// Treynor ratio: (mean(returns) - risk_free) / beta
double treynor_ratio(std::span<const double> returns, double risk_free, double beta);

// CAPM expected return: risk_free + beta * (market_return - risk_free)
double capm(double risk_free, double beta, double market_return);

// Max drawdown from equity curve
double max_drawdown(std::span<const double> equity_curve);

// --- Kelly criterion ---
double kelly_fraction(double win_prob, double win_loss_ratio);

// --- Compound interest ---
double compound(double principal, double rate, int n_periods, int compounds_per_period = 1);
double continuous_compound(double principal, double rate, double t);

// --- Portfolio metrics ---
// weights, cov matrix (flattened row-major), returns vector
double portfolio_variance(std::span<const double> weights,
                          std::span<const double> cov_matrix);
double portfolio_return(std::span<const double> weights,
                        std::span<const double> asset_returns);

// Simple-compounding implied forward rate between t1 and t2 given zero rates
// r1 (to t1) and r2 (to t2): (r2*t2 - r1*t1) / (t2 - t1)
double forward_rate(double r1, double t1, double r2, double t2);

// Cash-or-nothing digital option: pays payout if in-the-money at expiry
double digital_option(double S, double K, double T, double r, double sigma,
                      bool call, double payout = 1.0);

// Black-76 model for options on forwards/futures (F is the forward price)
double black76(double F, double K, double T, double r, double sigma, bool call);

// European barrier option (continuous monitoring, Haug closed-form).
// All 8 single-barrier types via call/put, knock_in/knock_out, up/down flags.
// Down barriers require S > B; up barriers require S < B.
double barrier_option(double S, double K, double B, double T, double r, double sigma,
                      bool call, bool knock_in, bool up);

// --- Pricing: binomial tree ---
double binomial_call(double S, double K, double T, double r, double sigma, int steps);
double binomial_put(double S, double K, double T, double r, double sigma, int steps);

// American option via binomial tree with early exercise
double american_option(double S, double K, double T, double r, double sigma,
                       bool call, int steps);

} // namespace finance
} // namespace ms
