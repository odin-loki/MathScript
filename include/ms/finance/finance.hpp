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

// --- Markowitz mean-variance portfolio optimization ---
// cov_matrix is flattened row-major n x n, same convention as portfolio_variance's
// cov_matrix. All three solve a dense linear system against cov_matrix internally
// and report a singular/near-singular matrix via ms::Result rather than NaN/crash.

// Global minimum-variance portfolio: the weights minimizing w^T*cov_matrix*w
// subject to sum(w) == 1, given by w = Sigma^-1*1 / (1^T*Sigma^-1*1).
Result<std::vector<double>> min_variance_portfolio(std::span<const double> cov_matrix, int n);

// Efficient-frontier portfolio: minimizes variance subject to sum(w) == 1 and
// w^T*mu == target_return (two-fund theorem via Lagrange multipliers).
Result<std::vector<double>> efficient_frontier_portfolio(std::span<const double> cov_matrix,
                                                          std::span<const double> mu,
                                                          double target_return, int n);

// Maximum Sharpe ratio (tangency) portfolio for risk-free rate risk_free, given by
// w = Sigma^-1*(mu - risk_free*1) / (1^T*Sigma^-1*(mu - risk_free*1)).
Result<std::vector<double>> max_sharpe_portfolio(std::span<const double> cov_matrix,
                                                 std::span<const double> mu,
                                                 double risk_free, int n);

// --- Black-Litterman model ---
// Combines a market-equilibrium prior with investor "views" to produce a
// blended (posterior) expected-return vector, for use with the mean-variance
// optimizers above (e.g. plug the posterior into efficient_frontier_portfolio
// or max_sharpe_portfolio as `mu`). cov_matrix/P/omega are flattened row-major,
// same convention as cov_matrix elsewhere in this file (cov_matrix is n x n,
// P is k x n, omega is k x k).

// Reverse-optimization: implied equilibrium excess returns Pi = delta*Sigma*w_mkt,
// where delta is the market risk-aversion coefficient and w_mkt is the vector of
// market-cap weights (size n).
Result<std::vector<double>> bl_implied_returns(std::span<const double> cov_matrix,
                                               std::span<const double> w_mkt,
                                               double delta, int n);

// Black-Litterman posterior expected returns:
//   E[R] = [(tau*Sigma)^-1 + P^T*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P^T*Omega^-1*Q]
// pi is the prior (size n, typically from bl_implied_returns), tau scales the
// uncertainty of the prior (typical 0.025-0.05), P is the k x n view "picking"
// matrix, Q is the k view returns, and omega is the k x k view-uncertainty
// covariance matrix. k=0 (empty P/Q/omega) is a valid "no views" case and
// returns pi unchanged, since the P^T*Omega^-1*(...) terms vanish and the
// formula collapses to the prior exactly.
Result<std::vector<double>> bl_posterior_returns(std::span<const double> pi,
                                                 std::span<const double> cov_matrix,
                                                 double tau,
                                                 std::span<const double> P,
                                                 std::span<const double> Q,
                                                 std::span<const double> omega,
                                                 int n, int k);

// Convenience overload: derives Omega the standard diagonal way (omega_ii =
// tau * P_i*Sigma*P_i^T, off-diagonals zero, i.e. views are assumed independent
// of each other) rather than requiring the caller to supply Omega directly.
Result<std::vector<double>> bl_posterior_returns_default_omega(
    std::span<const double> pi, std::span<const double> cov_matrix, double tau,
    std::span<const double> P, std::span<const double> Q, int n, int k);

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

// --- Pricing: Monte Carlo ---
// European option via Monte Carlo under GBM, sampling the terminal price
// directly from its known closed-form lognormal distribution (no path needed):
// S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z ~ N(0,1).
// Uses antithetic variates (each Z paired with -Z) to cut simulation variance.
// `seed` is explicit (not defaulted to a random device) so results are reproducible.
double mc_european_call(double S, double K, double T, double r, double sigma,
                        int n_paths, unsigned seed = 42);
double mc_european_put(double S, double K, double T, double r, double sigma,
                       int n_paths, unsigned seed = 42);

// Arithmetic-average Asian option via Monte Carlo: simulates a full discretized
// GBM path over n_steps of size dt=T/n_steps, averages the n_steps post-initial
// prices, and prices max(avg-K,0) (call) / max(K-avg,0) (put) discounted at r.
// Cheaper than the corresponding European option since averaging reduces the
// effective volatility of the payoff-determining quantity.
double mc_asian_call(double S, double K, double T, double r, double sigma,
                     int n_paths, int n_steps, unsigned seed = 42);
double mc_asian_put(double S, double K, double T, double r, double sigma,
                    int n_paths, int n_steps, unsigned seed = 42);

// Lookback options via Monte Carlo: simulates a full discretized GBM path over
// n_steps of size dt=T/n_steps (same convention as mc_asian_call/mc_asian_put)
// and tracks the running min/max over the path INCLUDING the initial price
// S_0=S, since the starting point can itself be the path extreme.
//
// Floating-strike: the strike is the path's own realized optimum, so the
// payoff is S_T-min(path) (call) / max(path)-S_T (put). Both are always
// non-negative pathwise (min(path)<=S_T<=max(path)), so these are strictly
// more valuable than the corresponding fixed-strike/vanilla option.
double mc_lookback_floating_call(double S, double T, double r, double sigma,
                                 int n_paths, int n_steps, unsigned seed = 42);
double mc_lookback_floating_put(double S, double T, double r, double sigma,
                                int n_paths, int n_steps, unsigned seed = 42);

// Fixed-strike: strike K is set upfront, but the payoff uses the path's
// extreme rather than the terminal price: max(max(path)-K,0) (call) /
// max(K-min(path),0) (put).
double mc_lookback_fixed_call(double S, double K, double T, double r, double sigma,
                              int n_paths, int n_steps, unsigned seed = 42);
double mc_lookback_fixed_put(double S, double K, double T, double r, double sigma,
                             int n_paths, int n_steps, unsigned seed = 42);

} // namespace finance
} // namespace ms
