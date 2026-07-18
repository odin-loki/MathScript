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

// Heston (1993) stochastic-volatility European call via Lewis/Fourier inversion.
// S=spot, K=strike, T=time_to_expiry(years), r=risk_free_rate, v0=initial
// variance, kappa=mean-reversion speed, theta=long-run variance, sigma_v=vol-of-vol,
// rho=spot-vol correlation. As sigma_v -> 0 with v0 ~= theta ~= sigma^2, the price
// converges to bs_call(S, K, T, r, sigma).
double heston_call(double S, double K, double T, double r, double v0, double kappa,
                   double theta, double sigma_v, double rho);

// Heston (1993) stochastic-volatility European put via put-call parity (q=0):
// P = C - S + K*e^{-rT}, using the same Lewis/Fourier call as heston_call.
double heston_put(double S, double K, double T, double r, double v0, double kappa,
                  double theta, double sigma_v, double rho);

// SABR (Hagan et al. 2002) European call via asymptotic Black implied volatility.
// S=spot, K=strike, T=time_to_expiry(years), r=risk_free_rate, alpha=initial vol,
// beta=CEV exponent in [0,1], rho=spot-vol correlation, nu=vol-of-vol. beta=1 gives
// lognormal-type dynamics; nu->0 freezes the smile at the alpha backbone.
double sabr_call(double S, double K, double T, double r, double alpha, double beta,
                 double rho, double nu);

// SABR (Hagan et al. 2002) European put via asymptotic Black implied volatility.
// Same parameters as sabr_call; price is bs_put with the Hagan SABR implied vol.
double sabr_put(double S, double K, double T, double r, double alpha, double beta,
                double rho, double nu);

// Greeks
double bs_delta(double S, double K, double T, double r, double sigma, bool call);
double bs_gamma(double S, double K, double T, double r, double sigma);
double bs_vega(double S, double K, double T, double r, double sigma);
double bs_theta(double S, double K, double T, double r, double sigma, bool call);
double bs_rho(double S, double K, double T, double r, double sigma, bool call);

// Implied volatility (Newton-Raphson)
Result<double> bs_implied_vol(double price, double S, double K, double T,
                               double r, bool call);

// --- Merton (1974) structural credit-risk model ---
// Firm equity is modeled as a European call option on firm assets with strike
// equal to the face value of debt. Distance to default and the implied
// probability of default reuse the same d2 / normal-CDF machinery as
// Black-Scholes above (S=asset value V, K=debt face value D).

struct MertonResult {
    double distance_to_default;
    double probability_of_default;
    double implied_asset_value;
    double implied_asset_volatility;
    bool converged;
    size_t iterations;
};

// Direct distance-to-default when asset value V and asset volatility sigma_V
// are known (not backed out from equity). distance_to_default is exactly
// Black-Scholes d2 with S=V, K=D, sigma=sigma_V; probability_of_default is
// N(-d2), the risk-neutral default probability over the horizon.
MertonResult merton_distance_to_default(double asset_value, double asset_volatility,
                                        double debt_face_value, double risk_free_rate,
                                        double time_horizon);

// Observable-equity version: given market equity value E and equity volatility
// sigma_E, solves the coupled Merton system
//   E = V*N(d1) - D*exp(-r*T)*N(d2)   (equity as a call on assets)
//   sigma_E*E = N(d1)*sigma_V*V         (Ito's lemma link)
// for (V, sigma_V) via fixed-point iteration, then delegates to
// merton_distance_to_default for the final risk metrics. Non-convergence is
// reported via converged=false (iterations holds the last iteration count).
MertonResult merton_implied_asset_params(double equity_value, double equity_volatility,
                                         double debt_face_value, double risk_free_rate,
                                         double time_horizon, int max_iter = 100,
                                         double tol = 1e-8);

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
// Parametric VaR/CVaR (variance-covariance method, assuming normally-distributed
// returns). alpha is the confidence level (e.g. 0.95); reported VaR is a positive
// loss magnitude.
double var(std::span<const double> returns, double alpha = 0.95);
double cvar(std::span<const double> returns, double alpha = 0.95); // ES

// Historical-simulation VaR/CVaR: uses the empirical distribution of actual
// return observations (no normality assumption). confidence in (0,1); VaR is
// the empirical (1-confidence) percentile of returns, reported as a positive
// loss. CVaR is the mean of all returns at or below the VaR threshold.
double historical_var(std::span<const double> returns, double confidence = 0.95);
double historical_cvar(std::span<const double> returns, double confidence = 0.95);

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

// --- Bachelier (normal) model for options on a forward F ---
// Same forward-based setup as black76 above, but assumes the forward follows
// ARITHMETIC Brownian motion dF = sigma*dW (normal/Gaussian dynamics) rather
// than black76's geometric Brownian motion dF/F = sigma*dW (lognormal
// dynamics). Because F itself is normally distributed under this model, F
// (and K) are allowed to be zero or NEGATIVE -- unlike bs_call/bs_put and
// black76, which require a strictly positive underlying/forward since they
// take log(F/K). This makes Bachelier the standard choice for interest-rate
// derivatives (e.g. swaption/cap/floor pricing) during negative-rate
// environments, where Black-76 breaks down.
//
// With d = (F - K) / (sigma*sqrt(T)), Phi/phi the standard normal CDF/PDF:
//   call = exp(-r*T) * [(F-K)*Phi(d) + sigma*sqrt(T)*phi(d)]
//   put  = exp(-r*T) * [(K-F)*Phi(-d) + sigma*sqrt(T)*phi(d)]
// Put-call parity call - put = exp(-r*T)*(F-K) holds exactly (model-
// independent identity). As sigma -> 0, both collapse to the discounted
// intrinsic value exp(-r*T)*max(+-(F-K), 0), and at F == K (at-the-money)
// they simplify to exp(-r*T) * sigma*sqrt(T) / sqrt(2*pi).
double bachelier_call(double F, double K, double T, double r, double sigma);
double bachelier_put(double F, double K, double T, double r, double sigma);

// --- Short-rate models: Vasicek / CIR zero-coupon bond pricing ---
// Both models below price a unit-face-value zero-coupon bond P(t,T) under a
// one-factor short-rate process for the instantaneous rate r, as the
// classic affine closed-form P(t,T) = A(tau)*exp(-B(tau)*r) with tau = T-t.
//
// Shared defensive conventions for both functions:
//   - tau <= 0: the bond is at or past maturity, so it is worth exactly its
//     unit face value regardless of the other parameters -- returns 1.0.
//   - a <= 0: mean-reversion speed must be strictly positive for the
//     "ordinary" closed-form below to be well-defined (B(tau)'s derivation
//     assumes a != 0, and a < 0 would make the process mean-diverging, not
//     mean-reverting). This is treated as malformed input: returns NaN
//     rather than silently producing a misleading number.

// Vasicek model: dr = a*(b - r)*dt + sigma*dW (Ornstein-Uhlenbeck short
// rate; can in principle go negative since the diffusion term is constant).
// Closed-form bond price:
//   B(tau) = (1 - exp(-a*tau)) / a
//   A(tau) = exp( (b - sigma^2/(2*a^2)) * (B(tau) - tau) - (sigma^2/(4*a)) * B(tau)^2 )
//   P(t,T) = A(tau) * exp(-B(tau) * r)
// At sigma == 0 this reduces exactly (no approximation) to the deterministic
// discount factor exp(-integral of r(s) ds), and in particular to
// exp(-b*tau) when r == b (already at the mean, zero drift).
// @param r current instantaneous short rate.
// @param a mean-reversion speed (a > 0 required; see defensive note above).
// @param b long-run mean rate level.
// @param sigma short-rate volatility (sigma >= 0; sign is irrelevant since
//        it only ever appears squared).
// @param tau time to maturity T-t (tau <= 0 returns 1.0; see note above).
// @return zero-coupon bond price P(t,T) for unit face value 1.
double vasicek_bond_price(double r, double a, double b, double sigma, double tau);

// Cox-Ingersoll-Ross (CIR) model: dr = a*(b - r)*dt + sigma*sqrt(r)*dW
// (mean-reverting with a level-dependent, non-negative-preserving diffusion
// term, unlike Vasicek). Closed-form bond price:
//   h = sqrt(a^2 + 2*sigma^2)
//   B(tau) = 2*(exp(h*tau) - 1) / ((h+a)*(exp(h*tau)-1) + 2*h)
//   A(tau) = ( 2*h*exp((a+h)*tau/2) / ((h+a)*(exp(h*tau)-1) + 2*h) ) ^ (2*a*b/sigma^2)
//   P(t,T) = A(tau) * exp(-B(tau) * r)
// sigma == 0 makes the A(tau) exponent 2*a*b/sigma^2 blow up while its base
// tends to exactly 1 (a removable 1^infinity singularity), which is
// numerically unsafe to evaluate directly via std::pow. This is handled
// defensively by falling back to the same deterministic zero-vol limit as
// vasicek_bond_price (B(tau) = (1-exp(-a*tau))/a, P = exp(b*(B(tau)-tau) -
// B(tau)*r)), which is the exact sigma -> 0 limit of the formula above.
// @param r, a, b, sigma, tau: as in vasicek_bond_price (same tau<=0/a<=0
//        defensive conventions; sigma == 0 handled explicitly as above).
// @return zero-coupon bond price P(t,T) for unit face value 1.
double cir_bond_price(double r, double a, double b, double sigma, double tau);

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

// --- Pricing: trinomial tree (Boyle 1986) ---
// Alternative discretization of the same continuous-time GBM process as the
// binomial tree above, with an extra "stay" branch per step: dt=T/n_steps,
// u=exp(sigma*sqrt(2*dt)), d=1/u, middle branch unchanged, and risk-neutral
// probabilities p_u/p_m/p_d chosen to match the first two moments of GBM
// (Boyle's discrete-time moment-matching construction). Converges to the
// closed-form bs_call/bs_put as n_steps -> infinity when is_american=false,
// and to the same limit as binomial_call/binomial_put/american_option since
// all three discretize the same process. Early exercise (is_american=true)
// is handled exactly like american_option: at each node, value is
// max(continuation, intrinsic).
double trinomial_option(double S, double K, double T, double r, double sigma,
                        int n_steps, bool is_call, bool is_american);

// --- Pricing: geometric-average Asian options (Kemna-Vorst 1990 closed form) ---
// Unlike the arithmetic-average Asian options below (mc_asian_call/put, no
// closed form), the GEOMETRIC average of lognormal GBM prices is itself
// lognormal, so the discretely-monitored geometric-average Asian option has
// an exact closed form. Fixings are the standard n equally-spaced
// observations EXCLUDING t=0 and INCLUDING t=T, i.e. t_i = i*T/n_fixings for
// i=1..n_fixings (so n_fixings=1 observes only S_T, reducing to a plain
// vanilla option).
//
// Averaging the (correlated) log-increments at those fixing times shows the
// geometric average G is lognormal with an adjusted effective volatility and
// drift:
//   sigma_adj = sigma * sqrt((n+1)*(2n+1) / (6*n^2))
//   mu_adj    = 0.5*sigma_adj^2 + (n+1)/(2n) * (r - 0.5*sigma^2)
// such that F = S*exp(mu_adj*T) is the risk-neutral expectation (the
// "forward price") of G. The option then prices exactly like Black-76 on
// that synthetic forward: price = black76(F, K, T, r, sigma_adj, call).
//
// Two hard sanity checks (see tests): as n_fixings -> infinity, sigma_adj ->
// sigma/sqrt(3) and mu_adj -> 0.5*(r - sigma^2/6), the well-known continuous
// geometric-average closed form; at n_fixings == 1, sigma_adj == sigma and
// mu_adj == r exactly, so F == S*exp(r*T) and the formula reduces exactly to
// plain Black-Scholes (bs_call/bs_put).
double geo_asian_call(double S, double K, double T, double r, double sigma, int n_fixings);
double geo_asian_put(double S, double K, double T, double r, double sigma, int n_fixings);

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
