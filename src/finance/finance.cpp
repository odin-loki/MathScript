#define _USE_MATH_DEFINES
#include "ms/finance/finance.hpp"
#include "ms/error/error_types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace ms {
namespace finance {

// Standard normal CDF
static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

static double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

static void bs_d1_d2(double S, double K, double T, double r, double sigma,
                     double& d1, double& d2) {
    d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    d2 = d1 - sigma * std::sqrt(T);
}

double bs_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double bs_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(K - S, 0.0);
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

double bs_delta(double S, double K, double T, double r, double sigma, bool call) {
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    return call ? norm_cdf(d1) : norm_cdf(d1) - 1.0;
}

double bs_gamma(double S, double K, double T, double r, double sigma) {
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    return norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

double bs_vega(double S, double K, double T, double r, double sigma) {
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    return S * norm_pdf(d1) * std::sqrt(T);
}

double bs_theta(double S, double K, double T, double r, double sigma, bool call) {
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    double t1 = -S * norm_pdf(d1) * sigma / (2.0 * std::sqrt(T));
    if (call)
        return t1 - r * K * std::exp(-r * T) * norm_cdf(d2);
    else
        return t1 + r * K * std::exp(-r * T) * norm_cdf(-d2);
}

double bs_rho(double S, double K, double T, double r, double sigma, bool call) {
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    if (call)
        return K * T * std::exp(-r * T) * norm_cdf(d2);
    else
        return -K * T * std::exp(-r * T) * norm_cdf(-d2);
}

Result<double> bs_implied_vol(double price, double S, double K, double T,
                               double r, bool call) {
    double sigma = 0.3;
    for (int i = 0; i < 100; ++i) {
        double theo = call ? bs_call(S, K, T, r, sigma) : bs_put(S, K, T, r, sigma);
        double diff = theo - price;
        if (std::abs(diff) < 1e-10) return sigma;
        double vega = bs_vega(S, K, T, r, sigma);
        if (std::abs(vega) < 1e-14)
            return std::unexpected(Error{ConvergenceFail{static_cast<size_t>(i), diff}});
        sigma -= diff / vega;
        if (sigma <= 0.0) sigma = 1e-6;
    }
    return std::unexpected(Error{ConvergenceFail{100, 0.0}});
}

double bond_price(double c, double y, int n, double fv) {
    double coupon = c * fv;
    double pv = 0.0;
    for (int i = 1; i <= n; ++i)
        pv += coupon / std::pow(1.0 + y, i);
    pv += fv / std::pow(1.0 + y, n);
    return pv;
}

double bond_duration(double c, double y, int n, double fv) {
    double coupon = c * fv;
    double bp = bond_price(c, y, n, fv);
    double dur = 0.0;
    for (int i = 1; i <= n; ++i)
        dur += static_cast<double>(i) * coupon / std::pow(1.0 + y, i);
    dur += static_cast<double>(n) * fv / std::pow(1.0 + y, n);
    return dur / bp;
}

double bond_modified_duration(double c, double y, int n, double fv) {
    return bond_duration(c, y, n, fv) / (1.0 + y);
}

double bond_convexity(double c, double y, int n, double fv) {
    double coupon = c * fv;
    double bp = bond_price(c, y, n, fv);
    double conv = 0.0;
    for (int i = 1; i <= n; ++i) {
        double t = static_cast<double>(i);
        conv += t * (t + 1.0) * coupon / std::pow(1.0 + y, i + 2);
    }
    conv += static_cast<double>(n) * (n + 1.0) * fv /
            std::pow(1.0 + y, n + 2);
    return conv / bp;
}

Result<double> bond_ytm(double price, double c, int n, double fv) {
    double lo = 0.0, hi = 1.0;
    for (int i = 0; i < 200; ++i) {
        double mid = 0.5 * (lo + hi);
        double p = bond_price(c, mid, n, fv);
        if (std::abs(p - price) < 1e-8) return mid;
        if (p > price) lo = mid; else hi = mid;
    }
    double mid = 0.5 * (lo + hi);
    return mid;
}

double npv(double rate, std::span<const double> cashflows) {
    double pv = 0.0;
    for (size_t t = 0; t < cashflows.size(); ++t)
        pv += cashflows[t] / std::pow(1.0 + rate, static_cast<double>(t));
    return pv;
}

Result<double> irr(std::span<const double> cashflows, double guess, int max_iter) {
    double r = guess;
    for (int i = 0; i < max_iter; ++i) {
        double f = 0.0, df = 0.0;
        for (size_t t = 0; t < cashflows.size(); ++t) {
            double dt = static_cast<double>(t);
            double disc = std::pow(1.0 + r, dt);
            f  += cashflows[t] / disc;
            df -= dt * cashflows[t] / ((1.0 + r) * disc);
        }
        if (std::abs(f) < 1e-10) return r;
        if (std::abs(df) < 1e-15)
            return std::unexpected(Error{ConvergenceFail{static_cast<size_t>(i), f}});
        r -= f / df;
    }
    return std::unexpected(Error{ConvergenceFail{static_cast<size_t>(max_iter), 0.0}});
}

double pv(double rate, int n, double pmt, double fv) {
    if (std::abs(rate) < 1e-12) return -pmt * n - fv;
    double disc = std::pow(1.0 + rate, n);
    return -(pmt * (1.0 - 1.0 / disc) / rate + fv / disc);
}

double fv_annuity(double rate, int n, double pmt, double pv0) {
    if (std::abs(rate) < 1e-12) return -pv0 - pmt * n;
    double disc = std::pow(1.0 + rate, n);
    return -(pv0 * disc + pmt * (disc - 1.0) / rate);
}

double pmt_annuity(double rate, int n, double pv0, double fv) {
    if (std::abs(rate) < 1e-12) return -(pv0 + fv) / n;
    double disc = std::pow(1.0 + rate, n);
    return -rate * (pv0 * disc + fv) / (disc - 1.0);
}

double var(std::span<const double> returns, double alpha) {
    std::vector<double> sorted(returns.begin(), returns.end());
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>((1.0 - alpha) * sorted.size());
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return -sorted[idx];
}

double cvar(std::span<const double> returns, double alpha) {
    std::vector<double> sorted(returns.begin(), returns.end());
    std::sort(sorted.begin(), sorted.end());
    size_t cutoff = static_cast<size_t>((1.0 - alpha) * sorted.size());
    if (cutoff == 0) return -sorted[0];
    double sum = 0.0;
    for (size_t i = 0; i < cutoff; ++i) sum += sorted[i];
    return -sum / static_cast<double>(cutoff);
}

double sharpe_ratio(std::span<const double> returns, double risk_free) {
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double var_r = 0.0;
    for (double r : returns) var_r += (r - mean) * (r - mean);
    var_r /= (returns.size() - 1);
    return (mean - risk_free) / std::sqrt(var_r);
}

double sortino_ratio(std::span<const double> returns, double risk_free) {
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double downside_var = 0.0;
    int count = 0;
    for (double r : returns)
        if (r < risk_free) { downside_var += (r - risk_free) * (r - risk_free); ++count; }
    if (count == 0) return 1e9;
    return (mean - risk_free) / std::sqrt(downside_var / count);
}

double max_drawdown(std::span<const double> equity_curve) {
    double peak = equity_curve[0], mdd = 0.0;
    for (double v : equity_curve) {
        if (v > peak) peak = v;
        double dd = (peak - v) / peak;
        if (dd > mdd) mdd = dd;
    }
    return mdd;
}

double kelly_fraction(double win_prob, double win_loss_ratio) {
    return win_prob - (1.0 - win_prob) / win_loss_ratio;
}

double compound(double principal, double rate, int n_periods, int compounds_per_period) {
    return principal * std::pow(1.0 + rate / compounds_per_period,
                                n_periods * compounds_per_period);
}

double continuous_compound(double principal, double rate, double t) {
    return principal * std::exp(rate * t);
}

double portfolio_variance(std::span<const double> weights,
                          std::span<const double> cov_matrix) {
    size_t n = weights.size();
    double var = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            var += weights[i] * weights[j] * cov_matrix[i * n + j];
    return var;
}

double portfolio_return(std::span<const double> weights,
                        std::span<const double> asset_returns) {
    double ret = 0.0;
    for (size_t i = 0; i < weights.size(); ++i)
        ret += weights[i] * asset_returns[i];
    return ret;
}

double binomial_call(double S, double K, double T, double r, double sigma, int steps) {
    double dt = T / steps;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double disc = std::exp(-r * dt);
    double p = (std::exp(r * dt) - d) / (u - d);
    std::vector<double> prices(steps + 1);
    for (int i = 0; i <= steps; ++i)
        prices[i] = std::max(S * std::pow(u, i) * std::pow(d, steps - i) - K, 0.0);
    for (int step = steps - 1; step >= 0; --step)
        for (int i = 0; i <= step; ++i)
            prices[i] = disc * (p * prices[i + 1] + (1.0 - p) * prices[i]);
    return prices[0];
}

double binomial_put(double S, double K, double T, double r, double sigma, int steps) {
    double dt = T / steps;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double disc = std::exp(-r * dt);
    double p = (std::exp(r * dt) - d) / (u - d);
    std::vector<double> prices(steps + 1);
    for (int i = 0; i <= steps; ++i)
        prices[i] = std::max(K - S * std::pow(u, i) * std::pow(d, steps - i), 0.0);
    for (int step = steps - 1; step >= 0; --step)
        for (int i = 0; i <= step; ++i)
            prices[i] = disc * (p * prices[i + 1] + (1.0 - p) * prices[i]);
    return prices[0];
}

} // namespace finance
} // namespace ms
