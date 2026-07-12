#define _USE_MATH_DEFINES
#include "ms/finance/finance.hpp"
#include "ms/error/error_types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
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

double information_ratio(std::span<const double> returns,
                         std::span<const double> benchmark) {
    double mean = 0.0;
    for (size_t i = 0; i < returns.size(); ++i)
        mean += returns[i] - benchmark[i];
    mean /= returns.size();
    double var_te = 0.0;
    for (size_t i = 0; i < returns.size(); ++i) {
        double ex = returns[i] - benchmark[i];
        var_te += (ex - mean) * (ex - mean);
    }
    var_te /= (returns.size() - 1);
    return mean / std::sqrt(var_te);
}

double treynor_ratio(std::span<const double> returns, double risk_free, double beta) {
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    return (mean - risk_free) / beta;
}

double capm(double risk_free, double beta, double market_return) {
    return risk_free + beta * (market_return - risk_free);
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

// Gauss-Jordan elimination on the augmented matrix K (n x n+n_rhs, last n_rhs
// columns are the RHS, one column per right-hand-side vector) with partial
// pivoting, reducing K in place to [I | solution(s)]. n_rhs defaults to 1 for
// the single-RHS callers below; bl_posterior_returns's matrix inversions pass
// n_rhs=n (RHS=identity) to solve A*X=I in one pass. Same pattern as
// gauss_solve in src/control/control.cpp, kept local here since small dense
// solves like this aren't routed through the generic Matrix<S,...> system
// elsewhere in the codebase either.
static bool gauss_solve_aug(std::vector<std::vector<double>>& K, int n, int n_rhs = 1) {
    int total_cols = n + n_rhs;
    for (int col = 0; col < n; ++col) {
        int pivot = -1;
        double best = 0.0;
        for (int row = col; row < n; ++row)
            if (std::abs(K[row][col]) > best) { best = std::abs(K[row][col]); pivot = row; }
        if (pivot < 0 || best < 1e-12) return false; // no usable pivot: singular to tolerance
        std::swap(K[col], K[pivot]);
        double sc = K[col][col];
        for (int j = col; j < total_cols; ++j) K[col][j] /= sc;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = K[row][col];
            for (int j = col; j < total_cols; ++j) K[row][j] -= f * K[col][j];
        }
    }
    return true;
}

// Solves the dense n x n system A*X = B for X, where B has n_rhs columns
// (flattened row-major, n x n_rhs). Generalizes solve_cov_system to multiple
// right-hand-sides so it can also compute a full matrix inverse (B = identity).
static Result<std::vector<double>> solve_linear_system(std::span<const double> A, int n,
                                                        std::span<const double> B, int n_rhs) {
    std::vector<std::vector<double>> K(static_cast<size_t>(n),
                                       std::vector<double>(static_cast<size_t>(n) +
                                                            static_cast<size_t>(n_rhs)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            K[i][j] = A[static_cast<size_t>(i) * n + j];
        for (int j = 0; j < n_rhs; ++j)
            K[i][n + j] = B[static_cast<size_t>(i) * n_rhs + j];
    }
    if (!gauss_solve_aug(K, n, n_rhs))
        return std::unexpected(Error{SingularMatrix{}});
    std::vector<double> X(static_cast<size_t>(n) * static_cast<size_t>(n_rhs));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n_rhs; ++j)
            X[static_cast<size_t>(i) * n_rhs + j] = K[i][n + j];
    return X;
}

// n x n identity matrix, flattened row-major.
static std::vector<double> identity_matrix(int n) {
    std::vector<double> I(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) I[static_cast<size_t>(i) * n + i] = 1.0;
    return I;
}

// Inverts an n x n matrix (flattened row-major) by solving A*X = I via
// solve_linear_system.
static Result<std::vector<double>> invert_matrix(std::span<const double> A, int n) {
    std::vector<double> I = identity_matrix(n);
    return solve_linear_system(A, n, I, n);
}

// Matrix-vector multiply: (rows x cols) * (cols) -> (rows). mat is flattened
// row-major.
static std::vector<double> mat_vec_mul(std::span<const double> mat, int rows, int cols,
                                       std::span<const double> vec) {
    std::vector<double> out(static_cast<size_t>(rows), 0.0);
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += mat[static_cast<size_t>(i) * cols + j] * vec[j];
        out[i] = s;
    }
    return out;
}

// Matrix-matrix multiply: (rows x inner) * (inner x cols) -> (rows x cols),
// all flattened row-major.
static std::vector<double> mat_mat_mul(std::span<const double> A, int rows, int inner,
                                       std::span<const double> B, int cols) {
    std::vector<double> out(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int t = 0; t < inner; ++t)
                s += A[static_cast<size_t>(i) * inner + t] * B[static_cast<size_t>(t) * cols + j];
            out[static_cast<size_t>(i) * cols + j] = s;
        }
    return out;
}

// Transpose a rows x cols matrix (flattened row-major) into cols x rows.
static std::vector<double> mat_transpose(std::span<const double> A, int rows, int cols) {
    std::vector<double> out(static_cast<size_t>(cols) * static_cast<size_t>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[static_cast<size_t>(j) * rows + i] = A[static_cast<size_t>(i) * cols + j];
    return out;
}

// Solves the dense n x n system cov_matrix*y = rhs for y. cov_matrix is the
// flattened row-major covariance matrix; it's copied into a local dense table
// since gauss_solve_aug needs to pivot/mutate rows in place.
static Result<std::vector<double>> solve_cov_system(std::span<const double> cov_matrix, int n,
                                                    std::span<const double> rhs) {
    std::vector<std::vector<double>> K(static_cast<size_t>(n),
                                       std::vector<double>(static_cast<size_t>(n) + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            K[i][j] = cov_matrix[static_cast<size_t>(i) * n + j];
        K[i][n] = rhs[i];
    }
    if (!gauss_solve_aug(K, n))
        return std::unexpected(Error{SingularMatrix{}});
    std::vector<double> y(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) y[i] = K[i][n];
    return y;
}

Result<std::vector<double>> min_variance_portfolio(std::span<const double> cov_matrix, int n) {
    if (n <= 0)
        return std::unexpected(Error{DomainError{"min_variance_portfolio", "n must be > 0"}});
    if (cov_matrix.size() != static_cast<size_t>(n) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{cov_matrix.size(),
                                                        static_cast<size_t>(n) * static_cast<size_t>(n)}});

    std::vector<double> ones(static_cast<size_t>(n), 1.0);
    auto y = solve_cov_system(cov_matrix, n, ones);
    if (!y) return std::unexpected(y.error());

    double s = std::accumulate(y->begin(), y->end(), 0.0);
    if (std::abs(s) < 1e-14)
        return std::unexpected(Error{SingularMatrix{}});

    std::vector<double> w(std::move(*y));
    for (double& wi : w) wi /= s;
    return w;
}

Result<std::vector<double>> efficient_frontier_portfolio(std::span<const double> cov_matrix,
                                                          std::span<const double> mu,
                                                          double target_return, int n) {
    if (n <= 0)
        return std::unexpected(Error{DomainError{"efficient_frontier_portfolio", "n must be > 0"}});
    if (cov_matrix.size() != static_cast<size_t>(n) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{cov_matrix.size(),
                                                        static_cast<size_t>(n) * static_cast<size_t>(n)}});
    if (mu.size() != static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{mu.size(), static_cast<size_t>(n)}});

    std::vector<double> ones(static_cast<size_t>(n), 1.0);
    auto y1 = solve_cov_system(cov_matrix, n, ones);
    if (!y1) return std::unexpected(y1.error());
    auto y2 = solve_cov_system(cov_matrix, n, mu);
    if (!y2) return std::unexpected(y2.error());

    double a = std::accumulate(y1->begin(), y1->end(), 0.0);
    double b = std::accumulate(y2->begin(), y2->end(), 0.0); // == mu^T*y1 by symmetry of Sigma^-1
    double c = 0.0;
    for (int i = 0; i < n; ++i) c += mu[i] * (*y2)[i];

    double det = a * c - b * b;
    if (std::abs(det) < 1e-12)
        return std::unexpected(Error{DomainError{"efficient_frontier_portfolio",
                                                  "degenerate two-fund system (near-zero determinant)"}});

    // Cramer's rule on [a b; b c][lambda; gamma] = [1; target_return].
    double lambda = (c - b * target_return) / det;
    double gamma  = (a * target_return - b) / det;

    std::vector<double> w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) w[i] = lambda * (*y1)[i] + gamma * (*y2)[i];
    return w;
}

Result<std::vector<double>> max_sharpe_portfolio(std::span<const double> cov_matrix,
                                                 std::span<const double> mu,
                                                 double risk_free, int n) {
    if (n <= 0)
        return std::unexpected(Error{DomainError{"max_sharpe_portfolio", "n must be > 0"}});
    if (cov_matrix.size() != static_cast<size_t>(n) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{cov_matrix.size(),
                                                        static_cast<size_t>(n) * static_cast<size_t>(n)}});
    if (mu.size() != static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{mu.size(), static_cast<size_t>(n)}});

    std::vector<double> excess(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) excess[i] = mu[i] - risk_free;

    auto y = solve_cov_system(cov_matrix, n, excess);
    if (!y) return std::unexpected(y.error());

    double s = std::accumulate(y->begin(), y->end(), 0.0);
    if (std::abs(s) < 1e-14)
        return std::unexpected(Error{DomainError{"max_sharpe_portfolio",
                                                  "excess-return weights sum to zero; tangency portfolio undefined"}});

    std::vector<double> w(std::move(*y));
    for (double& wi : w) wi /= s;
    return w;
}

Result<std::vector<double>> bl_implied_returns(std::span<const double> cov_matrix,
                                               std::span<const double> w_mkt,
                                               double delta, int n) {
    if (n <= 0)
        return std::unexpected(Error{DomainError{"bl_implied_returns", "n must be > 0"}});
    if (cov_matrix.size() != static_cast<size_t>(n) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{cov_matrix.size(),
                                                        static_cast<size_t>(n) * static_cast<size_t>(n)}});
    if (w_mkt.size() != static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{w_mkt.size(), static_cast<size_t>(n)}});
    if (delta <= 0.0)
        return std::unexpected(Error{DomainError{"bl_implied_returns", "delta must be > 0"}});

    std::vector<double> pi = mat_vec_mul(cov_matrix, n, n, w_mkt);
    for (double& v : pi) v *= delta;
    return pi;
}

Result<std::vector<double>> bl_posterior_returns(std::span<const double> pi,
                                                 std::span<const double> cov_matrix,
                                                 double tau,
                                                 std::span<const double> P,
                                                 std::span<const double> Q,
                                                 std::span<const double> omega,
                                                 int n, int k) {
    if (n <= 0)
        return std::unexpected(Error{DomainError{"bl_posterior_returns", "n must be > 0"}});
    if (k < 0)
        return std::unexpected(Error{DomainError{"bl_posterior_returns", "k must be >= 0"}});
    if (tau <= 0.0)
        return std::unexpected(Error{DomainError{"bl_posterior_returns", "tau must be > 0"}});
    if (pi.size() != static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{pi.size(), static_cast<size_t>(n)}});
    if (cov_matrix.size() != static_cast<size_t>(n) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{cov_matrix.size(),
                                                        static_cast<size_t>(n) * static_cast<size_t>(n)}});

    // No views: the P^T*Omega^-1*P and P^T*Omega^-1*Q terms vanish (empty
    // sums), so [(tau*Sigma)^-1]*E[R] = [(tau*Sigma)^-1]*Pi and the posterior
    // collapses exactly to the prior. Handled directly rather than inverting
    // a degenerate 0x0 Omega.
    if (k == 0) {
        if (!P.empty() || !Q.empty() || !omega.empty())
            return std::unexpected(Error{DomainError{"bl_posterior_returns",
                                                      "k=0 requires empty P, Q, and omega"}});
        return std::vector<double>(pi.begin(), pi.end());
    }

    if (P.size() != static_cast<size_t>(k) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{P.size(),
                                                        static_cast<size_t>(k) * static_cast<size_t>(n)}});
    if (Q.size() != static_cast<size_t>(k))
        return std::unexpected(Error{DimensionMismatch{Q.size(), static_cast<size_t>(k)}});
    if (omega.size() != static_cast<size_t>(k) * static_cast<size_t>(k))
        return std::unexpected(Error{DimensionMismatch{omega.size(),
                                                        static_cast<size_t>(k) * static_cast<size_t>(k)}});

    std::vector<double> tau_sigma(cov_matrix.begin(), cov_matrix.end());
    for (double& v : tau_sigma) v *= tau;

    auto inv_tau_sigma = invert_matrix(tau_sigma, n);
    if (!inv_tau_sigma) return std::unexpected(inv_tau_sigma.error());

    auto inv_omega = invert_matrix(omega, k);
    if (!inv_omega) return std::unexpected(inv_omega.error());

    std::vector<double> Pt = mat_transpose(P, k, n); // n x k
    std::vector<double> Pt_invOmega = mat_mat_mul(Pt, n, k, *inv_omega, k);       // n x k
    std::vector<double> Pt_invOmega_P = mat_mat_mul(Pt_invOmega, n, k, P, n);     // n x n

    // A = (tau*Sigma)^-1 + P^T*Omega^-1*P
    std::vector<double> A(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (size_t i = 0; i < A.size(); ++i) A[i] = (*inv_tau_sigma)[i] + Pt_invOmega_P[i];

    // b = (tau*Sigma)^-1*Pi + P^T*Omega^-1*Q
    std::vector<double> b1 = mat_vec_mul(*inv_tau_sigma, n, n, pi);
    std::vector<double> b2 = mat_vec_mul(Pt_invOmega, n, k, Q);
    std::vector<double> b(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) b[i] = b1[i] + b2[i];

    return solve_linear_system(A, n, b, 1);
}

Result<std::vector<double>> bl_posterior_returns_default_omega(
    std::span<const double> pi, std::span<const double> cov_matrix, double tau,
    std::span<const double> P, std::span<const double> Q, int n, int k) {
    if (n <= 0)
        return std::unexpected(Error{DomainError{"bl_posterior_returns_default_omega",
                                                  "n must be > 0"}});
    if (k < 0)
        return std::unexpected(Error{DomainError{"bl_posterior_returns_default_omega",
                                                  "k must be >= 0"}});
    if (tau <= 0.0)
        return std::unexpected(Error{DomainError{"bl_posterior_returns_default_omega",
                                                  "tau must be > 0"}});
    if (k == 0)
        return bl_posterior_returns(pi, cov_matrix, tau, P, Q, {}, n, 0);

    if (cov_matrix.size() != static_cast<size_t>(n) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{cov_matrix.size(),
                                                        static_cast<size_t>(n) * static_cast<size_t>(n)}});
    if (P.size() != static_cast<size_t>(k) * static_cast<size_t>(n))
        return std::unexpected(Error{DimensionMismatch{P.size(),
                                                        static_cast<size_t>(k) * static_cast<size_t>(n)}});
    if (Q.size() != static_cast<size_t>(k))
        return std::unexpected(Error{DimensionMismatch{Q.size(), static_cast<size_t>(k)}});

    // Standard He-Litterman simplification: Omega diagonal with
    // omega_ii = tau * (P_i . Sigma . P_i^T), i.e. each view's uncertainty is
    // proportional to the prior variance of that view's portfolio, and views
    // are uncorrelated with each other (off-diagonals zero).
    std::vector<double> omega(static_cast<size_t>(k) * static_cast<size_t>(k), 0.0);
    for (int i = 0; i < k; ++i) {
        std::span<const double> p_row = P.subspan(static_cast<size_t>(i) * n, n);
        std::vector<double> sigma_p = mat_vec_mul(cov_matrix, n, n, p_row);
        double view_var = 0.0;
        for (int j = 0; j < n; ++j) view_var += p_row[j] * sigma_p[j];
        omega[static_cast<size_t>(i) * k + i] = tau * view_var;
    }

    return bl_posterior_returns(pi, cov_matrix, tau, P, Q, omega, n, k);
}

double forward_rate(double r1, double t1, double r2, double t2) {
    return (r2 * t2 - r1 * t1) / (t2 - t1);
}

double digital_option(double S, double K, double T, double r, double sigma,
                      bool call, double payout) {
    if (T <= 0.0) return call ? (S > K ? payout : 0.0) : (S < K ? payout : 0.0);
    double d1, d2;
    bs_d1_d2(S, K, T, r, sigma, d1, d2);
    double disc = payout * std::exp(-r * T);
    return call ? disc * norm_cdf(d2) : disc * norm_cdf(-d2);
}

static void black76_d1_d2(double F, double K, double T, double sigma,
                            double& d1, double& d2) {
    d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / (sigma * std::sqrt(T));
    d2 = d1 - sigma * std::sqrt(T);
}

double black76(double F, double K, double T, double r, double sigma, bool call) {
    if (T <= 0.0) return call ? std::max(F - K, 0.0) : std::max(K - F, 0.0);
    double d1, d2;
    black76_d1_d2(F, K, T, sigma, d1, d2);
    double disc = std::exp(-r * T);
    if (call)
        return disc * (F * norm_cdf(d1) - K * norm_cdf(d2));
    return disc * (K * norm_cdf(-d2) - F * norm_cdf(-d1));
}

double bachelier_call(double F, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(F - K, 0.0);
    double disc = std::exp(-r * T);
    if (sigma <= 0.0) return disc * std::max(F - K, 0.0);
    double vol_sqrtT = sigma * std::sqrt(T);
    double d = (F - K) / vol_sqrtT;
    return disc * ((F - K) * norm_cdf(d) + vol_sqrtT * norm_pdf(d));
}

double bachelier_put(double F, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(K - F, 0.0);
    double disc = std::exp(-r * T);
    if (sigma <= 0.0) return disc * std::max(K - F, 0.0);
    double vol_sqrtT = sigma * std::sqrt(T);
    double d = (F - K) / vol_sqrtT;
    return disc * ((K - F) * norm_cdf(-d) + vol_sqrtT * norm_pdf(d));
}

struct BarrierCoeffs {
    double A, B, C, D;
};

static double barrier_safe_mul(double pow_hs, double n) {
    return n == 0.0 ? 0.0 : pow_hs * n;
}

static void barrier_coeffs(double S, double K, double B, double T, double r, double sigma,
                           double phi, double eta, BarrierCoeffs& c) {
    double vol_sqrtT = sigma * std::sqrt(T);
    double mu = r / (sigma * sigma) - 0.5;
    double muSigma = (1.0 + mu) * vol_sqrtT;
    double disc_r = std::exp(-r * T);
    double HS = B / S;
    double powHS0 = std::pow(HS, 2.0 * mu);
    double powHS1 = powHS0 * HS * HS;

    double x1 = std::log(S / K) / vol_sqrtT + muSigma;
    double x2 = std::log(S / B) / vol_sqrtT + muSigma;
    double y1 = std::log(B * HS / K) / vol_sqrtT + muSigma;
    double y2 = std::log(B / S) / vol_sqrtT + muSigma;

    c.A = phi * (S * norm_cdf(phi * x1) - K * disc_r * norm_cdf(phi * (x1 - vol_sqrtT)));
    c.B = phi * (S * norm_cdf(phi * x2) - K * disc_r * norm_cdf(phi * (x2 - vol_sqrtT)));
    c.C = phi * (S * barrier_safe_mul(powHS1, norm_cdf(eta * y1))
                 - K * disc_r * barrier_safe_mul(powHS0, norm_cdf(eta * (y1 - vol_sqrtT))));
    c.D = phi * (S * barrier_safe_mul(powHS1, norm_cdf(eta * y2))
                 - K * disc_r * barrier_safe_mul(powHS0, norm_cdf(eta * (y2 - vol_sqrtT))));
}

double barrier_option(double S, double K, double B, double T, double r, double sigma,
                      bool call, bool knock_in, bool up) {
    if (T <= 0.0)
        return call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);

    if (!up && S <= B) {
        if (knock_in) return call ? bs_call(S, K, T, r, sigma) : bs_put(S, K, T, r, sigma);
        return 0.0;
    }
    if (up && S >= B) {
        if (knock_in) return call ? bs_call(S, K, T, r, sigma) : bs_put(S, K, T, r, sigma);
        return 0.0;
    }

    double phi = call ? 1.0 : -1.0;
    double eta = up ? -1.0 : 1.0;
    BarrierCoeffs c;
    barrier_coeffs(S, K, B, T, r, sigma, phi, eta, c);

    bool strike_ge_barrier = K >= B;
    if (call) {
        if (!up) {
            if (knock_in)
                return strike_ge_barrier ? c.C : c.A - c.B + c.D;
            return strike_ge_barrier ? c.A - c.C : c.B - c.D;
        }
        if (knock_in)
            return strike_ge_barrier ? c.A : c.B - c.C + c.D;
        return strike_ge_barrier ? 0.0 : c.A - c.B + c.C - c.D;
    }
    if (!up) {
        if (knock_in)
            return strike_ge_barrier ? c.B - c.C + c.D : c.A;
        return strike_ge_barrier ? c.A - c.B + c.C - c.D : 0.0;
    }
    if (knock_in)
        return strike_ge_barrier ? c.A - c.B + c.D : c.C;
    return strike_ge_barrier ? c.B - c.D : c.A - c.C;
}

static double binomial_tree(double S, double K, double T, double r, double sigma,
                            int steps, bool call, bool american) {
    double dt = T / steps;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double disc = std::exp(-r * dt);
    double p = (std::exp(r * dt) - d) / (u - d);
    std::vector<double> prices(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        double spot = S * std::pow(u, i) * std::pow(d, steps - i);
        prices[i] = call ? std::max(spot - K, 0.0) : std::max(K - spot, 0.0);
    }
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            double continuation = disc * (p * prices[i + 1] + (1.0 - p) * prices[i]);
            if (american) {
                double spot = S * std::pow(u, i) * std::pow(d, step - i);
                double intrinsic = call ? std::max(spot - K, 0.0) : std::max(K - spot, 0.0);
                prices[i] = std::max(continuation, intrinsic);
            } else {
                prices[i] = continuation;
            }
        }
    }
    return prices[0];
}

double binomial_call(double S, double K, double T, double r, double sigma, int steps) {
    return binomial_tree(S, K, T, r, sigma, steps, true, false);
}

double binomial_put(double S, double K, double T, double r, double sigma, int steps) {
    return binomial_tree(S, K, T, r, sigma, steps, false, false);
}

double american_option(double S, double K, double T, double r, double sigma,
                       bool call, int steps) {
    return binomial_tree(S, K, T, r, sigma, steps, call, true);
}

double trinomial_option(double S, double K, double T, double r, double sigma,
                        int n_steps, bool is_call, bool is_american) {
    double dt = T / n_steps;
    double u = std::exp(sigma * std::sqrt(2.0 * dt));
    double d = 1.0 / u;
    double disc = std::exp(-r * dt);

    double up_leg = std::exp(sigma * std::sqrt(dt / 2.0));
    double down_leg = std::exp(-sigma * std::sqrt(dt / 2.0));
    double half_rate = std::exp(r * dt / 2.0);
    double denom = up_leg - down_leg;
    double pu_sqrt = (half_rate - down_leg) / denom;
    double pd_sqrt = (up_leg - half_rate) / denom;
    double p_u = pu_sqrt * pu_sqrt;
    double p_d = pd_sqrt * pd_sqrt;
    double p_m = 1.0 - p_u - p_d;

    // Node i at a level with `n` steps elapsed represents spot S*u^(i-n),
    // i.e. i=n is the middle (unchanged) node; i=0/i=2n are the extreme down/up
    // nodes. Terminal level has 2*n_steps+1 nodes.
    std::vector<double> prices(static_cast<size_t>(2 * n_steps + 1));
    for (int i = 0; i <= 2 * n_steps; ++i) {
        double spot = S * std::pow(u, i - n_steps);
        prices[i] = is_call ? std::max(spot - K, 0.0) : std::max(K - spot, 0.0);
    }
    for (int step = n_steps - 1; step >= 0; --step) {
        for (int i = 0; i <= 2 * step; ++i) {
            // Node i (exponent i-step) fans out to next-level indices i (down),
            // i+1 (middle), i+2 (up) -- see header derivation.
            double continuation = disc * (p_u * prices[i + 2] + p_m * prices[i + 1] +
                                          p_d * prices[i]);
            if (is_american) {
                double spot = S * std::pow(u, i - step);
                double intrinsic = is_call ? std::max(spot - K, 0.0) : std::max(K - spot, 0.0);
                prices[i] = std::max(continuation, intrinsic);
            } else {
                prices[i] = continuation;
            }
        }
    }
    return prices[0];
}

// Adjusted volatility/drift for the discrete geometric average of n equally-
// spaced fixings t_i=i*T/n, i=1..n (see header for the derivation). Shared by
// geo_asian_call/geo_asian_put.
static void geo_asian_adjusted_params(double sigma, double r, int n,
                                      double& sigma_adj, double& mu_adj) {
    double nd = static_cast<double>(n);
    sigma_adj = sigma * std::sqrt((nd + 1.0) * (2.0 * nd + 1.0) / (6.0 * nd * nd));
    mu_adj = 0.5 * sigma_adj * sigma_adj + (nd + 1.0) / (2.0 * nd) * (r - 0.5 * sigma * sigma);
}

double geo_asian_call(double S, double K, double T, double r, double sigma, int n_fixings) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    int n = std::max(n_fixings, 1);
    double sigma_adj, mu_adj;
    geo_asian_adjusted_params(sigma, r, n, sigma_adj, mu_adj);
    double F = S * std::exp(mu_adj * T);
    return black76(F, K, T, r, sigma_adj, true);
}

double geo_asian_put(double S, double K, double T, double r, double sigma, int n_fixings) {
    if (T <= 0.0) return std::max(K - S, 0.0);
    int n = std::max(n_fixings, 1);
    double sigma_adj, mu_adj;
    geo_asian_adjusted_params(sigma, r, n, sigma_adj, mu_adj);
    double F = S * std::exp(mu_adj * T);
    return black76(F, K, T, r, sigma_adj, false);
}

static double mc_european(double S, double K, double T, double r, double sigma,
                          int n_paths, unsigned seed, bool call) {
    if (n_paths <= 0 || S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) return 0.0;
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol_sqrtT = sigma * std::sqrt(T);
    int n_draws = (n_paths + 1) / 2;
    double sum = 0.0;
    int used = 0;
    for (int i = 0; i < n_draws; ++i) {
        double z = nd(rng);
        double s1 = S * std::exp(drift + vol_sqrtT * z);
        double payoff1 = call ? std::max(s1 - K, 0.0) : std::max(K - s1, 0.0);
        sum += payoff1;
        ++used;
        if (used >= n_paths) break;
        double s2 = S * std::exp(drift - vol_sqrtT * z);
        double payoff2 = call ? std::max(s2 - K, 0.0) : std::max(K - s2, 0.0);
        sum += payoff2;
        ++used;
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

double mc_european_call(double S, double K, double T, double r, double sigma,
                        int n_paths, unsigned seed) {
    return mc_european(S, K, T, r, sigma, n_paths, seed, true);
}

double mc_european_put(double S, double K, double T, double r, double sigma,
                       int n_paths, unsigned seed) {
    return mc_european(S, K, T, r, sigma, n_paths, seed, false);
}

static double mc_asian(double S, double K, double T, double r, double sigma,
                       int n_paths, int n_steps, unsigned seed, bool call) {
    if (n_paths <= 0 || n_steps <= 0 || S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0)
        return 0.0;
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    double dt = T / static_cast<double>(n_steps);
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol_sqrtdt = sigma * std::sqrt(dt);
    std::vector<double> z(static_cast<size_t>(n_steps));

    // Antithetic pairing over the whole path (mirror the full Z-sequence) halves
    // the number of independent path draws needed for a given sample count.
    int n_draws = (n_paths + 1) / 2;
    double sum = 0.0;
    int used = 0;
    for (int p = 0; p < n_draws; ++p) {
        for (auto& zi : z) zi = nd(rng);

        double s = S, avg = 0.0;
        for (double zi : z) {
            s *= std::exp(drift + vol_sqrtdt * zi);
            avg += s;
        }
        avg /= static_cast<double>(n_steps);
        double payoff1 = call ? std::max(avg - K, 0.0) : std::max(K - avg, 0.0);
        sum += payoff1;
        ++used;
        if (used >= n_paths) break;

        s = S; avg = 0.0;
        for (double zi : z) {
            s *= std::exp(drift - vol_sqrtdt * zi);
            avg += s;
        }
        avg /= static_cast<double>(n_steps);
        double payoff2 = call ? std::max(avg - K, 0.0) : std::max(K - avg, 0.0);
        sum += payoff2;
        ++used;
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

double mc_asian_call(double S, double K, double T, double r, double sigma,
                     int n_paths, int n_steps, unsigned seed) {
    return mc_asian(S, K, T, r, sigma, n_paths, n_steps, seed, true);
}

double mc_asian_put(double S, double K, double T, double r, double sigma,
                    int n_paths, int n_steps, unsigned seed) {
    return mc_asian(S, K, T, r, sigma, n_paths, n_steps, seed, false);
}

static double mc_lookback_floating(double S, double T, double r, double sigma,
                                   int n_paths, int n_steps, unsigned seed, bool call) {
    if (n_paths <= 0 || n_steps <= 0 || S <= 0.0 || T <= 0.0 || sigma <= 0.0) return 0.0;
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

        // path_min/path_max seeded at S rather than the first simulated point,
        // since the path includes S_0=S and it can itself be the extreme.
        double s = S, path_min = S, path_max = S;
        for (double zi : z) {
            s *= std::exp(drift + vol_sqrtdt * zi);
            path_min = std::min(path_min, s);
            path_max = std::max(path_max, s);
        }
        double payoff1 = call ? (s - path_min) : (path_max - s);
        sum += payoff1;
        ++used;
        if (used >= n_paths) break;

        s = S; path_min = S; path_max = S;
        for (double zi : z) {
            s *= std::exp(drift - vol_sqrtdt * zi);
            path_min = std::min(path_min, s);
            path_max = std::max(path_max, s);
        }
        double payoff2 = call ? (s - path_min) : (path_max - s);
        sum += payoff2;
        ++used;
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

double mc_lookback_floating_call(double S, double T, double r, double sigma,
                                 int n_paths, int n_steps, unsigned seed) {
    return mc_lookback_floating(S, T, r, sigma, n_paths, n_steps, seed, true);
}

double mc_lookback_floating_put(double S, double T, double r, double sigma,
                                int n_paths, int n_steps, unsigned seed) {
    return mc_lookback_floating(S, T, r, sigma, n_paths, n_steps, seed, false);
}

static double mc_lookback_fixed(double S, double K, double T, double r, double sigma,
                                int n_paths, int n_steps, unsigned seed, bool call) {
    if (n_paths <= 0 || n_steps <= 0 || S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0)
        return 0.0;
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

        double s = S, path_min = S, path_max = S;
        for (double zi : z) {
            s *= std::exp(drift + vol_sqrtdt * zi);
            path_min = std::min(path_min, s);
            path_max = std::max(path_max, s);
        }
        double payoff1 = call ? std::max(path_max - K, 0.0) : std::max(K - path_min, 0.0);
        sum += payoff1;
        ++used;
        if (used >= n_paths) break;

        s = S; path_min = S; path_max = S;
        for (double zi : z) {
            s *= std::exp(drift - vol_sqrtdt * zi);
            path_min = std::min(path_min, s);
            path_max = std::max(path_max, s);
        }
        double payoff2 = call ? std::max(path_max - K, 0.0) : std::max(K - path_min, 0.0);
        sum += payoff2;
        ++used;
    }
    return std::exp(-r * T) * sum / static_cast<double>(n_paths);
}

double mc_lookback_fixed_call(double S, double K, double T, double r, double sigma,
                              int n_paths, int n_steps, unsigned seed) {
    return mc_lookback_fixed(S, K, T, r, sigma, n_paths, n_steps, seed, true);
}

double mc_lookback_fixed_put(double S, double K, double T, double r, double sigma,
                             int n_paths, int n_steps, unsigned seed) {
    return mc_lookback_fixed(S, K, T, r, sigma, n_paths, n_steps, seed, false);
}

} // namespace finance
} // namespace ms
