#include "ms/poly/poly.hpp"
#include "ms/core/operations.hpp"
#include "ms/numthy/numthy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace ms {
namespace poly {

std::vector<double> poly_eval(const std::vector<double>& coeffs, double x) {
    double value = 0.0;
    for (size_t i = coeffs.size(); i > 0; --i) {
        value = value * x + coeffs[i - 1];
    }
    return {value};
}

std::vector<double> poly_add(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = (a.size() > b.size()) ? a.size() : b.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] += a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        out[i] += b[i];
    }
    return out;
}

std::vector<double> poly_sub(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = (a.size() > b.size()) ? a.size() : b.size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] += a[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
        out[i] -= b[i];
    }
    return out;
}

std::vector<double> poly_mul(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            out[i + j] += a[i] * b[j];
        }
    }
    return out;
}

std::vector<double> poly_deriv(const std::vector<double>& coeffs) {
    if (coeffs.size() <= 1) {
        return {0.0};
    }
    std::vector<double> out(coeffs.size() - 1);
    for (size_t i = 1; i < coeffs.size(); ++i) {
        out[i - 1] = coeffs[i] * static_cast<double>(i);
    }
    return out;
}

// Strip trailing near-zero coefficients
static std::vector<double> strip(std::vector<double> p, double eps = 1e-14) {
    while (p.size() > 1 && std::abs(p.back()) < eps) p.pop_back();
    return p;
}

std::vector<double> poly_div_quot(const std::vector<double>& a,
                                   const std::vector<double>& b) {
    auto A = strip(a), B = strip(b);
    if (A.size() < B.size()) return {0.0};
    const size_t n = A.size() - B.size() + 1;
    std::vector<double> q(n, 0.0);
    auto rem = A;
    while (rem.size() >= B.size()) {
        const size_t i = rem.size() - 1;
        if (std::abs(rem[i]) < 1e-14) {
            rem.pop_back();
            continue;
        }
        const size_t pos = i - (B.size() - 1);
        const double coef = rem[i] / B.back();
        q[pos] = coef;
        for (size_t j = 0; j < B.size(); ++j) {
            rem[i - (B.size() - 1 - j)] -= coef * B[j];
        }
        rem = strip(rem);
    }
    return strip(q);
}

std::vector<double> poly_mod(const std::vector<double>& a,
                              const std::vector<double>& b) {
    auto A = strip(a), B = strip(b);
    if (A.size() < B.size()) return A;
    auto rem = A;
    while (rem.size() >= B.size()) {
        const size_t i = rem.size() - 1;
        if (std::abs(rem[i]) < 1e-14) {
            rem.pop_back();
            continue;
        }
        const double coef = rem[i] / B.back();
        for (size_t j = 0; j < B.size(); ++j) {
            rem[i - (B.size() - 1 - j)] -= coef * B[j];
        }
        rem = strip(rem);
    }
    return rem;
}

std::vector<double> poly_integ(const std::vector<double>& coeffs, double c) {
    std::vector<double> out(coeffs.size() + 1, 0.0);
    out[0] = c;
    for (size_t i = 0; i < coeffs.size(); ++i)
        out[i + 1] = coeffs[i] / static_cast<double>(i + 1);
    return out;
}

std::vector<double> poly_compose(const std::vector<double>& p,
                                  const std::vector<double>& q) {
    // Horner: p(q(x)) = coeffs evaluated at q
    if (p.empty()) return {0.0};
    std::vector<double> result = {p.back()};
    for (int i = static_cast<int>(p.size()) - 2; i >= 0; --i) {
        result = poly_mul(result, q);
        result[0] += p[static_cast<size_t>(i)];
    }
    return result;
}

std::vector<double> poly_gcd(const std::vector<double>& a,
                               const std::vector<double>& b) {
    auto A = strip(a), B = strip(b);
    while (B.size() > 1) {
        auto R = poly_mod(A, B);
        R = strip(R);
        if (R.empty() || (R.size() == 1 && std::abs(R[0]) < 1e-10)) {
            A = B;
            break;
        }
        if (R.size() == 1) {
            A = {1.0};
            break;
        }
        A = B;
        B = R;
    }
    if (B.size() == 1 && std::abs(B[0]) < 1e-10) {
        // B is zero; A already holds the gcd candidate
    } else if (B.size() == 1 && A.size() <= 1) {
        A = {1.0};
    }
    // Normalize
    if (!A.empty() && std::abs(A.back()) > 1e-14) {
        double lc = A.back();
        for (auto& v : A) v /= lc;
    }
    return strip(A);
}

static double binom_double(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double r = 1.0;
    for (int i = 1; i <= k; ++i) {
        r *= static_cast<double>(n - k + i) / static_cast<double>(i);
    }
    return r;
}

std::vector<double> poly_pow(const std::vector<double>& p, int n) {
    if (n < 0) {
        throw std::invalid_argument("poly_pow: negative exponent unsupported");
    }
    if (n == 0) {
        return {1.0};
    }
    auto base = strip(p);
    if (base.empty()) {
        return {0.0};
    }
    std::vector<double> result = {1.0};
    std::vector<double> cur = base;
    int exp = n;
    while (exp > 0) {
        if (exp & 1) {
            result = poly_mul(result, cur);
        }
        exp >>= 1;
        if (exp > 0) {
            cur = poly_mul(cur, cur);
        }
    }
    return strip(result);
}

std::vector<double> poly_monic(const std::vector<double>& p) {
    auto P = strip(p);
    if (P.empty()) {
        return {0.0};
    }
    const double lc = P.back();
    if (std::abs(lc) < 1e-14) {
        return P;
    }
    for (auto& v : P) {
        v /= lc;
    }
    return strip(P);
}

std::vector<double> poly_reverse(const std::vector<double>& p) {
    auto P = strip(p);
    if (P.empty()) {
        return {0.0};
    }
    std::reverse(P.begin(), P.end());
    return strip(P);
}

std::vector<double> poly_shift(const std::vector<double>& p, double a) {
    auto P = strip(p);
    if (P.empty()) {
        return {0.0};
    }
    const size_t deg = P.size() - 1;
    std::vector<double> out(deg + 1, 0.0);
    for (size_t j = 0; j <= deg; ++j) {
        const double cj = P[j];
        for (size_t k = 0; k <= j; ++k) {
            const double sign = ((j - k) % 2 == 0) ? 1.0 : -1.0;
            out[k] += cj * binom_double(static_cast<int>(j), static_cast<int>(k)) *
                      sign * std::pow(a, static_cast<double>(j - k));
        }
    }
    return strip(out);
}

std::vector<double> poly_scale(const std::vector<double>& p, double a) {
    auto P = strip(p);
    if (P.empty()) {
        return {0.0};
    }
    double scale = 1.0;
    for (size_t i = 0; i < P.size(); ++i) {
        P[i] *= scale;
        scale *= a;
    }
    return strip(P);
}

std::vector<double> poly_lcm(const std::vector<double>& a,
                               const std::vector<double>& b) {
    auto A = strip(a), B = strip(b);
    if (A.empty() || (A.size() == 1 && std::abs(A[0]) < 1e-14)) {
        return {0.0};
    }
    if (B.empty() || (B.size() == 1 && std::abs(B[0]) < 1e-14)) {
        return {0.0};
    }
    const auto g = poly_gcd(A, B);
    const auto prod = poly_mul(A, B);
    const auto quot = poly_div_quot(prod, g);
    return poly_monic(quot);
}

ColMatrix<double> poly_sylvester(const std::vector<double>& p,
                                  const std::vector<double>& q) {
    auto P = strip(p);
    auto Q = strip(q);
    if (P.empty() || (P.size() == 1 && std::abs(P[0]) < 1e-14)) {
        P = {0.0};
    }
    if (Q.empty() || (Q.size() == 1 && std::abs(Q[0]) < 1e-14)) {
        Q = {0.0};
    }
    const size_t m = (P.size() > 1 || std::abs(P[0]) > 1e-14) ? P.size() - 1 : 0;
    const size_t n = (Q.size() > 1 || std::abs(Q[0]) > 1e-14) ? Q.size() - 1 : 0;
    const size_t sz = m + n;
    if (sz == 0) {
        return ColMatrix<double>(1, 1, 0.0);
    }
    ColMatrix<double> S(sz, sz, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k <= m; ++k) {
            const size_t col = i + k;
            if (col < sz) {
                S(i, col) = P[m - k];
            }
        }
    }
    for (size_t j = 0; j < m; ++j) {
        const size_t row = n + j;
        for (size_t k = 0; k <= n; ++k) {
            const size_t col = j + k;
            if (col < sz) {
                S(row, col) = Q[n - k];
            }
        }
    }
    return S;
}

double poly_resultant(const std::vector<double>& p,
                       const std::vector<double>& q) {
    const auto S = poly_sylvester(p, q);
    const auto d = det(S);
    if (!d) {
        // Singular Sylvester matrix <=> common root <=> resultant is zero
        return 0.0;
    }
    return *d;
}

double poly_discriminant(const std::vector<double>& p) {
    auto P = strip(p);
    if (P.size() <= 1) {
        return 0.0;
    }
    const int deg = static_cast<int>(P.size()) - 1;
    const double an = P.back();
    if (std::abs(an) < 1e-14) {
        return 0.0;
    }
    const auto dp = poly_deriv(P);
    const double res = poly_resultant(P, dp);
    const double sign = ((deg * (deg - 1) / 2) % 2 == 0) ? 1.0 : -1.0;
    return sign * res / an;
}

std::vector<double> poly_squarefree(const std::vector<double>& p) {
    auto P = strip(p);
    if (P.size() <= 1) {
        return P.empty() ? std::vector<double>{0.0} : P;
    }
    const auto dp = poly_deriv(P);
    const auto g = poly_gcd(P, dp);
    if (g.size() == 1 && std::abs(g[0] - 1.0) < 1e-10) {
        return poly_monic(P);
    }
    const auto quot = poly_div_quot(P, g);
    return poly_monic(quot);
}

double bernstein(int n, int i, double x) {
    if (n < 0 || i < 0 || i > n) {
        return 0.0;
    }
    return binom_double(n, i) * std::pow(x, i) * std::pow(1.0 - x, n - i);
}

std::vector<std::complex<double>> poly_roots(const std::vector<double>& coeffs) {
    auto p = strip(coeffs);
    int n = static_cast<int>(p.size()) - 1;
    if (n <= 0) return {};
    if (n == 1) return {{-p[0] / p[1], 0.0}};

    // Companion matrix eigenvalues via QR iteration (Aberth method approximation)
    // Use DKW (Durand-Kerner-Weierstrass) method
    std::vector<std::complex<double>> roots(static_cast<size_t>(n));
    const double PI = 3.14159265358979323846;
    double scale = std::pow(std::abs(p[0] / p.back()), 1.0 / n);
    for (int i = 0; i < n; ++i) {
        double theta = 2.0 * PI * i / n + 0.1;
        roots[static_cast<size_t>(i)] =
            std::polar(0.5 + scale, theta);
    }

    const auto eval_poly = [&](std::complex<double> x) {
        std::complex<double> v = p.back();
        for (int k = static_cast<int>(p.size()) - 2; k >= 0; --k)
            v = v * x + p[static_cast<size_t>(k)];
        return v;
    };

    for (int iter = 0; iter < 200; ++iter) {
        bool done = true;
        for (int i = 0; i < n; ++i) {
            std::complex<double> fi = eval_poly(roots[static_cast<size_t>(i)]);
            std::complex<double> denom = {1.0, 0.0};
            for (int j = 0; j < n; ++j) {
                if (j != i)
                    denom *= (roots[static_cast<size_t>(i)] -
                              roots[static_cast<size_t>(j)]);
            }
            if (std::abs(denom) < 1e-30) continue;
            std::complex<double> delta = fi / (p.back() * denom);
            roots[static_cast<size_t>(i)] -= delta;
            if (std::abs(delta) > 1e-10) done = false;
        }
        if (done) break;
    }
    return roots;
}

std::vector<double> poly_fit(const std::vector<double>& xs,
                              const std::vector<double>& ys, int degree) {
    // Vandermonde least-squares fit
    const int m = static_cast<int>(xs.size());
    const int n = degree + 1;
    // Build V (m x n)
    std::vector<std::vector<double>> V(
        static_cast<size_t>(m),
        std::vector<double>(static_cast<size_t>(n), 0.0));
    for (int i = 0; i < m; ++i) {
        double xp = 1.0;
        for (int j = 0; j < n; ++j) {
            V[static_cast<size_t>(i)][static_cast<size_t>(j)] = xp;
            xp *= xs[static_cast<size_t>(i)];
        }
    }
    // Normal equations V^T V c = V^T y
    std::vector<std::vector<double>> VtV(
        static_cast<size_t>(n),
        std::vector<double>(static_cast<size_t>(n), 0.0));
    std::vector<double> Vty(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Vty[static_cast<size_t>(j)] +=
                V[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                ys[static_cast<size_t>(i)];
            for (int k = 0; k < n; ++k)
                VtV[static_cast<size_t>(j)][static_cast<size_t>(k)] +=
                    V[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                    V[static_cast<size_t>(i)][static_cast<size_t>(k)];
        }
    }
    // Gaussian elimination
    auto A = VtV;
    auto b = Vty;
    for (int col = 0; col < n; ++col) {
        // Pivot
        int pivot = col;
        for (int row = col + 1; row < n; ++row)
            if (std::abs(A[static_cast<size_t>(row)][static_cast<size_t>(col)]) >
                std::abs(A[static_cast<size_t>(pivot)][static_cast<size_t>(col)]))
                pivot = row;
        std::swap(A[static_cast<size_t>(col)], A[static_cast<size_t>(pivot)]);
        std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot)]);
        double diag = A[static_cast<size_t>(col)][static_cast<size_t>(col)];
        if (std::abs(diag) < 1e-14) continue;
        for (int row = col + 1; row < n; ++row) {
            double factor = A[static_cast<size_t>(row)][static_cast<size_t>(col)] / diag;
            for (int k = col; k < n; ++k)
                A[static_cast<size_t>(row)][static_cast<size_t>(k)] -=
                    factor * A[static_cast<size_t>(col)][static_cast<size_t>(k)];
            b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
        }
    }
    // Back-substitution
    std::vector<double> c(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j)
            sum -= A[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                   c[static_cast<size_t>(j)];
        double diag = A[static_cast<size_t>(i)][static_cast<size_t>(i)];
        c[static_cast<size_t>(i)] = (std::abs(diag) > 1e-14) ? sum / diag : 0.0;
    }
    return c;
}

std::vector<double> poly_lagrange(const std::vector<double>& xs,
                                   const std::vector<double>& ys) {
    return poly_fit(xs, ys, static_cast<int>(xs.size()) - 1);
}

namespace {

// Expands the Newton form f[0,0] + f[0,1]*(x-nodes[0]) + f[0,2]*(x-nodes[0])*(x-nodes[1])
// + ... into flat ascending-power coefficients, given the top row f[0,k] of a
// divided-difference table and the node array it was built over.
std::vector<double> newton_form_to_coeffs(const std::vector<double>& top_row,
                                           const std::vector<double>& nodes) {
    std::vector<double> result = {top_row[0]};
    std::vector<double> basis = {1.0};
    for (size_t k = 1; k < top_row.size(); ++k) {
        basis = poly_mul(basis, {-nodes[k - 1], 1.0});
        std::vector<double> term(basis.size());
        for (size_t i = 0; i < basis.size(); ++i) {
            term[i] = top_row[k] * basis[i];
        }
        result = poly_add(result, term);
    }
    return result;
}

// True if any two entries of xs coincide within eps (invalid for interpolation
// with distinct nodes; division by zero in divided differences otherwise).
bool has_duplicate_nodes(const std::vector<double>& xs, double eps = 1e-12) {
    for (size_t i = 0; i < xs.size(); ++i) {
        for (size_t j = i + 1; j < xs.size(); ++j) {
            if (std::abs(xs[i] - xs[j]) < eps) return true;
        }
    }
    return false;
}

} // namespace

std::vector<double> interp_newton(const std::vector<double>& xs,
                                   const std::vector<double>& ys) {
    if (xs.size() != ys.size()) return {};
    const size_t n = xs.size();
    if (n == 0) return {};
    if (n == 1) return {ys[0]};
    if (has_duplicate_nodes(xs)) return {};

    // table[i][k] = f[x_i, ..., x_{i+k}]
    std::vector<std::vector<double>> table(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) table[i][0] = ys[i];
    for (size_t k = 1; k < n; ++k) {
        for (size_t i = 0; i + k < n; ++i) {
            const double denom = xs[i + k] - xs[i];
            if (std::abs(denom) < 1e-12) return {};
            table[i][k] = (table[i + 1][k - 1] - table[i][k - 1]) / denom;
        }
    }

    std::vector<double> top_row(n);
    for (size_t k = 0; k < n; ++k) top_row[k] = table[0][k];
    return newton_form_to_coeffs(top_row, xs);
}

std::vector<double> interp_hermite(const std::vector<double>& xs,
                                    const std::vector<double>& ys,
                                    const std::vector<double>& dys) {
    if (xs.size() != ys.size() || xs.size() != dys.size()) return {};
    const size_t n = xs.size();
    if (n == 0) return {};
    if (n == 1) return {ys[0] - dys[0] * xs[0], dys[0]};
    if (has_duplicate_nodes(xs)) return {};

    const size_t m = 2 * n;
    std::vector<double> z(m), q(m);
    for (size_t i = 0; i < n; ++i) {
        z[2 * i] = xs[i];
        z[2 * i + 1] = xs[i];
        q[2 * i] = ys[i];
        q[2 * i + 1] = ys[i];
    }

    std::vector<std::vector<double>> table(m, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < m; ++i) table[i][0] = q[i];

    // First-level differences: adjacent doubled entries (z[2i] == z[2i+1]) use
    // the derivative directly instead of dividing by zero.
    for (size_t i = 0; i + 1 < m; ++i) {
        const double denom = z[i + 1] - z[i];
        if (std::abs(denom) < 1e-12) {
            table[i][1] = dys[i / 2];
        } else {
            table[i][1] = (table[i + 1][0] - table[i][0]) / denom;
        }
    }
    // Higher-level differences never hit a zero gap once xs are distinct.
    for (size_t k = 2; k < m; ++k) {
        for (size_t i = 0; i + k < m; ++i) {
            const double denom = z[i + k] - z[i];
            if (std::abs(denom) < 1e-12) return {};
            table[i][k] = (table[i + 1][k - 1] - table[i][k - 1]) / denom;
        }
    }

    std::vector<double> top_row(m);
    for (size_t k = 0; k < m; ++k) top_row[k] = table[0][k];
    return newton_form_to_coeffs(top_row, z);
}

double poly_cheb_eval(const std::vector<double>& cheb_coeffs, double x) {
    // Clenshaw's algorithm for sum c_k T_k(x)
    double b1 = 0.0, b2 = 0.0;
    for (int i = static_cast<int>(cheb_coeffs.size()) - 1; i >= 1; --i) {
        double b0 = 2.0 * x * b1 - b2 + cheb_coeffs[static_cast<size_t>(i)];
        b2 = b1; b1 = b0;
    }
    return x * b1 - b2 + (cheb_coeffs.empty() ? 0.0 : cheb_coeffs[0]);
}

std::vector<double> poly_cheb_expand(std::function<double(double)> f, int n,
                                      double a, double b) {
    if (n < 0 || !f) return {};

    const double PI = 3.14159265358979323846;
    const int npts = n + 1;

    // Chebyshev points of the first kind on [-1, 1], mapped onto [a, b].
    std::vector<double> nodes(static_cast<size_t>(npts));
    std::vector<double> samples(static_cast<size_t>(npts));
    const double mid = 0.5 * (a + b);
    const double half = 0.5 * (b - a);
    for (int k = 0; k < npts; ++k) {
        double xk = std::cos((2.0 * k + 1.0) * PI / (2.0 * npts));
        nodes[static_cast<size_t>(k)] = xk;
        samples[static_cast<size_t>(k)] = f(mid + half * xk);
    }

    std::vector<double> coeffs(static_cast<size_t>(npts));
    for (int j = 0; j < npts; ++j) {
        double sum = 0.0;
        for (int k = 0; k < npts; ++k) {
            double xk = nodes[static_cast<size_t>(k)];
            double Tj = std::cos(j * std::acos(xk));
            sum += samples[static_cast<size_t>(k)] * Tj;
        }
        coeffs[static_cast<size_t>(j)] = (2.0 / npts) * sum;
    }
    // Match poly_cheb_eval's convention, which evaluates the series as
    // c_0 + sum_{j=1}^n c_j*T_j(x) (no implicit halving of coeffs[0]).
    coeffs[0] *= 0.5;
    return coeffs;
}

int poly_root_count(const std::vector<double>& p, double a, double b) {
    // Sturm's theorem: count sign changes of Sturm sequence at a and b
    auto build_sturm = [&]() {
        std::vector<std::vector<double>> seq;
        seq.push_back(strip(p));
        seq.push_back(poly_deriv(p));
        while (seq.back().size() > 1 ||
               std::abs(seq.back()[0]) > 1e-14) {
            auto r = poly_mod(seq[seq.size() - 2], seq.back());
            for (auto& v : r) v = -v;
            r = strip(r);
            if (r.empty()) r = {0.0};
            seq.push_back(r);
            if (seq.back().size() == 1 &&
                std::abs(seq.back()[0]) < 1e-14) break;
        }
        return seq;
    };

    const auto sign_changes = [](const std::vector<std::vector<double>>& seq,
                                  double x) {
        int changes = 0;
        double prev_sign = 0.0;
        for (auto& poly_k : seq) {
            double val = poly_k.back();
            for (int k = static_cast<int>(poly_k.size()) - 2; k >= 0; --k)
                val = val * x + poly_k[static_cast<size_t>(k)];
            double s = (val > 1e-14) ? 1.0 : ((val < -1e-14) ? -1.0 : 0.0);
            if (s != 0.0 && prev_sign != 0.0 && s != prev_sign) ++changes;
            if (s != 0.0) prev_sign = s;
        }
        return changes;
    };

    auto seq = build_sturm();
    return sign_changes(seq, a) - sign_changes(seq, b);
}

namespace {

// Upper bound on |a_0| / |a_n| (the rounded integer constant/leading
// coefficients) that we're willing to hand to numthy::divisors, which does
// O(sqrt(n)) trial division. 1e7 keeps that search well under a millisecond
// while comfortably covering any hand-constructed test polynomial; anything
// larger is rejected with a clear Result error instead of silently taking a
// long time.
constexpr double kRationalRootSafetyLimit = 1.0e7;

// Rounds every coefficient to the nearest integer; fails (returns false) if
// any coefficient is not within `tol` of an integer.
bool coeffs_to_integers(const std::vector<double>& p, std::vector<double>& out, double tol) {
    out.resize(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        const double r = std::round(p[i]);
        if (std::abs(p[i] - r) > tol) return false;
        out[i] = r;
    }
    return true;
}

// All reduced p/q candidates (q > 0) from the divisors of a0 and an, with
// both signs of the numerator.
std::vector<std::pair<int64_t, int64_t>> rational_root_candidates(uint64_t a0, uint64_t an) {
    std::set<std::pair<int64_t, int64_t>> uniq;
    const auto d0 = numthy::divisors(a0);
    const auto dn = numthy::divisors(an);
    for (uint64_t d1 : d0) {
        for (uint64_t d2 : dn) {
            int64_t num = static_cast<int64_t>(d1);
            int64_t den = static_cast<int64_t>(d2);
            const int64_t g = std::gcd(num, den);
            num /= g;
            den /= g;
            uniq.insert({num, den});
            uniq.insert({-num, den});
        }
    }
    return {uniq.begin(), uniq.end()};
}

// Evaluates the (integer-coefficient) poly at num/den and checks it's near
// zero relative to the polynomial's coefficient scale. This is a fast
// pre-filter; the actual accept/reject decision is made by checking that
// synthetic division leaves (near-)zero remainder, below.
bool looks_like_root(const std::vector<double>& p, int64_t num, int64_t den, double tol) {
    const double val = poly_eval(p, static_cast<double>(num) / static_cast<double>(den))[0];
    double scale = 1.0;
    for (double c : p) scale = std::max(scale, std::abs(c));
    return std::abs(val) <= tol * scale;
}

struct RationalRootState {
    std::vector<std::pair<int64_t, int64_t>> roots;
    std::vector<double> remainder;
};

Result<RationalRootState> rational_roots_core(const std::vector<double>& p_in, double tol) {
    std::vector<double> ip;
    if (!coeffs_to_integers(p_in, ip, tol)) {
        return std::unexpected(Error{DomainError{
            "poly_rational_roots",
            "coefficients are not within tolerance of integers; scale/clear "
            "denominators before calling"}});
    }
    auto p = strip(ip);
    if (p.empty() || (p.size() == 1 && std::abs(p[0]) < 1e-9)) {
        return std::unexpected(Error{DomainError{
            "poly_rational_roots", "zero polynomial has infinitely many roots"}});
    }

    RationalRootState state;
    if (p.size() == 1) {
        // Nonzero constant: no roots, nothing to factor.
        state.remainder = p;
        return state;
    }

    std::vector<double> rem = p;
    while (true) {
        rem = strip(rem);
        if (rem.size() <= 1) break;

        // x = 0 is a root iff the constant term is zero; handle separately
        // since 0 has no p/q divisor representation.
        if (std::abs(rem.front()) < 1e-9) {
            state.roots.push_back({0, 1});
            rem.erase(rem.begin());
            continue;
        }

        const double a0d = std::abs(rem.front());
        const double and_ = std::abs(rem.back());
        if (a0d > kRationalRootSafetyLimit || and_ > kRationalRootSafetyLimit) {
            return std::unexpected(Error{DomainError{
                "poly_rational_roots",
                "coefficient magnitude exceeds safety limit for divisor search"}});
        }
        const uint64_t a0 = static_cast<uint64_t>(std::llround(a0d));
        const uint64_t an = static_cast<uint64_t>(std::llround(and_));

        bool found = false;
        for (const auto& [num, den] : rational_root_candidates(a0, an)) {
            if (!looks_like_root(rem, num, den, 1e-7)) continue;

            const std::vector<double> divisor = {static_cast<double>(-num),
                                                   static_cast<double>(den)};
            auto q = poly_div_quot(rem, divisor);
            auto r = poly_mod(rem, divisor);
            const bool rem_is_zero =
                r.empty() || (r.size() == 1 && std::abs(r[0]) <= 1e-6 * and_);
            if (!rem_is_zero) continue; // pre-filter false positive; keep scanning

            for (auto& c : q) c = std::round(c); // exact by Gauss's lemma; clean fp noise
            rem = strip(q);
            state.roots.push_back({num, den});
            found = true;
            break;
        }
        if (!found) break;
    }

    state.remainder = strip(rem);
    return state;
}

} // namespace

Result<std::vector<std::pair<int64_t, int64_t>>> poly_rational_roots(
    const std::vector<double>& p, double tol) {
    auto res = rational_roots_core(p, tol);
    if (!res) return std::unexpected(res.error());
    return res->roots;
}

Result<RationalFactorization> poly_factor_rational(const std::vector<double>& p, double tol) {
    auto res = rational_roots_core(p, tol);
    if (!res) return std::unexpected(res.error());
    RationalFactorization out;
    out.linear_roots = std::move(res->roots);
    out.remainder = std::move(res->remainder);
    return out;
}

namespace {

// Poles/quadratic factors found via poly_roots are merged when clustering
// if within this relative tolerance of each other -- a numerical
// root-finder applied to a polynomial with an exact repeated root of
// multiplicity k will not return k EXACTLY equal roots in floating point,
// so nearby roots must be merged rather than compared for exact equality.
// This is also used (relative to the real part) as the threshold below
// which a root's imaginary part is treated as zero (a real root) rather
// than a genuine complex pole. 1e-4 rather than a tighter value like 1e-6
// is deliberate: poly_roots uses Durand-Kerner-Weierstrass iteration, whose
// convergence rate degrades for higher-multiplicity roots (a triple real
// root was empirically observed to converge to only ~1e-5 accuracy, with a
// similarly-sized spurious imaginary part), so a tighter tolerance would
// misclassify those as complex or fail to merge them into one pole. 1e-4
// is still many orders of magnitude tighter than the well-separated root
// gaps used throughout this module's test suite.
constexpr double kPartialFractionRootTol = 1e-4;

bool pf_near(double a, double b, double tol) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) <= tol * scale;
}

struct PFRealPole {
    double r;
    int mult;
};
struct PFQuadFactor {
    double p, q;
    int mult;
};

// Groups the roots of D(x) (as returned by poly_roots, one complex root per
// degree, with algebraic multiplicity) into distinct real poles and
// distinct complex-conjugate-pair quadratic factors x^2+p*x+q, each with a
// multiplicity, by clustering within kPartialFractionRootTol.
void pf_group_roots(const std::vector<std::complex<double>>& roots,
                     std::vector<PFRealPole>& real_poles,
                     std::vector<PFQuadFactor>& quad_factors) {
    std::vector<double> real_vals;
    std::vector<std::complex<double>> complex_vals; // only the +imag half of each pair

    for (const auto& rt : roots) {
        const double im_tol =
            std::max(kPartialFractionRootTol, kPartialFractionRootTol * std::abs(rt.real()));
        if (std::abs(rt.imag()) <= im_tol) {
            real_vals.push_back(rt.real());
        } else if (rt.imag() > 0.0) {
            complex_vals.push_back(rt);
        }
        // The rt.imag() < 0 half of each pair is the conjugate of an entry
        // already captured (or about to be captured) above.
    }

    std::vector<bool> used(real_vals.size(), false);
    for (size_t i = 0; i < real_vals.size(); ++i) {
        if (used[i]) continue;
        int count = 1;
        used[i] = true;
        double sum = real_vals[i];
        for (size_t j = i + 1; j < real_vals.size(); ++j) {
            if (!used[j] && pf_near(real_vals[i], real_vals[j], kPartialFractionRootTol)) {
                used[j] = true;
                sum += real_vals[j];
                ++count;
            }
        }
        real_poles.push_back({sum / count, count});
    }

    std::vector<bool> used_c(complex_vals.size(), false);
    for (size_t i = 0; i < complex_vals.size(); ++i) {
        if (used_c[i]) continue;
        int count = 1;
        used_c[i] = true;
        double re_sum = complex_vals[i].real();
        double im_sum = complex_vals[i].imag();
        for (size_t j = i + 1; j < complex_vals.size(); ++j) {
            if (!used_c[j] &&
                pf_near(complex_vals[i].real(), complex_vals[j].real(), kPartialFractionRootTol) &&
                pf_near(complex_vals[i].imag(), complex_vals[j].imag(), kPartialFractionRootTol)) {
                used_c[j] = true;
                re_sum += complex_vals[j].real();
                im_sum += complex_vals[j].imag();
                ++count;
            }
        }
        const double re = re_sum / count;
        const double im = im_sum / count;
        quad_factors.push_back({-2.0 * re, re * re + im * im, count});
    }
}

// Small dense linear solver (Gaussian elimination with partial pivoting),
// self-contained here rather than pulled from another module, matching the
// approach poly_fit already uses above for its normal-equations solve.
std::vector<double> pf_solve_linear(std::vector<std::vector<double>> A,
                                     std::vector<double> b) {
    const int n = static_cast<int>(b.size());
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(A[static_cast<size_t>(row)][static_cast<size_t>(col)]) >
                std::abs(A[static_cast<size_t>(pivot)][static_cast<size_t>(col)])) {
                pivot = row;
            }
        }
        std::swap(A[static_cast<size_t>(col)], A[static_cast<size_t>(pivot)]);
        std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot)]);
        const double diag = A[static_cast<size_t>(col)][static_cast<size_t>(col)];
        if (std::abs(diag) < 1e-14) continue;
        for (int row = col + 1; row < n; ++row) {
            const double factor = A[static_cast<size_t>(row)][static_cast<size_t>(col)] / diag;
            if (factor == 0.0) continue;
            for (int k = col; k < n; ++k) {
                A[static_cast<size_t>(row)][static_cast<size_t>(k)] -=
                    factor * A[static_cast<size_t>(col)][static_cast<size_t>(k)];
            }
            b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
        }
    }
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[static_cast<size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[static_cast<size_t>(i)][static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
        }
        const double diag = A[static_cast<size_t>(i)][static_cast<size_t>(i)];
        x[static_cast<size_t>(i)] = (std::abs(diag) > 1e-14) ? sum / diag : 0.0;
    }
    return x;
}

} // namespace

PartialFractionResult poly_partial_fractions(const std::vector<double>& numerator,
                                              const std::vector<double>& denominator) {
    PartialFractionResult out;
    auto D = strip(denominator);
    if (D.empty() || (D.size() == 1 && std::abs(D[0]) < 1e-14)) {
        return out; // zero/empty denominator: no finite decomposition
    }
    const int n = static_cast<int>(D.size()) - 1; // degree of D

    if (n == 0) {
        // D(x) is a nonzero constant: N(x)/D(x) is itself a polynomial;
        // there is no proper-fraction remainder to decompose into terms.
        std::vector<double> q(numerator.size(), 0.0);
        for (size_t i = 0; i < numerator.size(); ++i) q[i] = numerator[i] / D[0];
        out.quotient = strip(q.empty() ? std::vector<double>{0.0} : q);
        return out;
    }

    out.quotient = poly_div_quot(numerator, D);
    // poly_div_quot returns {0.0} when deg(numerator) < deg(D); treat that
    // as "no polynomial part" rather than a spurious zero quotient term.
    if (out.quotient.size() == 1 && std::abs(out.quotient[0]) < 1e-14) {
        out.quotient.clear();
    }
    const std::vector<double> Nr = strip(poly_mod(numerator, D));

    const auto roots = poly_roots(D);
    std::vector<PFRealPole> real_poles;
    std::vector<PFQuadFactor> quad_factors;
    pf_group_roots(roots, real_poles, quad_factors);

    // The declared unknowns must total deg(D); if root grouping produced a
    // mismatch (e.g. a pathological/near-degenerate denominator), bail out
    // with just the quotient rather than returning an incorrect decomposition.
    int total_unknowns = 0;
    for (const auto& rp : real_poles) total_unknowns += rp.mult;
    for (const auto& qf : quad_factors) total_unknowns += 2 * qf.mult;
    if (total_unknowns != n) {
        return out;
    }

    const double lead = D.back();

    // One monic primitive factor polynomial per grouped pole (ascending powers).
    std::vector<std::vector<double>> primitives;
    std::vector<int> mults;
    std::vector<bool> is_quad;
    for (const auto& rp : real_poles) {
        primitives.push_back({-rp.r, 1.0});
        mults.push_back(rp.mult);
        is_quad.push_back(false);
    }
    for (const auto& qf : quad_factors) {
        primitives.push_back({qf.q, qf.p, 1.0});
        mults.push_back(qf.mult);
        is_quad.push_back(true);
    }
    const size_t num_factors = primitives.size();

    // D_other[i] = lead * product_{j != i} primitives[j]^mults[j], i.e.
    // D(x) with factor i entirely divided out (a polynomial, since factor
    // i's full power exactly divides D by construction).
    std::vector<std::vector<double>> D_other(num_factors);
    for (size_t i = 0; i < num_factors; ++i) {
        std::vector<double> prod = {lead};
        for (size_t j = 0; j < num_factors; ++j) {
            if (j == i) continue;
            std::vector<double> fj = {1.0};
            for (int p = 0; p < mults[j]; ++p) fj = poly_mul(fj, primitives[j]);
            prod = poly_mul(prod, fj);
        }
        D_other[i] = prod;
    }

    // One basis polynomial per unknown coefficient (A_d for real poles;
    // B_d then C_d for quadratic factors), in the exact order the solved
    // unknowns are assigned back to PartialFractionTerm entries below.
    // For pole/factor i with multiplicity K, the d-th term's basis is
    // D_other[i](x) * primitives[i](x)^(K-d) (times an extra factor of x
    // for the B_d coefficient), since D(x)/primitives[i]^d ==
    // D_other[i](x) * primitives[i](x)^(K-d).
    std::vector<std::vector<double>> basis;
    for (size_t i = 0; i < num_factors; ++i) {
        const int K = mults[i];
        for (int d = 1; d <= K; ++d) {
            std::vector<double> rest = {1.0};
            for (int p = 0; p < K - d; ++p) rest = poly_mul(rest, primitives[i]);
            std::vector<double> base = poly_mul(D_other[i], rest);
            if (!is_quad[i]) {
                basis.push_back(base); // A_d
            } else {
                basis.push_back(poly_mul(base, {0.0, 1.0})); // B_d (times x)
                basis.push_back(base);                        // C_d
            }
        }
    }

    // Linear system: sum_m unknown_m * basis_m(x) == Nr(x), matched by
    // power of x (rows 0..n-1); solved via Gaussian elimination.
    std::vector<std::vector<double>> Amat(static_cast<size_t>(n),
                                           std::vector<double>(static_cast<size_t>(n), 0.0));
    for (size_t m = 0; m < basis.size(); ++m) {
        for (size_t row = 0; row < basis[m].size() && row < static_cast<size_t>(n); ++row) {
            Amat[row][m] = basis[m][row];
        }
    }
    std::vector<double> rhs(static_cast<size_t>(n), 0.0);
    for (size_t row = 0; row < Nr.size() && row < static_cast<size_t>(n); ++row) {
        rhs[row] = Nr[row];
    }

    const auto sol = pf_solve_linear(Amat, rhs);

    size_t idx = 0;
    for (const auto& rp : real_poles) {
        for (int d = 1; d <= rp.mult; ++d) {
            PartialFractionTerm term;
            term.is_quadratic = false;
            term.r = rp.r;
            term.k = d;
            term.A = sol[idx++];
            out.terms.push_back(term);
        }
    }
    for (const auto& qf : quad_factors) {
        for (int d = 1; d <= qf.mult; ++d) {
            PartialFractionTerm term;
            term.is_quadratic = true;
            term.p = qf.p;
            term.q = qf.q;
            term.k = d;
            term.B = sol[idx++];
            term.C = sol[idx++];
            out.terms.push_back(term);
        }
    }

    return out;
}

} // namespace poly
} // namespace ms
