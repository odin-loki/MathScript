#include "ms/poly/poly.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

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
    size_t n = A.size() - B.size() + 1;
    std::vector<double> q(n, 0.0);
    auto rem = A;
    for (size_t i = rem.size(); i-- > B.size() - 1;) {
        size_t pos = i - (B.size() - 1);
        double coef = rem[i] / B.back();
        q[pos] = coef;
        for (size_t j = 0; j < B.size(); ++j)
            rem[i - (B.size() - 1 - j)] -= coef * B[j];
    }
    return strip(q);
}

std::vector<double> poly_mod(const std::vector<double>& a,
                              const std::vector<double>& b) {
    auto A = strip(a), B = strip(b);
    if (A.size() < B.size()) return A;
    auto rem = A;
    for (size_t i = rem.size(); i-- > B.size() - 1;) {
        double coef = rem[i] / B.back();
        for (size_t j = 0; j < B.size(); ++j)
            rem[i - (B.size() - 1 - j)] -= coef * B[j];
    }
    rem.resize(B.size() - 1);
    return strip(rem);
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
    while (B.size() > 1 || std::abs(B[0]) > 1e-14) {
        auto R = poly_mod(A, B);
        R = strip(R);
        A = B;
        B = R;
        if (B.empty()) B = {0.0};
        if (B.size() == 1 && std::abs(B[0]) < 1e-14) break;
    }
    // Normalize
    if (!A.empty() && std::abs(A.back()) > 1e-14) {
        double lc = A.back();
        for (auto& v : A) v /= lc;
    }
    return strip(A);
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

double poly_cheb_eval(const std::vector<double>& cheb_coeffs, double x) {
    // Clenshaw's algorithm for sum c_k T_k(x)
    double b1 = 0.0, b2 = 0.0;
    for (int i = static_cast<int>(cheb_coeffs.size()) - 1; i >= 1; --i) {
        double b0 = 2.0 * x * b1 - b2 + cheb_coeffs[static_cast<size_t>(i)];
        b2 = b1; b1 = b0;
    }
    return x * b1 - b2 + (cheb_coeffs.empty() ? 0.0 : cheb_coeffs[0]);
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

} // namespace poly
} // namespace ms
