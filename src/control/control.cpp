#define _USE_MATH_DEFINES
#include "ms/control/control.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace control {

// ---- Helpers ----

// Matrix-vector multiplication (optional output buffer reuse)
static void matvec(const std::vector<std::vector<double>>& A,
                   const std::vector<double>& x,
                   std::vector<double>& y) {
    const int n = static_cast<int>(A.size());
    y.assign(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < static_cast<int>(x.size()); ++j)
            y[static_cast<size_t>(i)] += A[static_cast<size_t>(i)][static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
}

static std::vector<double> matvec(const std::vector<std::vector<double>>& A,
                                   const std::vector<double>& x) {
    std::vector<double> y;
    matvec(A, x, y);
    return y;
}

// Matrix-matrix multiply
static std::vector<std::vector<double>> matmul(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    int m = static_cast<int>(A.size());
    int k = static_cast<int>(B.size());
    int n = static_cast<int>(B[0].size());
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l)
                C[i][j] += A[i][l] * B[l][j];
    return C;
}

// Matrix transpose
static std::vector<std::vector<double>> transpose(
    const std::vector<std::vector<double>>& A) {
    int m = static_cast<int>(A.size());
    int n = static_cast<int>(A[0].size());
    std::vector<std::vector<double>> T(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

// Matrix add
static std::vector<std::vector<double>> matadd(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    int m = static_cast<int>(A.size());
    int n = static_cast<int>(A[0].size());
    auto C = A;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] += B[i][j];
    return C;
}

// Evaluate polynomial at complex s: poly[0]*s^n + ... + poly[n]
static std::complex<double> polyval(const std::vector<double>& p,
                                     std::complex<double> s) {
    std::complex<double> result = 0.0;
    for (double c : p) result = result * s + c;
    return result;
}

// Find roots of polynomial using companion matrix eigenvalues (simplified)
// Uses iterative approach for robustness
static std::vector<std::complex<double>> polyroots(const std::vector<double>& p) {
    int n = static_cast<int>(p.size()) - 1;
    if (n <= 0) return {};
    if (n == 1) return {std::complex<double>(-p[1] / p[0], 0.0)};
    if (n == 2) {
        double a = p[0], b = p[1], c = p[2];
        double disc = b * b - 4 * a * c;
        if (disc >= 0) {
            double sq = std::sqrt(disc);
            return {(-b + sq) / (2 * a), (-b - sq) / (2 * a)};
        } else {
            double re = -b / (2 * a);
            double im = std::sqrt(-disc) / (2 * a);
            return {{re, im}, {re, -im}};
        }
    }
    // Durand-Kerner-Weierstrass for higher degrees
    std::vector<std::complex<double>> roots(n);
    // Initialize
    std::complex<double> r(0.4, 0.9);
    std::complex<double> cur = 1.0;
    for (int i = 0; i < n; ++i) { roots[i] = cur; cur *= r; }
    // Iterate
    for (int iter = 0; iter < 200; ++iter) {
        bool converged = true;
        for (int i = 0; i < n; ++i) {
            std::complex<double> num = polyval(p, roots[i]) / p[0];
            std::complex<double> den = 1.0;
            for (int j = 0; j < n; ++j)
                if (j != i) den *= (roots[i] - roots[j]);
            if (std::abs(den) < 1e-30) continue;
            std::complex<double> delta = num / den;
            roots[i] -= delta;
            if (std::abs(delta) > 1e-12) converged = false;
        }
        if (converged) break;
    }
    return roots;
}

// StateSpace constructor
StateSpace::StateSpace(std::vector<std::vector<double>> a,
                       std::vector<std::vector<double>> b,
                       std::vector<std::vector<double>> c,
                       std::vector<std::vector<double>> d)
    : A(std::move(a)), B(std::move(b)), C(std::move(c)), D(std::move(d)) {
    n = static_cast<int>(A.size());
    m = static_cast<int>(B.empty() ? 0 : B[0].size());
    p = static_cast<int>(C.size());
}

// ---- Construction ----

TransferFunction tf(std::vector<double> num, std::vector<double> den) {
    return TransferFunction{std::move(num), std::move(den)};
}

StateSpace ss(std::vector<std::vector<double>> A,
              std::vector<std::vector<double>> B,
              std::vector<std::vector<double>> C,
              std::vector<std::vector<double>> D) {
    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D)};
}

static std::vector<double> polymul(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<double> c(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            c[i + j] += a[i] * b[j];
    return c;
}

static std::vector<double> polyadd(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    size_t sz = std::max(a.size(), b.size());
    std::vector<double> c(sz, 0.0);
    size_t off_a = sz - a.size(), off_b = sz - b.size();
    for (size_t i = 0; i < a.size(); ++i) c[i + off_a] += a[i];
    for (size_t i = 0; i < b.size(); ++i) c[i + off_b] += b[i];
    return c;
}

// Determinant of an n×n matrix whose entries are polynomials (descending coeffs).
static std::vector<double> poly_mat_det(
    const std::vector<std::vector<std::vector<double>>>& M) {
    const int n = static_cast<int>(M.size());
    if (n == 1) return M[0][0];
    if (n == 2) {
        auto ad = polymul(M[0][0], M[1][1]);
        auto bc = polymul(M[0][1], M[1][0]);
        std::vector<double> neg_bc(bc.size());
        for (size_t i = 0; i < bc.size(); ++i) neg_bc[i] = -bc[i];
        return polyadd(ad, neg_bc);
    }
    std::vector<double> det;
    for (int j = 0; j < n; ++j) {
        std::vector<std::vector<std::vector<double>>> minor(
            n - 1, std::vector<std::vector<double>>(n - 1));
        for (int r = 1; r < n; ++r) {
            int c = 0;
            for (int k = 0; k < n; ++k) {
                if (k == j) continue;
                minor[r - 1][c++] = M[r][k];
            }
        }
        auto term = polymul(M[0][j], poly_mat_det(minor));
        if (j % 2 == 1) {
            for (auto& c : term) c = -c;
        }
        det = polyadd(det, term);
    }
    return det;
}

// det(sI-A) with column col replaced by B — polynomial of degree n-1.
static std::vector<double> ss_col_det(const StateSpace& sys, int col) {
    const int n = sys.n;
    std::vector<std::vector<std::vector<double>>> M(
        n, std::vector<std::vector<double>>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j == col)
                M[i][j] = {sys.B[i][0]};
            else if (i == j)
                M[i][j] = {1.0, -sys.A[i][j]};
            else
                M[i][j] = {-sys.A[i][j]};
        }
    }
    return poly_mat_det(M);
}

TransferFunction tf2ss_tf(const StateSpace& sys) {
    // SISO: H(s) = C(sI - A)^{-1}B + D
    // num(s) = D*det(sI-A) + C*adj(sI-A)*B
    int n = sys.n;
    if (n == 0) {
        return TransferFunction{{sys.D[0][0]}, {1.0}};
    }
    // Characteristic polynomial of A via Faddeev-LeVerrier
    std::vector<double> char_poly(n + 1, 0.0);
    char_poly[0] = 1.0;
    std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) M[i][i] = 1.0; // Identity
    for (int k = 1; k <= n; ++k) {
        M = matmul(sys.A, M);
        double tr = 0.0;
        for (int i = 0; i < n; ++i) tr += M[i][i];
        char_poly[k] = -tr / k;
        for (int i = 0; i < n; ++i) M[i][i] += char_poly[k];
    }

    const double D = sys.D[0][0];
    std::vector<double> q;
    for (int i = 0; i < n; ++i) {
        auto col_det = ss_col_det(sys, i);
        if (std::abs(sys.C[0][i]) > 0.0) {
            std::vector<double> scaled(col_det.size());
            for (size_t k = 0; k < col_det.size(); ++k)
                scaled[k] = sys.C[0][i] * col_det[k];
            q = polyadd(q, scaled);
        }
    }

    std::vector<double> num(n + 1, 0.0);
    for (int i = 0; i <= n; ++i) num[i] = D * char_poly[i];
    const size_t off = static_cast<size_t>(n + 1) - q.size();
    for (size_t i = 0; i < q.size(); ++i) num[off + i] += q[i];
    return TransferFunction{std::move(num), std::move(char_poly)};
}

StateSpace tf2ss(const TransferFunction& sys) {
    int n = static_cast<int>(sys.den.size()) - 1;
    if (n <= 0) {
        double g = sys.num[0] / sys.den[0];
        return StateSpace{{{0.0}}, {{0.0}}, {{0.0}}, {{g}}};
    }
    // Controllable canonical form
    double a0 = sys.den[0];
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    // Last column: -den coefficients (normalised)
    for (int i = 0; i < n; ++i) {
        if (i + 1 < n) A[i][i + 1] = 1.0;
        A[n-1][i] = -sys.den[n - i] / a0;
    }
    std::vector<std::vector<double>> B(n, std::vector<double>(1, 0.0));
    B[n-1][0] = 1.0 / a0;
    // Build C from numerator (zero-pad or trim to match den degree)
    std::vector<double> num = sys.num;
    // Normalize numerator by den[0]
    for (auto& x : num) x /= a0;
    // Pad or trim to n elements
    while ((int)num.size() < n + 1) num.insert(num.begin(), 0.0);
    double D_val = num[0];
    num.erase(num.begin());
    // subtract D_val * den from num
    for (int i = 0; i < n; ++i) {
        num[i] -= D_val * sys.den[i + 1] / a0;
    }
    std::vector<std::vector<double>> C(1, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) C[0][n - 1 - i] = num[i];
    std::vector<std::vector<double>> D_mat = {{D_val}};
    return StateSpace{A, B, C, D_mat};
}

TransferFunction ss2tf(const StateSpace& sys) {
    return tf2ss_tf(sys);
}

// ---- Discretization helpers ----

static Matrix<double> to_ms_matrix(const std::vector<std::vector<double>>& M) {
    const int rows = static_cast<int>(M.size());
    const int cols = rows > 0 ? static_cast<int>(M[0].size()) : 0;
    Matrix<double> out(static_cast<size_t>(rows), static_cast<size_t>(cols), 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out(static_cast<size_t>(i), static_cast<size_t>(j)) = M[i][j];
    return out;
}

static std::vector<std::vector<double>> from_ms_matrix(const Matrix<double>& M) {
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[i][j] = M(static_cast<size_t>(i), static_cast<size_t>(j));
    return out;
}

static std::vector<std::vector<double>> identity_mat(int n) {
    auto I = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

static std::vector<std::vector<double>> mat_scale(
    const std::vector<std::vector<double>>& A, double s) {
    auto C = A;
    for (auto& row : C)
        for (auto& x : row) x *= s;
    return C;
}

static std::vector<std::vector<double>> mat_sub(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    auto C = A;
    for (size_t i = 0; i < C.size(); ++i)
        for (size_t j = 0; j < C[i].size(); ++j)
            C[i][j] -= B[i][j];
    return C;
}

static std::vector<std::vector<double>> mat_inv(
    const std::vector<std::vector<double>>& A) {
    const int n = static_cast<int>(A.size());
    auto Am = to_ms_matrix(A);
    auto I = eye<double>(static_cast<size_t>(n));
    auto inv = solve(Am, I);
    if (!inv)
        throw std::runtime_error("matrix inversion failed");
    return from_ms_matrix(*inv);
}

static Matrix<double> expm_scaled(const Matrix<double>& A) {
    const size_t n = A.rows();
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            norm = std::max(norm, std::abs(A(i, j)));
    int s = 0;
    while (norm > 0.5) {
        norm *= 0.5;
        ++s;
    }
    const double scale = std::pow(0.5, static_cast<double>(s));
    Matrix<double> M(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            M(i, j) = A(i, j) * scale;
    Matrix<double> result = eye<double>(n);
    Matrix<double> term = eye<double>(n);
    for (int k = 1; k <= 20; ++k) {
        auto next = matmul(term, M);
        if (!next)
            throw std::runtime_error("matmul failed");
        term = *next;
        const double inv_k = 1.0 / static_cast<double>(k);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                term(i, j) *= inv_k;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                result(i, j) += term(i, j);
    }
    for (int i = 0; i < s; ++i) {
        auto sq = matmul(result, result);
        if (!sq)
            throw std::runtime_error("matmul failed");
        result = *sq;
    }
    return result;
}

static double matrix_inf_norm(const std::vector<std::vector<double>>& A) {
    double n = 0.0;
    for (const auto& row : A)
        for (double x : row)
            n = std::max(n, std::abs(x));
    return n;
}

static std::vector<std::vector<double>> logm_series(
    const std::vector<std::vector<double>>& Ad, int terms = 40) {
    const int n = static_cast<int>(Ad.size());
    auto N = mat_sub(Ad, identity_mat(n));
    auto L = N;
    auto term = N;
    for (int k = 2; k <= terms; ++k) {
        term = matmul(term, N);
        const double coef = ((k % 2 == 0) ? -1.0 : 1.0) / static_cast<double>(k);
        L = matadd(L, mat_scale(term, coef));
    }
    return L;
}

static void c2d_zoh_ab(const std::vector<std::vector<double>>& A,
                       const std::vector<std::vector<double>>& B,
                       double Ts,
                       std::vector<std::vector<double>>& Ad,
                       std::vector<std::vector<double>>& Bd) {
    const int n = static_cast<int>(A.size());
    const int m = static_cast<int>(B[0].size());
    const int nm = n + m;
    Matrix<double> Mc(static_cast<size_t>(nm), static_cast<size_t>(nm), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            Mc(static_cast<size_t>(i), static_cast<size_t>(j)) = A[i][j] * Ts;
        for (int j = 0; j < m; ++j)
            Mc(static_cast<size_t>(i), static_cast<size_t>(n + j)) = B[i][j] * Ts;
    }
    const auto E = expm_scaled(Mc);
    Ad.assign(n, std::vector<double>(n, 0.0));
    Bd.assign(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            Ad[i][j] = E(static_cast<size_t>(i), static_cast<size_t>(j));
        for (int j = 0; j < m; ++j)
            Bd[i][j] = E(static_cast<size_t>(i), static_cast<size_t>(n + j));
    }
}

static void d2c_zoh_ab(const std::vector<std::vector<double>>& Ad,
                       const std::vector<std::vector<double>>& Bd,
                       double Ts,
                       std::vector<std::vector<double>>& A,
                       std::vector<std::vector<double>>& B) {
    const int n = static_cast<int>(Ad.size());
    auto Ad_minus_I = mat_sub(Ad, identity_mat(n));
    if (matrix_inf_norm(Ad_minus_I) < 1.0) {
        A = mat_scale(logm_series(Ad), 1.0 / Ts);
    } else {
        auto log_result = logm(to_ms_matrix(Ad));
        if (!log_result)
            throw std::runtime_error("logm failed in d2c ZOH");
        A = mat_scale(from_ms_matrix(*log_result), 1.0 / Ts);
    }

    if (matrix_inf_norm(Ad_minus_I) > 1e-12) {
        auto inv = mat_inv(Ad_minus_I);
        B = matmul(inv, matmul(A, Bd));
    } else {
        B = mat_scale(Bd, 1.0 / Ts);
    }
}

static void c2d_tustin(const std::vector<std::vector<double>>& A,
                       const std::vector<std::vector<double>>& B,
                       const std::vector<std::vector<double>>& C,
                       const std::vector<std::vector<double>>& D,
                       double Ts,
                       std::vector<std::vector<double>>& Ad,
                       std::vector<std::vector<double>>& Bd,
                       std::vector<std::vector<double>>& Cd,
                       std::vector<std::vector<double>>& Dd) {
    const int n = static_cast<int>(A.size());
    auto I = identity_mat(n);
    auto half = 0.5 * Ts;
    auto T = mat_sub(I, mat_scale(A, half));
    auto Tinv = mat_inv(T);
    Ad = matmul(Tinv, matadd(I, mat_scale(A, half)));
    Bd = matmul(Tinv, mat_scale(B, Ts));
    Cd = matmul(C, Tinv);
    auto CTinvB = matmul(Cd, B);
    Dd = D;
    for (int i = 0; i < static_cast<int>(D.size()); ++i)
        for (int j = 0; j < static_cast<int>(D[0].size()); ++j)
            Dd[i][j] = D[i][j] + half * CTinvB[i][j];
}

static void d2c_tustin(const std::vector<std::vector<double>>& Ad,
                       const std::vector<std::vector<double>>& Bd,
                       const std::vector<std::vector<double>>& Cd,
                       const std::vector<std::vector<double>>& Dd,
                       double Ts,
                       std::vector<std::vector<double>>& A,
                       std::vector<std::vector<double>>& B,
                       std::vector<std::vector<double>>& C,
                       std::vector<std::vector<double>>& D) {
    const int n = static_cast<int>(Ad.size());
    auto I = identity_mat(n);
    auto half = 0.5 * Ts;
    auto IplusAd = matadd(I, Ad);
    auto IplusAd_inv = mat_inv(IplusAd);
    A = mat_scale(matmul(IplusAd_inv, mat_sub(Ad, I)), 2.0 / Ts);
    auto T = mat_sub(I, mat_scale(A, half));
    B = mat_scale(matmul(T, Bd), 1.0 / Ts);
    C = matmul(Cd, T);
    auto CdB = matmul(Cd, B);
    D = Dd;
    for (int i = 0; i < static_cast<int>(D.size()); ++i)
        for (int j = 0; j < static_cast<int>(D[0].size()); ++j)
            D[i][j] = Dd[i][j] - half * CdB[i][j];
}

StateSpace c2d(const StateSpace& sys, double Ts, DiscretizationMethod method) {
    std::vector<std::vector<double>> Ad, Bd, Cd = sys.C, Dd = sys.D;
    switch (method) {
    case DiscretizationMethod::ZOH:
        c2d_zoh_ab(sys.A, sys.B, Ts, Ad, Bd);
        break;
    case DiscretizationMethod::Tustin:
        c2d_tustin(sys.A, sys.B, sys.C, sys.D, Ts, Ad, Bd, Cd, Dd);
        break;
    case DiscretizationMethod::Euler:
        Ad = matadd(identity_mat(sys.n), mat_scale(sys.A, Ts));
        Bd = mat_scale(sys.B, Ts);
        break;
    }
    return StateSpace{std::move(Ad), std::move(Bd), std::move(Cd), std::move(Dd)};
}

StateSpace d2c(const StateSpace& sys, double Ts, DiscretizationMethod method) {
    std::vector<std::vector<double>> A, B, C = sys.C, D = sys.D;
    switch (method) {
    case DiscretizationMethod::ZOH:
        d2c_zoh_ab(sys.A, sys.B, Ts, A, B);
        break;
    case DiscretizationMethod::Tustin:
        d2c_tustin(sys.A, sys.B, sys.C, sys.D, Ts, A, B, C, D);
        break;
    case DiscretizationMethod::Euler:
        A = mat_scale(mat_sub(sys.A, identity_mat(sys.n)), 1.0 / Ts);
        B = mat_scale(sys.B, 1.0 / Ts);
        break;
    }
    return StateSpace{std::move(A), std::move(B), std::move(C), std::move(D)};
}

TransferFunction c2d(const TransferFunction& sys, double Ts, DiscretizationMethod method) {
    return ss2tf(c2d(tf2ss(sys), Ts, method));
}

TransferFunction d2c(const TransferFunction& sys, double Ts, DiscretizationMethod method) {
    return ss2tf(d2c(tf2ss(sys), Ts, method));
}

// ---- Interconnections ----

TransferFunction series(const TransferFunction& g, const TransferFunction& h) {
    return TransferFunction{polymul(g.num, h.num), polymul(g.den, h.den)};
}

TransferFunction parallel(const TransferFunction& g, const TransferFunction& h) {
    return TransferFunction{polyadd(polymul(g.num, h.den), polymul(h.num, g.den)),
                            polymul(g.den, h.den)};
}

TransferFunction feedback(const TransferFunction& g, const TransferFunction& h, int sign) {
    // G / (1 - sign*G*H)
    auto num = polymul(g.num, h.den);
    auto denom_gh = polymul(g.num, h.num);
    auto denom_base = polymul(g.den, h.den);
    // den = g.den * h.den - sign * g.num * h.num
    size_t sz = std::max(denom_gh.size(), denom_base.size());
    std::vector<double> den(sz, 0.0);
    size_t off_base = sz - denom_base.size(), off_gh = sz - denom_gh.size();
    for (size_t i = 0; i < denom_base.size(); ++i) den[i + off_base] += denom_base[i];
    for (size_t i = 0; i < denom_gh.size(); ++i)   den[i + off_gh]   -= sign * denom_gh[i];
    return TransferFunction{num, den};
}

// ---- Analysis ----

std::vector<std::complex<double>> poles(const TransferFunction& sys) {
    return polyroots(sys.den);
}

std::vector<std::complex<double>> zeros(const TransferFunction& sys) {
    return polyroots(sys.num);
}

double dcgain(const TransferFunction& sys) {
    // H(0) = num(0) / den(0) = sum(num) / sum(den)
    double n = sys.num.back(), d = sys.den.back();
    return (std::abs(d) > 1e-30) ? n / d : 0.0;
}

bool is_stable(const TransferFunction& sys) {
    for (auto& p : poles(sys))
        if (p.real() >= 0.0) return false;
    return true;
}

// ---- Frequency response ----

static std::complex<double> freqresp(const TransferFunction& sys,
                                      double w) {
    auto s = std::complex<double>(0.0, w);
    return polyval(sys.num, s) / polyval(sys.den, s);
}

BodeData bode(const TransferFunction& sys,
              double w_lo, double w_hi, int n_pts) {
    BodeData bd;
    bd.w.resize(n_pts);
    bd.magnitude.resize(n_pts);
    bd.phase.resize(n_pts);
    double lw_lo = std::log10(w_lo), lw_hi = std::log10(w_hi);
    for (int i = 0; i < n_pts; ++i) {
        double w = std::pow(10.0, lw_lo + (lw_hi - lw_lo) * i / (n_pts - 1));
        auto H = freqresp(sys, w);
        bd.w[i]         = w;
        bd.magnitude[i] = 20.0 * std::log10(std::abs(H));
        bd.phase[i]     = std::arg(H) * 180.0 / M_PI;
    }
    return bd;
}

std::vector<std::pair<double,double>> nyquist(const TransferFunction& sys,
              double w_lo, double w_hi, int n_pts) {
    std::vector<std::pair<double,double>> pts;
    double lw_lo = std::log10(w_lo), lw_hi = std::log10(w_hi);
    for (int i = 0; i < n_pts; ++i) {
        double w = std::pow(10.0, lw_lo + (lw_hi - lw_lo) * i / (n_pts - 1));
        auto H = freqresp(sys, w);
        pts.push_back({H.real(), H.imag()});
    }
    return pts;
}

Margins margin(const TransferFunction& sys) {
    const int N = 5000;
    const double w_lo = 1e-3, w_hi = 1e5;
    Margins m{};
    m.gain_margin_db = std::numeric_limits<double>::infinity();
    m.phase_margin_deg = std::numeric_limits<double>::infinity();
    m.gain_crossover_freq = 0.0;
    m.phase_crossover_freq = 0.0;

    const double dc = dcgain(sys);
    const double dc_mag = std::abs(dc);
    if (dc_mag > 1e-12 && std::abs(20.0 * std::log10(dc_mag)) < 0.05) {
        const double ph_deg = std::arg(std::complex<double>(dc, 0.0)) * 180.0 / M_PI;
        m.phase_margin_deg = ph_deg + 180.0;
        m.gain_crossover_freq = 0.0;
    }

    double prev_phase = 0.0;
    double prev_mag_db = 0.0;
    bool first = true;

    for (int i = 0; i < N; ++i) {
        double w = w_lo * std::pow(w_hi / w_lo, static_cast<double>(i) / (N - 1));
        auto H = freqresp(sys, w);
        double mag_db = 20.0 * std::log10(std::abs(H));
        double ph_deg = std::arg(H) * 180.0 / M_PI;

        if (!first) {
            // Gain crossover: |H| = 1 (0 dB)
            if ((prev_mag_db >= 0.0 && mag_db < 0.0) ||
                (prev_mag_db <= 0.0 && mag_db > 0.0)) {
                m.gain_crossover_freq = w;
                m.phase_margin_deg = ph_deg + 180.0;
            }
            // Phase crossover: phase = -180
            if ((prev_phase >= -180.0 && ph_deg < -180.0) ||
                (prev_phase <= -180.0 && ph_deg > -180.0)) {
                m.phase_crossover_freq = w;
                m.gain_margin_db = -mag_db;
            }
        }
        prev_mag_db = mag_db;
        prev_phase  = ph_deg;
        first = false;
    }
    return m;
}

// ---- Time response via matrix exponential (trapezoidal integration) ----

StepData step_response(const TransferFunction& sys, double t_end, int n_pts) {
    auto s = tf2ss(sys);
    int n = s.n;
    StepData data;
    data.t.resize(n_pts);
    data.y.resize(n_pts);
    double dt = t_end / (n_pts - 1);

    // State vector x
    std::vector<double> x(n, 0.0);
    std::vector<double> xdot;
    xdot.reserve(static_cast<size_t>(n));
    // Simple forward Euler integration: u(t) = 1 (step)
    for (int i = 0; i < n_pts; ++i) {
        data.t[i] = i * dt;
        // y = C*x + D*u
        double y = s.D[0][0]; // D*u, u=1
        for (int j = 0; j < n; ++j) y += s.C[0][j] * x[j];
        data.y[i] = y;
        // dx = A*x + B*u
        matvec(s.A, x, xdot);
        for (int j = 0; j < n; ++j) xdot[j] += s.B[j][0]; // B*1
        for (int j = 0; j < n; ++j) x[j] += dt * xdot[j];
    }
    return data;
}

StepData impulse_response(const TransferFunction& sys, double t_end, int n_pts) {
    auto s = tf2ss(sys);
    int n = s.n;
    StepData data;
    data.t.resize(n_pts);
    data.y.resize(n_pts);
    double dt = t_end / (n_pts - 1);

    // Impulse: x(0) = B*u(0) = B (apply B at t=0), then free response
    std::vector<double> x(n, 0.0);
    for (int j = 0; j < n; ++j) x[j] = s.B[j][0];
    std::vector<double> xdot;
    xdot.reserve(static_cast<size_t>(n));

    for (int i = 0; i < n_pts; ++i) {
        data.t[i] = i * dt;
        double y = 0.0;
        for (int j = 0; j < n; ++j) y += s.C[0][j] * x[j];
        data.y[i] = y;
        matvec(s.A, x, xdot);
        for (int j = 0; j < n; ++j) x[j] += dt * xdot[j];
    }
    return data;
}

// ---- Step-response metrics ----

static double step_info_tail_mean(const std::vector<double>& y) {
    const size_t n = y.size();
    if (n == 0) return 0.0;
    const size_t tail = std::max<size_t>(1, n / 20);  // last 5%
    double sum = 0.0;
    for (size_t i = n - tail; i < n; ++i) sum += y[i];
    return sum / static_cast<double>(tail);
}

// Linear interpolation of crossing time: find first index where y crosses
// `threshold` in the direction given by `rising` (true: y >= thresh).
static double interp_cross_time(const std::vector<double>& t,
                                const std::vector<double>& y,
                                size_t start_idx,
                                double threshold,
                                bool rising) {
    const size_t n = y.size();
    for (size_t i = start_idx + 1; i < n; ++i) {
        const bool crossed = rising
            ? (y[i - 1] < threshold && y[i] >= threshold)
            : (y[i - 1] > threshold && y[i] <= threshold);
        if (!crossed) continue;
        const double dy = y[i] - y[i - 1];
        if (std::abs(dy) < 1e-30) return t[i];
        const double frac = (threshold - y[i - 1]) / dy;
        return t[i - 1] + frac * (t[i] - t[i - 1]);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

static StepInfo step_info_impl(const std::vector<double>& t,
                               const std::vector<double>& y,
                               double final_value,
                               double settling_tol_pct) {
    StepInfo info{};
    info.rise_time = std::numeric_limits<double>::quiet_NaN();
    info.settling_time = std::numeric_limits<double>::infinity();
    info.overshoot_pct = 0.0;
    info.peak_time = 0.0;
    info.peak_value = 0.0;

    if (t.empty() || y.empty() || t.size() != y.size()) return info;

    const double y0 = y.front();
    const double yf = final_value;
    const double step = yf - y0;
    const double abs_step = std::abs(step);
    const double abs_yf = std::abs(yf);

    info.peak_value = y[0];
    info.peak_time = t[0];
    for (size_t i = 1; i < y.size(); ++i) {
        if (y[i] > info.peak_value) {
            info.peak_value = y[i];
            info.peak_time = t[i];
        }
    }

    if (abs_step < 1e-12) {
        info.rise_time = std::numeric_limits<double>::quiet_NaN();
        info.settling_time = 0.0;
        info.overshoot_pct = 0.0;
        info.peak_value = y0;
        info.peak_time = t.front();
        return info;
    }

    const bool rising = step > 0.0;
    const double lo = y0 + 0.1 * step;
    const double hi = y0 + 0.9 * step;

    const double t_lo = interp_cross_time(t, y, 0, lo, rising);
    size_t hi_start = 0;
    if (std::isfinite(t_lo)) {
        for (size_t i = 0; i + 1 < y.size(); ++i) {
            if (t[i] <= t_lo && t[i + 1] >= t_lo) {
                hi_start = i;
                break;
            }
        }
    }
    const double t_hi = interp_cross_time(t, y, hi_start, hi, rising);
    if (std::isfinite(t_lo) && std::isfinite(t_hi) && t_hi >= t_lo)
        info.rise_time = t_hi - t_lo;

    if (info.peak_value > yf && abs_yf > 1e-12)
        info.overshoot_pct = 100.0 * (info.peak_value - yf) / abs_yf;

    const double tol = (settling_tol_pct / 100.0) * (abs_yf > 1e-12 ? abs_yf : abs_step);
    int last_outside = -1;
    for (size_t i = 0; i < y.size(); ++i) {
        if (std::abs(y[i] - yf) > tol)
            last_outside = static_cast<int>(i);
    }

    if (last_outside < 0) {
        info.settling_time = t.front();
    } else if (static_cast<size_t>(last_outside) + 1 >= y.size()) {
        info.settling_time = std::numeric_limits<double>::infinity();
    } else {
        info.settling_time = t[static_cast<size_t>(last_outside) + 1];
    }

    return info;
}

StepInfo step_info(const StepData& data, double settling_tol_pct) {
    return step_info_impl(data.t, data.y, step_info_tail_mean(data.y), settling_tol_pct);
}

StepInfo step_info(const std::vector<double>& t,
                   const std::vector<double>& y,
                   double final_value,
                   double settling_tol_pct) {
    return step_info_impl(t, y, final_value, settling_tol_pct);
}

// ---- Gaussian elimination helper for augmented matrix ----
static bool gauss_solve_flat(std::vector<double>& K, int n, int cols) {
    for (int col = 0; col < n; ++col) {
        int pivot = -1;
        double best = 0.0;
        for (int row = col; row < n; ++row) {
            const double v = std::abs(K[static_cast<size_t>(row * cols + col)]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (pivot < 0 || best < 1e-12) return false;
        if (pivot != col) {
            for (int j = 0; j < cols; ++j)
                std::swap(K[static_cast<size_t>(col * cols + j)],
                          K[static_cast<size_t>(pivot * cols + j)]);
        }
        const double sc = K[static_cast<size_t>(col * cols + col)];
        for (int j = col; j < cols; ++j)
            K[static_cast<size_t>(col * cols + j)] /= sc;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            const double f = K[static_cast<size_t>(row * cols + col)];
            for (int j = col; j < cols; ++j)
                K[static_cast<size_t>(row * cols + j)] -= f * K[static_cast<size_t>(col * cols + j)];
        }
    }
    return true;
}

static bool gauss_solve(std::vector<std::vector<double>>& K, int n) {
    const int cols = n + 1;
    std::vector<double> flat(static_cast<size_t>(n * cols));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < cols; ++j)
            flat[static_cast<size_t>(i * cols + j)] = K[static_cast<size_t>(i)][static_cast<size_t>(j)];
    if (!gauss_solve_flat(flat, n, cols)) return false;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < cols; ++j)
            K[static_cast<size_t>(i)][static_cast<size_t>(j)] = flat[static_cast<size_t>(i * cols + j)];
    return true;
}

// ---- Lyapunov solver via Kronecker product ----
// Solve A*X + X*A^T + Q = 0
// Vectorized: K * vec(X) = -vec(Q), where
// K[(i*n+j),(k*n+l)] = A[i][k]*δ_{jl} + A[j][l]*δ_{ik}  (row-major vec)
Result<std::vector<std::vector<double>>>
lyap(const std::vector<std::vector<double>>& A,
     const std::vector<std::vector<double>>& Q) {
    int n = static_cast<int>(A.size());
    int n2 = n * n;
    const int cols = n2 + 1;
    std::vector<double> K(static_cast<size_t>(n2 * cols), 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const int row = i * n + j;
            for (int k = 0; k < n; ++k)  K[static_cast<size_t>(row * cols + k * n + j)] += A[i][k];
            for (int l = 0; l < n; ++l)  K[static_cast<size_t>(row * cols + i * n + l)] += A[j][l];
            K[static_cast<size_t>(row * cols + n2)] = -Q[i][j];
        }
    if (!gauss_solve_flat(K, n2, cols))
        return std::unexpected(Error{DomainError{"lyap", "singular — system may be unstable"}});
    std::vector<std::vector<double>> X(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            X[i][j] = K[static_cast<size_t>((i * n + j) * cols + n2)];
    return X;
}

// ---- Discrete Lyapunov: A*X*A^T - X + Q = 0 via Kronecker ----
// vec form: (A⊗A - I) vec(X) = -vec(Q)
// K[(i*n+j),(k*n+l)] = A[i][k]*A[j][l] - δ_{ik}δ_{jl}
Result<std::vector<std::vector<double>>>
dlyap(const std::vector<std::vector<double>>& A,
      const std::vector<std::vector<double>>& Q) {
    int n = static_cast<int>(A.size());
    int n2 = n * n;
    const int cols = n2 + 1;
    std::vector<double> K(static_cast<size_t>(n2 * cols), 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const int row = i * n + j;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    K[static_cast<size_t>(row * cols + k * n + l)] += A[i][k] * A[j][l];
            K[static_cast<size_t>(row * cols + row)] -= 1.0; // -I term
            K[static_cast<size_t>(row * cols + n2)] = -Q[i][j];
        }
    if (!gauss_solve_flat(K, n2, cols))
        return std::unexpected(Error{DomainError{"dlyap", "singular"}});
    std::vector<std::vector<double>> X(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            X[i][j] = K[static_cast<size_t>((i * n + j) * cols + n2)];
    return X;
}

// Forward declaration — pole placement used for stabilising Kleinman initial gain
Result<std::vector<double>>
place(const std::vector<std::vector<double>>& A,
      const std::vector<std::vector<double>>& B,
      const std::vector<double>& desired_poles);

static std::vector<std::vector<double>>
q_from_k(const std::vector<std::vector<double>>& Q,
         const std::vector<std::vector<double>>& K,
         const std::vector<std::vector<double>>& R) {
    int n = static_cast<int>(Q.size());
    auto KT = transpose(K);
    auto KTR = matmul(KT, R);
    auto KTRK = matmul(KTR, K);
    std::vector<std::vector<double>> Qaug(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Qaug[i][j] = Q[i][j] + KTRK[i][j];
    return Qaug;
}

// ---- Riccati (Kleinman policy iteration) ----
// Solve: A^T X + X A - X S X + Q = 0 where S = B R^{-1} B^T
// With stabilising K_0, iterate:
//   (A - B K)^T X + X (A - B K) + Q + K^T R K = 0
//   K <- R^{-1} B^T X
Result<std::vector<std::vector<double>>>
riccati(const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B,
        const std::vector<std::vector<double>>& Q,
        const std::vector<std::vector<double>>& R) {
    int n = static_cast<int>(A.size());
    int m = static_cast<int>(B[0].size());
    auto BT = transpose(B);
    int r_size = static_cast<int>(R.size());
    std::vector<std::vector<double>> Rinv(r_size, std::vector<double>(r_size, 0.0));
    for (int i = 0; i < r_size; ++i)
        Rinv[i][i] = 1.0 / R[i][i];

    std::vector<std::vector<double>> K(m, std::vector<double>(n, 0.0));
    bool have_k = false;

    if (m == 1) {
        std::vector<double> desired(n);
        for (int i = 0; i < n; ++i)
            desired[i] = -(static_cast<double>(i) + 1.0);
        auto k0 = place(A, B, desired);
        if (k0) {
            for (int j = 0; j < n; ++j)
                K[0][j] = k0.value()[j];
            have_k = true;
        }
    }

    if (!have_k) {
        auto Kdir = matmul(Rinv, BT);
        for (int attempt = 0; attempt < 30 && !have_k; ++attempt) {
            double scale = std::pow(10.0, static_cast<double>(attempt) - 2.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    K[i][j] = scale * Kdir[i][j];
            auto BK = matmul(B, K);
            std::vector<std::vector<double>> Acl(n, std::vector<double>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Acl[i][j] = A[i][j] - BK[i][j];
            if (lyap(transpose(Acl), q_from_k(Q, K, R)))
                have_k = true;
        }
    }

    if (!have_k)
        return std::unexpected(Error{DomainError{"riccati", "no stabilising initial gain"}});

    for (int iter = 0; iter < 100; ++iter) {
        auto BK = matmul(B, K);
        std::vector<std::vector<double>> Acl(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Acl[i][j] = A[i][j] - BK[i][j];

        auto X = lyap(transpose(Acl), q_from_k(Q, K, R));
        if (!X) return std::unexpected(X.error());

        auto Knew = matmul(matmul(Rinv, BT), X.value());
        double maxdiff = 0.0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                maxdiff = std::max(maxdiff, std::abs(Knew[i][j] - K[i][j]));
        K = Knew;
        if (maxdiff < 1e-10)
            return X;
    }
    return std::unexpected(Error{ConvergenceFail{100, 0.0}});
}

Result<std::vector<std::vector<double>>>
dare(const std::vector<std::vector<double>>& A,
     const std::vector<std::vector<double>>& B,
     const std::vector<std::vector<double>>& Q,
     const std::vector<std::vector<double>>& R) {
    int n = static_cast<int>(A.size());
    auto AT = transpose(A);
    auto BT = transpose(B);
    int r_sz = static_cast<int>(R.size());
    std::vector<std::vector<double>> Rinv(r_sz, std::vector<double>(r_sz, 0.0));
    for (int i = 0; i < r_sz; ++i) Rinv[i][i] = 1.0 / R[i][i];

    // S = B R^{-1} B^T
    auto S = matmul(matmul(B, Rinv), BT);
    auto X = Q;
    for (int iter = 0; iter < 500; ++iter) {
        // X_{k+1} = Q + A^T * X_k * (I + S * X_k)^{-1} * A
        auto SX = matmul(S, X);
        int sz = n;
        // Invert (I + SX) using augmented Gauss-Jordan
        std::vector<std::vector<double>> IpSX(sz, std::vector<double>(2 * sz, 0.0));
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) IpSX[i][j] = SX[i][j];
            IpSX[i][i] += 1.0;
            IpSX[i][sz + i] = 1.0;
        }
        bool ok = true;
        for (int col = 0; col < sz && ok; ++col) {
            int pivot2 = col;
            for (int row = col+1; row < sz; ++row)
                if (std::abs(IpSX[row][col]) > std::abs(IpSX[pivot2][col])) pivot2 = row;
            std::swap(IpSX[col], IpSX[pivot2]);
            double sc2 = IpSX[col][col];
            if (std::abs(sc2) < 1e-14) { ok = false; break; }
            for (int j = 0; j < 2*sz; ++j) IpSX[col][j] /= sc2;
            for (int row = 0; row < sz; ++row) {
                if (row == col) continue;
                double ff = IpSX[row][col];
                for (int j = 0; j < 2*sz; ++j) IpSX[row][j] -= ff * IpSX[col][j];
            }
        }
        if (!ok) return std::unexpected(Error{DomainError{"dare","singular"}});
        std::vector<std::vector<double>> IpSX_inv(sz, std::vector<double>(sz));
        for (int i = 0; i < sz; ++i)
            for (int j = 0; j < sz; ++j)
                IpSX_inv[i][j] = IpSX[i][sz + j];

        auto XnewCore = matmul(matmul(X, IpSX_inv), A);
        auto X_new = matmul(AT, XnewCore);
        double maxdiff = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                X_new[i][j] += Q[i][j];
                maxdiff = std::max(maxdiff, std::abs(X_new[i][j] - X[i][j]));
            }
        X = X_new;
        if (maxdiff < 1e-10) return X;
    }
    return std::unexpected(Error{ConvergenceFail{500, 0.0}});
}

// ---- LQR ----
Result<std::vector<std::vector<double>>>
lqr(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    const std::vector<std::vector<double>>& Q,
    const std::vector<std::vector<double>>& R) {
    auto X = riccati(A, B, Q, R);
    if (!X) return std::unexpected(X.error());
    // K = R^{-1} B^T X
    int r_sz = static_cast<int>(R.size());
    std::vector<std::vector<double>> Rinv(r_sz, std::vector<double>(r_sz, 0.0));
    for (int i = 0; i < r_sz; ++i) Rinv[i][i] = 1.0 / R[i][i];
    auto BT = transpose(B);
    return matmul(matmul(Rinv, BT), X.value());
}

// ---- LQE (dual of LQR) ----
// Filter ARE: A*P + P*A^T - P*C^T*R^{-1}*C*P + Q = 0
// Dual control ARE with (A, B) -> (A^T, C^T) gives the same equation for P.
Result<LQEResult>
lqe(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& C,
    const std::vector<std::vector<double>>& Q,
    const std::vector<std::vector<double>>& R) {
    auto P = riccati(transpose(A), transpose(C), Q, R);
    if (!P) return std::unexpected(P.error());

    const int r_sz = static_cast<int>(R.size());
    std::vector<std::vector<double>> Rinv(r_sz, std::vector<double>(r_sz, 0.0));
    for (int i = 0; i < r_sz; ++i) Rinv[i][i] = 1.0 / R[i][i];

    LQEResult out;
    out.P = P.value();
    // L = P * C^T * R^{-1}  (n×p; dual of K = R^{-1} B^T X)
    out.L = matmul(matmul(out.P, transpose(C)), Rinv);
    return out;
}

// ---- Controllability & Observability ----

std::vector<std::vector<double>>
ctrb(const std::vector<std::vector<double>>& A,
     const std::vector<std::vector<double>>& B) {
    int n = static_cast<int>(A.size());
    int m = static_cast<int>(B[0].size());
    std::vector<std::vector<double>> C(n, std::vector<double>(n * m, 0.0));
    auto AB = B;
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                C[i][k * m + j] = AB[i][j];
        AB = matmul(A, AB);
    }
    return C;
}

std::vector<std::vector<double>>
obsv(const std::vector<std::vector<double>>& A,
     const std::vector<std::vector<double>>& C) {
    int n = static_cast<int>(A.size());
    int p = static_cast<int>(C.size());
    std::vector<std::vector<double>> O(n * p, std::vector<double>(n, 0.0));
    auto CA = C;
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < n; ++j)
                O[k * p + i][j] = CA[i][j];
        CA = matmul(CA, A);
    }
    return O;
}

static double matrix_rank(const std::vector<std::vector<double>>& M) {
    // Approximate rank via count of non-negligible singular values
    int rows = static_cast<int>(M.size());
    int cols = static_cast<int>(M[0].size());
    // Compute Frobenius norm
    double fnorm = 0.0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            fnorm += M[i][j] * M[i][j];
    fnorm = std::sqrt(fnorm);
    // Approximate: check if rows are linearly independent via Gram-Schmidt
    int rank = 0;
    std::vector<std::vector<double>> basis;
    for (int i = 0; i < rows && rank < std::min(rows, cols); ++i) {
        std::vector<double> v = M[i];
        for (auto& b : basis) {
            double dot = 0.0;
            for (int j = 0; j < (int)v.size(); ++j) dot += v[j] * b[j];
            for (int j = 0; j < (int)v.size(); ++j) v[j] -= dot * b[j];
        }
        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-10 * fnorm) {
            for (auto& x : v) x /= norm;
            basis.push_back(v);
            ++rank;
        }
    }
    return rank;
}

bool is_controllable(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B) {
    int n = static_cast<int>(A.size());
    auto C = ctrb(A, B);
    return matrix_rank(C) == n;
}

bool is_observable(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& C) {
    int n = static_cast<int>(A.size());
    auto O = obsv(A, C);
    return matrix_rank(O) == n;
}

// ---- Gramians ----
// lyap(A, Q) solves A*X + X*A^T + Q = 0 (see the Lyapunov solver above).
// Wc: A*Wc + Wc*A^T + B*B^T = 0            -> lyap(A, B*B^T)
// Wo: A^T*Wo + Wo*A + C^T*C = 0 = A'*Wo + Wo*(A')^T + C^T*C -> lyap(A^T, C^T*C)
Result<std::vector<std::vector<double>>>
gram(const StateSpace& sys, GramianType type) {
    if (type == GramianType::Controllability) {
        auto BBt = matmul(sys.B, transpose(sys.B));
        return lyap(sys.A, BBt);
    }
    auto CtC = matmul(transpose(sys.C), sys.C);
    return lyap(transpose(sys.A), CtC);
}

Result<std::vector<std::vector<double>>> ctrb_gram(const StateSpace& sys) {
    return gram(sys, GramianType::Controllability);
}

Result<std::vector<std::vector<double>>> obsv_gram(const StateSpace& sys) {
    return gram(sys, GramianType::Observability);
}

// ---- Pole placement (Ackermann's formula for SISO) ----
Result<std::vector<double>>
place(const std::vector<std::vector<double>>& A,
      const std::vector<std::vector<double>>& B,
      const std::vector<double>& desired_poles) {
    int n = static_cast<int>(A.size());
    // Build desired characteristic polynomial
    std::vector<double> char_poly = {1.0};
    for (double p : desired_poles) {
        std::vector<double> factor = {1.0, -p};
        char_poly = polymul(char_poly, factor);
    }
    // Evaluate desired poly at A (Cayley-Hamilton): phi_d(A)
    // phi_d(A) = char_poly[0]*A^n + ... + char_poly[n]*I
    std::vector<std::vector<double>> phi_A(n, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> Apow(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Apow[i][i] = 1.0; // A^0 = I
    // Build from A^n down
    std::vector<std::vector<std::vector<double>>> powers(n + 1);
    powers[0] = Apow;
    for (int k = 1; k <= n; ++k) powers[k] = matmul(powers[k-1], A);
    for (int k = 0; k <= n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                phi_A[i][j] += char_poly[k] * powers[n - k][i][j];
    // Controllability matrix
    auto C = ctrb(A, B);
    // Ackermann: k^T = e_n^T * C^{-1} * phi_d(A)
    // e_n = last standard basis vector (1x1 output: we want last row of C^{-1})
    // Simplified: use e_n^T * C^{-1} numerically for small systems
    // For simplicity compute k^T = e_n^T * C_inv * phi_A via Gaussian elimination
    // C is n x n (1 input)
    std::vector<std::vector<double>> C_sq(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C_sq[i][j] = C[i][j];
    // Augment with phi_A row (last row of phi_A needed)
    std::vector<double> en_phi(n, 0.0);
    for (int j = 0; j < n; ++j) en_phi[j] = phi_A[n-1][j];
    // Solve C_sq^T * k = (last row of phi_A) in least-squares sense
    // Simple: for 1-input use Ackermann formula numerically
    std::vector<double> k(n, 0.0);
    // k^T = e_n^T * C^{-1} * phi(A) → solve using basic approach
    // For small n, use Cramer's rule / LU via forward elimination
    // augmented matrix [C^T | e_n]
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = C_sq[j][i]; // C^T
        aug[i][n] = (i == n - 1) ? 1.0 : 0.0; // e_n
    }
    // Gauss-Jordan
    for (int col = 0; col < n; ++col) {
        int pivot = -1;
        double best = 0.0;
        for (int row = col; row < n; ++row)
            if (std::abs(aug[row][col]) > best) { best = std::abs(aug[row][col]); pivot = row; }
        if (pivot < 0 || best < 1e-12)
            return std::unexpected(Error{DomainError{"place", "system not controllable"}});
        std::swap(aug[col], aug[pivot]);
        double sc = aug[col][col];
        for (int j = col; j <= n; ++j) aug[col][j] /= sc;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = aug[row][col];
            for (int j = col; j <= n; ++j) aug[row][j] -= f * aug[col][j];
        }
    }
    // aug[i][n] is now C^{-T} e_n
    std::vector<double> Cinv_en(n);
    for (int i = 0; i < n; ++i) Cinv_en[i] = aug[i][n];
    // k = phi_A^T * Cinv_en
    for (int j = 0; j < n; ++j) {
        k[j] = 0.0;
        for (int i = 0; i < n; ++i) k[j] += phi_A[i][j] * Cinv_en[i];
    }
    return k;
}

PIDGains pidtune(const TransferFunction& plant, double bandwidth) {
    // Ziegler-Nichols style: approximate based on plant at crossover
    double Kp = std::abs(polyval(plant.den, std::complex<double>(0.0, bandwidth))) /
                std::abs(polyval(plant.num, std::complex<double>(0.0, bandwidth)));
    return {Kp, Kp * bandwidth / 10.0, Kp / (10.0 * bandwidth)};
}

// ---- Kalman filter ----

// Small dense matrix inverse via Gauss-Jordan with partial pivoting, in the
// same defensive style as gauss_solve() above: never throws, and reports
// singularity through `ok` instead (unlike mat_inv(), which throws — not
// usable here since a singular innovation covariance is an expected,
// recoverable degenerate input rather than a programming error).
static std::vector<std::vector<double>> kalman_safe_inverse(
    const std::vector<std::vector<double>>& M, bool& ok) {
    const int n = static_cast<int>(M.size());
    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = M[i][j];
        aug[i][n + i] = 1.0;
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::abs(aug[col][col]);
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(aug[row][col]) > best) { best = std::abs(aug[row][col]); pivot = row; }
        }
        if (best < 1e-12) { ok = false; return {}; }
        std::swap(aug[col], aug[pivot]);
        double sc = aug[col][col];
        for (int j = col; j < 2 * n; ++j) aug[col][j] /= sc;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = aug[row][col];
            for (int j = col; j < 2 * n; ++j) aug[row][j] -= f * aug[col][j];
        }
    }
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            inv[i][j] = aug[i][n + j];
    ok = true;
    return inv;
}

// Checks A is r x c (non-ragged) with the given dimensions.
static bool mat_has_shape(const std::vector<std::vector<double>>& M, int r, int c) {
    if (static_cast<int>(M.size()) != r) return false;
    for (const auto& row : M)
        if (static_cast<int>(row.size()) != c) return false;
    return true;
}

KalmanState kalman_predict(const KalmanState& state,
                           const std::vector<std::vector<double>>& A,
                           const std::vector<std::vector<double>>& Q) {
    const int n = static_cast<int>(state.x.size());
    if (n == 0) return state;
    if (!mat_has_shape(A, n, n)) return state;
    if (!mat_has_shape(Q, n, n)) return state;
    if (!mat_has_shape(state.P, n, n)) return state;

    KalmanState out;
    out.x = matvec(A, state.x);
    auto AP = matmul(A, state.P);
    auto APAt = matmul(AP, transpose(A));
    out.P = matadd(APAt, Q);
    return out;
}

KalmanState kalman_update(const KalmanState& state,
                          const std::vector<double>& z,
                          const std::vector<std::vector<double>>& H,
                          const std::vector<std::vector<double>>& R) {
    const int n = static_cast<int>(state.x.size());
    if (n == 0) return state;
    const int m = static_cast<int>(H.size());
    if (m == 0) return state;
    if (!mat_has_shape(H, m, n)) return state;
    if (z.size() != static_cast<size_t>(m)) return state;
    if (!mat_has_shape(R, m, m)) return state;
    if (!mat_has_shape(state.P, n, n)) return state;

    const auto Ht = transpose(H);
    const auto Hx = matvec(H, state.x);
    std::vector<double> y(m);
    for (int i = 0; i < m; ++i) y[i] = z[i] - Hx[i];

    const auto PHt = matmul(state.P, Ht);
    const auto HPHt = matmul(H, PHt);
    const auto S = matadd(HPHt, R);

    bool ok = false;
    const auto Sinv = kalman_safe_inverse(S, ok);
    if (!ok) return state;  // singular innovation covariance: safe fallback

    const auto K = matmul(PHt, Sinv);

    KalmanState out;
    const auto Ky = matvec(K, y);
    out.x.resize(n);
    for (int i = 0; i < n; ++i) out.x[i] = state.x[i] + Ky[i];

    const auto KH = matmul(K, H);
    out.P = matmul(mat_sub(identity_mat(n), KH), state.P);
    return out;
}

} // namespace control
} // namespace ms
