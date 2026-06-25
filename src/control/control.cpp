#define _USE_MATH_DEFINES
#include "ms/control/control.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace control {

// ---- Helpers ----

// Matrix-vector multiplication
static std::vector<double> matvec(const std::vector<std::vector<double>>& A,
                                   const std::vector<double>& x) {
    int n = static_cast<int>(A.size());
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < (int)x.size(); ++j)
            y[i] += A[i][j] * x[j];
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
    // Simple forward Euler integration: u(t) = 1 (step)
    for (int i = 0; i < n_pts; ++i) {
        data.t[i] = i * dt;
        // y = C*x + D*u
        double y = s.D[0][0]; // D*u, u=1
        for (int j = 0; j < n; ++j) y += s.C[0][j] * x[j];
        data.y[i] = y;
        // dx = A*x + B*u
        auto xdot = matvec(s.A, x);
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

    for (int i = 0; i < n_pts; ++i) {
        data.t[i] = i * dt;
        double y = 0.0;
        for (int j = 0; j < n; ++j) y += s.C[0][j] * x[j];
        data.y[i] = y;
        auto xdot = matvec(s.A, x);
        for (int j = 0; j < n; ++j) x[j] += dt * xdot[j];
    }
    return data;
}

// ---- Gaussian elimination helper for augmented matrix ----
static bool gauss_solve(std::vector<std::vector<double>>& K, int n) {
    for (int col = 0; col < n; ++col) {
        int pivot = -1; double best = 0.0;
        for (int row = col; row < n; ++row)
            if (std::abs(K[row][col]) > best) { best = std::abs(K[row][col]); pivot = row; }
        if (pivot < 0 || best < 1e-12) return false;
        std::swap(K[col], K[pivot]);
        double sc = K[col][col];
        for (int j = col; j <= n; ++j) K[col][j] /= sc;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = K[row][col];
            for (int j = col; j <= n; ++j) K[row][j] -= f * K[col][j];
        }
    }
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
    std::vector<std::vector<double>> K(n2, std::vector<double>(n2 + 1, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int row = i * n + j;
            for (int k = 0; k < n; ++k)  K[row][k * n + j] += A[i][k];
            for (int l = 0; l < n; ++l)  K[row][i * n + l] += A[j][l];
            K[row][n2] = -Q[i][j];
        }
    if (!gauss_solve(K, n2))
        return std::unexpected(Error{DomainError{"lyap", "singular — system may be unstable"}});
    std::vector<std::vector<double>> X(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            X[i][j] = K[i * n + j][n2];
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
    std::vector<std::vector<double>> K(n2, std::vector<double>(n2 + 1, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int row = i * n + j;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l)
                    K[row][k * n + l] += A[i][k] * A[j][l];
            K[row][row] -= 1.0; // -I term
            K[row][n2]   = -Q[i][j];
        }
    if (!gauss_solve(K, n2))
        return std::unexpected(Error{DomainError{"dlyap", "singular"}});
    std::vector<std::vector<double>> X(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            X[i][j] = K[i * n + j][n2];
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

} // namespace control
} // namespace ms
