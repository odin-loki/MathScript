// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ms::cpu::lapack {

namespace {

double generate_reflector(int n, double* x) {
    if (n <= 1) {
        return 0.0;
    }

    double sigma = 0.0;
    for (int i = 1; i < n; ++i) {
        sigma += x[i] * x[i];
    }
    sigma = std::sqrt(sigma);
    if (sigma == 0.0) {
        return 0.0;
    }

    const double x0 = x[0];
    const double beta = -std::copysign(std::hypot(x0, sigma), x0);
    const double tau = (beta - x0) / beta;
    const double denom = x0 - beta;

    for (int i = 1; i < n; ++i) {
        x[i] /= denom;
    }
    x[0] = beta;
    return tau;
}

// Symmetric rank-2 update realising the similarity transform A22 := H * A22 * H,
// where H = I - tau * v * v^T is the reflector produced by generate_reflector for
// step i.  Everything is expressed in the LAPACK 'L' convention used by dsytrd:
//
//   * v is stored in column i of A, rows i+1 .. n-1.  v[0] is implicitly 1 (that
//     slot holds the new subdiagonal entry beta), v[k] = A(i+1+k, i) for k >= 1.
//   * A22 is the trailing block rows/cols i+1 .. n-1, referenced through its
//     LOWER triangle only; the upper triangle still holds untouched input data.
//
// With p = tau * A22 * v and w = p - (tau/2) * (p^T v) * v, the exact identity
//   H A22 H = A22 - v w^T - w v^T
// holds; dropping the second-order term would only give H * A22.
void sym_rank2_lower(int n, int i, double* A, int lda, double tau) {
    const int nk = n - i - 1;
    if (nk <= 0 || tau == 0.0) {
        return;
    }

    const std::size_t ld = static_cast<std::size_t>(lda);
    const std::size_t base = static_cast<std::size_t>(i + 1);

    // v[k] for k in [0, nk): stored in column i below the (implicit) unit entry.
    auto vec = [&](int k) -> double {
        if (k == 0) {
            return 1.0;
        }
        return A[static_cast<std::size_t>(i) * ld + base + static_cast<std::size_t>(k)];
    };

    // A22(r, c) read from the lower triangle only, using symmetry for r < c.
    auto a22 = [&](int r, int c) -> double {
        const int rr = (r >= c) ? r : c;
        const int cc = (r >= c) ? c : r;
        return A[(base + static_cast<std::size_t>(cc)) * ld + base + static_cast<std::size_t>(rr)];
    };

    std::vector<double> w(static_cast<std::size_t>(nk), 0.0);
    double pv = 0.0;
    for (int r = 0; r < nk; ++r) {
        double sum = 0.0;
        for (int c = 0; c < nk; ++c) {
            sum += a22(r, c) * vec(c);
        }
        const double p_r = tau * sum;
        w[static_cast<std::size_t>(r)] = p_r;
        pv += p_r * vec(r);
    }

    const double half = -0.5 * tau * pv;
    for (int r = 0; r < nk; ++r) {
        w[static_cast<std::size_t>(r)] += half * vec(r);
    }

    for (int c = 0; c < nk; ++c) {
        const double v_c = vec(c);
        const double w_c = w[static_cast<std::size_t>(c)];
        for (int r = c; r < nk; ++r) {
            A[(base + static_cast<std::size_t>(c)) * ld + base + static_cast<std::size_t>(r)] -=
                vec(r) * w_c + w[static_cast<std::size_t>(r)] * v_c;
        }
    }
}

int sturm_count(int n, const double* d, const double* e, double mu) {
    double pm2 = 1.0;
    double pm1 = d[0] - mu;
    if (pm1 == 0.0) {
        pm1 = -1e-200;
    }
    int count = (pm2 * pm1 < 0.0) ? 1 : 0;
    for (int k = 1; k < n; ++k) {
        double pk = (d[k] - mu) * pm1 - e[k - 1] * e[k - 1] * pm2;
        if (pk == 0.0) {
            pk = -1e-200;
        }
        if (pm1 * pk < 0.0) {
            ++count;
        }
        pm2 = pm1;
        pm1 = pk;
    }
    return count;
}

void tridiag_bounds(int n, const double* d, const double* e, double& lo, double& hi) {
    lo = std::numeric_limits<double>::max();
    hi = -std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        double off = 0.0;
        if (i > 0) {
            off += std::abs(e[i - 1]);
        }
        if (i + 1 < n) {
            off += std::abs(e[i]);
        }
        lo = (std::min)(lo, d[i] - off);
        hi = (std::max)(hi, d[i] + off);
    }
}

void tridiag_solve(int n, const double* d, const double* e, double sigma, const double* rhs, double* x) {
    std::vector<double> a(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i)] = d[i] - sigma;
    }

    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        scale = (std::max)(scale, std::abs(a[static_cast<std::size_t>(i)]));
    }
    for (int i = 0; i < n - 1; ++i) {
        scale = (std::max)(scale, std::abs(e[i]));
    }
    const double pivot_tol = 1e-14 * (1.0 + scale);

    std::vector<double> b(rhs, rhs + n);
    for (int i = 1; i < n; ++i) {
        double pivot = a[static_cast<std::size_t>(i - 1)];
        if (std::abs(pivot) < pivot_tol) {
            pivot = (pivot >= 0.0) ? pivot_tol : -pivot_tol;
        }
        const double w = e[i - 1] / pivot;
        a[static_cast<std::size_t>(i)] -= w * e[i - 1];
        b[static_cast<std::size_t>(i)] -= w * b[static_cast<std::size_t>(i - 1)];
    }

    double last_pivot = a[static_cast<std::size_t>(n - 1)];
    if (std::abs(last_pivot) < pivot_tol) {
        last_pivot = (last_pivot >= 0.0) ? pivot_tol : -pivot_tol;
    }
    x[n - 1] = b[static_cast<std::size_t>(n - 1)] / last_pivot;
    for (int i = n - 2; i >= 0; --i) {
        double pivot = a[static_cast<std::size_t>(i)];
        if (std::abs(pivot) < pivot_tol) {
            pivot = (pivot >= 0.0) ? pivot_tol : -pivot_tol;
        }
        x[i] = (b[static_cast<std::size_t>(i)] - e[i] * x[i + 1]) / pivot;
    }
}

double tridiag_rayleigh_quotient(int n, const double* d, const double* e, const double* x) {
    double num = 0.0;
    for (int i = 0; i < n; ++i) {
        num += x[i] * d[i] * x[i];
    }
    for (int i = 0; i < n - 1; ++i) {
        num += 2.0 * e[i] * x[i] * x[i + 1];
    }
    double den = 0.0;
    for (int i = 0; i < n; ++i) {
        den += x[i] * x[i];
    }
    return num / den;
}

void tridiag_eigenvector(int n, const double* d, const double* e, double lambda, double* x) {
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 + 0.37 * static_cast<double>(i);
    }

    double sigma = lambda + 1e-7 * (1.0 + std::abs(lambda));
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int iter = 0; iter < 12; ++iter) {
        tridiag_solve(n, d, e, sigma, x, y.data());
        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += y[static_cast<std::size_t>(i)] * y[static_cast<std::size_t>(i)];
        }
        norm = std::sqrt(norm);
        if (norm == 0.0) {
            sigma += 1e-7 * (1.0 + std::abs(sigma));
            continue;
        }
        for (int i = 0; i < n; ++i) {
            x[i] = y[static_cast<std::size_t>(i)] / norm;
        }
        sigma = tridiag_rayleigh_quotient(n, d, e, x);
    }
}

} // namespace

void dsteqr(int n, double* d, double* e, double* Z, int ldz) {
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        if (Z != nullptr) {
            Z[0] = 1.0;
        }
        return;
    }

    if (n == 2) {
        const double a = d[0];
        const double b = e[0];
        const double c = d[1];
        const double half = 0.5 * (a + c);
        const double rad = std::hypot(0.5 * (a - c), b);
        const double w0 = half + rad;
        const double w1 = half - rad;
        d[0] = w1;
        d[1] = w0;
        if (Z != nullptr) {
            auto set_vector = [&](double lambda, int col) {
                double v0 = b;
                double v1 = lambda - a;
                if (std::abs(v0) + std::abs(v1) == 0.0) {
                    v0 = 1.0;
                    v1 = 0.0;
                }
                const double nv = std::hypot(v0, v1);
                Z[static_cast<std::size_t>(col) * static_cast<std::size_t>(ldz) + 0] = v0 / nv;
                Z[static_cast<std::size_t>(col) * static_cast<std::size_t>(ldz) + 1] = v1 / nv;
            };
            set_vector(w1, 0);
            set_vector(w0, 1);
        }
        e[0] = 0.0;
        return;
    }

    bool diagonal = true;
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(e[i]) > 1e-15 * (1.0 + std::abs(d[i]) + std::abs(d[i + 1]))) {
            diagonal = false;
            break;
        }
    }
    if (diagonal) {
        if (Z != nullptr) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    Z[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldz) + static_cast<std::size_t>(i)] =
                        (i == j) ? 1.0 : 0.0;
                }
            }
        }
        for (int i = 0; i < n - 1; ++i) {
            e[i] = 0.0;
        }
        return;
    }

    const std::vector<double> diag(d, d + n);
    const std::vector<double> subd(e, e + n - 1);

    double lo = 0.0;
    double hi = 0.0;
    tridiag_bounds(n, diag.data(), subd.data(), lo, hi);

    const double tol = 1e-14;
    for (int k = 0; k < n; ++k) {
        double a = lo;
        double b = hi;
        for (int step = 0; step < 200; ++step) {
            const double mid = 0.5 * (a + b);
            if (sturm_count(n, diag.data(), subd.data(), mid) < k + 1) {
                a = mid;
            } else {
                b = mid;
            }
            if (b - a <= tol * (1.0 + std::abs(a))) {
                break;
            }
        }
        d[k] = 0.5 * (a + b);
    }

    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        scale = (std::max)(scale, std::abs(diag[static_cast<std::size_t>(i)]));
    }
    for (int i = 0; i < n - 1; ++i) {
        scale = (std::max)(scale, std::abs(subd[static_cast<std::size_t>(i)]));
    }
    const double zero_tol = 1e-15 * scale * static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        if (std::abs(d[k]) < zero_tol) {
            d[k] = 0.0;
        }
    }

    if (Z != nullptr) {
        std::vector<double> vec(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            tridiag_eigenvector(n, diag.data(), subd.data(), d[k], vec.data());
            for (int i = 0; i < n; ++i) {
                Z[static_cast<std::size_t>(k) * static_cast<std::size_t>(ldz) + static_cast<std::size_t>(i)] =
                    vec[static_cast<std::size_t>(i)];
            }
        }
    }

    for (int i = 0; i < n - 1; ++i) {
        e[i] = 0.0;
    }
}

// Z := Q * Z, where Q = H(0) H(1) ... H(n-2) is the orthogonal factor implied by
// the dsytrd reduction (A = Q T Q^T).  Reflector i is stored in column i of A:
// v(i+1) = 1 implicitly (that slot holds the subdiagonal entry), v(r) = A(r, i)
// for r > i+1.  This offset-by-one layout is the LAPACK dsytrd 'L' convention and
// differs from the dgeqrf layout that dormqr expects, so the sweep is done here.
void apply_q_from_dsytrd(int n, const double* A, int lda, const double* tau, double* Z, int ldz) {
    const std::size_t la = static_cast<std::size_t>(lda);
    const std::size_t lz = static_cast<std::size_t>(ldz);
    for (int i = n - 2; i >= 0; --i) {
        const double t = tau[static_cast<std::size_t>(i)];
        if (t == 0.0) {
            continue;
        }
        for (int col = 0; col < n; ++col) {
            const std::size_t zc = static_cast<std::size_t>(col) * lz;
            double dot = Z[zc + static_cast<std::size_t>(i + 1)];
            for (int r = i + 2; r < n; ++r) {
                dot += A[static_cast<std::size_t>(i) * la + static_cast<std::size_t>(r)] *
                       Z[zc + static_cast<std::size_t>(r)];
            }
            if (dot == 0.0) {
                continue;
            }
            Z[zc + static_cast<std::size_t>(i + 1)] -= t * dot;
            for (int r = i + 2; r < n; ++r) {
                Z[zc + static_cast<std::size_t>(r)] -=
                    t * dot * A[static_cast<std::size_t>(i) * la + static_cast<std::size_t>(r)];
            }
        }
    }
}

void dsytrd(int n, double* A, int lda, double* d, double* e, double* tau) {
    for (int i = 0; i < n - 1; ++i) {
        d[i] = A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
        const int len = n - i - 1;
        double* x = &A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i + 1)];
        if (len > 1) {
            tau[i] = generate_reflector(len, x);
            e[i] = x[0];
            sym_rank2_lower(n, i, A, lda, tau[i]);
        } else {
            e[i] = x[0];
            tau[i] = 0.0;
        }
    }
    d[n - 1] = A[static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(n - 1)];
}

int dsyev(char jobz, int n, double* A, int lda, double* w) {
    if (n <= 0) {
        return 0;
    }
    if (A == nullptr || w == nullptr) {
        return 1;
    }

    std::vector<double> d(static_cast<std::size_t>(n));
    std::vector<double> e(static_cast<std::size_t>((std::max)(0, n - 1)));
    std::vector<double> tau(static_cast<std::size_t>((std::max)(0, n - 1)));

    dsytrd(n, A, lda, d.data(), e.data(), tau.data());

    std::vector<double> z;
    double* z_ptr = nullptr;
    const int ldz = n;
    if (jobz == 'V' || jobz == 'v') {
        z.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        z_ptr = z.data();
    }

    dsteqr(n, d.data(), e.data(), z_ptr, ldz);

    for (int i = 0; i < n; ++i) {
        w[i] = d[static_cast<std::size_t>(i)];
    }

    if (jobz == 'V' || jobz == 'v') {
        apply_q_from_dsytrd(n, A, lda, tau.data(), z_ptr, ldz);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] =
                    z[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldz) + static_cast<std::size_t>(i)];
            }
        }
    }

    return 0;
}

} // namespace ms::cpu::lapack
