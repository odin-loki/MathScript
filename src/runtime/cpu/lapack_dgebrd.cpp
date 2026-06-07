#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>
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

double generate_reflector_strided(int n, double* alpha, int incx) {
    if (n <= 1) {
        return 0.0;
    }

    double sigma = 0.0;
    for (int i = 1; i < n; ++i) {
        const double xi = alpha[i * incx];
        sigma += xi * xi;
    }
    sigma = std::sqrt(sigma);
    if (sigma == 0.0) {
        return 0.0;
    }

    const double x0 = alpha[0];
    const double beta = -std::copysign(std::hypot(x0, sigma), x0);
    const double tau = (beta - x0) / beta;
    const double denom = x0 - beta;

    for (int i = 1; i < n; ++i) {
        alpha[i * incx] /= denom;
    }
    alpha[0] = beta;
    return tau;
}

void apply_reflector_left_sub(
    double* A,
    int lda,
    int r0,
    int r1,
    int c0,
    int c1,
    int v_r,
    int v_c,
    int v_inc,
    double tau) {
    const int m = r1 - r0;
    const int n = c1 - c0;
    if (tau == 0.0 || n == 0) {
        return;
    }

    int lastv = m;
    while (lastv > 1) {
        const double vi = (v_inc == 1) ? A[(v_r + lastv - 1) + v_c * lda] : A[v_r + (v_c + lastv - 1) * lda];
        if (vi != 0.0) {
            break;
        }
        --lastv;
    }

    int lastc = 0;
    for (int j = c0; j < c1; ++j) {
        for (int i = r0; i < r0 + lastv; ++i) {
            if (A[i + j * lda] != 0.0) {
                lastc = j - c0 + 1;
            }
        }
    }
    if (lastc == 0) {
        return;
    }

    if (lastv == 1) {
        for (int j = c0; j < c0 + lastc; ++j) {
            A[r0 + j * lda] *= 1.0 - tau;
        }
        return;
    }

    std::vector<double> work(static_cast<std::size_t>(lastc));
    for (int jj = 0; jj < lastc; ++jj) {
        const int j = c0 + jj;
        double sum = A[r0 + j * lda];
        for (int ii = 1; ii < lastv; ++ii) {
            const int i = r0 + ii;
            const double vi = (v_inc == 1) ? A[(v_r + ii) + v_c * lda] : A[v_r + (v_c + ii) * lda];
            sum += A[i + j * lda] * vi;
        }
        work[static_cast<std::size_t>(jj)] = sum;
    }

    for (int jj = 0; jj < lastc; ++jj) {
        const int j = c0 + jj;
        A[r0 + j * lda] -= tau * work[static_cast<std::size_t>(jj)];
    }

    for (int ii = 1; ii < lastv; ++ii) {
        const int i = r0 + ii;
        const double vi = (v_inc == 1) ? A[(v_r + ii) + v_c * lda] : A[v_r + (v_c + ii) * lda];
        for (int jj = 0; jj < lastc; ++jj) {
            const int j = c0 + jj;
            A[i + j * lda] -= tau * vi * work[static_cast<std::size_t>(jj)];
        }
    }
}

void apply_reflector_right_sub(
    double* A,
    int lda,
    int r0,
    int r1,
    int c0,
    int c1,
    int v_r,
    int v_c,
    int v_inc,
    double tau) {
    const int m = r1 - r0;
    const int n = c1 - c0;
    if (tau == 0.0 || m == 0) {
        return;
    }

    int lastv = n;
    while (lastv > 1) {
        const double vj = (v_inc == 1) ? A[(v_r + lastv - 1) + v_c * lda] : A[v_r + (v_c + lastv - 1) * lda];
        if (vj != 0.0) {
            break;
        }
        --lastv;
    }

    int lastc = 0;
    for (int i = r0; i < r1; ++i) {
        for (int j = c0; j < c0 + lastv; ++j) {
            if (A[i + j * lda] != 0.0) {
                lastc = i - r0 + 1;
            }
        }
    }
    if (lastc == 0) {
        return;
    }

    if (lastv == 1) {
        for (int i = r0; i < r0 + lastc; ++i) {
            A[i + c0 * lda] *= 1.0 - tau;
        }
        return;
    }

    std::vector<double> work(static_cast<std::size_t>(lastc));
    for (int ii = 0; ii < lastc; ++ii) {
        const int i = r0 + ii;
        double sum = A[i + c0 * lda];
        for (int jj = 1; jj < lastv; ++jj) {
            const int j = c0 + jj;
            const double vj = (v_inc == 1) ? A[(v_r + jj) + v_c * lda] : A[v_r + (v_c + jj) * lda];
            sum += A[i + j * lda] * vj;
        }
        work[static_cast<std::size_t>(ii)] = sum;
    }

    for (int ii = 0; ii < lastc; ++ii) {
        const int i = r0 + ii;
        A[i + c0 * lda] -= tau * work[static_cast<std::size_t>(ii)];
    }

    for (int jj = 1; jj < lastv; ++jj) {
        const int j = c0 + jj;
        const double vj = (v_inc == 1) ? A[(v_r + jj) + v_c * lda] : A[v_r + (v_c + jj) * lda];
        for (int ii = 0; ii < lastc; ++ii) {
            const int i = r0 + ii;
            A[i + j * lda] -= tau * work[static_cast<std::size_t>(ii)] * vj;
        }
    }
}

int dgebd2(int m, int n, double* A, int lda, double* D, double* E, double* tauq, double* taup) {
    if (m <= 0 || n <= 0) {
        return 0;
    }
    if (A == nullptr || D == nullptr || E == nullptr || tauq == nullptr || taup == nullptr) {
        return 1;
    }

    const int k = (std::min)(m, n);

    if (m >= n) {
        for (int i = 0; i < n; ++i) {
            double* pivot = &A[i + i * lda];
            tauq[i] = generate_reflector(m - i, pivot);

            D[i] = A[i + i * lda];

            if (i < n - 1) {
                apply_reflector_left_sub(A, lda, i, m, i + 1, n, i, i, 1, tauq[i]);

                double* row_pivot = &A[i + (i + 1) * lda];
                taup[i] = generate_reflector_strided(n - i - 1, row_pivot, lda);
                E[i] = A[i + (i + 1) * lda];

                apply_reflector_right_sub(A, lda, i + 1, m, i + 1, n, i, i + 1, lda, taup[i]);
            } else {
                taup[i] = 0.0;
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            double* pivot = &A[i + i * lda];
            taup[i] = generate_reflector_strided(n - i, pivot, lda);

            D[i] = A[i + i * lda];

            if (i < m - 1) {
                apply_reflector_right_sub(A, lda, i + 1, m, i, n, i, i, lda, taup[i]);

                double* col_pivot = &A[(i + 1) + i * lda];
                tauq[i] = generate_reflector(m - i - 1, col_pivot);
                E[i] = A[(i + 1) + i * lda];

                apply_reflector_left_sub(A, lda, i + 1, m, i + 1, n, i + 1, i, 1, tauq[i]);
            } else {
                tauq[i] = 0.0;
            }
        }
    }

    return 0;
}

} // namespace

int dgebrd(int m, int n, double* A, int lda, double* D, double* E, double* tauq, double* taup) {
    return dgebd2(m, n, A, lda, D, E, tauq, taup);
}

} // namespace ms::cpu::lapack
