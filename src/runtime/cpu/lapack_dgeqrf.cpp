#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>

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

void apply_reflector_left(
    int m,
    int n,
    int k,
    double* A,
    int lda,
    double tau) {
    if (tau == 0.0) {
        return;
    }

    for (int j = k + 1; j < n; ++j) {
        double dot = A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)];
        for (int i = k + 1; i < m; ++i) {
            dot += A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                   A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
        }

        A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)] -=
            tau * dot;
        for (int i = k + 1; i < m; ++i) {
            A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] -=
                tau * dot *
                A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
        }
    }
}

void apply_reflector_left_q(
    int m,
    int q_cols,
    int i,
    double* Q,
    int ldq,
    const double* A,
    int lda,
    double tau) {
    if (tau == 0.0) {
        return;
    }

    for (int j = i; j < q_cols; ++j) {
        double dot = Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(i)];
        for (int r = i + 1; r < m; ++r) {
            dot += A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(r)] *
                   Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(r)];
        }

        Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(i)] -= tau * dot;
        for (int r = i + 1; r < m; ++r) {
            Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(r)] -=
                tau * dot *
                A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(r)];
        }
    }
}

} // namespace

int dgeqrf(int m, int n, double* A, int lda, double* tau) {
    if (m <= 0 || n <= 0) {
        return 0;
    }
    if (A == nullptr || tau == nullptr) {
        return 1;
    }

    const int k_end = (std::min)(m, n);
    for (int k = 0; k < k_end; ++k) {
        double* pivot = &A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)];
        tau[k] = generate_reflector(m - k, pivot);
        apply_reflector_left(m, n, k, A, lda, tau[k]);
    }

    return 0;
}

void dorgqr(int m, int n, int k, const double* A, int lda, const double* tau, double* Q, int ldq) {
    if (m <= 0 || k <= 0 || A == nullptr || tau == nullptr || Q == nullptr) {
        return;
    }

    const int q_cols = (std::min)(k, n);
    for (int j = 0; j < q_cols; ++j) {
        for (int i = 0; i < m; ++i) {
            Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(i)] =
                (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = q_cols - 1; i >= 0; --i) {
        apply_reflector_left_q(m, q_cols, i, Q, ldq, A, lda, tau[i]);
    }
}

} // namespace ms::cpu::lapack
