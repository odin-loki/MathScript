#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ms::cpu::lapack {

namespace {

void swap_rows(double* A, int lda, int n, int r0, int r1) {
    for (int j = 0; j < n; ++j) {
        double* col = A + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda);
        std::swap(col[r0], col[r1]);
    }
}

} // namespace

int dgetrf(int m, int n, double* A, int lda, int* ipiv) {
    if (m <= 0 || n <= 0) {
        return 0;
    }
    if (A == nullptr || ipiv == nullptr) {
        return 1;
    }

    const int k_end = (std::min)(m, n);
    for (int k = 0; k < k_end; ++k) {
        ipiv[k] = k;
        int pivot = k;
        double max_val = std::abs(A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)]);
        for (int i = k + 1; i < m; ++i) {
            const double val =
                std::abs(A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)]);
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }

        if (max_val < std::numeric_limits<double>::epsilon()) {
            return k + 1;
        }

        if (pivot != k) {
            swap_rows(A, lda, n, k, pivot);
            ipiv[k] = pivot;
        }

        const double pivot_val =
            A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)];
        for (int i = k + 1; i < m; ++i) {
            double* lik = &A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
            *lik /= pivot_val;
            for (int j = k + 1; j < n; ++j) {
                A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] -=
                    *lik * A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)];
            }
        }
    }

    return 0;
}

void dgetrs(char trans, int n, int nrhs, const double* A, int lda, const int* ipiv, double* B, int ldb) {
    if (n <= 0 || nrhs <= 0 || A == nullptr || ipiv == nullptr || B == nullptr) {
        return;
    }
    if (trans == 'T' || trans == 't') {
        return;
    }

    for (int rhs = 0; rhs < nrhs; ++rhs) {
        double* x = B + static_cast<std::size_t>(rhs) * static_cast<std::size_t>(ldb);

        for (int k = 0; k < n; ++k) {
            if (ipiv[k] != k) {
                std::swap(x[k], x[ipiv[k]]);
            }
        }

        for (int k = 0; k < n - 1; ++k) {
            for (int i = k + 1; i < n; ++i) {
                const double lik =
                    A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
                x[i] -= lik * x[k];
            }
        }

        for (int i = n - 1; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                x[i] -= A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                        x[j];
            }
            x[i] /= A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
        }
    }
}

} // namespace ms::cpu::lapack
