#include "ms/cpu/blas.hpp"
#include "ms/simd/simd.hpp"

#include <algorithm>
#include <cstddef>

namespace ms::cpu::blas {

namespace {

bool is_lower(char uplo) {
    return uplo == 'L' || uplo == 'l';
}

bool is_upper(char uplo) {
    return uplo == 'U' || uplo == 'u';
}

bool is_left(char side) {
    return side == 'L' || side == 'l';
}

bool is_no_transpose(char trans) {
    return trans == 'N' || trans == 'n';
}

bool is_unit_diag(char diag) {
    return diag == 'U' || diag == 'u';
}

void scale_triangle(char uplo, int n, double beta, double* C, int ldc) {
    if (beta == 1.0) {
        return;
    }
    if (is_lower(uplo)) {
        for (int j = 0; j < n; ++j) {
            double* col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(j);
            const int len = n - j;
            if (beta == 0.0) {
                std::fill(col, col + len, 0.0);
            } else {
                ms::simd::scale(beta, {col, static_cast<std::size_t>(len)}, {col, static_cast<std::size_t>(len)});
            }
        }
        return;
    }
    for (int j = 0; j < n; ++j) {
        double* col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc);
        const int len = j + 1;
        if (beta == 0.0) {
            std::fill(col, col + len, 0.0);
        } else {
            ms::simd::scale(beta, {col, static_cast<std::size_t>(len)}, {col, static_cast<std::size_t>(len)});
        }
    }
}

void dsyrk_ln(
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double* C,
    int ldc) {
    if (alpha == 0.0 || k == 0) {
        return;
    }
    for (int p = 0; p < k; ++p) {
        const double* ap = A + static_cast<std::size_t>(p) * static_cast<std::size_t>(lda);
        for (int j = 0; j < n; ++j) {
            const double scale = alpha * ap[j];
            if (scale == 0.0) {
                continue;
            }
            double* c_col = C + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(j);
            ms::simd::axpy(scale, {ap + j, static_cast<std::size_t>(n - j)}, {c_col, static_cast<std::size_t>(n - j)});
        }
    }
}

void dsyrk_generic(
    char uplo,
    char trans,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double beta,
    double* C,
    int ldc) {
    scale_triangle(uplo, n, beta, C, ldc);
    if (alpha == 0.0 || k == 0) {
        return;
    }

    for (int j = 0; j < n; ++j) {
        const int i_start = is_lower(uplo) ? j : 0;
        const int i_end = is_lower(uplo) ? n : j + 1;
        for (int i = i_start; i < i_end; ++i) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                const double a_ip = is_no_transpose(trans)
                                        ? A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                                            static_cast<std::size_t>(i)]
                                        : A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) +
                                            static_cast<std::size_t>(p)];
                const double a_jp = is_no_transpose(trans)
                                        ? A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) +
                                            static_cast<std::size_t>(j)]
                                        : A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) +
                                            static_cast<std::size_t>(p)];
                sum += a_ip * a_jp;
            }
            C[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i)] +=
                alpha * sum;
        }
    }
}

void dtrsm_left_lower(
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    char diag,
    double* B,
    int ldb) {
    for (int j = 0; j < n; ++j) {
        double* b_col = B + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
        for (int i = 0; i < m; ++i) {
            double sum = alpha * b_col[i];
            for (int k = 0; k < i; ++k) {
                sum -= A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                       b_col[k];
            }
            const double diag_val =
                is_unit_diag(diag) ? 1.0
                                   : A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) +
                                       static_cast<std::size_t>(i)];
            b_col[i] = sum / diag_val;
        }
    }
}

void dtrsm_generic(
    char side,
    char uplo,
    char transa,
    char diag,
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    double* B,
    int ldb) {
    auto fetch_a = [&](int i, int k) {
        if (is_no_transpose(transa)) {
            return A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
        }
        return A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(k)];
    };
    auto diag_at = [&](int i) {
        if (is_unit_diag(diag)) {
            return 1.0;
        }
        return fetch_a(i, i);
    };

    if (is_left(side) && is_lower(uplo) && is_no_transpose(transa)) {
        dtrsm_left_lower(m, n, alpha, A, lda, diag, B, ldb);
        return;
    }

    if (is_left(side) && is_no_transpose(transa)) {
        for (int j = 0; j < n; ++j) {
            double* b_col = B + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
            for (int i = m; i-- > 0;) {
                double sum = alpha * b_col[i];
                const int k_start = is_lower(uplo) ? 0 : i + 1;
                const int k_end = is_lower(uplo) ? i : m;
                for (int k = k_start; k < k_end; ++k) {
                    if (k == i) {
                        continue;
                    }
                    sum -= fetch_a(i, k) * b_col[k];
                }
                if (is_lower(uplo)) {
                    sum -= fetch_a(i, i) * b_col[i];
                    b_col[i] = sum / diag_at(i);
                } else {
                    b_col[i] = sum / diag_at(i);
                }
            }
        }
        return;
    }

    for (int j = 0; j < n; ++j) {
        double* b_col = B + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
        for (int i = 0; i < m; ++i) {
            double sum = alpha * b_col[i];
            for (int k = 0; k < m; ++k) {
                if (k == i) {
                    continue;
                }
                sum -= fetch_a(k, i) * b_col[k];
            }
            sum -= fetch_a(i, i) * b_col[i];
            b_col[i] = sum / diag_at(i);
        }
    }
}

} // namespace

void dsyrk(
    char uplo,
    char trans,
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double beta,
    double* C,
    int ldc) {
    if (n <= 0) {
        return;
    }
    if (A == nullptr || C == nullptr || k < 0) {
        return;
    }

    scale_triangle(uplo, n, beta, C, ldc);
    if (is_lower(uplo) && is_no_transpose(trans)) {
        dsyrk_ln(n, k, alpha, A, lda, C, ldc);
        return;
    }
    dsyrk_generic(uplo, trans, n, k, alpha, A, lda, 1.0, C, ldc);
}

void dtrsm(
    char side,
    char uplo,
    char transa,
    char diag,
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    double* B,
    int ldb) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (A == nullptr || B == nullptr) {
        return;
    }
    if (alpha == 0.0) {
        for (int j = 0; j < n; ++j) {
            double* b_col = B + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
            std::fill(b_col, b_col + m, 0.0);
        }
        return;
    }

    if (is_left(side) && is_lower(uplo) && is_no_transpose(transa)) {
        dtrsm_left_lower(m, n, alpha, A, lda, diag, B, ldb);
        return;
    }

    dtrsm_generic(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

} // namespace ms::cpu::blas
