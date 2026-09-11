// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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

// Full triangular solve covering every (side, uplo, transa, diag) combination the
// header documents:
//
//   side='L':  op(A) * X = alpha * B,  A is m-by-m, B is m-by-n
//   side='R':  X * op(A) = alpha * B,  A is n-by-n, B is m-by-n
//
// op(A) = A for transa='N' and A^T otherwise.  Only the triangle named by uplo is
// referenced; with diag='U' the diagonal is not read and is taken to be 1.
// op(A) is lower triangular exactly when (uplo names the lower triangle) matches
// (transa is 'N'), which fixes the substitution direction.
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
    const bool notrans = is_no_transpose(transa);
    const bool unit = is_unit_diag(diag);
    const std::size_t la = static_cast<std::size_t>(lda);
    const std::size_t lb = static_cast<std::size_t>(ldb);

    // op(A)(r, c) read from the stored triangle.
    auto op_a = [&](int r, int c) -> double {
        if (notrans) {
            return A[static_cast<std::size_t>(c) * la + static_cast<std::size_t>(r)];
        }
        return A[static_cast<std::size_t>(r) * la + static_cast<std::size_t>(c)];
    };
    auto op_diag = [&](int i) -> double {
        if (unit) {
            return 1.0;
        }
        return op_a(i, i);
    };

    const bool op_lower = (is_lower(uplo) == notrans);

    if (is_left(side)) {
        for (int j = 0; j < n; ++j) {
            double* b_col = B + static_cast<std::size_t>(j) * lb;
            if (op_lower) {
                // Forward substitution.
                for (int i = 0; i < m; ++i) {
                    double sum = alpha * b_col[i];
                    for (int k = 0; k < i; ++k) {
                        sum -= op_a(i, k) * b_col[k];
                    }
                    b_col[i] = sum / op_diag(i);
                }
            } else {
                // Back substitution.
                for (int i = m - 1; i >= 0; --i) {
                    double sum = alpha * b_col[i];
                    for (int k = i + 1; k < m; ++k) {
                        sum -= op_a(i, k) * b_col[k];
                    }
                    b_col[i] = sum / op_diag(i);
                }
            }
        }
        return;
    }

    // side='R': X * op(A) = alpha * B.  Row i of X satisfies
    //   sum_k X(i,k) * op(A)(k,j) = alpha * B(i,j),
    // so each row is a triangular solve against op(A)^T, swept over columns in the
    // order dictated by which triangle op(A) occupies.
    for (int i = 0; i < m; ++i) {
        if (op_lower) {
            for (int j = n - 1; j >= 0; --j) {
                double sum = alpha * B[static_cast<std::size_t>(j) * lb + static_cast<std::size_t>(i)];
                for (int k = j + 1; k < n; ++k) {
                    sum -= B[static_cast<std::size_t>(k) * lb + static_cast<std::size_t>(i)] * op_a(k, j);
                }
                B[static_cast<std::size_t>(j) * lb + static_cast<std::size_t>(i)] = sum / op_diag(j);
            }
        } else {
            for (int j = 0; j < n; ++j) {
                double sum = alpha * B[static_cast<std::size_t>(j) * lb + static_cast<std::size_t>(i)];
                for (int k = 0; k < j; ++k) {
                    sum -= B[static_cast<std::size_t>(k) * lb + static_cast<std::size_t>(i)] * op_a(k, j);
                }
                B[static_cast<std::size_t>(j) * lb + static_cast<std::size_t>(i)] = sum / op_diag(j);
            }
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
    // Reject argument combinations outside the documented domain rather than
    // silently solving a different system; dtrsm is void, so B is left untouched.
    if (!is_left(side) && !(side == 'R' || side == 'r')) {
        return;
    }
    if (!is_lower(uplo) && !is_upper(uplo)) {
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
