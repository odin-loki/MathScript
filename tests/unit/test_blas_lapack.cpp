#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "ms/cpu/blas.hpp"
#include "ms/cpu/lapack.hpp"
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;

namespace {

void expect_orthogonal_columns(const ColMatrix<double>& M, double tol = 1e-9) {
    const ColMatrix<double> mt_m = transpose(M) * M;
    const int n = static_cast<int>(M.cols());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(mt_m(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), expected, tol);
        }
    }
}

void expect_orthogonal_rows(const ColMatrix<double>& M, double tol = 1e-9) {
    const ColMatrix<double> m_mt = M * transpose(M);
    const int m = static_cast<int>(M.rows());
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(m_mt(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), expected, tol);
        }
    }
}

ColMatrix<double> spd_matrix(int n) {
    ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            const double v = 0.05 * static_cast<double>(i + 1) + 0.01 * static_cast<double>(j + 1);
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = v;
            A(static_cast<std::size_t>(j), static_cast<std::size_t>(i)) = v;
        }
        A(static_cast<std::size_t>(j), static_cast<std::size_t>(j)) += static_cast<double>(n);
    }
    return A;
}

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

void reference_dsyrk(
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
    auto fetch = [&](int i, int p) {
        if (is_no_transpose(trans)) {
            return A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
        }
        return A[static_cast<std::size_t>(i) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(p)];
    };
    for (int j = 0; j < n; ++j) {
        const int i_start = is_lower(uplo) ? j : 0;
        const int i_end = is_lower(uplo) ? n : j + 1;
        for (int i = i_start; i < i_end; ++i) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += fetch(i, p) * fetch(j, p);
            }
            const std::size_t idx =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i);
            C[idx] = alpha * sum + beta * C[idx];
        }
    }
}

double tri_entry(const ColMatrix<double>& A, char uplo, char trans, int i, int j) {
    int ri = i;
    int rj = j;
    if (!is_no_transpose(trans)) {
        std::swap(ri, rj);
    }
    if (is_lower(uplo)) {
        if (ri < rj) {
            return 0.0;
        }
    } else if (ri > rj) {
        return 0.0;
    }
    return A(static_cast<std::size_t>(ri), static_cast<std::size_t>(rj));
}

ColMatrix<double> op_triangular(const ColMatrix<double>& A, char uplo, char trans) {
    const int n = static_cast<int>(A.rows());
    ColMatrix<double> op(static_cast<std::size_t>(n), static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            op(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = tri_entry(A, uplo, trans, i, j);
        }
    }
    return op;
}

void reference_dtrsm_left(
    char uplo,
    char trans,
    char diag,
    int m,
    int n,
    double alpha,
    const ColMatrix<double>& A,
    const ColMatrix<double>& B_in,
    ColMatrix<double>& X_out) {
    const ColMatrix<double> op = op_triangular(A, uplo, trans);
    X_out = B_in;
    for (int col = 0; col < n; ++col) {
        if (is_lower(uplo) && is_no_transpose(trans)) {
            for (int i = 0; i < m; ++i) {
                double sum = alpha * B_in(static_cast<std::size_t>(i), static_cast<std::size_t>(col));
                for (int k = 0; k < i; ++k) {
                    sum -= op(static_cast<std::size_t>(i), static_cast<std::size_t>(k)) *
                           X_out(static_cast<std::size_t>(k), static_cast<std::size_t>(col));
                }
                const double d = is_unit_diag(diag) ? 1.0 : op(static_cast<std::size_t>(i), static_cast<std::size_t>(i));
                X_out(static_cast<std::size_t>(i), static_cast<std::size_t>(col)) = sum / d;
            }
        } else if (is_upper(uplo) && is_no_transpose(trans)) {
            for (int i = m - 1; i >= 0; --i) {
                double sum = alpha * B_in(static_cast<std::size_t>(i), static_cast<std::size_t>(col));
                for (int k = i + 1; k < m; ++k) {
                    sum -= op(static_cast<std::size_t>(i), static_cast<std::size_t>(k)) *
                           X_out(static_cast<std::size_t>(k), static_cast<std::size_t>(col));
                }
                const double d = is_unit_diag(diag) ? 1.0 : op(static_cast<std::size_t>(i), static_cast<std::size_t>(i));
                X_out(static_cast<std::size_t>(i), static_cast<std::size_t>(col)) = sum / d;
            }
        } else {
            for (int i = 0; i < m; ++i) {
                double sum = alpha * B_in(static_cast<std::size_t>(i), static_cast<std::size_t>(col));
                for (int k = 0; k < i; ++k) {
                    sum -= op(static_cast<std::size_t>(i), static_cast<std::size_t>(k)) *
                           X_out(static_cast<std::size_t>(k), static_cast<std::size_t>(col));
                }
                const double d = is_unit_diag(diag) ? 1.0 : op(static_cast<std::size_t>(i), static_cast<std::size_t>(i));
                X_out(static_cast<std::size_t>(i), static_cast<std::size_t>(col)) = sum / d;
            }
        }
    }
}

void reference_dtrsm_generic(
    char side,
    char uplo,
    char trans,
    char diag,
    int m,
    int n,
    double alpha,
    const double* A,
    int lda,
    const double* B_in,
    int ldb,
    double* B_out) {
    auto fetch_a = [&](int i, int k) {
        if (is_no_transpose(trans)) {
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

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            B_out[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(i)] =
                B_in[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb) + static_cast<std::size_t>(i)];
        }
    }

    if (is_left(side) && is_no_transpose(trans)) {
        for (int j = 0; j < n; ++j) {
            double* b_col = B_out + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
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
        double* b_col = B_out + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldb);
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

void expect_dtrsm_matches_reference(
    char side,
    char uplo,
    char trans,
    char diag,
    int m,
    int n,
    double alpha,
    const ColMatrix<double>& A,
    const ColMatrix<double>& B_in) {
    ColMatrix<double> got = B_in;
    ColMatrix<double> ref = B_in;
    const int lda = static_cast<int>(A.rows());
    const int ldb = static_cast<int>(got.rows());
    cpu::blas::dtrsm(side, uplo, trans, diag, m, n, alpha, A.data(), lda, got.data(), ldb);
    if (is_left(side) && is_lower(uplo) && is_no_transpose(trans)) {
        reference_dtrsm_left(uplo, trans, diag, m, n, alpha, A, B_in, ref);
    } else {
        reference_dtrsm_generic(
            side, uplo, trans, diag, m, n, alpha, A.data(), lda, B_in.data(), ldb, ref.data());
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            EXPECT_NEAR(got(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                        ref(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                        1e-10)
                << "side=" << side << " uplo=" << uplo << " trans=" << trans << " at (" << i << "," << j << ")";
        }
    }
}

void reference_dsyrk_ln(
    int n,
    int k,
    double alpha,
    const double* A,
    int lda,
    double beta,
    double* C,
    int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                       A[static_cast<std::size_t>(p) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(j)];
            }
            const std::size_t idx =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(ldc) + static_cast<std::size_t>(i);
            C[idx] = alpha * sum + beta * C[idx];
        }
    }
}

void check_gebrd_svals(int m, int n, const ColMatrix<double>& M, const std::vector<double>& s_ref_in) {
    ColMatrix<double> A = M;
    const int k = (std::min)(m, n);
    std::vector<double> D(static_cast<std::size_t>(k));
    std::vector<double> E(static_cast<std::size_t>((std::max)(0, k - 1)));
    std::vector<double> tauq(static_cast<std::size_t>(k));
    std::vector<double> taup(static_cast<std::size_t>(k));

    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> gram_d(static_cast<std::size_t>(k));
    std::vector<double> gram_e(static_cast<std::size_t>((std::max)(0, k - 1)));
    for (int i = 0; i < k; ++i) {
        gram_d[static_cast<std::size_t>(i)] =
            D[static_cast<std::size_t>(i)] * D[static_cast<std::size_t>(i)] +
            (i > 0 ? E[static_cast<std::size_t>(i - 1)] * E[static_cast<std::size_t>(i - 1)] : 0.0);
    }
    for (int i = 0; i < k - 1; ++i) {
        gram_e[static_cast<std::size_t>(i)] = D[static_cast<std::size_t>(i)] * E[static_cast<std::size_t>(i)];
    }

    cpu::lapack::dsteqr(k, gram_d.data(), gram_e.data(), nullptr, 0);

    std::vector<double> s_ref = s_ref_in;
    std::sort(s_ref.begin(), s_ref.end(), std::greater<double>());

    ASSERT_EQ(static_cast<int>(s_ref.size()), k);
    for (int i = 0; i < k; ++i) {
        const double got = std::sqrt((std::max)(0.0, gram_d[static_cast<std::size_t>(k - 1 - i)]));
        EXPECT_NEAR(got, s_ref[static_cast<std::size_t>(i)], 1e-9);
    }
}

void exercise_dbdsqr_from_matrix(int n, char uplo, bool with_vectors) {
    ColMatrix<double> M(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            M(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                1.0 / static_cast<double>(i + j + 1) + 0.01 * static_cast<double>(i - j);
        }
    }

    ColMatrix<double> A = M;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    if (!with_vectors) {
        ASSERT_EQ(cpu::lapack::dbdsqr(uplo, n, d.data(), e.data()), 0);
        for (int i = 0; i < n; ++i) {
            EXPECT_GT(d[static_cast<std::size_t>(i)], 0.0);
        }
        return;
    }

    ColMatrix<double> Ub(n, n);
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr(uplo, n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);
    for (int i = 0; i < n - 1; ++i) {
        EXPECT_GE(d[static_cast<std::size_t>(i)], d[static_cast<std::size_t>(i + 1)] - 1e-9);
    }
}

void exercise_dbdsqr_tall(int m, int n, char uplo) {
    ColMatrix<double> M(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            M(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                std::sin(0.3 * static_cast<double>(i + 1)) + 0.1 * static_cast<double>(j);
        }
    }

    ColMatrix<double> A = M;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ColMatrix<double> Ub(n, n);
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr(uplo, n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);
    EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
}

} // namespace

TEST(BlasLevel3Test, dsyrk_lower_no_transpose) {
    constexpr int n = 8;
    constexpr int k = 5;
    ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.1 * static_cast<double>(i) - 0.02 * static_cast<double>(j);
        }
    }

    std::vector<double> C(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.25);
    std::vector<double> ref = C;
    cpu::blas::dsyrk('L', 'N', n, k, 1.0, A.data(), n, 1.0, C.data(), n);
    reference_dsyrk_ln(n, k, 1.0, A.data(), n, 1.0, ref.data(), n);

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            EXPECT_NEAR(C[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        ref[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        1e-10);
        }
    }
}

TEST(BlasLevel3Test, dtrsm_left_lower_solve) {
    ColMatrix<double> L{{2, 0, 0}, {1, 2, 0}, {0, 1, 3}};
    ColMatrix<double> B{{4, 2}, {6, 4}, {3, 6}};

    cpu::blas::dtrsm('L', 'L', 'N', 'N', 3, 2, 1.0, L.data(), 3, B.data(), 3);

    EXPECT_NEAR(B(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(B(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(B(2, 0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(B(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(B(1, 1), 1.5, 1e-12);
    EXPECT_NEAR(B(2, 1), 1.5, 1e-12);
}

TEST(BlasLevel3Test, dsyrk_upper_no_transpose) {
    constexpr int n = 6;
    constexpr int k = 4;
    ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.07 * static_cast<double>(i) + 0.03 * static_cast<double>(j);
        }
    }

    std::vector<double> C(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.5);
    std::vector<double> ref = C;
    cpu::blas::dsyrk('U', 'N', n, k, 1.0, A.data(), n, 1.0, C.data(), n);
    reference_dsyrk('U', 'N', n, k, 1.0, A.data(), n, 1.0, ref.data(), n);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            EXPECT_NEAR(C[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        ref[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        1e-10);
        }
    }
}

TEST(BlasLevel3Test, dsyrk_lower_transpose) {
    constexpr int n = 5;
    constexpr int k = 3;
    ColMatrix<double> A(static_cast<std::size_t>(k), static_cast<std::size_t>(n));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.11 * static_cast<double>(i) - 0.04 * static_cast<double>(j);
        }
    }

    std::vector<double> C(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    std::vector<double> ref = C;
    cpu::blas::dsyrk('L', 'T', n, k, 1.0, A.data(), k, 0.0, C.data(), n);
    reference_dsyrk('L', 'T', n, k, 1.0, A.data(), k, 0.0, ref.data(), n);

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            EXPECT_NEAR(C[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        ref[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        1e-10);
        }
    }
}

TEST(BlasLevel3Test, dsyrk_upper_beta_zero_clears_triangle) {
    constexpr int n = 4;
    constexpr int k = 2;
    ColMatrix<double> A{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::vector<double> C(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 99.0);
    cpu::blas::dsyrk('U', 'N', n, k, 1.0, A.data(), n, 0.0, C.data(), n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            EXPECT_NE(C[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)], 99.0);
        }
    }
}

TEST(BlasLevel3Test, dtrsm_left_upper_recovers_solution) {
    ColMatrix<double> U{{2, 1, 0}, {0, 2, 1}, {0, 0, 3}};
    const ColMatrix<double> X{{1, 2}, {3, 4}, {5, 6}};
    ColMatrix<double> B = U * X;
    cpu::blas::dtrsm('L', 'U', 'N', 'N', 3, 2, 1.0, U.data(), 3, B.data(), 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(B(i, j), X(i, j), 1e-12);
        }
    }
}

TEST(BlasLevel3Test, dtrsm_right_lower_matches_reference) {
    ColMatrix<double> L{{2, 0}, {1, 3}};
    ColMatrix<double> B{{4, 2}, {6, 4}};
    expect_dtrsm_matches_reference('R', 'L', 'N', 'N', 2, 2, 1.0, L, B);
}

TEST(BlasLevel3Test, dtrsm_alpha_zero_zeros_rhs) {
    ColMatrix<double> L{{2, 0}, {1, 3}};
    ColMatrix<double> B{{5, 6}, {7, 8}};
    cpu::blas::dtrsm('R', 'L', 'N', 'N', 2, 2, 0.0, L.data(), 2, B.data(), 2);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(B(i, j), 0.0);
        }
    }
}

TEST(LapackPotrfTest, dpotrf_and_chol_agree) {
    const ColMatrix<double> A = spd_matrix(6);
    ColMatrix<double> factor = A;
    ASSERT_EQ(cpu::lapack::dpotrf('L', 6, factor.data(), 6), 0);

    const auto L = chol(A).value();
    for (int j = 0; j < 6; ++j) {
        for (int i = j; i < 6; ++i) {
            EXPECT_NEAR(L(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                        factor(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), 1e-10);
        }
    }

    for (int j = 0; j < 6; ++j) {
        for (int i = 0; i < 6; ++i) {
            double sum = 0.0;
            for (int k = 0; k < 6; ++k) {
                const double lik = i >= k ? L(static_cast<std::size_t>(i), static_cast<std::size_t>(k)) : 0.0;
                const double ljk = j >= k ? L(static_cast<std::size_t>(j), static_cast<std::size_t>(k)) : 0.0;
                sum += lik * ljk;
            }
            EXPECT_NEAR(sum, A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), 1e-9);
        }
    }
}

TEST(LapackPotrfTest, detects_indefinite_matrix) {
    ColMatrix<double> A{{1, 2}, {2, 1}};
    EXPECT_EQ(cpu::lapack::dpotrf('L', 2, A.data(), 2), 2);
    EXPECT_FALSE(chol(A).has_value());
}

TEST(LapackGetrfTest, dgetrs_solves_linear_system) {
    ColMatrix<double> A{{2, 1}, {1, 3}};
    ColMatrix<double> B{{4}, {7}};
    std::vector<int> ipiv(2);
    std::iota(ipiv.begin(), ipiv.end(), 0);

    ASSERT_EQ(cpu::lapack::dgetrf(2, 2, A.data(), 2, ipiv.data()), 0);
    cpu::lapack::dgetrs('N', 2, 1, A.data(), 2, ipiv.data(), B.data(), 2);

    EXPECT_NEAR(B(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(B(1, 0), 2.0, 1e-10);
}

TEST(BlasLevel2Test, dger_rank1_update) {
    ColMatrix<double> A{{1, 2}, {3, 4}, {5, 6}};
    const std::vector<double> x{1.0, -1.0, 2.0};
    const std::vector<double> y{3.0, -2.0};

    cpu::blas::dger(3, 2, 2.0, x.data(), 1, y.data(), 1, A.data(), 3);

    EXPECT_NEAR(A(0, 0), 1.0 + 2.0 * 1.0 * 3.0, 1e-12);
    EXPECT_NEAR(A(1, 0), 3.0 + 2.0 * (-1.0) * 3.0, 1e-12);
    EXPECT_NEAR(A(2, 0), 5.0 + 2.0 * 2.0 * 3.0, 1e-12);
    EXPECT_NEAR(A(0, 1), 2.0 + 2.0 * 1.0 * (-2.0), 1e-12);
    EXPECT_NEAR(A(1, 1), 4.0 + 2.0 * (-1.0) * (-2.0), 1e-12);
    EXPECT_NEAR(A(2, 1), 6.0 + 2.0 * 2.0 * (-2.0), 1e-12);
}

TEST(BlasLevel2Test, dger_non_unit_stride_and_zero_alpha) {
    const std::vector<double> x{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> y{1.0, 0.5, 1.5, 0.25};
    ColMatrix<double> A(2, 2, 0.0);
    cpu::blas::dger(2, 2, 0.0, x.data(), 2, y.data(), 2, A.data(), 2);
    EXPECT_DOUBLE_EQ(A(0, 0), 0.0);
    cpu::blas::dger(2, 2, 1.0, x.data(), 2, y.data(), 2, A.data(), 2);
    EXPECT_NEAR(A(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(A(1, 1), 4.5, 1e-12);
}

TEST(LapackGetrfTest, dgesv_solves_multiple_rhs) {
    ColMatrix<double> A{{2, 1}, {1, 3}};
    ColMatrix<double> B{{4, 1}, {7, 2}};
    std::vector<int> ipiv(2);

    ASSERT_EQ(cpu::lapack::dgesv(2, 2, A.data(), 2, ipiv.data(), B.data(), 2), 0);
    EXPECT_NEAR(B(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(B(1, 0), 2.0, 1e-10);
    EXPECT_NEAR(B(0, 1), 0.2, 1e-10);
    EXPECT_NEAR(B(1, 1), 0.6, 1e-10);
}

TEST(LapackQrfTest, dgeqrf_dorgqr_reconstructs_matrix) {
    ColMatrix<double> A{{4, 2, 1}, {1, 3, 2}, {2, 1, 5}, {0, 2, 3}};
    const ColMatrix<double> A_ref = A;
    std::vector<double> tau(3);

    ASSERT_EQ(cpu::lapack::dgeqrf(4, 3, A.data(), 4, tau.data()), 0);

    ColMatrix<double> R(3, 3, 0.0);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i <= j; ++i) {
            R(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }

    ColMatrix<double> Q(4, 3, 0.0);
    cpu::lapack::dorgqr(4, 3, 3, A.data(), 4, tau.data(), Q.data(), 4);

    const ColMatrix<double> prod = Q * R;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(prod(i, j), A_ref(i, j), 1e-10);
        }
    }
}

TEST(LapackQrfTest, qr_uses_householder_path) {
    ColMatrix<double> A{{4, 2, 1}, {1, 3, 2}, {2, 1, 5}, {0, 2, 3}};
    const auto [Q, R] = qr(A).value();
    const ColMatrix<double> prod = Q * R;
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(prod(i, j), A(i, j), 1e-10);
        }
    }

    const ColMatrix<double> QTQ = transpose(Q) * Q;
    for (std::size_t i = 0; i < Q.cols(); ++i) {
        for (std::size_t j = 0; j < Q.cols(); ++j) {
            EXPECT_NEAR(QTQ(i, j), (i == j) ? 1.0 : 0.0, 1e-10);
        }
    }
}

TEST(LapackGelsTest, dgels_least_squares) {
    ColMatrix<double> A{{1, 1}, {1, 2}, {1, 3}};
    ColMatrix<double> B{{6}, {9}, {12}};

    ASSERT_EQ(cpu::lapack::dgels(3, 2, 1, A.data(), 3, B.data(), 3), 0);
    EXPECT_NEAR(B(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(B(1, 0), 3.0, 1e-9);
}

TEST(LapackGelsTest, dgels_rejects_underdetermined) {
    ColMatrix<double> A{{1, 2, 3}};
    ColMatrix<double> B{{1}};
    EXPECT_EQ(cpu::lapack::dgels(1, 3, 1, A.data(), 1, B.data(), 1), 2);
}

TEST(LapackSyevTest, dsyev_matches_reference) {
    const ColMatrix<double> ref{{4, 1}, {1, 3}};
    ColMatrix<double> A = ref;
    std::vector<double> w(2);
    ASSERT_EQ(cpu::lapack::dsyev('V', 2, A.data(), 2, w.data()), 0);
    EXPECT_NEAR(w[0], 2.381966011250105, 1e-9);
    EXPECT_NEAR(w[1], 4.618033988749895, 1e-9);

    for (std::size_t j = 0; j < 2; ++j) {
        ColMatrix<double> v{{A(0, j)}, {A(1, j)}};
        const ColMatrix<double> Av = ref * v;
        for (std::size_t i = 0; i < 2; ++i) {
            EXPECT_NEAR(Av(i, 0), w[j] * A(i, j), 1e-6);
        }
    }
}

TEST(LapackSyevTest, dsyev_symmetric_4x4) {
    const ColMatrix<double> ref{{4, 1, 0, 0}, {1, 3, 1, 0}, {0, 1, 2, 1}, {0, 0, 1, 1}};
    ColMatrix<double> A = ref;
    std::vector<double> w(4);
    ASSERT_EQ(cpu::lapack::dsyev('V', 4, A.data(), 4, w.data()), 0);

    std::vector<double> expected{0.254718762, 1.822717082, 3.177282918, 4.745281238};
    for (int i = 0; i < 4; ++i) {
        bool matched = false;
        for (int j = 0; j < 4; ++j) {
            if (std::abs(w[static_cast<std::size_t>(i)] - expected[static_cast<std::size_t>(j)]) < 1e-6) {
                matched = true;
                break;
            }
        }
        EXPECT_TRUE(matched);
    }

    for (std::size_t j = 0; j < 4; ++j) {
        ColMatrix<double> v(4, 1);
        for (std::size_t i = 0; i < 4; ++i) {
            v(i, 0) = A(i, j);
        }
        const ColMatrix<double> Av = ref * v;
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(Av(i, 0), w[j] * A(i, j), 1e-5);
        }
    }
}

TEST(LapackSteqrTest, dsteqr_tridiagonal_matches_dsyev) {
    std::vector<double> d{4.0, 3.0, 2.0, 1.0};
    std::vector<double> e{1.0, 0.5, 0.25};

    std::vector<double> d_copy = d;
    std::vector<double> e_copy = e;
    cpu::lapack::dsteqr(4, d_copy.data(), e_copy.data(), nullptr, 0);

    ColMatrix<double> A{{4, 1, 0, 0}, {1, 3, 0.5, 0}, {0, 0.5, 2, 0.25}, {0, 0, 0.25, 1}};
    std::vector<double> w(4);
    ASSERT_EQ(cpu::lapack::dsyev('N', 4, A.data(), 4, w.data()), 0);

    for (int i = 0; i < 4; ++i) {
        bool matched = false;
        for (int j = 0; j < 4; ++j) {
            if (std::abs(d_copy[static_cast<std::size_t>(i)] - w[static_cast<std::size_t>(j)]) < 1e-8) {
                matched = true;
                break;
            }
        }
        EXPECT_TRUE(matched);
    }
}

TEST(LapackGebrdTest, dgebrd_2x2) {
    check_gebrd_svals(2, 2, ColMatrix<double>{{3, 1}, {1, 2}}, {3.6180339887498953, 1.3819660112501049});
}

TEST(LapackGebrdTest, dgebrd_3x2) {
    check_gebrd_svals(
        3,
        2,
        ColMatrix<double>{{0.35, -1.30}, {0.82, 0.91}, {0.33, 0.45}},
        {1.6797085501065459, 0.8960910593789928});
}

TEST(LapackGebrdTest, dgebrd_2x3) {
    check_gebrd_svals(2, 3, ColMatrix<double>{{1.0, 2.0, -1.0}, {0.5, -1.0, 2.0}}, {3.0240753892854886, 1.4508507986412036});
}

TEST(LapackGebrdTest, dgebrd_4x3) {
    check_gebrd_svals(
        4,
        3,
        ColMatrix<double>{
            {0.19, 0.82, -1.30},
            {0.52, -0.54, 0.91},
            {0.78, 0.31, 0.45},
            {0.42, 0.23, -0.76}},
        {2.035319606563809, 1.0908508989139205, 0.41233289400147577});
}

TEST(LapackBdsqrTest, dbdsqr_upper_matches_reference) {
    {
        std::vector<double> d{-0.6526478833927157, 0.14983893167549497};
        std::vector<double> e{-0.07748563762582951};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), e.data()), 0);
        EXPECT_NEAR(d[0], 0.657481712002918, 1e-6);
        EXPECT_NEAR(d[1], 0.148737310594672, 1e-6);
    }
    {
        std::vector<double> d{-1.0238585548996153, 1.584075647578786};
        std::vector<double> e{-0.2514552556898243};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), e.data()), 0);
        EXPECT_NEAR(d[0], 1.617045691104904, 1e-6);
        EXPECT_NEAR(d[1], 1.002983043895096, 1e-6);
    }
}

TEST(LapackBdsqrTest, dbdsqr_upper_2x2_vectors_reconstruct_bidiagonal) {
    std::vector<double> d{-0.6526478833927157, 0.14983893167549497};
    std::vector<double> e{-0.07748563762582951};
    const ColMatrix<double> B{{d[0], e[0]}, {0.0, d[1]}};

    ColMatrix<double> Ub(2, 2);
    ColMatrix<double> VTb(2, 2);
    ASSERT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), e.data(), Ub.data(), 2, VTb.data(), 2), 0);

    ColMatrix<double> Sigma = zeros<double>(2, 2);
    for (int i = 0; i < 2; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = d[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(2, 2);
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                VTb(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    const ColMatrix<double> prod = Ub * Sigma * transpose(V);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), B(i, j), 1e-9);
        }
    }
}

TEST(LapackBdsqrTest, dbdsqr_upper_3x3_from_gebrd) {
    ColMatrix<double> M{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const int m = 4;
    const int n = 3;
    ColMatrix<double> A = M;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ASSERT_EQ(cpu::lapack::dbdsqr('U', n, d.data(), e.data()), 0);
    EXPECT_NEAR(d[0], 2.035319606563809, 1e-6);
    EXPECT_NEAR(d[1], 1.0908508989139205, 1e-6);
    EXPECT_NEAR(d[2], 0.41233289400147577, 1e-6);
}

TEST(LapackBdsqrTest, dbdsqr_upper_3x3_from_gebrd_vectors) {
    ColMatrix<double> M{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const int m = 4;
    const int n = 3;
    ColMatrix<double> A = M;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> B(n, n, 0.0);
    for (int j = 0; j < n; ++j) {
        B(static_cast<std::size_t>(j), static_cast<std::size_t>(j)) = D[static_cast<std::size_t>(j)];
        if (j + 1 < n) {
            B(static_cast<std::size_t>(j), static_cast<std::size_t>(j + 1)) = E[static_cast<std::size_t>(j)];
        }
    }

    std::vector<double> d = D;
    std::vector<double> e = E;
    ColMatrix<double> Ub(n, n);
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr('U', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);

    ColMatrix<double> Sigma = zeros<double>(n, n);
    for (int i = 0; i < n; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = d[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                VTb(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    const ColMatrix<double> prod = Ub * Sigma * transpose(V);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), B(i, j), 1e-9)
                << " at (" << i << "," << j << ")";
        }
    }
}

TEST(LapackBdsqrTest, dbdsqr_upper_3x3_diagonal_vectors) {
    std::vector<double> d{3.0, 2.0, 1.0};
    std::vector<double> e{0.0, 0.0};
    ColMatrix<double> Ub(3, 3);
    ColMatrix<double> VTb(3, 3);
    ASSERT_EQ(cpu::lapack::dbdsqr('U', 3, d.data(), e.data(), Ub.data(), 3, VTb.data(), 3), 0);
    EXPECT_NEAR(d[0], 3.0, 1e-9);
    EXPECT_NEAR(d[1], 2.0, 1e-9);
    EXPECT_NEAR(d[2], 1.0, 1e-9);

    ColMatrix<double> Sigma = zeros<double>(3, 3);
    for (int i = 0; i < 3; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = d[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(3, 3);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                VTb(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    const ColMatrix<double> B{{3.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 1.0}};
    const ColMatrix<double> prod = Ub * Sigma * transpose(V);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), B(i, j), 1e-9);
        }
    }
}

TEST(LapackBdsqrTest, dbdsqr_upper_3x3_offdiag_singular_values) {
    std::vector<double> d{3.0, 2.0, 1.0};
    std::vector<double> e{0.5, 0.3};
    ASSERT_EQ(cpu::lapack::dbdsqr('U', 3, d.data(), e.data()), 0);
    EXPECT_NEAR(d[0], 3.0720321, 1e-6);
    EXPECT_NEAR(d[1], 1.98308508, 1e-6);
    EXPECT_NEAR(d[2], 0.98488189, 1e-6);
}

TEST(LapackBdsqrTest, dbdsqr_upper_3x3_offdiag_vectors) {
    std::vector<double> d{3.0, 2.0, 1.0};
    std::vector<double> e{0.5, 0.3};
    const ColMatrix<double> B{{3.0, 0.5, 0.0}, {0.0, 2.0, 0.3}, {0.0, 0.0, 1.0}};

    ColMatrix<double> Ub(3, 3);
    ColMatrix<double> VTb(3, 3);
    ASSERT_EQ(cpu::lapack::dbdsqr('U', 3, d.data(), e.data(), Ub.data(), 3, VTb.data(), 3), 0);

    ColMatrix<double> Sigma = zeros<double>(3, 3);
    for (int i = 0; i < 3; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = d[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(3, 3);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                VTb(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    const ColMatrix<double> prod = Ub * Sigma * transpose(V);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), B(i, j), 1e-9);
        }
    }
}

TEST(LapackGebrdTest, dgebrd_tall_4x3_factorization) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const ColMatrix<double> A_ref = A;
    const int m = 4;
    const int n = 3;
    ColMatrix<double> work = A;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, work.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> B(static_cast<std::size_t>(m), static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        B(static_cast<std::size_t>(j), static_cast<std::size_t>(j)) = D[static_cast<std::size_t>(j)];
        if (j + 1 < n) {
            B(static_cast<std::size_t>(j), static_cast<std::size_t>(j + 1)) = E[static_cast<std::size_t>(j)];
        }
    }

    ColMatrix<double> Q(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    cpu::lapack::dorgbr('Q', m, n, n, work.data(), m, tauq.data(), Q.data(), m);

    ColMatrix<double> Pt(n, n);
    cpu::lapack::dorgbr('P', n, n, n, work.data(), m, taup.data(), Pt.data(), n);

    const ColMatrix<double> prod = Q * B * Pt;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), A_ref(i, j), 1e-9);
        }
    }
}

TEST(LapackDorgbrTest, dorgbr_q_is_orthonormal) {
    ColMatrix<double> A{{0.19, 0.82, -1.30}, {0.52, -0.54, 0.91}, {0.78, 0.31, 0.45}, {0.42, 0.23, -0.76}};
    const int m = 4;
    const int n = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> Q(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    cpu::lapack::dorgbr('Q', m, n, n, A.data(), m, tauq.data(), Q.data(), m);
    const ColMatrix<double> qtq = transpose(Q) * Q;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(qtq(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), expected, 1e-9);
        }
    }
}

TEST(LapackGesvdTest, dgesvd_tall_4x3_singular_values) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const int m = 4;
    const int n = 3;
    const int k = 3;
    std::vector<double> S(k);
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
    ColMatrix<double> VT(static_cast<std::size_t>(k), static_cast<std::size_t>(n));

    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);
    EXPECT_NEAR(S[0], 2.035319606563809, 1e-9);
    EXPECT_NEAR(S[1], 1.0908508989139205, 1e-9);
    EXPECT_NEAR(S[2], 0.41233289400147577, 1e-9);
}

TEST(LapackGesvdTest, dgesvd_tall_4x3_reconstructs_matrix) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const ColMatrix<double> A_ref = A;
    const int m = 4;
    const int n = 3;
    const int k = 3;
    std::vector<double> S(k);
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
    ColMatrix<double> VT(static_cast<std::size_t>(k), static_cast<std::size_t>(n));

    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);

    ColMatrix<double> Sigma = zeros<double>(k, k);
    for (int i = 0; i < k; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = S[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                VT(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    const ColMatrix<double> prod = U * Sigma * transpose(V);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), A_ref(i, j), 1e-9);
        }
    }
}

TEST(LapackGesvdTest, dgesvd_wide_3x4_reconstructs_matrix) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30, 0.42},
        {0.52, -0.54, 0.91, 0.23},
        {0.78, 0.31, 0.45, -0.76}};
    const ColMatrix<double> A_ref = A;
    const int m = 3;
    const int n = 4;
    const int k = 3;
    std::vector<double> S(k);
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
    ColMatrix<double> VT(static_cast<std::size_t>(k), static_cast<std::size_t>(n));

    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);

    ColMatrix<double> Sigma = zeros<double>(k, k);
    for (int i = 0; i < k; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = S[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                VT(static_cast<std::size_t>(j), static_cast<std::size_t>(i));
        }
    }
    const ColMatrix<double> prod = U * Sigma * transpose(V);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), A_ref(i, j), 1e-9);
        }
    }
}

TEST(LapackSteqrTest, dsteqr_bidiagonal_btb_2x2) {
    std::vector<double> diag{10.0, 5.0};
    std::vector<double> off{5.0};
    std::vector<double> Z(4);
    cpu::lapack::dsteqr(2, diag.data(), off.data(), Z.data(), 2);
    EXPECT_FALSE(std::isnan(Z[0]));
    EXPECT_FALSE(std::isnan(Z[1]));
    EXPECT_FALSE(std::isnan(Z[2]));
    EXPECT_FALSE(std::isnan(Z[3]));
}

TEST(LapackGesvdTest, gebrd_btb_vectors_not_nan) {
    std::vector<double> diag{10.000000000000002, 5.0};
    std::vector<double> off{5.000000000000001};
    std::vector<double> V(4);
    cpu::lapack::dsteqr(2, diag.data(), off.data(), V.data(), 2);
    for (double x : V) {
        EXPECT_FALSE(std::isnan(x)) << x;
    }
}

TEST(LapackGesvdTest, dgesvd_reconstructs_matrix) {
    ColMatrix<double> A{{3, 1}, {1, 2}};
    const ColMatrix<double> A_ref = A;
    const int m = 2;
    const int n = 2;
    const int k = 2;
    std::vector<double> S(k);
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
    ColMatrix<double> VT(static_cast<std::size_t>(k), static_cast<std::size_t>(n));

    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);
    EXPECT_NEAR(S[0], 3.618033988749895, 1e-9);
    EXPECT_NEAR(S[1], 1.381966011250105, 1e-9);

    ColMatrix<double> Sigma = zeros<double>(k, k);
    for (int i = 0; i < k; ++i) {
        Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = S[static_cast<std::size_t>(i)];
    }
    ColMatrix<double> V(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = VT(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    const ColMatrix<double> prod = U * Sigma * transpose(V);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(prod(i, j), A_ref(i, j), 1e-9);
        }
    }
}

TEST(LapackGesvdTest, svd_uses_dgesvd_path) {
    ColMatrix<double> A{{3, 1}, {1, 2}};
    const auto result = svd(A).value();
    ColMatrix<double> Sigma = zeros<double>(result.S.rows(), result.S.rows());
    for (size_t i = 0; i < result.S.rows(); ++i) {
        Sigma(i, i) = result.S(i, 0);
    }
    const ColMatrix<double> prod = result.U * Sigma * transpose(result.V);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(prod(i, j), A(i, j), 1e-9);
        }
    }
}

TEST(LapackGetrfTest, lu_uses_dgetrf_path) {
    ColMatrix<double> A{{4, 3, 2}, {6, 7, 5}, {2, 8, 3}};
    const auto [L, U, P] = lu(A).value();
    const ColMatrix<double> PA = P * A;
    const ColMatrix<double> LU = L * U;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(PA(i, j), LU(i, j), 1e-10);
        }
    }
}

TEST(LapackBdsqrTest, dbdsqr_lower_3x3_offdiag_singular_values) {
    std::vector<double> d{3.0, 2.0, 1.0};
    std::vector<double> e{0.5, 0.3};
    ASSERT_EQ(cpu::lapack::dbdsqr('L', 3, d.data(), e.data()), 0);
    EXPECT_GT(d[0], d[1]);
    EXPECT_GT(d[1], d[2]);
}

TEST(LapackBdsqrTest, dbdsqr_lower_3x3_offdiag_vectors) {
    std::vector<double> d{3.0, 2.0, 1.0};
    std::vector<double> e{0.5, 0.3};
    ColMatrix<double> Ub(3, 3);
    ColMatrix<double> VTb(3, 3);
    ASSERT_EQ(cpu::lapack::dbdsqr('L', 3, d.data(), e.data(), Ub.data(), 3, VTb.data(), 3), 0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GT(d[static_cast<std::size_t>(i)], 0.0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_invalid_uplo) {
    std::vector<double> d{1.0, 2.0};
    std::vector<double> e{0.5};
    EXPECT_EQ(cpu::lapack::dbdsqr('X', 2, d.data(), e.data()), 1);
}

TEST(LapackBdsqrTest, dbdsqr_single_element) {
    std::vector<double> d{2.5};
    ASSERT_EQ(cpu::lapack::dbdsqr('U', 1, d.data(), nullptr), 0);
    EXPECT_NEAR(d[0], 2.5, 1e-12);
}

TEST(LapackBdsqrTest, dbdsqr_4x4_from_gebrd) {
    ColMatrix<double> M{
        {0.19, 0.82, -1.30, 0.42},
        {0.52, -0.54, 0.91, 0.23},
        {0.78, 0.31, 0.45, -0.76},
        {0.42, 0.23, -0.76, 0.11}};
    const int n = 4;
    ColMatrix<double> A = M;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ASSERT_EQ(cpu::lapack::dbdsqr('U', n, d.data(), e.data()), 0);
    for (int i = 0; i < n - 1; ++i) {
        EXPECT_GT(d[static_cast<std::size_t>(i)], d[static_cast<std::size_t>(i + 1)]);
    }

    d = D;
    e = E;
    ColMatrix<double> Ub(n, n);
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr('L', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);
    for (int i = 0; i < n; ++i) {
        EXPECT_GT(d[static_cast<std::size_t>(i)], 0.0);
    }
}

TEST(LapackDorgbrTest, dorgbr_p_square_3x3) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const int n = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> P(n, n);
    cpu::lapack::dorgbr('P', n, n, n, A.data(), n, taup.data(), P.data(), n);
    expect_orthogonal_columns(P);
}

TEST(LapackDorgbrTest, dorgbr_p_wide_2x3) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}};
    const int m = 2;
    const int n = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> P(m, n);
    cpu::lapack::dorgbr('P', m, n, m, A.data(), m, taup.data(), P.data(), m);
    expect_orthogonal_rows(P);
}

TEST(LapackDormbrTest, apply_q_left_and_p_right) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const int m = 4;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> C(m, n, 1.0);
    cpu::lapack::dormbr('Q', 'L', 'N', m, n, k, A.data(), m, tauq.data(), C.data(), m);
    EXPECT_NE(C(0, 0), 1.0);

    ColMatrix<double> Dmat(n, m, 1.0);
    cpu::lapack::dormbr('P', 'R', 'T', m, n, k, A.data(), m, taup.data(), Dmat.data(), n);
    EXPECT_NE(Dmat(0, 0), 1.0);
}

TEST(LapackSyevTest, dsyev_values_only_3x3) {
    ColMatrix<double> A{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    std::vector<double> w(3);
    ASSERT_EQ(cpu::lapack::dsyev('N', 3, A.data(), 3, w.data()), 0);
    EXPECT_GT(w[2], w[0]);
}

TEST(LapackSyevTest, dsyev_vectors_4x4) {
    ColMatrix<double> A = spd_matrix(4);
    std::vector<double> w(4);
    ASSERT_EQ(cpu::lapack::dsyev('V', 4, A.data(), 4, w.data()), 0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(w[static_cast<std::size_t>(i + 1)], w[static_cast<std::size_t>(i)] - 1e-9);
    }
}

TEST(LapackBdsqrTest, dbdsqr_hilbert_sizes_5_through_8) {
    for (int n = 5; n <= 8; ++n) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', true);
    }
    exercise_dbdsqr_from_matrix(9, 'U', true);
}

TEST(LapackBdsqrTest, dbdsqr_tall_gebrd_6x4_and_7x5) {
    exercise_dbdsqr_tall(6, 4, 'U');
    exercise_dbdsqr_tall(7, 5, 'L');
}

TEST(LapackDorgbrTest, dorgbr_invalid_vect_noop) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    ColMatrix<double> Q(2, 2, 0.0);
    std::vector<double> tau{0.1, 0.2};
    cpu::lapack::dorgbr('X', 2, 2, 2, A.data(), 2, tau.data(), Q.data(), 2);
    EXPECT_DOUBLE_EQ(Q(0, 0), 0.0);
}

TEST(LapackDorgbrTest, dorgbr_q_wide_3x2) {
    ColMatrix<double> A{{1, 2}, {3, 4}, {5, 6}};
    const int m = 3;
    const int n = 2;
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> Q(m, n);
    cpu::lapack::dorgbr('Q', m, n, n, A.data(), m, tauq.data(), Q.data(), m);
    const ColMatrix<double> qtq = transpose(Q) * Q;
    EXPECT_NEAR(qtq(0, 0), 1.0, 1e-9);
}

TEST(LapackDormbrTest, dormbr_p_left_wide_smoke) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76}};
    const int m = 4;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> Cp(m, n, 1.0);
    cpu::lapack::dormbr('P', 'L', 'N', m, n, k, A.data(), m, taup.data(), Cp.data(), m);
    EXPECT_FALSE(std::isnan(Cp(0, 0)));
}

TEST(LapackDormbrTest, dormbr_p_right_tall_smoke) {
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    const int n = 4;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(n, n, 1.0);
    cpu::lapack::dormbr('P', 'R', 'T', n, n, n, A.data(), n, taup.data(), C.data(), n);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(BlasLevel3Test, dtrsm_left_upper_transpose_matches_reference) {
    ColMatrix<double> U{{2, 1}, {0, 3}};
    ColMatrix<double> B{{4, 2}, {6, 4}};
    expect_dtrsm_matches_reference('L', 'U', 'T', 'N', 2, 2, 1.0, U, B);
}

TEST(LapackBdsqrTest, dbdsqr_hilbert_sizes_10_through_12) {
    for (int n = 10; n <= 12; ++n) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', n % 2 == 0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_tall_gebrd_8x6_9x5_and_10x7) {
    exercise_dbdsqr_tall(8, 6, 'U');
    exercise_dbdsqr_tall(9, 5, 'L');
    exercise_dbdsqr_tall(10, 7, 'U');
}

TEST(LapackDorgbrTest, dorgbr_p_square_5x5) {
    const int n = 5;
    ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.1 * static_cast<double>(i + 1) + 0.05 * static_cast<double>(j);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> P(n, n);
    cpu::lapack::dorgbr('P', n, n, n, A.data(), n, taup.data(), P.data(), n);
    expect_orthogonal_columns(P);
}

TEST(LapackDormbrTest, dormbr_q_left_transpose_square) {
    const int n = 5;
    ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                std::cos(0.2 * static_cast<double>(i)) + 0.1 * static_cast<double>(j);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(n, n, 1.0);
    cpu::lapack::dormbr('Q', 'L', 'T', n, n, n, A.data(), n, tauq.data(), C.data(), n);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(BlasLevel3Test, dtrsm_right_upper_matches_reference) {
    ColMatrix<double> U{{2, 1, 0}, {0, 3, 1}, {0, 0, 4}};
    ColMatrix<double> B{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    expect_dtrsm_matches_reference('R', 'U', 'N', 'N', 3, 3, 1.0, U, B);
}

TEST(BlasLevel3Test, dsyrk_upper_transpose) {
    constexpr int n = 4;
    constexpr int k = 3;
    ColMatrix<double> A(static_cast<std::size_t>(k), static_cast<std::size_t>(n));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = 0.05 * static_cast<double>(i + j);
        }
    }
    std::vector<double> C(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    std::vector<double> ref = C;
    cpu::blas::dsyrk('U', 'T', n, k, 1.0, A.data(), k, 0.0, C.data(), n);
    reference_dsyrk('U', 'T', n, k, 1.0, A.data(), k, 0.0, ref.data(), n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            EXPECT_NEAR(C[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        ref[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)],
                        1e-10);
        }
    }
}

TEST(LapackBdsqrTest, dbdsqr_hilbert_sizes_16_through_20) {
    for (int n = 16; n <= 20; ++n) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', n % 3 == 0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_hilbert_sizes_13_through_15) {
    for (int n = 13; n <= 15; ++n) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', true);
    }
}

TEST(LapackBdsqrTest, dbdsqr_tall_gebrd_12x8_and_11x6) {
    exercise_dbdsqr_tall(12, 8, 'U');
    exercise_dbdsqr_tall(11, 6, 'L');
}

TEST(LapackDorgbrTest, dorgbr_p_wide_3x5) {
    const int m = 3;
    const int n = 5;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = 0.2 * static_cast<double>(i + j + 1);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> P(m, n);
    cpu::lapack::dorgbr('P', m, n, m, A.data(), m, taup.data(), P.data(), m);
    expect_orthogonal_rows(P);
}

TEST(LapackDorgbrTest, dorgbr_p_wide_5x7) {
    const int m = 5;
    const int n = 7;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                std::sin(0.17 * static_cast<double>(i + 1)) + 0.13 * static_cast<double>(j);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> P(m, n);
    cpu::lapack::dorgbr('P', m, n, m, A.data(), m, taup.data(), P.data(), m);
    expect_orthogonal_rows(P);
}

TEST(LapackDormbrTest, dormbr_p_left_transpose_on_wide) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76},
        {0.11, 0.67, 0.33}};
    const int m = 5;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 1.0);
    cpu::lapack::dormbr('P', 'L', 'T', m, n, k, A.data(), m, taup.data(), C.data(), m);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_p_right_on_tall_gebrd) {
    ColMatrix<double> A{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
        {13, 14, 15}};
    const int m = 5;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(n, m, 1.0);
    cpu::lapack::dormbr('P', 'R', 'N', m, n, k, A.data(), m, taup.data(), C.data(), n);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_q_left_transpose_on_wide) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76},
        {0.11, 0.67, 0.33}};
    const int m = 5;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 1.0);
    cpu::lapack::dormbr('Q', 'L', 'T', m, n, k, A.data(), m, tauq.data(), C.data(), m);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackGesvdTest, dgesvd_square_7x7_and_8x8_smoke) {
    for (int n : {7, 8}) {
        ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    0.08 * static_cast<double>(i + 1) + 0.04 * static_cast<double>(j);
            }
        }
        std::vector<double> S(static_cast<std::size_t>(n));
        ColMatrix<double> U(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        ColMatrix<double> VT(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgesvd(n, n, A.data(), n, S.data(), U.data(), n, VT.data(), n), 0);
        EXPECT_GT(S[0], S[static_cast<std::size_t>(n - 1)]);
    }
}

TEST(LapackGesvdTest, dgesvd_square_6x6_smoke) {
    const int n = 6;
    ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                std::sin(0.17 * static_cast<double>(i + 1)) + 0.05 * static_cast<double>(j);
        }
    }
    std::vector<double> S(static_cast<std::size_t>(n));
    ColMatrix<double> U(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    ColMatrix<double> VT(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgesvd(n, n, A.data(), n, S.data(), U.data(), n, VT.data(), n), 0);
    EXPECT_GT(S[0], S[static_cast<std::size_t>(n - 1)]);
}

TEST(LapackGesvdTest, dgesvd_tall_8x5_and_wide_5x8_smoke) {
    for (auto [m, n] : std::vector<std::pair<int, int>>{{8, 5}, {5, 8}}) {
        ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    0.1 * static_cast<double>(i + 1) - 0.03 * static_cast<double>(j);
            }
        }
        const int k = (std::min)(m, n);
        std::vector<double> S(static_cast<std::size_t>(k));
        ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
        ColMatrix<double> VT(static_cast<std::size_t>(k), static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);
        EXPECT_GT(S[0], 0.0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_hilbert_sizes_25_through_28) {
    for (int n = 25; n <= 28; ++n) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', n % 2 == 1);
    }
}

TEST(LapackBdsqrTest, dbdsqr_hilbert_sizes_21_through_24) {
    for (int n = 21; n <= 24; ++n) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', n % 2 == 0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_tall_gebrd_15x10_and_14x9) {
    exercise_dbdsqr_tall(15, 10, 'U');
    exercise_dbdsqr_tall(14, 9, 'L');
}

TEST(LapackDormbrTest, dormbr_q_right_square_and_tall) {
    {
        ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
        const int n = 4;
        std::vector<double> D(static_cast<std::size_t>(n));
        std::vector<double> E(static_cast<std::size_t>(n - 1));
        std::vector<double> tauq(static_cast<std::size_t>(n));
        std::vector<double> taup(static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
        ColMatrix<double> C(n, n, 1.0);
        cpu::lapack::dormbr('Q', 'R', 'N', n, n, n, A.data(), n, tauq.data(), C.data(), n);
        EXPECT_FALSE(std::isnan(C(0, 0)));
        cpu::lapack::dormbr('Q', 'R', 'T', n, n, n, A.data(), n, tauq.data(), C.data(), n);
        EXPECT_FALSE(std::isnan(C(0, 0)));
    }
    ColMatrix<double> tall{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76},
        {0.11, 0.67, 0.33}};
    const int m = 5;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, tall.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> Ct(m, n, 1.0);
    cpu::lapack::dormbr('Q', 'R', 'N', m, n, k, tall.data(), m, tauq.data(), Ct.data(), m);
    EXPECT_FALSE(std::isnan(Ct(0, 0)));
}

TEST(LapackGelsTest, dgels_multiple_rhs) {
    ColMatrix<double> A{{1, 1}, {1, 2}, {1, 3}};
    ColMatrix<double> B{{6, 5}, {9, 8}, {12, 11}};
    ASSERT_EQ(cpu::lapack::dgels(3, 2, 2, A.data(), 3, B.data(), 3), 0);
    EXPECT_NEAR(B(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(B(1, 0), 3.0, 1e-9);
}

TEST(LapackBdsqrTest, dbdsqr_guards_and_single_negative_vt) {
    std::vector<double> d{1.0, 2.0};
    std::vector<double> e{0.5};
    EXPECT_EQ(cpu::lapack::dbdsqr('U', 0, d.data(), e.data()), 0);
    EXPECT_EQ(cpu::lapack::dbdsqr('U', 2, nullptr, e.data()), 1);
    EXPECT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), nullptr), 1);

    std::vector<double> d1{-4.0};
    ColMatrix<double> VT(1, 1, 1.0);
    ASSERT_EQ(cpu::lapack::dbdsqr('U', 1, d1.data(), nullptr, nullptr, 0, VT.data(), 1), 0);
    EXPECT_GT(d1[0], 0.0);
    EXPECT_LT(VT(0, 0), 0.0);
}

TEST(LapackBdsqrTest, dbdsqr_zero_entries_and_backward_sweep) {
    {
        std::vector<double> d{1e-14, 50.0, 40.0, 30.0};
        std::vector<double> e{0.0, 2.0, 1.0};
        ColMatrix<double> Ub(4, 4);
        ColMatrix<double> VTb(4, 4);
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 4, d.data(), e.data(), Ub.data(), 4, VTb.data(), 4), 0);
        for (double s : d) {
            EXPECT_GE(s, 0.0);
        }
    }
    {
        std::vector<double> d{100.0, 80.0, 1e-12, 5.0};
        std::vector<double> e{3.0, 0.0, 2.0};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 4, d.data(), e.data()), 0);
        EXPECT_GT(d[0], d[3]);
    }
    {
        std::vector<double> d{0.0, 2.0, 1.0};
        std::vector<double> e{0.0, 0.5};
        ColMatrix<double> Ub(3, 3);
        ASSERT_EQ(cpu::lapack::dbdsqr('L', 3, d.data(), e.data(), Ub.data(), 3, nullptr, 0), 0);
        EXPECT_GE(d[0], 0.0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_lower_u_only_and_custom_nru) {
    std::vector<double> d{4.0, 3.0, 2.0, 1.0};
    std::vector<double> e{0.5, 0.25, 0.1};
    ColMatrix<double> Ub(4, 4);
    ASSERT_EQ(cpu::lapack::dbdsqr('L', 4, d.data(), e.data(), Ub.data(), 4, nullptr, 0, 4, true), 0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(d[static_cast<std::size_t>(i)], d[static_cast<std::size_t>(i + 1)] - 1e-9);
    }

    std::vector<double> d2{2.0, 1.0, 0.5};
    std::vector<double> e2{0.3, 0.2};
    ColMatrix<double> Ub2(3, 3);
    ASSERT_EQ(cpu::lapack::dbdsqr('L', 3, d2.data(), e2.data(), Ub2.data(), 3, nullptr, 0, 2), 0);
    EXPECT_GT(d2[0], 0.0);
}

TEST(LapackBdsqrTest, dbdsqr_dlasv2_edge_2x2_blocks) {
    {
        std::vector<double> d{0.0, 0.0};
        std::vector<double> e{5.0};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), e.data()), 0);
        EXPECT_GE(d[0], 0.0);
        EXPECT_GE(d[1], 0.0);
    }
    {
        std::vector<double> d{1e-200, 1e200};
        std::vector<double> e{1.0};
        ColMatrix<double> Ub(2, 2);
        ColMatrix<double> VTb(2, 2);
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), e.data(), Ub.data(), 2, VTb.data(), 2), 0);
        EXPECT_GT(d[0], 0.0);
    }
    {
        std::vector<double> d{1.0, 1e-14};
        std::vector<double> e{1e14};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 2, d.data(), e.data()), 0);
        EXPECT_GT(d[0], 0.0);
    }
}

TEST(LapackBdsqrTest, dbdsqr_custom_bidiagonal_edge_sizes) {
    {
        std::vector<double> d{1e-12, 5.0, 0.25};
        std::vector<double> e{2.0, 3.0};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 3, d.data(), e.data()), 0);
        for (double s : d) {
            EXPECT_GE(s, 0.0);
        }
    }
    {
        std::vector<double> d{0.0, 1e-14, 2.0};
        std::vector<double> e{0.0, 1.0};
        ColMatrix<double> Ub(3, 3);
        ColMatrix<double> VTb(3, 3);
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 3, d.data(), e.data(), Ub.data(), 3, VTb.data(), 3), 0);
        EXPECT_GE(d[2], d[1] - 1e-9);
    }
    {
        std::vector<double> d{3.0, 1e-8, 2.0, 0.5};
        std::vector<double> e{1.0, 4.0, 0.25};
        ASSERT_EQ(cpu::lapack::dbdsqr('U', 4, d.data(), e.data()), 0);
        EXPECT_GT(d[0], 0.0);
    }
    {
        std::vector<double> d{0.5, 2.0, 1e-10};
        std::vector<double> e{1.5, 0.75};
        ColMatrix<double> Ub(3, 3);
        ASSERT_EQ(cpu::lapack::dbdsqr('L', 3, d.data(), e.data(), Ub.data(), 3, nullptr, 0), 0);
        EXPECT_GT(d[0], 0.0);
    }
    {
        std::vector<double> d{2.0, 0.01};
        std::vector<double> e{100.0};
        ColMatrix<double> Ub(2, 2);
        ColMatrix<double> VTb(2, 2);
        ASSERT_EQ(cpu::lapack::dbdsqr('L', 2, d.data(), e.data(), Ub.data(), 2, VTb.data(), 2), 0);
        EXPECT_GT(d[0], d[1] - 1e-9);
    }
}

TEST(LapackBdsqrTest, dbdsqr_init_vectors_false_preserves_u) {
    const int n = 3;
    std::vector<double> d{2.0, 1.0, 0.5};
    std::vector<double> e{0.3, 0.2};
    ColMatrix<double> Ub(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Ub(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = (i == j) ? 1.0 : 0.0;
        }
    }
    ColMatrix<double> VTb(n, n, 0.0);
    ASSERT_EQ(
        cpu::lapack::dbdsqr('U', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n, 0, false),
        0);
    EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
}

TEST(LapackBdsqrTest, dbdsqr_sizes_2_and_5_singular_mix) {
    for (int n : {2, 5}) {
        exercise_dbdsqr_from_matrix(n, 'U', false);
        exercise_dbdsqr_from_matrix(n, 'L', true);
    }
}

TEST(LapackDorgbrTest, dorgbr_and_dormbr_invalid_guards) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    std::vector<double> tau{0.5, 0.25};
    ColMatrix<double> Q(2, 2);
    cpu::lapack::dorgbr('X', 2, 2, 2, A.data(), 2, tau.data(), Q.data(), 2);
    cpu::lapack::dorgbr('Q', 0, 2, 2, A.data(), 2, tau.data(), Q.data(), 2);

    ColMatrix<double> C(2, 2, 1.0);
    EXPECT_EQ(cpu::lapack::dormbr('X', 'L', 'N', 2, 2, 2, A.data(), 2, tau.data(), C.data(), 2), 1);
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'L', 'N', 0, 2, 2, A.data(), 2, tau.data(), C.data(), 2), 0);
}

TEST(LapackDormbrTest, dormbr_p_left_square_no_transpose) {
    const int n = 4;
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> C(n, n, 1.0);
    ASSERT_EQ(cpu::lapack::dormbr('P', 'L', 'N', n, n, n, A.data(), n, taup.data(), C.data(), n), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_p_left_square_transpose) {
    const int n = 4;
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> C(n, n, 1.0);
    ASSERT_EQ(cpu::lapack::dormbr('P', 'L', 'T', n, n, n, A.data(), n, taup.data(), C.data(), n), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_p_left_rejects_c_wide) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}};
    const int m = 2;
    const int n = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);

    ColMatrix<double> C(m, n, 1.0);
    EXPECT_EQ(cpu::lapack::dormbr('P', 'L', 'N', m, n, m, A.data(), m, taup.data(), C.data(), m), 1);
}

TEST(LapackDormbrTest, dormbr_q_right_transpose_on_wide) {
    ColMatrix<double> A{
        {0.19, 0.82, -1.30},
        {0.52, -0.54, 0.91},
        {0.78, 0.31, 0.45},
        {0.42, 0.23, -0.76},
        {0.11, 0.67, 0.33}};
    const int m = 5;
    const int n = 3;
    const int k = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 0.5);
    cpu::lapack::dormbr('Q', 'R', 'T', m, n, k, A.data(), m, tauq.data(), C.data(), m);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackGesvdTest, dgesvd_dimension_and_null_guards) {
    std::vector<double> A{1.0, 2.0, 3.0, 4.0};
    std::vector<double> S(2);
    std::vector<double> U(4);
    std::vector<double> VT(4);
    EXPECT_EQ(cpu::lapack::dgesvd(0, 2, A.data(), 2, S.data(), U.data(), 2, VT.data(), 2), 0);
    EXPECT_EQ(cpu::lapack::dgesvd(2, 0, A.data(), 2, S.data(), U.data(), 2, VT.data(), 2), 0);
    EXPECT_EQ(cpu::lapack::dgesvd(2, 2, nullptr, 2, S.data(), U.data(), 2, VT.data(), 2), 1);
    EXPECT_EQ(cpu::lapack::dgesvd(2, 2, A.data(), 2, nullptr, U.data(), 2, VT.data(), 2), 1);
    EXPECT_EQ(cpu::lapack::dgesvd(2, 2, A.data(), 2, S.data(), nullptr, 2, VT.data(), 2), 1);
    EXPECT_EQ(cpu::lapack::dgesvd(2, 2, A.data(), 2, S.data(), U.data(), 2, nullptr, 2), 1);
}

TEST(LapackGesvdTest, dgesvd_square_9x9_and_wide_5x8_smoke) {
    for (int n : {9, 10}) {
        ColMatrix<double> A(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    0.2 * static_cast<double>(i + 1) - 0.05 * static_cast<double>(j);
            }
        }
        std::vector<double> S(static_cast<std::size_t>(n));
        ColMatrix<double> U(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        ColMatrix<double> VT(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgesvd(n, n, A.data(), n, S.data(), U.data(), n, VT.data(), n), 0);
        EXPECT_GT(S[0], S[static_cast<std::size_t>(n - 1)]);
    }

    const int m = 5;
    const int n = 8;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                std::cos(0.17 * static_cast<double>(i)) + 0.08 * static_cast<double>(j);
        }
    }
    std::vector<double> S(static_cast<std::size_t>(m));
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(m));
    ColMatrix<double> VT(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);
    EXPECT_GT(S[0], 0.0);
}

TEST(LapackBdsqrTest, dbdsqr_vt_only_from_gebrd) {
    const int n = 4;
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr('U', n, d.data(), e.data(), nullptr, 0, VTb.data(), n), 0);
    for (int i = 0; i < n - 1; ++i) {
        EXPECT_GE(d[static_cast<std::size_t>(i)], d[static_cast<std::size_t>(i + 1)] - 1e-9);
    }
}

TEST(LapackBdsqrTest, dbdsqr_lower_vectors_sizes_6_through_10) {
    for (int n : {6, 7, 8, 9, 10}) {
        exercise_dbdsqr_from_matrix(n, 'L', true);
        exercise_dbdsqr_from_matrix(n, 'U', true);
    }
}

TEST(LapackDorgbrTest, dorgbr_q_tall_8x5_10x6_and_12x7) {
    for (auto [m, n] : std::vector<std::pair<int, int>>{{8, 5}, {10, 6}, {12, 7}}) {
        ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    0.15 * static_cast<double>(i + 1) + 0.07 * static_cast<double>(j);
            }
        }
        std::vector<double> D(static_cast<std::size_t>(n));
        std::vector<double> E(static_cast<std::size_t>(n - 1));
        std::vector<double> tauq(static_cast<std::size_t>(n));
        std::vector<double> taup(static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
        ColMatrix<double> Q(m, n);
        cpu::lapack::dorgbr('Q', m, n, n, A.data(), m, tauq.data(), Q.data(), m);
        expect_orthogonal_columns(Q);
    }
}

TEST(LapackDorgbrTest, dorgbr_p_wide_6x8_and_8x10) {
    for (auto [m, n] : std::vector<std::pair<int, int>>{{6, 8}, {8, 10}}) {
        ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    std::sin(0.11 * static_cast<double>(i)) + 0.09 * static_cast<double>(j);
            }
        }
        std::vector<double> D(static_cast<std::size_t>(n));
        std::vector<double> E(static_cast<std::size_t>(n - 1));
        std::vector<double> tauq(static_cast<std::size_t>(n));
        std::vector<double> taup(static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
        ColMatrix<double> P(m, n);
        cpu::lapack::dorgbr('P', m, n, m, A.data(), m, taup.data(), P.data(), m);
        expect_orthogonal_rows(P);
    }
}

TEST(LapackDormbrTest, dormbr_q_right_no_transpose_on_tall) {
    const int m = 8;
    const int n = 5;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.2 * static_cast<double>(i) - 0.05 * static_cast<double>(j);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 0.75);
    ASSERT_EQ(cpu::lapack::dormbr('Q', 'R', 'N', m, n, n, A.data(), m, tauq.data(), C.data(), m), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDorgbrTest, dorgbr_guards_zero_dimensions) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    ColMatrix<double> Q(2, 2);
    std::vector<double> tau{0.1, 0.2};
    cpu::lapack::dorgbr('Q', 0, 2, 2, A.data(), 2, tau.data(), Q.data(), 2);
    cpu::lapack::dorgbr('P', 2, 0, 2, A.data(), 2, tau.data(), Q.data(), 2);
    cpu::lapack::dorgbr('P', 2, 2, 0, A.data(), 2, tau.data(), Q.data(), 2);
    EXPECT_FALSE(std::isnan(Q(0, 0)));
}

TEST(LapackDormbrTest, dormbr_zero_k_and_zero_c_no_op) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<double> tau(3, 0.0);
    ColMatrix<double> C(3, 3, 0.0);
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'R', 'N', 3, 3, 0, A.data(), 3, tau.data(), C.data(), 3), 0);
    EXPECT_DOUBLE_EQ(C(0, 0), 0.0);
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'R', 'T', 3, 3, 3, A.data(), 3, tau.data(), C.data(), 3), 0);
    EXPECT_DOUBLE_EQ(C(0, 0), 0.0);
}

TEST(LapackDormbrTest, dormbr_p_right_wide_single_row_no_reflectors) {
    ColMatrix<double> A{{1}, {2}};
    const int m = 2;
    const int n = 1;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(1, 0.0);
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 1.0);
    ASSERT_EQ(cpu::lapack::dormbr('P', 'R', 'N', m, n, 1, A.data(), m, taup.data(), C.data(), m), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_q_right_transpose_on_zero_matrix) {
    const int n = 4;
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(n, n, 0.0);
    ASSERT_EQ(cpu::lapack::dormbr('Q', 'R', 'T', n, n, n, A.data(), n, tauq.data(), C.data(), n), 0);
    EXPECT_DOUBLE_EQ(C(0, 0), 0.0);
}

TEST(LapackDorgbrTest, dorgbr_p_square_single_column) {
    ColMatrix<double> A{{3}, {4}, {5}};
    const int m = 3;
    const int n = 1;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(1, 0.0);
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> P(m, n);
    cpu::lapack::dorgbr('P', m, n, m, A.data(), m, taup.data(), P.data(), m);
    for (int i = 0; i < m; ++i) {
        EXPECT_FALSE(std::isnan(P(static_cast<std::size_t>(i), 0)));
    }
}

TEST(LapackGesvdTest, dgesvd_wide_2x5_and_3x6_reconstruct) {
    for (auto [m, n] : std::vector<std::pair<int, int>>{{2, 5}, {3, 6}}) {
        ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    0.13 * static_cast<double>(i + 1) - 0.07 * static_cast<double>(j);
            }
        }
        const ColMatrix<double> A_ref = A;
        const int k = m;
        std::vector<double> S(static_cast<std::size_t>(k));
        ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(k));
        ColMatrix<double> VT(static_cast<std::size_t>(k), static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);

        ColMatrix<double> Sigma = zeros<double>(k, k);
        for (int i = 0; i < k; ++i) {
            Sigma(static_cast<std::size_t>(i), static_cast<std::size_t>(i)) = S[static_cast<std::size_t>(i)];
        }
        ColMatrix<double> V(static_cast<std::size_t>(n), static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    VT(static_cast<std::size_t>(j), static_cast<std::size_t>(i));
            }
        }
        const ColMatrix<double> prod = U * Sigma * transpose(V);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                EXPECT_NEAR(prod(static_cast<std::size_t>(i), static_cast<std::size_t>(j)), A_ref(i, j), 1e-8);
            }
        }
    }
}

TEST(LapackDormbrTest, dormbr_q_right_transpose_on_zero_tau) {
    const int n = 3;
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n), 0.0);
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(n, n, 1.0);
    ASSERT_EQ(cpu::lapack::dormbr('Q', 'R', 'T', n, n, n, A.data(), n, tauq.data(), C.data(), n), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_p_left_no_transpose_on_identity_block) {
    const int n = 4;
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(n, n, 0.0);
    for (int j = 0; j < n; ++j) {
        C(static_cast<std::size_t>(j), static_cast<std::size_t>(j)) = 1.0;
    }
    ASSERT_EQ(cpu::lapack::dormbr('P', 'L', 'N', n, n, n, A.data(), n, taup.data(), C.data(), n), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackGesvdTest, dgesvd_wide_4x7_smoke) {
    const int m = 4;
    const int n = 7;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.09 * static_cast<double>(i + 1) + 0.04 * static_cast<double>(j);
        }
    }
    std::vector<double> S(static_cast<std::size_t>(m));
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(m));
    ColMatrix<double> VT(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);
    EXPECT_GT(S[0], S[static_cast<std::size_t>(m - 1)]);
}

TEST(LapackGesvdTest, dgesvd_tall_6x4_singular_values) {
    ColMatrix<double> A(static_cast<std::size_t>(6), static_cast<std::size_t>(4));
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.11 * static_cast<double>(i + 1) + 0.06 * static_cast<double>(j);
        }
    }
    const int m = 6;
    const int n = 4;
    std::vector<double> S(static_cast<std::size_t>(n));
    ColMatrix<double> U(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    ColMatrix<double> VT(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgesvd(m, n, A.data(), m, S.data(), U.data(), m, VT.data(), n), 0);
    EXPECT_GT(S[0], S[static_cast<std::size_t>(n - 1)]);
}

TEST(LapackBdsqrTest, dbdsqr_vt_only_lower_from_gebrd) {
    const int n = 5;
    ColMatrix<double> A{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr('L', n, d.data(), e.data(), nullptr, 0, VTb.data(), n), 0);
    EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
}

TEST(LapackBdsqrTest, dbdsqr_init_vectors_false_vt_only_and_lower) {
    {
        const int n = 4;
        std::vector<double> d{3.0, 2.0, 1.0, 0.5};
        std::vector<double> e{0.4, 0.3, 0.2};
        ColMatrix<double> VTb(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                VTb(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = (i == j) ? 1.0 : 0.0;
            }
        }
        ASSERT_EQ(
            cpu::lapack::dbdsqr('U', n, d.data(), e.data(), nullptr, 0, VTb.data(), n, 0, false),
            0);
        EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
    }
    {
        const int n = 3;
        std::vector<double> d{2.5, 1.5, 0.75};
        std::vector<double> e{0.35, 0.25};
        ColMatrix<double> Ub(n, n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Ub(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = (i == j) ? 1.0 : 0.0;
            }
        }
        ASSERT_EQ(
            cpu::lapack::dbdsqr('L', n, d.data(), e.data(), Ub.data(), n, nullptr, 0, 0, false),
            0);
        EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
    }
}

TEST(LapackDorgbrTest, dorgbr_null_pointer_guards) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    ColMatrix<double> Q(2, 2, 0.0);
    std::vector<double> tau{0.1, 0.2};
    cpu::lapack::dorgbr('Q', 2, 2, 2, nullptr, 2, tau.data(), Q.data(), 2);
    cpu::lapack::dorgbr('Q', 2, 2, 2, A.data(), 2, nullptr, Q.data(), 2);
    cpu::lapack::dorgbr('Q', 2, 2, 2, A.data(), 2, tau.data(), nullptr, 2);
    cpu::lapack::dorgbr('P', 2, 2, 2, nullptr, 2, tau.data(), Q.data(), 2);
    EXPECT_DOUBLE_EQ(Q(0, 0), 0.0);
}

TEST(LapackDormbrTest, dormbr_null_and_dimension_guards) {
    ColMatrix<double> A{{1, 2}, {3, 4}};
    ColMatrix<double> C(2, 2, 1.0);
    std::vector<double> tau{0.1, 0.2};
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'L', 'N', 2, 2, 2, nullptr, 2, tau.data(), C.data(), 2), 0);
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'L', 'N', 2, 2, 2, A.data(), 2, nullptr, C.data(), 2), 0);
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'L', 'N', 2, 2, 2, A.data(), 2, tau.data(), nullptr, 2), 0);
    EXPECT_EQ(cpu::lapack::dormbr('Q', 'L', 'N', 2, 0, 2, A.data(), 2, tau.data(), C.data(), 2), 0);
    EXPECT_EQ(cpu::lapack::dormbr('P', 'R', 'N', 0, 2, 2, A.data(), 2, tau.data(), C.data(), 2), 0);
}

TEST(LapackDormbrTest, dormbr_p_right_transpose_on_wide_gebrd) {
    const int m = 3;
    const int n = 5;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.17 * static_cast<double>(i + 1) - 0.08 * static_cast<double>(j);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 0.5);
    ASSERT_EQ(cpu::lapack::dormbr('P', 'R', 'T', m, n, m, A.data(), m, taup.data(), C.data(), m), 0);
    EXPECT_FALSE(std::isnan(C(0, 0)));
}

TEST(LapackDormbrTest, dormbr_q_left_no_transpose_on_wide) {
    const int m = 5;
    const int n = 3;
    ColMatrix<double> A(static_cast<std::size_t>(m), static_cast<std::size_t>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                0.12 * static_cast<double>(i) + 0.06 * static_cast<double>(j);
        }
    }
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(m, n, A.data(), m, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> C(m, n, 1.0);
    ASSERT_EQ(cpu::lapack::dormbr('Q', 'L', 'N', m, n, n, A.data(), m, tauq.data(), C.data(), m), 0);
    EXPECT_NE(C(0, 0), 1.0);
}

TEST(LapackBdsqrTest, dbdsqr_gebrd_lower_full_vectors_26_and_28) {
    for (int n : {26, 28}) {
        ColMatrix<double> M(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                M(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                    1.0 / static_cast<double>(i + j + 1) + 0.01 * static_cast<double>(i - j);
            }
        }
        ColMatrix<double> A = M;
        std::vector<double> D(static_cast<std::size_t>(n));
        std::vector<double> E(static_cast<std::size_t>(n - 1));
        std::vector<double> tauq(static_cast<std::size_t>(n));
        std::vector<double> taup(static_cast<std::size_t>(n));
        ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

        std::vector<double> d = D;
        std::vector<double> e = E;
        ColMatrix<double> Ub(n, n);
        ColMatrix<double> VTb(n, n);
        ASSERT_EQ(cpu::lapack::dbdsqr('L', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);
        for (int i = 0; i < n - 1; ++i) {
            EXPECT_GE(d[static_cast<std::size_t>(i)], d[static_cast<std::size_t>(i + 1)] - 1e-9);
        }
    }
}

TEST(LapackBdsqrTest, dbdsqr_init_vectors_false_from_gebrd_28) {
    const int n = 28;
    ColMatrix<double> M(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            M(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                1.0 / static_cast<double>(i + j + 1) + 0.01 * static_cast<double>(i - j);
        }
    }
    ColMatrix<double> A = M;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ColMatrix<double> Ub(n, n);
    ColMatrix<double> VTb(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Ub(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            VTb(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = (i == j) ? 1.0 : 0.0;
        }
    }
    ASSERT_EQ(
        cpu::lapack::dbdsqr('U', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n, 0, false),
        0);
    EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
}

TEST(LapackBdsqrTest, dbdsqr_lowercase_uplo_from_gebrd) {
    const int n = 6;
    ColMatrix<double> A{{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18},
                        {19, 20, 21, 22, 23, 24}, {25, 26, 27, 28, 29, 30}, {31, 32, 33, 34, 35, 36}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);

    std::vector<double> d = D;
    std::vector<double> e = E;
    ColMatrix<double> Ub(n, n);
    ColMatrix<double> VTb(n, n);
    ASSERT_EQ(cpu::lapack::dbdsqr('u', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);
    EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);

    d = D;
    e = E;
    ASSERT_EQ(cpu::lapack::dbdsqr('l', n, d.data(), e.data(), Ub.data(), n, VTb.data(), n), 0);
    EXPECT_GT(d[0], d[static_cast<std::size_t>(n - 1)] - 1e-9);
}

TEST(LapackDorgbrTest, dorgbr_lowercase_vect_chars) {
    ColMatrix<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const int n = 3;
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> Q(n, n);
    ColMatrix<double> P(n, n);
    cpu::lapack::dorgbr('q', n, n, n, A.data(), n, tauq.data(), Q.data(), n);
    cpu::lapack::dorgbr('p', n, n, n, A.data(), n, taup.data(), P.data(), n);
    expect_orthogonal_columns(Q);
    expect_orthogonal_columns(P);
}

TEST(LapackDormbrTest, dormbr_lowercase_side_and_trans) {
    const int n = 4;
    ColMatrix<double> A{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>(n - 1));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    ASSERT_EQ(cpu::lapack::dgebrd(n, n, A.data(), n, D.data(), E.data(), tauq.data(), taup.data()), 0);
    ColMatrix<double> Cq(n, n, 1.0);
    ColMatrix<double> Cp(n, n, 1.0);
    ASSERT_EQ(cpu::lapack::dormbr('q', 'l', 't', n, n, n, A.data(), n, tauq.data(), Cq.data(), n), 0);
    ASSERT_EQ(cpu::lapack::dormbr('p', 'r', 'n', n, n, n, A.data(), n, taup.data(), Cp.data(), n), 0);
    EXPECT_FALSE(std::isnan(Cq(0, 0)));
    EXPECT_FALSE(std::isnan(Cp(0, 0)));
}
