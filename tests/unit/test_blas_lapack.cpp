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
