// Unit tests for the real QMR, TFQMR, LSMR and SSOR/ILU(0)/PCG implementations.
//
// These four used to be stand-ins: qmr() ran a BiCGSTAB loop, tfqmr() forwarded
// to bicgstab(), lsmr() forwarded to lsqr(), and precond_ssor() returned
// diag(A)/omega. The tests below pin the properties that tell each real
// algorithm apart from the stand-in it replaced, alongside the shape and
// degenerate-input contracts the older suites already rely on.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <initializer_list>
#include <vector>

#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

DMatrix col(std::initializer_list<double> vals) {
    DMatrix v(vals.size(), 1, 0.0);
    size_t i = 0;
    for (const double d : vals) {
        v(i, 0) = d;
        ++i;
    }
    return v;
}

// ||A*x - b||_2
double residual_norm(const DMatrix& A, const DMatrix& x, const DMatrix& b) {
    double acc = 0.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        double row = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) {
            row += A(i, j) * x(j, 0);
        }
        const double d = row - b(i, 0);
        acc += d * d;
    }
    return std::sqrt(acc);
}

// ||A^T * (b - A*x)||_2 -- the quantity LSMR minimises.
double normal_eq_residual(const DMatrix& A, const DMatrix& x, const DMatrix& b) {
    std::vector<double> r(A.rows(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        double row = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) {
            row += A(i, j) * x(j, 0);
        }
        r[i] = b(i, 0) - row;
    }
    double acc = 0.0;
    for (size_t j = 0; j < A.cols(); ++j) {
        double col_sum = 0.0;
        for (size_t i = 0; i < A.rows(); ++i) {
            col_sum += A(i, j) * r[i];
        }
        acc += col_sum * col_sum;
    }
    return std::sqrt(acc);
}

DMatrix apply(const DMatrix& A, const DMatrix& x) {
    DMatrix y(A.rows(), 1, 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        double row = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) {
            row += A(i, j) * x(j, 0);
        }
        y(i, 0) = row;
    }
    return y;
}

DMatrix spd_tridiagonal() {
    return DMatrix{{4.0, 1.0, 0.0}, {1.0, 4.0, 1.0}, {0.0, 1.0, 4.0}};
}

DMatrix nonsymmetric_3x3() {
    return DMatrix{{4.0, 1.0, 0.0}, {2.0, 5.0, 1.0}, {0.0, 1.0, 6.0}};
}

DMatrix poisson_1d() {
    return DMatrix{{4.0, -1.0, 0.0}, {-1.0, 4.0, -1.0}, {0.0, -1.0, 4.0}};
}

} // namespace

// ---------------------------------------------------------------------------
// QMR -- two-sided Lanczos + quasi-minimal residual smoothing
// ---------------------------------------------------------------------------

TEST(QmrRealTest, identity_system_returns_rhs) {
    const DMatrix I = eye<double>(3);
    const DMatrix b = col({5.0, 3.0, 7.0});
    const auto x = qmr(I, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 5.0, 1e-10);
    EXPECT_NEAR((*x)(1, 0), 3.0, 1e-10);
    EXPECT_NEAR((*x)(2, 0), 7.0, 1e-10);
}

TEST(QmrRealTest, spd_tridiagonal_matches_exact_solution) {
    const DMatrix A = spd_tridiagonal();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const auto x = qmr(A, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 5.0 / 28.0, 1e-10);
    EXPECT_NEAR((*x)(1, 0), 2.0 / 7.0, 1e-10);
    EXPECT_NEAR((*x)(2, 0), 19.0 / 28.0, 1e-10);
    EXPECT_LT(residual_norm(A, *x, b), 1e-10);
}

TEST(QmrRealTest, nonsymmetric_system_matches_exact_solution) {
    // A^T is genuinely different from A here, so a solver that quietly ignored
    // the transpose half of the Lanczos recurrence would not land on this.
    const DMatrix A = nonsymmetric_3x3();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const auto x = qmr(A, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 5.0 / 26.0, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 3.0 / 13.0, 1e-8);
    EXPECT_NEAR((*x)(2, 0), 6.0 / 13.0, 1e-8);
    EXPECT_LT(residual_norm(A, *x, b), 1e-8);
}

TEST(QmrRealTest, terminates_in_exactly_n_lanczos_steps) {
    // Unsymmetric Lanczos spans the whole Krylov space after n steps, so QMR
    // is exact at k = n = 3 and cannot be exact at k = 2.
    const DMatrix A = spd_tridiagonal();
    const DMatrix b = col({1.0, 2.0, 3.0});
    EXPECT_TRUE(qmr(A, b, 3, 1e-12).has_value());
    EXPECT_FALSE(qmr(A, b, 2, 1e-12).has_value());
}

TEST(QmrRealTest, zero_rhs_returns_zero_vector) {
    const DMatrix A{{4.0, 1.0}, {1.0, 3.0}};
    const DMatrix zero_b(2, 1, 0.0);
    const auto x = qmr(A, zero_b, 50, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 2u);
    EXPECT_NEAR((*x)(0, 0), 0.0, 1e-15);
    EXPECT_NEAR((*x)(1, 0), 0.0, 1e-15);
}

TEST(QmrRealTest, shape_errors) {
    const DMatrix rect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const DMatrix b3(3, 1, 1.0);
    EXPECT_FALSE(qmr(rect, b3, 10, 1e-10).has_value());

    const DMatrix empty(0, 0);
    const DMatrix b1(1, 1, 1.0);
    EXPECT_FALSE(qmr(empty, b1, 10, 1e-10).has_value());

    const DMatrix A{{4.0, 1.0}, {1.0, 3.0}};
    const DMatrix b_tall(3, 1, 1.0);
    EXPECT_FALSE(qmr(A, b_tall, 10, 1e-10).has_value());
}

TEST(QmrRealTest, reports_convergence_fail_when_iterations_exhausted) {
    const DMatrix A{{4.0, 1.0}, {1.0, 3.0}};
    const DMatrix b = col({1.0, 2.0});
    EXPECT_FALSE(qmr(A, b, 1, 1e-14).has_value());
    EXPECT_FALSE(qmr(A, b, 0, 1e-14).has_value());
}

TEST(QmrRealTest, lanczos_breakdown_is_reported_not_hidden) {
    // Skew-symmetric A makes q^T A p vanish on the first step. Without
    // look-ahead that is a genuine breakdown, and it must surface as an error
    // rather than as a silently wrong iterate.
    const DMatrix skew{{0.0, 1.0}, {-1.0, 0.0}};
    const DMatrix b = col({1.0, 1.0});
    EXPECT_FALSE(qmr(skew, b, 100, 1e-12).has_value());
}

TEST(QmrRealTest, singular_system_does_not_leak_a_bogus_iterate) {
    const DMatrix singular{{1.0, 1.0}, {1.0, 1.0}};
    const DMatrix b = col({1.0, 2.0});
    EXPECT_FALSE(qmr(singular, b, 50, 1e-12).has_value());
}

// ---------------------------------------------------------------------------
// TFQMR -- transpose-free QMR (Freund 1993)
// ---------------------------------------------------------------------------

TEST(TfqmrRealTest, identity_system_returns_rhs) {
    // Regression: the historical half-step loop tested the quasi-residual
    // before applying the x update. On A = I tau hits exactly zero in the very
    // half step that first makes x correct, so it used to return x = 0.
    const DMatrix I = eye<double>(3);
    const DMatrix b = col({5.0, 3.0, 7.0});
    const auto x = tfqmr(I, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 5.0, 1e-10);
    EXPECT_NEAR((*x)(1, 0), 3.0, 1e-10);
    EXPECT_NEAR((*x)(2, 0), 7.0, 1e-10);
}

TEST(TfqmrRealTest, diagonal_system_exact) {
    const DMatrix A{{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}};
    const DMatrix b = col({4.0, 5.0, 6.0});
    const auto x = tfqmr(A, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((*x)(i, 0), 1.0, 1e-8);
    }
}

TEST(TfqmrRealTest, spd_tridiagonal_matches_exact_solution) {
    const DMatrix A = spd_tridiagonal();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const auto x = tfqmr(A, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 5.0 / 28.0, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 2.0 / 7.0, 1e-8);
    EXPECT_NEAR((*x)(2, 0), 19.0 / 28.0, 1e-8);
    EXPECT_LT(residual_norm(A, *x, b), 1e-8);
}

TEST(TfqmrRealTest, nonsymmetric_agrees_with_bicgstab) {
    const DMatrix A = nonsymmetric_3x3();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const auto xt = tfqmr(A, b, 200, 1e-12);
    const auto xb = bicgstab(A, b, 200, 1e-12);
    ASSERT_TRUE(xt.has_value());
    ASSERT_TRUE(xb.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((*xt)(i, 0), (*xb)(i, 0), 1e-6);
    }
}

TEST(TfqmrRealTest, max_iter_counts_outer_iterations_of_two_half_steps) {
    // Convergence on this system needs 5 half steps, i.e. 3 outer iterations.
    const DMatrix A = poisson_1d();
    const DMatrix b = col({1.0, 2.0, 3.0});
    EXPECT_FALSE(tfqmr(A, b, 2, 1e-12).has_value());
    EXPECT_TRUE(tfqmr(A, b, 3, 1e-12).has_value());
}

TEST(TfqmrRealTest, zero_rhs_and_shape_errors) {
    const DMatrix A{{4.0, 1.0}, {1.0, 3.0}};
    const DMatrix zero_b(2, 1, 0.0);
    const auto x = tfqmr(A, zero_b, 50, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 0.0, 1e-15);
    EXPECT_NEAR((*x)(1, 0), 0.0, 1e-15);

    const DMatrix rect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const DMatrix b3(3, 1, 1.0);
    EXPECT_FALSE(tfqmr(rect, b3, 10, 1e-10).has_value());

    const DMatrix b_tall(3, 1, 1.0);
    EXPECT_FALSE(tfqmr(A, b_tall, 10, 1e-10).has_value());
}

TEST(TfqmrRealTest, reports_convergence_fail_when_iterations_exhausted) {
    const DMatrix A{{4.0, 1.0}, {1.0, 3.0}};
    const DMatrix b = col({1.0, 2.0});
    EXPECT_FALSE(tfqmr(A, b, 1, 1e-14).has_value());
}

TEST(TfqmrRealTest, divergent_singular_system_does_not_leak_an_iterate) {
    // The CGS half of TFQMR diverges wildly on this rank-deficient system; the
    // isfinite/stagnation guards plus the final true-residual check must keep
    // the garbage iterate from being returned as a solution.
    const DMatrix singular{{1.0, 1.0}, {1.0, 1.0}};
    const DMatrix b = col({1.0, 2.0});
    EXPECT_FALSE(tfqmr(singular, b, 50, 1e-12).has_value());
}

// ---------------------------------------------------------------------------
// LSMR -- Golub-Kahan bidiagonalisation minimising ||A^T r||
// ---------------------------------------------------------------------------

TEST(LsmrRealTest, identity_system_returns_rhs) {
    const DMatrix I = eye<double>(3);
    const DMatrix b = col({5.0, 3.0, 7.0});
    const auto x = lsmr(I, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 5.0, 1e-10);
    EXPECT_NEAR((*x)(1, 0), 3.0, 1e-10);
    EXPECT_NEAR((*x)(2, 0), 7.0, 1e-10);
}

TEST(LsmrRealTest, square_diagonal_system) {
    const DMatrix A{{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}};
    const DMatrix b = col({4.0, 5.0, 6.0});
    const auto x = lsmr(A, b, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((*x)(i, 0), 1.0, 1e-8);
    }
}

TEST(LsmrRealTest, overdetermined_least_squares_solution) {
    // Straight-line fit: A = [1 t] with t = 0,1,2,3 and b = (1,3,4,6).
    // Normal equations [[4,6],[6,14]] x = (14,29) give x = (1.1, 1.6).
    const DMatrix A{{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};
    const DMatrix b = col({1.0, 3.0, 4.0, 6.0});
    const auto x = lsmr(A, b, 100, 1e-14);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 2u);
    EXPECT_NEAR((*x)(0, 0), 1.1, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 1.6, 1e-8);
    EXPECT_LT(normal_eq_residual(A, *x, b), 1e-8);
    // The system is inconsistent, so the residual itself stays finite.
    EXPECT_NEAR(residual_norm(A, *x, b), std::sqrt(0.2), 1e-8);
}

TEST(LsmrRealTest, singular_system_gives_minimum_norm_solution) {
    // A = [[1,1],[1,1]] has rank 1; the least-squares solutions form the line
    // x0 + x1 = 1.5, whose minimum-norm point is (0.75, 0.75).
    const DMatrix A{{1.0, 1.0}, {1.0, 1.0}};
    const DMatrix b = col({1.0, 2.0});
    const auto x = lsmr(A, b, 50, 1e-14);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 0.75, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 0.75, 1e-8);
}

TEST(LsmrRealTest, monotonically_decreases_normal_equation_residual) {
    // Quadratic fit on five points; tol = 0 forces exactly k iterations.
    const DMatrix A{{1.0, 0.0, 0.0},
                    {1.0, 1.0, 1.0},
                    {1.0, 2.0, 4.0},
                    {1.0, 3.0, 9.0},
                    {1.0, 4.0, 16.0}};
    const DMatrix b = col({1.0, 2.0, 3.0, 5.0, 8.0});
    std::vector<double> normar;
    for (size_t k = 1; k <= 5; ++k) {
        const auto x = lsmr(A, b, k, 0.0);
        ASSERT_TRUE(x.has_value());
        normar.push_back(normal_eq_residual(A, *x, b));
    }
    for (size_t k = 1; k < normar.size(); ++k) {
        EXPECT_LE(normar[k], normar[k - 1] * (1.0 + 1e-6) + 1e-12)
            << "||A^T r|| increased at iteration " << k;
    }
    EXPECT_LT(normar[1], 0.5 * normar[0]);
    EXPECT_LT(normar[2], 1e-6);
}

TEST(LsmrRealTest, converges_to_the_exact_quadratic_fit) {
    const DMatrix A{{1.0, 0.0, 0.0},
                    {1.0, 1.0, 1.0},
                    {1.0, 2.0, 4.0},
                    {1.0, 3.0, 9.0},
                    {1.0, 4.0, 16.0}};
    const DMatrix b = col({1.0, 2.0, 3.0, 5.0, 8.0});
    const auto x = lsmr(A, b, 100, 1e-14);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 39.0 / 35.0, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 19.0 / 70.0, 1e-8);
    EXPECT_NEAR((*x)(2, 0), 5.0 / 14.0, 1e-8);
}

TEST(LsmrRealTest, is_not_lsqr) {
    // LSMR minimises ||A^T r|| over the same Krylov space in which LSQR
    // minimises ||r||, so after two steps each wins on its own functional.
    // While lsmr() forwarded to lsqr() these four numbers were pairwise equal.
    const DMatrix A{{1.0, 0.0, 0.0},
                    {1.0, 1.0, 1.0},
                    {1.0, 2.0, 4.0},
                    {1.0, 3.0, 9.0},
                    {1.0, 4.0, 16.0}};
    const DMatrix b = col({1.0, 2.0, 3.0, 5.0, 8.0});
    const auto xm = lsmr(A, b, 2, 0.0);
    const auto xq = lsqr(A, b, 2, 0.0);
    ASSERT_TRUE(xm.has_value());
    ASSERT_TRUE(xq.has_value());
    EXPECT_LT(normal_eq_residual(A, *xm, b), normal_eq_residual(A, *xq, b));
    EXPECT_LT(residual_norm(A, *xq, b), residual_norm(A, *xm, b));
}

TEST(LsmrRealTest, degenerate_right_hand_sides_return_zero) {
    const DMatrix A = poisson_1d();
    const auto x_zero_b = lsmr(A, DMatrix(3, 1, 0.0), 10, 1e-12);
    ASSERT_TRUE(x_zero_b.has_value());
    EXPECT_EQ(x_zero_b->rows(), 3u);
    EXPECT_NEAR((*x_zero_b)(0, 0), 0.0, 1e-15);

    // A = 0 means A^T b = 0: x = 0 is already the least-squares solution.
    const DMatrix zero_A(3, 3, 0.0);
    const auto x_zero_A = lsmr(zero_A, col({1.0, 2.0, 3.0}), 10, 1e-12);
    ASSERT_TRUE(x_zero_A.has_value());
    EXPECT_NEAR((*x_zero_A)(0, 0), 0.0, 1e-15);
    EXPECT_NEAR((*x_zero_A)(2, 0), 0.0, 1e-15);

    // Zero iterations still yields a value, never ConvergenceFail.
    const auto x_no_iter = lsmr(A, col({1.0, 2.0, 3.0}), 0, 1e-12);
    ASSERT_TRUE(x_no_iter.has_value());
    EXPECT_NEAR((*x_no_iter)(0, 0), 0.0, 1e-15);
}

TEST(LsmrRealTest, rejects_row_count_mismatch_but_allows_rectangular_a) {
    const DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    const DMatrix b3(3, 1, 1.0);
    EXPECT_FALSE(lsmr(A, b3, 10, 1e-10).has_value());

    const DMatrix empty(0, 0);
    const DMatrix b1(1, 1, 1.0);
    EXPECT_FALSE(lsmr(empty, b1, 10, 1e-10).has_value());

    // A rectangular A with matching rows is legal and yields A.cols() unknowns.
    const DMatrix tall{{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}};
    const auto x = lsmr(tall, col({1.0, 2.0, 3.0}), 20, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 2u);
}

// ---------------------------------------------------------------------------
// SSOR / ILU(0) preconditioners
// ---------------------------------------------------------------------------

TEST(PrecondSsorRealTest, diagonal_matrix_reproduces_scaled_diagonal) {
    const DMatrix A{{4.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 6.0}};
    const DMatrix M = precond_ssor(A, 1.0);
    ASSERT_EQ(M.rows(), 3u);
    ASSERT_EQ(M.cols(), 3u);
    EXPECT_NEAR(M(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(M(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(M(2, 2), 6.0, 1e-12);
    EXPECT_NEAR(M(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(M(2, 0), 0.0, 1e-12);
}

TEST(PrecondSsorRealTest, tridiagonal_matrix_has_full_ssor_factors) {
    // M = (D/w + L) (D/w)^-1 (D/w + U). For w = 1 and A = tridiag(-1,4,-1)
    // that is [[4,-1,0],[-1,4.25,-1],[0,-1,4.25]] -- NOT diag(A).
    const DMatrix A = poisson_1d();
    const DMatrix M1 = precond_ssor(A, 1.0);
    EXPECT_NEAR(M1(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(M1(0, 1), -1.0, 1e-12);
    EXPECT_NEAR(M1(0, 2), 0.0, 1e-12);
    EXPECT_NEAR(M1(1, 0), -1.0, 1e-12);
    EXPECT_NEAR(M1(1, 1), 4.25, 1e-12);
    EXPECT_NEAR(M1(1, 2), -1.0, 1e-12);
    EXPECT_NEAR(M1(2, 0), 0.0, 1e-12);
    EXPECT_NEAR(M1(2, 1), -1.0, 1e-12);
    EXPECT_NEAR(M1(2, 2), 4.25, 1e-12);

    const DMatrix M12 = precond_ssor(A, 1.2);
    EXPECT_NEAR(M12(0, 0), 10.0 / 3.0, 1e-12);
    EXPECT_NEAR(M12(1, 1), 109.0 / 30.0, 1e-12);
    EXPECT_NEAR(M12(2, 2), 109.0 / 30.0, 1e-12);
    EXPECT_NEAR(M12(1, 0), -1.0, 1e-12);
    EXPECT_NEAR(M12(0, 1), -1.0, 1e-12);
    EXPECT_NEAR(M12(0, 2), 0.0, 1e-12);
}

TEST(PrecondSsorRealTest, is_symmetric_when_a_is) {
    const DMatrix A = poisson_1d();
    const DMatrix M = precond_ssor(A, 1.35);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(M(i, j), M(j, i), 1e-12);
        }
    }
}

TEST(PrecondSsorRealTest, degenerate_inputs_are_unchanged) {
    const DMatrix A11{{4.0}};
    EXPECT_NEAR(precond_ssor(A11, 2.0)(0, 0), 2.0, 1e-15);

    // omega = 2 must not divide by zero: no 1/(omega*(2-omega)) factor.
    const DMatrix Z{{0.0, 1.0}, {0.0, 0.0}};
    EXPECT_NEAR(precond_ssor(Z, 2.0)(0, 0), 0.0, 1e-15);

    const DMatrix empty(0, 0);
    const DMatrix ME = precond_ssor(empty, 1.0);
    EXPECT_EQ(ME.rows(), 0u);
    EXPECT_EQ(ME.cols(), 0u);

    const DMatrix rect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const DMatrix MR = precond_ssor(rect, 1.0);
    EXPECT_EQ(MR.rows(), 2u);
    EXPECT_EQ(MR.cols(), 2u);
    EXPECT_NEAR(MR(0, 0), 1.0, 1e-12);

    // omega = 0 degrades to omega = 1 rather than dividing by zero.
    const DMatrix A = poisson_1d();
    const DMatrix M0 = precond_ssor(A, 0.0);
    const DMatrix M1 = precond_ssor(A, 1.0);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(M0(i, j), M1(i, j), 1e-12);
        }
    }
}

TEST(PrecondSsorApplyTest, applies_the_inverse_of_the_explicit_matrix) {
    const DMatrix A = poisson_1d();
    const DMatrix r = col({1.0, 2.0, 3.0});
    const DMatrix z = precond_ssor_apply(A, 1.0, r);
    ASSERT_EQ(z.rows(), 3u);
    ASSERT_EQ(z.cols(), 1u);
    EXPECT_NEAR(z(0, 0), 0.4462890625, 1e-12);
    EXPECT_NEAR(z(1, 0), 0.78515625, 1e-12);
    EXPECT_NEAR(z(2, 0), 0.890625, 1e-12);

    const DMatrix M = precond_ssor(A, 1.0);
    const DMatrix Mz = apply(M, z);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Mz(i, 0), r(i, 0), 1e-10);
    }
}

TEST(PrecondSsorApplyTest, agrees_with_the_explicit_matrix_for_other_omega) {
    const DMatrix A = poisson_1d();
    const DMatrix r = col({-2.0, 5.0, 0.5});
    const DMatrix M = precond_ssor(A, 1.4);
    const DMatrix z = precond_ssor_apply(A, 1.4, r);
    const DMatrix Mz = apply(M, z);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(Mz(i, 0), r(i, 0), 1e-10);
    }
}

TEST(PrecondSsorApplyTest, wrong_shaped_rhs_returns_zero_vector) {
    const DMatrix A = poisson_1d();
    const DMatrix z = precond_ssor_apply(A, 1.0, col({1.0, 2.0}));
    ASSERT_EQ(z.rows(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(z(i, 0), 0.0, 1e-15);
    }
}

TEST(PrecondIlu0Test, is_exact_for_a_tridiagonal_matrix) {
    const DMatrix A = poisson_1d();
    const DMatrix LU = precond_ilu0(A);
    EXPECT_NEAR(LU(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(LU(0, 1), -1.0, 1e-12);
    EXPECT_NEAR(LU(1, 0), -0.25, 1e-12);
    EXPECT_NEAR(LU(1, 1), 3.75, 1e-12);
    EXPECT_NEAR(LU(1, 2), -1.0, 1e-12);
    EXPECT_NEAR(LU(2, 1), -4.0 / 15.0, 1e-12);
    EXPECT_NEAR(LU(2, 2), 56.0 / 15.0, 1e-12);

    // A tridiagonal pattern admits no fill-in, so L*U == A exactly and
    // (L*U)^-1 A e == e.
    const DMatrix e = col({1.0, 2.0, 3.0});
    const DMatrix back = precond_ilu0_apply(LU, apply(A, e));
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(back(i, 0), e(i, 0), 1e-10);
    }
}

TEST(PrecondIlu0Test, preserves_the_sparsity_pattern) {
    const DMatrix A{{4.0, 1.0, 0.0, 1.0},
                    {1.0, 4.0, 1.0, 0.0},
                    {0.0, 1.0, 4.0, 1.0},
                    {1.0, 0.0, 1.0, 4.0}};
    const DMatrix LU = precond_ilu0(A);
    EXPECT_NEAR(LU(0, 2), 0.0, 1e-15);
    EXPECT_NEAR(LU(2, 0), 0.0, 1e-15);
    EXPECT_NEAR(LU(1, 3), 0.0, 1e-15);
    EXPECT_NEAR(LU(3, 1), 0.0, 1e-15);
    EXPECT_NEAR(LU(1, 0), 0.25, 1e-12);
    EXPECT_NEAR(LU(1, 1), 3.75, 1e-12);
}

TEST(PrecondIlu0Test, degenerate_inputs) {
    const DMatrix empty(0, 0);
    const DMatrix LUE = precond_ilu0(empty);
    EXPECT_EQ(LUE.rows(), 0u);
    EXPECT_EQ(LUE.cols(), 0u);

    // Rectangular input is truncated to its leading square block.
    const DMatrix rect{{4.0, 1.0, 2.0}, {1.0, 3.0, 5.0}};
    const DMatrix LUR = precond_ilu0(rect);
    EXPECT_EQ(LUR.rows(), 2u);
    EXPECT_EQ(LUR.cols(), 2u);
    EXPECT_NEAR(LUR(1, 0), 0.25, 1e-12);

    // A zero pivot is skipped rather than divided by.
    const DMatrix zero_pivot{{0.0, 1.0}, {1.0, 2.0}};
    const DMatrix LUZ = precond_ilu0(zero_pivot);
    EXPECT_NEAR(LUZ(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(LUZ(1, 1), 2.0, 1e-12);

    const DMatrix LU = precond_ilu0(poisson_1d());
    const DMatrix z = precond_ilu0_apply(LU, col({1.0, 2.0}));
    ASSERT_EQ(z.rows(), 3u);
    EXPECT_NEAR(z(0, 0), 0.0, 1e-15);
}

// ---------------------------------------------------------------------------
// PCG
// ---------------------------------------------------------------------------

TEST(PcgRealTest, ssor_preconditioned_solves_spd_system) {
    const DMatrix A = poisson_1d();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const std::function<DMatrix(const DMatrix&)> M =
        [&A](const DMatrix& r) { return precond_ssor_apply(A, 1.0, r); };
    const auto x = pcg(A, b, M, 100, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 13.0 / 28.0, 1e-8);
    EXPECT_NEAR((*x)(1, 0), 6.0 / 7.0, 1e-8);
    EXPECT_NEAR((*x)(2, 0), 27.0 / 28.0, 1e-8);
}

TEST(PcgRealTest, ilu0_preconditioned_solves_spd_system_in_one_step) {
    const DMatrix A = poisson_1d();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const DMatrix LU = precond_ilu0(A);
    const std::function<DMatrix(const DMatrix&)> M =
        [&LU](const DMatrix& r) { return precond_ilu0_apply(LU, r); };
    // ILU(0) is the exact factorisation of a tridiagonal A, so M^-1 == A^-1
    // and a single PCG step must land on the solution.
    const auto x = pcg(A, b, M, 1, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 13.0 / 28.0, 1e-10);
    EXPECT_NEAR((*x)(1, 0), 6.0 / 7.0, 1e-10);
    EXPECT_NEAR((*x)(2, 0), 27.0 / 28.0, 1e-10);
}

TEST(PcgRealTest, unpreconditioned_matches_cg) {
    const DMatrix A = poisson_1d();
    const DMatrix b = col({1.0, 2.0, 3.0});
    const std::function<DMatrix(const DMatrix&)> M =
        [](const DMatrix& r) { return r; };
    const auto xp = pcg(A, b, M, 100, 1e-12);
    const auto xc = cg(A, b, 100, 1e-12);
    ASSERT_TRUE(xp.has_value());
    ASSERT_TRUE(xc.has_value());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR((*xp)(i, 0), (*xc)(i, 0), 1e-8);
    }
}

TEST(PcgRealTest, rejects_nonsymmetric_mismatched_and_exhausted) {
    const std::function<DMatrix(const DMatrix&)> M =
        [](const DMatrix& r) { return r; };

    const DMatrix nonsym{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_FALSE(pcg(nonsym, col({1.0, 1.0}), M, 10, 1e-10).has_value());

    const DMatrix A = poisson_1d();
    const DMatrix b4(4, 1, 1.0);
    EXPECT_FALSE(pcg(A, b4, M, 10, 1e-10).has_value());

    const DMatrix rect{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    EXPECT_FALSE(pcg(rect, col({1.0, 2.0}), M, 10, 1e-10).has_value());

    EXPECT_FALSE(pcg(A, col({1.0, 2.0, 3.0}), M, 1, 1e-12).has_value());
}

TEST(PcgRealTest, rejects_bad_preconditioners) {
    const DMatrix A = poisson_1d();
    const DMatrix b = col({1.0, 2.0, 3.0});

    const std::function<DMatrix(const DMatrix&)> empty_fn;
    EXPECT_FALSE(pcg(A, b, empty_fn, 10, 1e-12).has_value());

    const std::function<DMatrix(const DMatrix&)> wrong_shape =
        [](const DMatrix&) { return DMatrix(2, 1, 1.0); };
    EXPECT_FALSE(pcg(A, b, wrong_shape, 10, 1e-12).has_value());

    // r^T z < 0 for every non-zero r: not an SPD operator.
    const std::function<DMatrix(const DMatrix&)> indefinite =
        [](const DMatrix& r) {
            DMatrix z(r.rows(), 1, 0.0);
            for (size_t i = 0; i < r.rows(); ++i) {
                z(i, 0) = -r(i, 0);
            }
            return z;
        };
    EXPECT_FALSE(pcg(A, b, indefinite, 10, 1e-12).has_value());
}

TEST(PcgRealTest, zero_rhs_returns_zero_vector) {
    const DMatrix A = poisson_1d();
    const std::function<DMatrix(const DMatrix&)> M =
        [](const DMatrix& r) { return r; };
    const auto x = pcg(A, DMatrix(3, 1, 0.0), M, 10, 1e-12);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 3u);
    EXPECT_NEAR((*x)(0, 0), 0.0, 1e-15);
}
