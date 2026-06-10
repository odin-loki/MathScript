#include <gtest/gtest.h>
#include <cmath>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

double frobenius_norm(const DMatrix& A) {
    double sum = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < A.cols(); ++j) {
            sum += A(i, j) * A(i, j);
        }
    }
    return std::sqrt(sum);
}

} // namespace

TEST(NumericalLinalg, LU_3x3_PA_equals_LU) {
    const DMatrix A{{4, 3, 2}, {6, 7, 5}, {2, 8, 3}};
    const auto [L, U, P] = lu(A).value();
    const DMatrix residual = P * A - L * U;
    EXPECT_LT(frobenius_norm(residual), 1e-14);
}

TEST(NumericalLinalg, LU_4x4_PA_equals_LU) {
    const DMatrix A{
        {11, 2, 3, 4},
        {5, 16, 7, 8},
        {9, 10, 21, 12},
        {13, 14, 15, 26}};
    const auto [L, U, P] = lu(A).value();
    const DMatrix residual = P * A - L * U;
    EXPECT_LT(frobenius_norm(residual), 1e-14);
}

TEST(NumericalLinalg, SVD_2x2_reconstruction) {
    const DMatrix A{{3, 1}, {1, 2}};
    const auto s = svd(A).value();

    for (std::size_t i = 1; i < s.S.rows(); ++i) {
        EXPECT_GE(s.S(i - 1, 0), s.S(i, 0));
    }

    DMatrix Sigma = zeros<double>(s.S.rows(), s.S.rows());
    for (std::size_t i = 0; i < s.S.rows(); ++i) {
        Sigma(i, i) = s.S(i, 0);
    }
    const DMatrix reconstructed = s.U * Sigma * transpose(s.V);
    const DMatrix residual = A - reconstructed;
    EXPECT_LT(frobenius_norm(residual), 1e-13);
}

TEST(NumericalLinalg, SVD_3x3_diagonal_reconstruction) {
    const DMatrix A = diag(std::vector<double>{5, 3, 1});
    const auto s = svd(A).value();

    for (std::size_t i = 1; i < s.S.rows(); ++i) {
        EXPECT_GE(s.S(i - 1, 0), s.S(i, 0));
    }

    DMatrix Sigma = zeros<double>(s.S.rows(), s.S.rows());
    for (std::size_t i = 0; i < s.S.rows(); ++i) {
        Sigma(i, i) = s.S(i, 0);
    }
    const DMatrix reconstructed = s.U * Sigma * transpose(s.V);
    const DMatrix residual = A - reconstructed;
    EXPECT_LT(frobenius_norm(residual), 1e-13);
}

TEST(NumericalLinalg, SVD_4x4_reconstruction) {
    const DMatrix A = diag(std::vector<double>{9, 7, 5, 3});
    const auto s = svd(A).value();

    for (std::size_t i = 1; i < s.S.rows(); ++i) {
        EXPECT_GE(s.S(i - 1, 0), s.S(i, 0));
    }

    DMatrix Sigma = zeros<double>(s.S.rows(), s.S.rows());
    for (std::size_t i = 0; i < s.S.rows(); ++i) {
        Sigma(i, i) = s.S(i, 0);
    }
    const DMatrix reconstructed = s.U * Sigma * transpose(s.V);
    const DMatrix residual = A - reconstructed;
    EXPECT_LT(frobenius_norm(residual), 1e-13);
}

TEST(NumericalLinalg, Solve_3x3_residual) {
    const DMatrix A{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    const DMatrix b{{1}, {2}, {3}};
    const auto x = solve(A, b).value();
    const DMatrix residual = A * x - b;
    EXPECT_LT(frobenius_norm(residual), 1e-12);
}

TEST(NumericalLinalg, Solve_4x4_residual) {
    const DMatrix A{
        {8, 1, 0, 0},
        {1, 7, 1, 0},
        {0, 1, 6, 1},
        {0, 0, 1, 5}};
    const DMatrix b{{1}, {2}, {3}, {4}};
    const auto x = solve(A, b).value();
    const DMatrix residual = A * x - b;
    EXPECT_LT(frobenius_norm(residual), 1e-12);
}

TEST(NumericalLinalg, EigSym_3x3_spd) {
    const DMatrix A = diag(std::vector<double>{2, 5, 8});
    const auto result = eig_sym(A).value();

    for (std::size_t i = 1; i < result.values.rows(); ++i) {
        EXPECT_GE(result.values(i - 1, 0), result.values(i, 0));
    }

    EXPECT_NEAR(result.values(0, 0), 8.0, 1e-12);
    EXPECT_NEAR(result.values(1, 0), 5.0, 1e-12);
    EXPECT_NEAR(result.values(2, 0), 2.0, 1e-12);
}

TEST(NumericalLinalg, EigSym_4x4_diagonal) {
    const DMatrix A = diag(std::vector<double>{1, 4, 9, 16});
    const auto result = eig_sym(A).value();

    EXPECT_NEAR(result.values(0, 0), 16.0, 1e-12);
    EXPECT_NEAR(result.values(1, 0), 9.0, 1e-12);
    EXPECT_NEAR(result.values(2, 0), 4.0, 1e-12);
    EXPECT_NEAR(result.values(3, 0), 1.0, 1e-12);
}

TEST(NumericalLinalg, Chol_3x3_reconstruction) {
    const DMatrix A{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    const auto L = chol(A).value();
    const DMatrix residual = A - L * transpose(L);
    EXPECT_LT(frobenius_norm(residual), 1e-13);
}

TEST(NumericalLinalg, Chol_4x4_reconstruction) {
    const DMatrix A{
        {9, 1, 0, 0},
        {1, 8, 1, 0},
        {0, 1, 7, 1},
        {0, 0, 1, 6}};
    const auto L = chol(A).value();
    const DMatrix residual = A - L * transpose(L);
    EXPECT_LT(frobenius_norm(residual), 1e-13);
}
