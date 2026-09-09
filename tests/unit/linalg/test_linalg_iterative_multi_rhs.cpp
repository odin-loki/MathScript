// qmr, tfqmr, lsqr and lsmr with a multi-column right-hand side.
//
// Six of the ten iterative solvers route a multi-column b through solve_per_column, which
// runs the single-vector kernel once per column. These four did not: they built their work
// vectors at b's full shape while every matrix-vector product they take produces a single
// column, so axpy() read past the shorter operand. AddressSanitizer caught it as a
// heap-buffer-overflow on lsmr(A, B) with a 3x3 B -- an eight-byte read one element past a
// three-element buffer.
//
// Each solver must now agree, column for column, with what it returns for that column on
// its own, and the result must actually solve the system.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "ms/linalg/linalg.hpp"

namespace {

using ms::Matrix;

// A symmetric positive definite 3x3, so every solver here applies.
Matrix<double> spd3() {
    Matrix<double> a(3, 3);
    a(0, 0) = 4.0; a(0, 1) = 1.0; a(0, 2) = 0.0;
    a(1, 0) = 1.0; a(1, 1) = 3.0; a(1, 2) = 1.0;
    a(2, 0) = 0.0; a(2, 1) = 1.0; a(2, 2) = 2.0;
    return a;
}

Matrix<double> rhs3x2() {
    Matrix<double> b(3, 2);
    b(0, 0) = 1.0;  b(0, 1) = 0.0;
    b(1, 0) = 2.0;  b(1, 1) = -1.0;
    b(2, 0) = -1.0; b(2, 1) = 3.0;
    return b;
}

Matrix<double> column(const Matrix<double>& m, std::size_t j) {
    Matrix<double> c(m.rows(), 1);
    for (std::size_t i = 0; i < m.rows(); ++i) c(i, 0) = m(i, j);
    return c;
}

double residual(const Matrix<double>& a, const Matrix<double>& x, const Matrix<double>& b,
                std::size_t j) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        double ax = 0.0;
        for (std::size_t k = 0; k < a.cols(); ++k) ax += a(i, k) * x(k, j);
        worst = std::max(worst, std::abs(ax - b(i, j)));
    }
    return worst;
}

}  // namespace

TEST(LinalgIterativeMultiRhs, SquareSolversMatchColumnByColumn) {
    const Matrix<double> a = spd3();
    const Matrix<double> b = rhs3x2();

    struct Case {
        const char* name;
        ms::Result<Matrix<double>> (*solve)(const Matrix<double>&, const Matrix<double>&,
                                            std::size_t, double);
    };
    const Case cases[] = {
        {"qmr", [](const Matrix<double>& A, const Matrix<double>& B, std::size_t it, double t) {
             return ms::qmr(A, B, it, t);
         }},
        {"tfqmr", [](const Matrix<double>& A, const Matrix<double>& B, std::size_t it, double t) {
             return ms::tfqmr(A, B, it, t);
         }},
    };

    for (const auto& c : cases) {
        const auto x = c.solve(a, b, 200, 1e-12);
        ASSERT_TRUE(x.has_value()) << c.name;
        ASSERT_EQ(x->rows(), 3u) << c.name;
        ASSERT_EQ(x->cols(), 2u) << c.name;
        for (std::size_t j = 0; j < 2; ++j) {
            const auto single = c.solve(a, column(b, j), 200, 1e-12);
            ASSERT_TRUE(single.has_value()) << c.name << " column " << j;
            for (std::size_t i = 0; i < 3; ++i) {
                EXPECT_NEAR((*x)(i, j), (*single)(i, 0), 1e-9)
                    << c.name << " (" << i << "," << j << ")";
            }
            EXPECT_LT(residual(a, *x, b, j), 1e-8) << c.name << " column " << j;
        }
    }
}

TEST(LinalgIterativeMultiRhs, LeastSquaresSolversMatchColumnByColumn) {
    // Overdetermined 4x2, so the result has A.cols() rows rather than A.rows().
    Matrix<double> a(4, 2);
    a(0, 0) = 1.0; a(0, 1) = 0.0;
    a(1, 0) = 1.0; a(1, 1) = 1.0;
    a(2, 0) = 1.0; a(2, 1) = 2.0;
    a(3, 0) = 1.0; a(3, 1) = 3.0;
    Matrix<double> b(4, 3);
    for (std::size_t i = 0; i < 4; ++i) {
        b(i, 0) = static_cast<double>(i) * 2.0 + 1.0;   // exactly on the model
        b(i, 1) = static_cast<double>(i * i);
        b(i, 2) = 1.0;
    }

    struct Case {
        const char* name;
        ms::Result<Matrix<double>> (*solve)(const Matrix<double>&, const Matrix<double>&,
                                            std::size_t, double);
    };
    const Case cases[] = {
        {"lsqr", [](const Matrix<double>& A, const Matrix<double>& B, std::size_t it, double t) {
             return ms::lsqr(A, B, it, t);
         }},
        {"lsmr", [](const Matrix<double>& A, const Matrix<double>& B, std::size_t it, double t) {
             return ms::lsmr(A, B, it, t);
         }},
    };

    for (const auto& c : cases) {
        const auto x = c.solve(a, b, 200, 1e-12);
        ASSERT_TRUE(x.has_value()) << c.name;
        ASSERT_EQ(x->rows(), 2u) << c.name << ": one row per column of A";
        ASSERT_EQ(x->cols(), 3u) << c.name;
        for (std::size_t j = 0; j < 3; ++j) {
            const auto single = c.solve(a, column(b, j), 200, 1e-12);
            ASSERT_TRUE(single.has_value()) << c.name << " column " << j;
            for (std::size_t i = 0; i < 2; ++i) {
                EXPECT_NEAR((*x)(i, j), (*single)(i, 0), 1e-8)
                    << c.name << " (" << i << "," << j << ")";
            }
        }
        // Column 0 lies exactly on the model y = 1 + 2t, so the fit is exact.
        EXPECT_NEAR((*x)(0, 0), 1.0, 1e-6) << c.name;
        EXPECT_NEAR((*x)(1, 0), 2.0, 1e-6) << c.name;
    }
}

TEST(LinalgIterativeMultiRhs, MismatchedRowsAreStillReported) {
    const Matrix<double> a = spd3();
    Matrix<double> b(4, 2);
    EXPECT_FALSE(ms::qmr(a, b, 50, 1e-10).has_value());
    EXPECT_FALSE(ms::tfqmr(a, b, 50, 1e-10).has_value());
    EXPECT_FALSE(ms::lsqr(a, b, 50, 1e-10).has_value());
    EXPECT_FALSE(ms::lsmr(a, b, 50, 1e-10).has_value());
}
