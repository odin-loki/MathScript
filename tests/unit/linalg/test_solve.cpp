// MathScript Linear Solver Unit Test

#include <gtest/gtest.h>
#include <string>
#include <variant>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(SolveTest, basic_2x2) {
    DMatrix A{{4, 3}, {6, 3}};
    DMatrix b{{1}, {1}};

    auto x = solve(A, b).value();

    DMatrix Ax = A * x;
    for (size_t i = 0; i < b.rows(); ++i) {
        EXPECT_NEAR(Ax(i, 0), b(i, 0), 1e-10);
    }
}

TEST(SolveTest, 3x3_identity) {
    DMatrix A = eye<double>(3);
    DMatrix b{{1}, {2}, {3}};

    auto x = solve(A, b).value();

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(x(i, 0), b(i, 0), 1e-10);
    }
}

TEST(SolveTest, overdetermined_leastsquares) {
    DMatrix A2{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
    DMatrix b2{{14}, {32}, {50}};

    auto x = solve(A2, b2).value();

    DMatrix Ax = A2 * x;
    for (size_t i = 0; i < b2.rows(); ++i) {
        EXPECT_NEAR(Ax(i, 0), b2(i, 0), 1e-10);
    }
}

TEST(SolveTest, multiple_right_hand_sides) {
    DMatrix A{{2, 1}, {1, 3}};
    DMatrix B{{4, 1}, {7, 5}};

    auto x = solve(A, B).value();
    const DMatrix Ax = A * x;

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR(Ax(i, j), B(i, j), 1e-10);
        }
    }
}

TEST(SolveTest, singular_matrix) {
    DMatrix A{{1, 2}, {2, 4}};
    DMatrix b{{5}, {10}};

    auto result = solve(A, b);
    EXPECT_FALSE(result.has_value());
}

TEST(SolveTest, dimension_mismatch_matrix_and_rhs) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    DMatrix b{{1}, {2}};
    EXPECT_FALSE(solve(A, b).has_value());
    DMatrix B{{1, 2}, {3, 4}, {5, 6}};
    EXPECT_FALSE(solve(A, B).has_value());
}

TEST(SolveTest, one_by_one) {
    DMatrix A{{4}};
    DMatrix b{{12}};
    auto x = solve(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 3.0, 1e-12);
}

TEST(SolveTest, one_by_one_singular) {
    DMatrix A{{0}};
    DMatrix b{{1}};
    EXPECT_FALSE(solve(A, b).has_value());
}

TEST(SolveTest, non_square_and_empty_mismatch) {
    DMatrix A32{{1, 2}, {3, 4}, {5, 6}};
    DMatrix b3{{1}, {2}, {3}};
    EXPECT_FALSE(solve(A32, b3).has_value());
    DMatrix A01(0, 1);
    DMatrix b0(0, 1);
    EXPECT_FALSE(solve(A01, b0).has_value());
    DMatrix I = eye<double>(3);
    DMatrix b2{{1}, {2}};
    EXPECT_FALSE(solve(I, b2).has_value());
}

TEST(SolveTest, zero_by_zero_square) {
    DMatrix A(0, 0);
    DMatrix b(0, 1);
    auto x = solve(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 0u);
    EXPECT_EQ(x->cols(), 1u);
}

TEST(SolveTest, float_2x2_and_singular) {
    ColMatrix<float> A{{2.f, 1.f}, {1.f, 3.f}};
    ColMatrix<float> b{{4.f}, {7.f}};
    auto x = solve(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_NEAR((*x)(0, 0), 1.f, 1e-5);
    EXPECT_NEAR((*x)(1, 0), 2.f, 1e-5);

    ColMatrix<float> S{{1.f, 2.f}, {2.f, 4.f}};
    ColMatrix<float> sb{{1.f}, {2.f}};
    EXPECT_FALSE(solve(S, sb).has_value());
}

TEST(SolveTest, float_dimension_mismatch) {
    ColMatrix<float> A{{1.f, 2.f}, {3.f, 4.f}};
    ColMatrix<float> b{{1.f}, {2.f}, {3.f}};
    EXPECT_FALSE(solve(A, b).has_value());
    ColMatrix<float> R{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    ColMatrix<float> br{{1.f}, {2.f}};
    EXPECT_FALSE(solve(R, br).has_value());
}

TEST(SolveTest, identity_multiple_rhs) {
    DMatrix I = eye<double>(2);
    DMatrix B{{1, 0, 3}, {0, 1, -1}};
    auto X = solve(I, B);
    ASSERT_TRUE(X.has_value());
    EXPECT_NEAR((*X)(0, 0), 1.0, 1e-12);
    EXPECT_NEAR((*X)(1, 1), 1.0, 1e-12);
    EXPECT_NEAR((*X)(0, 2), 3.0, 1e-12);
    EXPECT_NEAR((*X)(1, 2), -1.0, 1e-12);
}

TEST(SolveTest, singular_3x3) {
    DMatrix A{{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};
    DMatrix b{{1}, {2}, {3}};
    EXPECT_FALSE(solve(A, b).has_value());
}

TEST(SolveTest, square_rhs_row_mismatch) {
    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix b{{1.0}, {2.0}, {3.0}};
    auto result = solve(A, b);
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(result.error()));
    const auto mismatch = std::get<DimensionMismatch>(result.error());
    EXPECT_EQ(mismatch.got_rows, A.rows());
    EXPECT_EQ(mismatch.got_cols, b.rows());
}

TEST(SolveTest, cuda_copy_path_if_gpu) {
    if (!has_cuda()) {
        GTEST_SKIP() << "CUDA not available";
    }
    const size_t n = 256;
    const auto decision = decide(n, ExecPolicy::AUTO);
    if (decision.backend != Backend::CUDA) {
        GTEST_SKIP() << "decide() did not select CUDA";
    }
    DMatrix A = eye<double>(n);
    DMatrix b(n, 1);
    for (size_t i = 0; i < n; ++i) {
        b(i, 0) = static_cast<double>(i + 1);
    }
    auto x = solve(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), n);
    EXPECT_EQ(x->cols(), 1u);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR((*x)(i, 0), static_cast<double>(i + 1), 1e-8);
    }
}

TEST(SolveTest, cpu_singular_holds_singular_matrix) {
    DMatrix A{{1, 2}, {2, 4}};
    DMatrix b{{3}, {6}};
    auto result = solve(A, b);
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<SingularMatrix>(result.error()));
}

TEST(SolveTest, non_square_holds_dimension_mismatch) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    DMatrix b{{1}, {2}};
    auto result = solve(A, b);
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(result.error()));
    const auto mismatch = std::get<DimensionMismatch>(result.error());
    EXPECT_EQ(mismatch.got_rows, A.rows());
    EXPECT_EQ(mismatch.got_cols, A.cols());
}

TEST(SolveTest, det_trace_row_vector_dimension_mismatch) {
    DMatrix row{{1.0, 2.0, 3.0}};
    const auto d = det(row);
    ASSERT_FALSE(d.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(d.error()));
    EXPECT_FALSE(std::holds_alternative<DomainError>(d.error()));
    const std::string det_msg = format_error(d.error());
    EXPECT_NE(det_msg.find("dimension mismatch"), std::string::npos);
    EXPECT_EQ(det_msg.find("square"), std::string::npos);

    const auto t = trace(row);
    ASSERT_FALSE(t.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(t.error()));
    EXPECT_FALSE(std::holds_alternative<DomainError>(t.error()));
    const std::string tr_msg = format_error(t.error());
    EXPECT_NE(tr_msg.find("dimension mismatch"), std::string::npos);
    EXPECT_EQ(tr_msg.find("square"), std::string::npos);
}

TEST(SolveTest, singular_zero_matrix) {
    DMatrix A{{0.0, 0.0}, {0.0, 0.0}};
    DMatrix b{{1.0}, {2.0}};
    auto result = solve(A, b);
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<SingularMatrix>(result.error()));
}

TEST(SolveTest, rhs_row_mismatch) {
    DMatrix A{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    DMatrix b{{1.0}, {2.0}};
    auto result = solve(A, b);
    ASSERT_FALSE(result.has_value());
    ASSERT_TRUE(std::holds_alternative<DimensionMismatch>(result.error()));
    const auto mismatch = std::get<DimensionMismatch>(result.error());
    EXPECT_EQ(mismatch.got_rows, A.rows());
    EXPECT_EQ(mismatch.got_cols, b.rows());
}
