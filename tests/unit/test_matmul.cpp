// MathScript Matrix Multiply Unit Test

#include <gtest/gtest.h>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/topology.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;
using RMatrix = RowMatrix<double>;

TEST(MatmulTest, basic_2x2) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{5, 6}, {7, 8}};
    auto C = matmul(A, B).value();

    EXPECT_DOUBLE_EQ(C(0, 0), 19);
    EXPECT_DOUBLE_EQ(C(0, 1), 22);
    EXPECT_DOUBLE_EQ(C(1, 0), 43);
    EXPECT_DOUBLE_EQ(C(1, 1), 50);
}

TEST(MatmulTest, 3x3_identity) {
    DMatrix I = eye<double>(3);
    DMatrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto C = matmul(I, A).value();

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(C(i, j), A(i, j), 1e-12);
        }
    }
}

TEST(MatmulTest, dimension_mismatch) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto result = matmul(A, B);
    EXPECT_FALSE(result.has_value());
}

TEST(MatmulTest, gpu_policy_smoke_without_hardware) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{5, 6}, {7, 8}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 19);
}

TEST(MatmulTest, row_major_uses_generic_path) {
    RMatrix A{{1, 2, 3}, {4, 5, 6}};
    RMatrix B{{7, 8}, {9, 10}, {11, 12}};
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 58);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 154);
}

TEST(MatmulTest, row_and_col_major_same_result) {
    const DMatrix A_col{{1, 2, 3}, {4, 5, 6}};
    const DMatrix B_col{{7, 8}, {9, 10}, {11, 12}};
    const RMatrix A_row{{1, 2, 3}, {4, 5, 6}};
    const RMatrix B_row{{7, 8}, {9, 10}, {11, 12}};

    const auto C_col = matmul(A_col, B_col, static_cast<int>(ExecPolicy::CPU)).value();
    const auto C_row = matmul(A_row, B_row, static_cast<int>(ExecPolicy::CPU)).value();

    ASSERT_EQ(C_col.rows(), C_row.rows());
    ASSERT_EQ(C_col.cols(), C_row.cols());
    for (size_t i = 0; i < C_col.rows(); ++i) {
        for (size_t j = 0; j < C_col.cols(); ++j) {
            EXPECT_NEAR(C_col(i, j), C_row(i, j), 1e-12);
        }
    }
}

TEST(MatmulTest, float_uses_generic_path) {
    ColMatrix<float> A{{1.f, 2.f}, {3.f, 4.f}};
    ColMatrix<float> B{{5.f, 6.f}, {7.f, 8.f}};
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_FLOAT_EQ((*C)(0, 0), 19.f);
    EXPECT_FLOAT_EQ((*C)(1, 1), 50.f);
}

TEST(MatmulTest, gpu_policy_falls_back_to_cpu_without_cuda) {
    if (has_cuda()) {
        GTEST_SKIP() << "CUDA available; CPU fallback path not under test";
    }

    const DMatrix A{{1, 2, 3}, {4, 5, 6}};
    const DMatrix B{{7, 8}, {9, 10}, {11, 12}};
    const auto decision = decide(
        (std::max)({A.rows(), A.cols(), B.cols()}),
        ExecPolicy::GPU);
    EXPECT_EQ(decision.policy, ExecPolicy::GPU);
    EXPECT_EQ(decision.backend, Backend::CPU);

    auto gpu_result = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    auto cpu_result = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(gpu_result.has_value());
    ASSERT_TRUE(cpu_result.has_value());
    for (size_t i = 0; i < cpu_result->rows(); ++i) {
        for (size_t j = 0; j < cpu_result->cols(); ++j) {
            EXPECT_NEAR((*gpu_result)(i, j), (*cpu_result)(i, j), 1e-12);
        }
    }
}

TEST(MatmulTest, gpu_policy_row_major_uses_generic_cpu_path) {
    RMatrix A{{1, 2, 3}, {4, 5, 6}};
    RMatrix B{{7, 8}, {9, 10}, {11, 12}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 58);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 64);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 139);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 154);
}

TEST(MatmulTest, gpu_policy_dimension_mismatch) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{1, 2, 3}};
    auto result = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    EXPECT_FALSE(result.has_value());
}
