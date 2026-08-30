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

TEST(MatmulTest, cpu_policy_2x2) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{5, 6}, {7, 8}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 19);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 22);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 43);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 50);
}

TEST(MatmulTest, auto_policy_2x2) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{5, 6}, {7, 8}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 19);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 22);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 43);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 50);
}

TEST(MatmulTest, one_by_one) {
    DMatrix A{{7}};
    DMatrix B{{3}};
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 1u);
    EXPECT_EQ(C->cols(), 1u);
    EXPECT_DOUBLE_EQ((*C)(0, 0), 21);
}

TEST(MatmulTest, one_by_one_all_policies) {
    DMatrix A{{2}};
    DMatrix B{{9}};
    auto cpu = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    auto gpu = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    auto aut = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(gpu.has_value());
    ASSERT_TRUE(aut.has_value());
    EXPECT_DOUBLE_EQ((*cpu)(0, 0), 18);
    EXPECT_DOUBLE_EQ((*gpu)(0, 0), 18);
    EXPECT_DOUBLE_EQ((*aut)(0, 0), 18);
}

TEST(MatmulTest, empty_zero_rows) {
    DMatrix A(0, 3);
    DMatrix B(3, 2);
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 0u);
    EXPECT_EQ(C->cols(), 2u);
}

TEST(MatmulTest, empty_zero_inner) {
    DMatrix A(4, 0);
    DMatrix B(0, 3);
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 4u);
    EXPECT_EQ(C->cols(), 3u);
    for (size_t i = 0; i < C->rows(); ++i) {
        for (size_t j = 0; j < C->cols(); ++j) {
            EXPECT_DOUBLE_EQ((*C)(i, j), 0.0);
        }
    }
}

TEST(MatmulTest, empty_zero_by_zero) {
    DMatrix A(0, 0);
    DMatrix B(0, 0);
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 0u);
    EXPECT_EQ(C->cols(), 0u);
}

TEST(MatmulTest, empty_gpu_policy) {
    DMatrix A(0, 2);
    DMatrix B(2, 0);
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 0u);
    EXPECT_EQ(C->cols(), 0u);
}

TEST(MatmulTest, inner_product_shape) {
    DMatrix A{{1, 2, 3}};
    DMatrix B{{4}, {5}, {6}};
    auto C = matmul(A, B).value();
    EXPECT_EQ(C.rows(), 1u);
    EXPECT_EQ(C.cols(), 1u);
    EXPECT_DOUBLE_EQ(C(0, 0), 32);
}

TEST(MatmulTest, outer_product_shape) {
    DMatrix A{{1}, {2}, {3}};
    DMatrix B{{4, 5}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::CPU)).value();
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 2u);
    EXPECT_DOUBLE_EQ(C(0, 0), 4);
    EXPECT_DOUBLE_EQ(C(0, 1), 5);
    EXPECT_DOUBLE_EQ(C(2, 0), 12);
    EXPECT_DOUBLE_EQ(C(2, 1), 15);
}

TEST(MatmulTest, transpose_like_non_square) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    DMatrix At{{1, 4}, {2, 5}, {3, 6}};
    auto AAt = matmul(A, At, static_cast<int>(ExecPolicy::CPU)).value();
    auto AtA = matmul(At, A, static_cast<int>(ExecPolicy::AUTO)).value();
    EXPECT_EQ(AAt.rows(), 2u);
    EXPECT_EQ(AAt.cols(), 2u);
    EXPECT_EQ(AtA.rows(), 3u);
    EXPECT_EQ(AtA.cols(), 3u);
    EXPECT_DOUBLE_EQ(AAt(0, 0), 14);
    EXPECT_DOUBLE_EQ(AAt(0, 1), 32);
    EXPECT_DOUBLE_EQ(AAt(1, 0), 32);
    EXPECT_DOUBLE_EQ(AAt(1, 1), 77);
    EXPECT_DOUBLE_EQ(AtA(0, 0), 17);
    EXPECT_DOUBLE_EQ(AtA(1, 1), 29);
    EXPECT_DOUBLE_EQ(AtA(2, 2), 45);
}

TEST(MatmulTest, row_major_one_by_one) {
    RMatrix A{{5}};
    RMatrix B{{7}};
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 35);
}

TEST(MatmulTest, row_major_empty) {
    RMatrix A(0, 2);
    RMatrix B(2, 1);
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 0u);
    EXPECT_EQ(C->cols(), 1u);
}

TEST(MatmulTest, float_one_by_one) {
    ColMatrix<float> A{{1.5f}};
    ColMatrix<float> B{{2.0f}};
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_FLOAT_EQ((*C)(0, 0), 3.0f);
}

TEST(MatmulTest, cpu_policy_dimension_mismatch) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{1, 2, 3}};
    auto result = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    EXPECT_FALSE(result.has_value());
}

TEST(MatmulTest, auto_policy_dimension_mismatch) {
    DMatrix A{{1, 2}};
    DMatrix B{{1, 2}, {3, 4}, {5, 6}};
    auto result = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    EXPECT_FALSE(result.has_value());
}

TEST(MatmulTest, rectangular_2x3_by_3x4_all_policies) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    DMatrix B{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
    auto cpu = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    auto gpu = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    auto aut = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(gpu.has_value());
    ASSERT_TRUE(aut.has_value());
    EXPECT_EQ(cpu->rows(), 2u);
    EXPECT_EQ(cpu->cols(), 4u);
    EXPECT_DOUBLE_EQ((*cpu)(0, 0), 1);
    EXPECT_DOUBLE_EQ((*cpu)(0, 1), 2);
    EXPECT_DOUBLE_EQ((*cpu)(0, 2), 3);
    EXPECT_DOUBLE_EQ((*cpu)(0, 3), 6);
    EXPECT_DOUBLE_EQ((*cpu)(1, 0), 4);
    EXPECT_DOUBLE_EQ((*cpu)(1, 3), 15);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ((*gpu)(i, j), (*cpu)(i, j));
            EXPECT_DOUBLE_EQ((*aut)(i, j), (*cpu)(i, j));
        }
    }
}

TEST(MatmulTest, row_major_dimension_mismatch_all_policies) {
    RMatrix A{{1, 2}};
    RMatrix B{{1, 2}, {3, 4}, {5, 6}};
    EXPECT_FALSE(matmul(A, B).has_value());
    EXPECT_FALSE(matmul(A, B, static_cast<int>(ExecPolicy::CPU)).has_value());
    EXPECT_FALSE(matmul(A, B, static_cast<int>(ExecPolicy::GPU)).has_value());
    EXPECT_FALSE(matmul(A, B, static_cast<int>(ExecPolicy::AUTO)).has_value());
}

TEST(MatmulTest, float_empty_rect_and_mismatch) {
    ColMatrix<float> A(0, 2);
    ColMatrix<float> B(2, 3);
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 0u);
    EXPECT_EQ(C->cols(), 3u);

    ColMatrix<float> At{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    ColMatrix<float> Bt{{1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}};
    auto R = matmul(At, Bt, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(R.has_value());
    EXPECT_FLOAT_EQ((*R)(0, 0), 4.f);
    EXPECT_FLOAT_EQ((*R)(1, 1), 11.f);

    ColMatrix<float> badA{{1.f, 2.f}};
    ColMatrix<float> badB{{1.f, 2.f}};
    EXPECT_FALSE(matmul(badA, badB).has_value());
    EXPECT_FALSE(matmul(badA, badB, static_cast<int>(ExecPolicy::GPU)).has_value());
}

TEST(MatmulTest, empty_mismatch_and_auto_zero_rows) {
    DMatrix A(2, 0);
    DMatrix B(1, 3);
    EXPECT_FALSE(matmul(A, B).has_value());
    EXPECT_FALSE(matmul(A, B, static_cast<int>(ExecPolicy::CPU)).has_value());
    DMatrix A00(0, 0);
    DMatrix B10(1, 0);
    EXPECT_FALSE(matmul(A00, B10, static_cast<int>(ExecPolicy::AUTO)).has_value());

    DMatrix Zr(0, 2);
    DMatrix Br(2, 1);
    auto C = matmul(Zr, Br, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 0u);
    EXPECT_EQ(C->cols(), 1u);

    DMatrix Zc(3, 0);
    DMatrix Bc(0, 2);
    auto C2 = matmul(Zc, Bc, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(C2.has_value());
    EXPECT_EQ(C2->rows(), 3u);
    EXPECT_EQ(C2->cols(), 2u);
}

TEST(MatmulTest, row_major_empty_inner_and_auto_rect) {
    RMatrix A(2, 0);
    RMatrix B(0, 3);
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 2u);
    EXPECT_EQ(C->cols(), 3u);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ((*C)(i, j), 0.0);
        }
    }

    RMatrix Ar{{1, 2, 3}, {4, 5, 6}};
    RMatrix Br{{1}, {0}, {1}};
    auto R = matmul(Ar, Br, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(R.has_value());
    EXPECT_DOUBLE_EQ((*R)(0, 0), 4);
    EXPECT_DOUBLE_EQ((*R)(1, 0), 10);
}

TEST(MatmulTest, gpu_policy_rect_and_empty_inner) {
    DMatrix A{{1, 0}, {0, 1}, {1, 1}};
    DMatrix B{{2, 3}, {4, 5}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 3u);
    EXPECT_EQ(C->cols(), 2u);
    EXPECT_DOUBLE_EQ((*C)(0, 0), 2);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 5);
    EXPECT_DOUBLE_EQ((*C)(2, 0), 6);
    EXPECT_DOUBLE_EQ((*C)(2, 1), 8);

    DMatrix Z(4, 0);
    DMatrix W(0, 1);
    auto E = matmul(Z, W, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(E.has_value());
    EXPECT_EQ(E->rows(), 4u);
    EXPECT_EQ(E->cols(), 1u);
    EXPECT_DOUBLE_EQ((*E)(0, 0), 0.0);
}

TEST(MatmulTest, tall_4x2_by_2x3) {
    DMatrix A{{1, 0}, {0, 1}, {1, 1}, {2, 0}};
    DMatrix B{{1, 2, 3}, {4, 5, 6}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 4u);
    EXPECT_EQ(C->cols(), 3u);
    EXPECT_DOUBLE_EQ((*C)(0, 0), 1);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 5);
    EXPECT_DOUBLE_EQ((*C)(2, 2), 9);
    EXPECT_DOUBLE_EQ((*C)(3, 0), 2);
}

TEST(MatmulTest, float_all_policies_2x2) {
    ColMatrix<float> A{{1.f, 2.f}, {3.f, 4.f}};
    ColMatrix<float> B{{5.f, 6.f}, {7.f, 8.f}};
    auto cpu = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    auto gpu = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    auto aut = matmul(A, B, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(cpu.has_value());
    ASSERT_TRUE(gpu.has_value());
    ASSERT_TRUE(aut.has_value());
    EXPECT_FLOAT_EQ((*cpu)(0, 0), 19.f);
    EXPECT_FLOAT_EQ((*gpu)(1, 1), 50.f);
    EXPECT_FLOAT_EQ((*aut)(0, 1), 22.f);
}

TEST(MatmulTest, row_major_explicit_2x2) {
    using RowM = Matrix<double, StorageOrder::RowMajor>;
    RowM A{{1.0, 2.0}, {3.0, 4.0}};
    RowM B{{5.0, 6.0}, {7.0, 8.0}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 19.0);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 22.0);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 43.0);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 50.0);
}

TEST(MatmulTest, row_major_cuda_copy_to_col_if_gpu) {
    using RowM = Matrix<double, StorageOrder::RowMajor>;
    RowM A{{1.0, 2.0}, {3.0, 4.0}};
    RowM B{{5.0, 6.0}, {7.0, 8.0}};
    const auto decision = decide(
        (std::max)({A.rows(), A.cols(), B.cols()}),
        ExecPolicy::GPU);
    if (decision.backend != Backend::CUDA) {
        GTEST_SKIP() << "decide() did not select CUDA";
    }
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_DOUBLE_EQ((*C)(0, 0), 19.0);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 22.0);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 43.0);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 50.0);
}

TEST(MatmulTest, row_major_rect_3x2_by_2x4) {
    RMatrix A{{1, 2}, {3, 4}, {5, 6}};
    RMatrix B{{1, 0, 1, 0}, {0, 1, 0, 1}};
    auto C = matmul(A, B, static_cast<int>(ExecPolicy::CPU));
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 3u);
    EXPECT_EQ(C->cols(), 4u);
    EXPECT_DOUBLE_EQ((*C)(0, 0), 1);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 2);
    EXPECT_DOUBLE_EQ((*C)(0, 2), 1);
    EXPECT_DOUBLE_EQ((*C)(0, 3), 2);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 3);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 4);
    EXPECT_DOUBLE_EQ((*C)(2, 0), 5);
    EXPECT_DOUBLE_EQ((*C)(2, 3), 6);
}

TEST(MatmulTest, row_major_transpose_like_gpu_copy) {
    RMatrix A{{1, 2, 3}, {4, 5, 6}};
    RMatrix At{{1, 4}, {2, 5}, {3, 6}};
    auto AAt = matmul(A, At, static_cast<int>(ExecPolicy::GPU));
    ASSERT_TRUE(AAt.has_value());
    EXPECT_EQ(AAt->rows(), 2u);
    EXPECT_EQ(AAt->cols(), 2u);
    EXPECT_DOUBLE_EQ((*AAt)(0, 0), 14);
    EXPECT_DOUBLE_EQ((*AAt)(0, 1), 32);
    EXPECT_DOUBLE_EQ((*AAt)(1, 0), 32);
    EXPECT_DOUBLE_EQ((*AAt)(1, 1), 77);

    auto AtA = matmul(At, A, static_cast<int>(ExecPolicy::AUTO));
    ASSERT_TRUE(AtA.has_value());
    EXPECT_EQ(AtA->rows(), 3u);
    EXPECT_EQ(AtA->cols(), 3u);
    EXPECT_DOUBLE_EQ((*AtA)(0, 0), 17);
    EXPECT_DOUBLE_EQ((*AtA)(1, 1), 29);
    EXPECT_DOUBLE_EQ((*AtA)(2, 2), 45);
}

TEST(MatmulTest, col_times_row_major_copies) {
    using RowM = Matrix<double, StorageOrder::RowMajor>;
    const DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const RowM B{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    auto C = matmul(A, B);
    ASSERT_TRUE(C.has_value());
    EXPECT_EQ(C->rows(), 2u);
    EXPECT_EQ(C->cols(), 2u);
    EXPECT_DOUBLE_EQ((*C)(0, 0), 58.0);
    EXPECT_DOUBLE_EQ((*C)(0, 1), 64.0);
    EXPECT_DOUBLE_EQ((*C)(1, 0), 139.0);
    EXPECT_DOUBLE_EQ((*C)(1, 1), 154.0);
}

TEST(MatmulTest, inner_dim_mismatch) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    auto result = matmul(A, B);
    ASSERT_FALSE(result.has_value());
}
