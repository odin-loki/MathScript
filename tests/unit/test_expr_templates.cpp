#include <gtest/gtest.h>
#include "ms/core/expr.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

// Concrete leaf expression wrapping a Matrix
class MatLeaf : public expr::MatExpr<MatLeaf> {
    const DMatrix& mat_;
public:
    explicit MatLeaf(const DMatrix& m) : mat_(m) {}
    size_t rows_impl() const { return mat_.rows(); }
    size_t cols_impl() const { return mat_.cols(); }
    double eval_at(size_t r, size_t c) const { return mat_(r, c); }
};

} // namespace

TEST(ExprTemplates, MatAdd_evaluates_correctly) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B{{5.0, 6.0}, {7.0, 8.0}};

    const MatLeaf la(A), lb(B);
    const auto sum = la + lb;

    const DMatrix result = static_cast<DMatrix>(sum);
    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 12.0);
}

TEST(ExprTemplates, MatScale_evaluates_correctly) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    const MatLeaf la(A);
    const auto scaled = la * 3.0;

    const DMatrix result = static_cast<DMatrix>(scaled);
    EXPECT_DOUBLE_EQ(result(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 6.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 9.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 12.0);
}

TEST(ExprTemplates, rows_cols_propagated) {
    DMatrix A(3, 4, 1.0);
    DMatrix B(3, 4, 2.0);
    const MatLeaf la(A), lb(B);
    const auto sum = la + lb;
    EXPECT_EQ(sum.rows(), 3u);
    EXPECT_EQ(sum.cols(), 4u);
}

TEST(ExprTemplates, chained_add_and_scale) {
    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix B{{2.0, 0.0}, {0.0, 2.0}};
    const MatLeaf la(A), lb(B);

    // (A + B) * 2 = [[6,0],[0,6]]
    // Store intermediate to avoid dangling reference in expression template chain
    const auto add_expr = la + lb;
    const auto chained = add_expr * 2.0;
    const DMatrix result = static_cast<DMatrix>(chained);
    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 6.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 0.0);
}

TEST(ExprTemplates, zero_scale_gives_zeros) {
    DMatrix A{{5.0, 3.0}, {2.0, 7.0}};
    const MatLeaf la(A);
    const auto zeroed = la * 0.0;
    const DMatrix result = static_cast<DMatrix>(zeroed);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(result(i, j), 0.0);
}

TEST(ExprTemplates, MatMul_evaluates_correctly) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    DMatrix B{{5.0, 6.0}, {7.0, 8.0}};
    const MatLeaf la(A), lb(B);
    const auto product = la * lb;

    const DMatrix result = static_cast<DMatrix>(product);
    EXPECT_DOUBLE_EQ(result(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 43.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 50.0);
}

TEST(ExprTemplates, MatMul_rectangular_shape) {
    DMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};  // 2x3
    DMatrix B{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};  // 3x2
    const MatLeaf la(A), lb(B);
    const auto product = la * lb;

    EXPECT_EQ(product.rows(), 2u);
    EXPECT_EQ(product.cols(), 2u);

    const DMatrix result = static_cast<DMatrix>(product);
    EXPECT_DOUBLE_EQ(result(0, 0), 22.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 28.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 49.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 64.0);
}

TEST(ExprTemplates, MatMul_add_then_mul_matches_dense) {
    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix B{{1.0, 1.0}, {1.0, 1.0}};
    DMatrix C{{2.0, 0.0}, {0.0, 2.0}};
    const MatLeaf la(A), lb(B), lc(C);

    const auto add_expr = la + lb;
    const auto lazy = add_expr * lc;
    const DMatrix lazy_result = static_cast<DMatrix>(lazy);
    const DMatrix dense_result = (A + B) * C;

    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            EXPECT_DOUBLE_EQ(lazy_result(r, c), dense_result(r, c));
}

TEST(ExprTemplates, MatMul_dimension_mismatch_throws) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};                              // 2x2
    DMatrix B{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};  // 3x3
    const MatLeaf la(A), lb(B);

    EXPECT_THROW(static_cast<void>(la * lb), DimensionMismatch);
}

TEST(ExprTemplates, MatMul_identity_preserves_operand) {
    DMatrix I{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix A{{2.0, 3.0}, {4.0, 5.0}};
    const MatLeaf li(I), la(A);

    const DMatrix result = static_cast<DMatrix>(li * la);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            EXPECT_DOUBLE_EQ(result(r, c), A(r, c));
}

TEST(ExprTemplates, MatScale_still_works_after_matmul_operator) {
    DMatrix A{{1.0, 2.0}, {3.0, 4.0}};
    const MatLeaf la(A);
    const auto scaled = la * 2.5;

    const DMatrix result = static_cast<DMatrix>(scaled);
    EXPECT_DOUBLE_EQ(result(0, 0), 2.5);
    EXPECT_DOUBLE_EQ(result(1, 1), 10.0);
}

TEST(ExprTemplates, MatMul_chained_with_scale) {
    DMatrix A{{1.0, 0.0}, {0.0, 1.0}};
    DMatrix B{{2.0, 0.0}, {0.0, 2.0}};
    const MatLeaf la(A), lb(B);

    const auto mul_expr = la * lb;
    const auto scaled = mul_expr * 3.0;
    const DMatrix result = static_cast<DMatrix>(scaled);

    EXPECT_DOUBLE_EQ(result(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 6.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 0.0);
}
