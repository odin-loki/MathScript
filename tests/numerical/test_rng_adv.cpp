// MathScript Advanced RNG and Random Matrix Tests (Wave 52)
// Tests statistical properties of rand/randn and seed reproducibility.

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "ms/linalg/linalg.hpp"
#include "ms/core/matrix.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// rand: basic properties
// ---------------------------------------------------------------------------

TEST(RngAdv, Rand_Rows_Cols_Correct) {
    auto A = rand<double>(5, 7, 42u);
    EXPECT_EQ(A.rows(), 5u);
    EXPECT_EQ(A.cols(), 7u);
}

TEST(RngAdv, Rand_Values_In_Unit_Interval) {
    auto A = rand<double>(20, 20, 1u);
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_GE(A(i, j), 0.0);
            EXPECT_LE(A(i, j), 1.0);
        }
}

TEST(RngAdv, Rand_Reproducible_With_Same_Seed) {
    auto A1 = rand<double>(4, 4, 777u);
    auto A2 = rand<double>(4, 4, 777u);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_DOUBLE_EQ(A1(i, j), A2(i, j))
                << "rand not reproducible with same seed at (" << i << "," << j << ")";
}

TEST(RngAdv, Rand_Different_Seeds_Different_Values) {
    auto A1 = rand<double>(4, 4, 1u);
    auto A2 = rand<double>(4, 4, 2u);
    int diff_count = 0;
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (A1(i, j) != A2(i, j)) ++diff_count;
    EXPECT_GT(diff_count, 0) << "Different seeds should produce different values";
}

TEST(RngAdv, Rand_Mean_Near_Half) {
    auto A = rand<double>(50, 50, 99u);
    double sum = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            sum += A(i, j);
    double m = sum / static_cast<double>(A.rows() * A.cols());
    EXPECT_NEAR(m, 0.5, 0.05) << "Mean of Uniform[0,1] should be near 0.5";
}

TEST(RngAdv, Rand_AllFinite) {
    auto A = rand<double>(10, 10, 55u);
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            EXPECT_TRUE(std::isfinite(A(i, j)));
}

// ---------------------------------------------------------------------------
// randn: statistical properties
// ---------------------------------------------------------------------------

TEST(RngAdv, Randn_Rows_Cols_Correct) {
    auto A = randn<double>(6, 8, 11u);
    EXPECT_EQ(A.rows(), 6u);
    EXPECT_EQ(A.cols(), 8u);
}

TEST(RngAdv, Randn_AllFinite) {
    auto A = randn<double>(15, 15, 22u);
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            EXPECT_TRUE(std::isfinite(A(i, j)));
}

TEST(RngAdv, Randn_Reproducible_With_Same_Seed) {
    auto A1 = randn<double>(5, 5, 333u);
    auto A2 = randn<double>(5, 5, 333u);
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            EXPECT_DOUBLE_EQ(A1(i, j), A2(i, j))
                << "randn not reproducible with same seed at (" << i << "," << j << ")";
}

TEST(RngAdv, Randn_Mean_Near_Zero) {
    auto A = randn<double>(50, 50, 77u);
    double sum = 0.0;
    size_t N = A.rows() * A.cols();
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            sum += A(i, j);
    EXPECT_NEAR(sum / N, 0.0, 0.2) << "Mean of N(0,1) should be near 0";
}

TEST(RngAdv, Randn_StdDev_Near_One) {
    auto A = randn<double>(50, 50, 88u);
    size_t N = A.rows() * A.cols();
    double sum = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j) {
            sum  += A(i, j);
            sum2 += A(i, j) * A(i, j);
        }
    double m   = sum / N;
    double var = sum2 / N - m * m;
    EXPECT_NEAR(std::sqrt(var), 1.0, 0.15) << "Std dev of N(0,1) should be near 1";
}

TEST(RngAdv, Randn_Different_Seeds_Different_Values) {
    auto A1 = randn<double>(5, 5, 10u);
    auto A2 = randn<double>(5, 5, 20u);
    int diff_count = 0;
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            if (A1(i, j) != A2(i, j)) ++diff_count;
    EXPECT_GT(diff_count, 0);
}

// ---------------------------------------------------------------------------
// diag construction
// ---------------------------------------------------------------------------

TEST(RngAdv, Diag_From_Vector_Shape) {
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto D = diag<double>(v);
    EXPECT_EQ(D.rows(), 4u);
    EXPECT_EQ(D.cols(), 4u);
}

TEST(RngAdv, Diag_From_Vector_Values) {
    std::vector<double> v = {5.0, 7.0, 11.0};
    auto D = diag<double>(v);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(D(i, i), v[i], 1e-14);
}

TEST(RngAdv, Diag_From_Vector_OffDiagZero) {
    std::vector<double> v = {1.0, 2.0, 3.0};
    auto D = diag<double>(v);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            if (i != j) EXPECT_NEAR(D(i, j), 0.0, 1e-14);
}

TEST(RngAdv, Diag_From_Matrix_ExtractsDiagonal) {
    std::vector<double> v = {3.0, 7.0, 2.0};
    auto D = diag<double>(v);
    auto extracted = diag(D);
    ASSERT_EQ(extracted.size(), 3u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(extracted[i], v[i], 1e-14);
}

// ---------------------------------------------------------------------------
// rand → SPD matrix construction
// ---------------------------------------------------------------------------

TEST(RngAdv, Randn_SPD_Construction_Positive_Diag) {
    // A = randn(4,4); B = A^T * A + n*I → SPD
    auto A = randn<double>(4, 4, 42u);
    DMatrix B(4, 4);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < 4; ++k) s += A(k, i) * A(k, j);
            B(i, j) = s;
        }
        B(i, i) += 8.0; // ensure positive definiteness
    }
    // Diagonal elements should all be positive
    for (size_t i = 0; i < 4; ++i)
        EXPECT_GT(B(i, i), 0.0);
}

TEST(RngAdv, Rand_Matrix_NotAllSameValue) {
    auto A = rand<double>(5, 5, 5u);
    double first = A(0, 0);
    bool found_different = false;
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            if (std::abs(A(i, j) - first) > 1e-12) found_different = true;
    EXPECT_TRUE(found_different) << "Random matrix should not be constant";
}
