// MathScript: Advanced Polynomial Operation Tests
// Tests for poly_add, poly_sub, poly_mul, poly_eval, poly_deriv edge cases

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/poly/poly.hpp"

using namespace ms;
using Poly = std::vector<double>;

// ---------------------------------------------------------------------------
// poly_add
// ---------------------------------------------------------------------------

TEST(PolyAdv, Add_Zero_PolynomialIsIdentity) {
    Poly a = {1.0, 2.0, 3.0};
    Poly zero = {0.0, 0.0, 0.0};
    auto r = poly_add(a, zero);
    ASSERT_EQ(r.size(), 3u);
    EXPECT_NEAR(r[0], 1.0, 1e-10);
    EXPECT_NEAR(r[1], 2.0, 1e-10);
    EXPECT_NEAR(r[2], 3.0, 1e-10);
}

TEST(PolyAdv, Add_Commutative) {
    Poly a = {1.0, -2.0, 3.0};
    Poly b = {0.5, 1.5, -1.0};
    auto r1 = poly_add(a, b);
    auto r2 = poly_add(b, a);
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0; i < r1.size(); ++i)
        EXPECT_NEAR(r1[i], r2[i], 1e-10) << "at index " << i;
}

TEST(PolyAdv, Add_OppositeCoeffs) {
    Poly a = {1.0, 2.0, 3.0};
    Poly b = {-1.0, -2.0, -3.0};
    auto r = poly_add(a, b);
    for (size_t i = 0; i < r.size(); ++i)
        EXPECT_NEAR(r[i], 0.0, 1e-10);
}

TEST(PolyAdv, Add_DifferentSizes) {
    Poly a = {1.0, 2.0};
    Poly b = {3.0, 4.0, 5.0};
    auto r = poly_add(a, b);
    // Longer result should be at least 3 coefficients
    EXPECT_GE(r.size(), 3u);
}

// ---------------------------------------------------------------------------
// poly_sub
// ---------------------------------------------------------------------------

TEST(PolyAdv, Sub_Self_IsZero) {
    Poly a = {3.0, -1.0, 2.5};
    auto r = poly_sub(a, a);
    for (size_t i = 0; i < r.size(); ++i)
        EXPECT_NEAR(r[i], 0.0, 1e-10);
}

TEST(PolyAdv, Sub_ZeroSubtracted_IsIdentity) {
    Poly a = {1.0, 2.0, 3.0};
    Poly zero = {0.0, 0.0, 0.0};
    auto r = poly_sub(a, zero);
    ASSERT_EQ(r.size(), a.size());
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(r[i], a[i], 1e-10);
}

TEST(PolyAdv, Sub_IsAntiCommutative_AddInverse) {
    Poly a = {1.0, 2.0, 3.0};
    Poly b = {4.0, 5.0, 6.0};
    auto r_sub = poly_sub(a, b);
    auto r_add_neg = poly_add(a, {-4.0, -5.0, -6.0});
    ASSERT_EQ(r_sub.size(), r_add_neg.size());
    for (size_t i = 0; i < r_sub.size(); ++i)
        EXPECT_NEAR(r_sub[i], r_add_neg[i], 1e-10);
}

TEST(PolyAdv, Sub_DifferentSizes) {
    Poly a = {1.0, 2.0, 3.0};
    Poly b = {1.0};
    auto r = poly_sub(a, b);
    EXPECT_GE(r.size(), 3u);
}

// ---------------------------------------------------------------------------
// poly_mul
// ---------------------------------------------------------------------------

TEST(PolyAdv, Mul_Commutative) {
    Poly a = {1.0, 2.0};
    Poly b = {3.0, 4.0, 5.0};
    auto r1 = poly_mul(a, b);
    auto r2 = poly_mul(b, a);
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0; i < r1.size(); ++i)
        EXPECT_NEAR(r1[i], r2[i], 1e-10);
}

TEST(PolyAdv, Mul_ByConstant) {
    // (1 + x) * 3 = 3 + 3x
    Poly a = {1.0, 1.0};
    Poly c = {3.0};
    auto r = poly_mul(a, c);
    ASSERT_GE(r.size(), 2u);
    EXPECT_NEAR(r[0], 3.0, 1e-10);
    EXPECT_NEAR(r[1], 3.0, 1e-10);
}

TEST(PolyAdv, Mul_ByMonomial) {
    // (1 + x) * x = x + x^2
    Poly a = {1.0, 1.0};
    Poly x = {0.0, 1.0};  // x^1
    auto r = poly_mul(a, x);
    // coefficients: [0, 1, 1]
    ASSERT_GE(r.size(), 3u);
    EXPECT_NEAR(r[0], 0.0, 1e-10);
    EXPECT_NEAR(r[1], 1.0, 1e-10);
    EXPECT_NEAR(r[2], 1.0, 1e-10);
}

TEST(PolyAdv, Mul_ResultSize) {
    // Degree-2 * degree-3 = degree-5, size = 6
    Poly a = {1.0, 2.0, 3.0};       // degree 2, size 3
    Poly b = {1.0, 1.0, 1.0, 1.0}; // degree 3, size 4
    auto r = poly_mul(a, b);
    EXPECT_EQ(r.size(), 6u);  // degree 5, size 6
}

TEST(PolyAdv, Mul_TwoLinears) {
    // (1 + x) * (2 + 3x) = 2 + 5x + 3x^2
    Poly a = {1.0, 1.0};
    Poly b = {2.0, 3.0};
    auto r = poly_mul(a, b);
    ASSERT_GE(r.size(), 3u);
    EXPECT_NEAR(r[0], 2.0, 1e-10);
    EXPECT_NEAR(r[1], 5.0, 1e-10);
    EXPECT_NEAR(r[2], 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// poly_eval
// ---------------------------------------------------------------------------

TEST(PolyAdv, Eval_ConstantPolynomial) {
    Poly p = {7.0};
    auto r = poly_eval(p, 3.0);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 7.0, 1e-10);
}

TEST(PolyAdv, Eval_LinearAt2) {
    // 3 + 2x at x=2 = 7
    Poly p = {3.0, 2.0};
    auto r = poly_eval(p, 2.0);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 7.0, 1e-10);
}

TEST(PolyAdv, Eval_QuadraticAtZero) {
    // 5 + 3x + x^2 at x=0 = 5
    Poly p = {5.0, 3.0, 1.0};
    auto r = poly_eval(p, 0.0);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 5.0, 1e-10);
}

TEST(PolyAdv, Eval_HighDegreeFinite) {
    // High-degree polynomial at x=0.5 should give finite result
    Poly p(20, 1.0);  // All coefficients 1, degree 19
    auto r = poly_eval(p, 0.5);
    ASSERT_FALSE(r.empty());
    EXPECT_TRUE(std::isfinite(r[0]));
}

TEST(PolyAdv, Eval_NegativeX) {
    // 1 - x + x^2 at x=-1 = 1+1+1 = 3
    Poly p = {1.0, -1.0, 1.0};
    auto r = poly_eval(p, -1.0);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 3.0, 1e-10);
}

// ---------------------------------------------------------------------------
// poly_deriv
// ---------------------------------------------------------------------------

TEST(PolyAdv, Deriv_Constant_IsZero) {
    Poly p = {5.0};
    auto r = poly_deriv(p);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 0.0, 1e-10);
}

TEST(PolyAdv, Deriv_Linear_IsConstant) {
    // d/dx(3 + 2x) = 2
    Poly p = {3.0, 2.0};
    auto r = poly_deriv(p);
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(r[0], 2.0, 1e-10);
}

TEST(PolyAdv, Deriv_Quadratic) {
    // d/dx(1 + 2x + 3x^2) = 2 + 6x
    Poly p = {1.0, 2.0, 3.0};
    auto r = poly_deriv(p);
    ASSERT_GE(r.size(), 2u);
    EXPECT_NEAR(r[0], 2.0, 1e-10);
    EXPECT_NEAR(r[1], 6.0, 1e-10);
}

TEST(PolyAdv, Deriv_ReducesDegree) {
    // Degree-n poly derivative has degree n-1 (size n)
    Poly p = {1.0, 2.0, 3.0, 4.0};  // degree 3, size 4
    auto r = poly_deriv(p);
    EXPECT_LT(r.size(), p.size());
}

TEST(PolyAdv, Deriv_PowerRule) {
    // d/dx(x^4) = 4x^3, coeffs: [0, 0, 0, 0, 1] -> [0, 0, 0, 4]
    Poly p = {0.0, 0.0, 0.0, 0.0, 1.0};  // x^4
    auto r = poly_deriv(p);
    ASSERT_GE(r.size(), 4u);
    EXPECT_NEAR(r[0], 0.0, 1e-10);
    EXPECT_NEAR(r[1], 0.0, 1e-10);
    EXPECT_NEAR(r[2], 0.0, 1e-10);
    EXPECT_NEAR(r[3], 4.0, 1e-10);
}

TEST(PolyAdv, Deriv_ConsistentWithEval) {
    // Numerical derivative check: p'(x) ≈ (p(x+h) - p(x-h)) / (2h)
    Poly p = {1.0, 0.0, 1.0};  // 1 + x^2, deriv = 2x
    auto d = poly_deriv(p);
    double x = 2.0;
    auto actual = poly_eval(d, x);
    ASSERT_FALSE(actual.empty());
    EXPECT_NEAR(actual[0], 2.0 * x, 1e-8);  // 2x at x=2 should be 4
}
