// MathScript: Advanced Polynomial Tests (Wave 48)
// Tests for poly_eval, poly_add, poly_sub, poly_mul, poly_deriv

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/poly/poly.hpp"

using namespace ms;

// Helper: evaluate polynomial using Horner's method for reference
static double horner(const std::vector<double>& coeffs, double x) {
    // coeffs[0] is constant term, coeffs[n] is degree-n coefficient
    if (coeffs.empty()) return 0.0;
    double result = coeffs.back();
    for (int i = static_cast<int>(coeffs.size()) - 2; i >= 0; --i)
        result = result * x + coeffs[i];
    return result;
}

// Check if two coefficient vectors are near-equal
static void expect_poly_near(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-10) {
    size_t n = std::max(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double ai = i < a.size() ? a[i] : 0.0;
        double bi = i < b.size() ? b[i] : 0.0;
        EXPECT_NEAR(ai, bi, tol) << "Polynomial coefficient mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// poly_eval
// ---------------------------------------------------------------------------

TEST(PolyAdv2, Eval_Constant) {
    std::vector<double> c = {7.0};
    auto result = poly_eval(c, 5.0);
    ASSERT_FALSE(result.empty());
    EXPECT_NEAR(result[0], 7.0, 1e-10);
}

TEST(PolyAdv2, Eval_Linear_At0) {
    // 2 + 3x at x=0 => 2
    std::vector<double> c = {2.0, 3.0};
    auto result = poly_eval(c, 0.0);
    ASSERT_FALSE(result.empty());
    EXPECT_NEAR(result[0], 2.0, 1e-10);
}

TEST(PolyAdv2, Eval_Quadratic) {
    // x^2 - x + 1 at x=2 => 4-2+1=3
    std::vector<double> c = {1.0, -1.0, 1.0};
    auto result = poly_eval(c, 2.0);
    ASSERT_FALSE(result.empty());
    EXPECT_NEAR(result[0], 3.0, 1e-10);
}

TEST(PolyAdv2, Eval_MatchesHorner) {
    std::vector<double> c = {3.0, -1.0, 2.0, 0.5};
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0}) {
        auto result = poly_eval(c, x);
        ASSERT_FALSE(result.empty());
        EXPECT_NEAR(result[0], horner(c, x), 1e-9) << "Mismatch at x=" << x;
    }
}

TEST(PolyAdv2, Eval_ZeroCoeffs_IsZero) {
    std::vector<double> c = {0.0, 0.0, 0.0};
    auto result = poly_eval(c, 5.0);
    ASSERT_FALSE(result.empty());
    EXPECT_NEAR(result[0], 0.0, 1e-10);
}

TEST(PolyAdv2, Eval_HighDegree_Finite) {
    // x^5 + 2x^4 + ... at x=1.5
    std::vector<double> c = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto result = poly_eval(c, 1.5);
    ASSERT_FALSE(result.empty());
    EXPECT_TRUE(std::isfinite(result[0]));
}

// ---------------------------------------------------------------------------
// poly_add
// ---------------------------------------------------------------------------

TEST(PolyAdv2, Add_SameSize) {
    // (1 + 2x) + (3 + 4x) = 4 + 6x
    auto r = poly_add({1.0, 2.0}, {3.0, 4.0});
    expect_poly_near(r, {4.0, 6.0});
}

TEST(PolyAdv2, Add_DifferentSizes) {
    // (1 + 2x + 3x^2) + (5 + 6x) = 6 + 8x + 3x^2
    auto r = poly_add({1.0, 2.0, 3.0}, {5.0, 6.0});
    expect_poly_near(r, {6.0, 8.0, 3.0});
}

TEST(PolyAdv2, Add_Zero_Unchanged) {
    auto r = poly_add({3.0, 1.0, 2.0}, {0.0, 0.0, 0.0});
    expect_poly_near(r, {3.0, 1.0, 2.0});
}

TEST(PolyAdv2, Add_Commutative_Result) {
    // poly_add(a,b) and poly_add(b,a) should have same values at any x
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0};
    auto ab = poly_add(a, b);
    auto ba = poly_add(b, a);
    for (double x : {0.0, 1.0, -1.0, 2.0}) {
        double vab = horner(ab, x);
        double vba = horner(ba, x);
        EXPECT_NEAR(vab, vba, 1e-9) << "poly_add not commutative at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// poly_sub
// ---------------------------------------------------------------------------

TEST(PolyAdv2, Sub_SameSize) {
    // (5 + 3x) - (2 + x) = 3 + 2x
    auto r = poly_sub({5.0, 3.0}, {2.0, 1.0});
    expect_poly_near(r, {3.0, 2.0});
}

TEST(PolyAdv2, Sub_Self_IsZero) {
    std::vector<double> c = {1.0, 2.0, 3.0};
    auto r = poly_sub(c, c);
    for (auto v : r) EXPECT_NEAR(v, 0.0, 1e-10);
}

TEST(PolyAdv2, Sub_Inverse_Of_Add) {
    // (a + b) - b = a (check at evaluation points)
    std::vector<double> a = {2.0, -1.0, 3.0};
    std::vector<double> b = {1.0, 4.0, -2.0};
    auto sum = poly_add(a, b);
    auto back = poly_sub(sum, b);
    for (double x : {0.0, 1.0, -1.0, 2.0}) {
        double va = horner(a, x);
        double vb = horner(back, x);
        EXPECT_NEAR(vb, va, 1e-9) << "Sub-inverse failed at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// poly_mul
// ---------------------------------------------------------------------------

TEST(PolyAdv2, Mul_Constant_Scales) {
    // 3 * (2 + x) = 6 + 3x
    auto r = poly_mul({3.0}, {2.0, 1.0});
    expect_poly_near(r, {6.0, 3.0});
}

TEST(PolyAdv2, Mul_Linear_Times_Linear) {
    // (1 + x) * (1 - x) = 1 - x^2
    auto r = poly_mul({1.0, 1.0}, {1.0, -1.0});
    // Evaluate at x=2: (1+2)*(1-2) = 3*(-1) = -3; polynomial: 1-x^2 = 1-4 = -3
    double val = horner(r, 2.0);
    EXPECT_NEAR(val, -3.0, 1e-10);
    // Evaluate at x=0: 1
    EXPECT_NEAR(horner(r, 0.0), 1.0, 1e-10);
}

TEST(PolyAdv2, Mul_Commutative) {
    std::vector<double> a = {2.0, 1.0};
    std::vector<double> b = {3.0, -1.0, 2.0};
    auto ab = poly_mul(a, b);
    auto ba = poly_mul(b, a);
    for (double x : {0.0, 1.0, -1.0, 2.0}) {
        EXPECT_NEAR(horner(ab, x), horner(ba, x), 1e-9) << "mul not commutative at x=" << x;
    }
}

TEST(PolyAdv2, Mul_DegreeIsSum) {
    // Degree n1 * degree n2 = degree n1+n2
    std::vector<double> a = {1.0, 2.0, 3.0};   // degree 2
    std::vector<double> b = {1.0, 1.0, 1.0, 1.0};  // degree 3
    auto r = poly_mul(a, b);
    EXPECT_EQ(r.size(), 6u);  // degree 5 => 6 coefficients
}

TEST(PolyAdv2, Mul_MatchesAnalytic) {
    // (1 + x) * (1 + x) = 1 + 2x + x^2
    auto r = poly_mul({1.0, 1.0}, {1.0, 1.0});
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        EXPECT_NEAR(horner(r, x), (1.0 + x) * (1.0 + x), 1e-10) << "at x=" << x;
    }
}

// ---------------------------------------------------------------------------
// poly_deriv
// ---------------------------------------------------------------------------

TEST(PolyAdv2, Deriv_Constant_IsZero) {
    auto r = poly_deriv({5.0});
    // Derivative of constant should be zero or empty
    double val = r.empty() ? 0.0 : horner(r, 1.0);
    EXPECT_NEAR(val, 0.0, 1e-10);
}

TEST(PolyAdv2, Deriv_Linear_IsConstant) {
    // d/dx(3 + 2x) = 2
    auto r = poly_deriv({3.0, 2.0});
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(horner(r, 0.0), 2.0, 1e-10);
    EXPECT_NEAR(horner(r, 5.0), 2.0, 1e-10);
}

TEST(PolyAdv2, Deriv_Quadratic) {
    // d/dx(1 + 2x + 3x^2) = 2 + 6x
    auto r = poly_deriv({1.0, 2.0, 3.0});
    ASSERT_FALSE(r.empty());
    EXPECT_NEAR(horner(r, 0.0), 2.0, 1e-10);  // 2+6*0=2
    EXPECT_NEAR(horner(r, 1.0), 8.0, 1e-10);  // 2+6*1=8
    EXPECT_NEAR(horner(r, 2.0), 14.0, 1e-10); // 2+6*2=14
}

TEST(PolyAdv2, Deriv_Matches_FiniteDiff) {
    // d/dx(1 + x + x^2 + x^3) at x=1.5
    std::vector<double> c = {1.0, 1.0, 1.0, 1.0};
    auto dc = poly_deriv(c);
    double x = 1.5, h = 1e-6;
    double fd = (horner(c, x + h) - horner(c, x - h)) / (2.0 * h);
    EXPECT_NEAR(horner(dc, x), fd, 1e-5);
}

TEST(PolyAdv2, Deriv_Cubic_DegreeReduces) {
    // d/dx(degree-3 poly) = degree-2 poly (4 - 1 coefficients)
    auto r = poly_deriv({1.0, 2.0, 3.0, 4.0});
    // 3 or 4 coefficients (implementation may not strip trailing zero)
    EXPECT_LE(r.size(), 4u);
    // Value at x=0 should be coeffs[1] = 2
    EXPECT_NEAR(horner(r, 0.0), 2.0, 1e-10);
}
