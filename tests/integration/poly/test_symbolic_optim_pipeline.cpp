// MathScript Integration Tests: Symbolic + Optimization + Poly Pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <limits>

#include "ms/symbolic/symbolic.hpp"
#include "ms/optim/optim.hpp"
#include "ms/poly/poly.hpp"
#include "ms/prob/prob.hpp"
#include "ms/domain/domain.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

// ---------------------------------------------------------------------------
// Symbolic + Optim: find minimum of f(x) = (x-2)^2 using golden section
// ---------------------------------------------------------------------------

TEST(SymbolicOptimPipeline, GoldenSection_OnQuadratic) {
    // f(x) = (x-2)^2 = x^2 - 4x + 4, minimum at x=2
    // Use golden_section on a plain lambda (not symbolic to avoid copy issues)
    auto func = [](double x) -> double { return (x - 2.0) * (x - 2.0); };
    double result = golden_section(func, 0.0, 4.0, 1e-8);
    EXPECT_NEAR(result, 2.0, 1e-4);
}

TEST(SymbolicOptimPipeline, Symbolic_Quadratic_Eval) {
    // f(3) = 3^2 - 4*3 + 4 = 9 - 12 + 4 = 1
    auto expr = sym_add(
        sym_add(
            sym_pow(sym_var("x"), sym_const(2.0)),
            sym_mul(sym_const(-4.0), sym_var("x"))
        ),
        sym_const(4.0)
    );
    double val = sym_eval(expr, {{"x", 3.0}});
    EXPECT_NEAR(val, 1.0, 1e-8);
}

TEST(SymbolicOptimPipeline, Symbolic_Differentiation_At_Critical_Point) {
    // f(x) = x^2 + (-4)*x + 4, f'(x) = 2x - 4, zero at x=2
    auto expr = sym_add(
        sym_add(
            sym_pow(sym_var("x"), sym_const(2.0)),
            sym_mul(sym_const(-4.0), sym_var("x"))
        ),
        sym_const(4.0)
    );
    auto deriv = sym_diff(std::move(expr), "x");
    double deriv_at_2 = sym_eval(deriv, {{"x", 2.0}});
    EXPECT_NEAR(deriv_at_2, 0.0, 1e-8);
}

TEST(SymbolicOptimPipeline, Symbolic_Sin_Eval) {
    double val = sym_eval(sym_sin(sym_var("x")), {{"x", M_PI / 2.0}});
    EXPECT_NEAR(val, 1.0, 1e-8);
}

TEST(SymbolicOptimPipeline, Symbolic_Cos_Eval) {
    double val = sym_eval(sym_cos(sym_var("x")), {{"x", 0.0}});
    EXPECT_NEAR(val, 1.0, 1e-8);
}

TEST(SymbolicOptimPipeline, Symbolic_Exp_Log_Roundtrip) {
    // exp(log(3)) = 3
    double val = sym_eval(sym_exp(sym_log(sym_var("x"))), {{"x", 3.0}});
    EXPECT_NEAR(val, 3.0, 1e-8);
}

// ---------------------------------------------------------------------------
// Polynomial arithmetic pipeline
// ---------------------------------------------------------------------------

TEST(PolyPipeline, Add_And_Evaluate) {
    // p1 = 1 + 0*x + 2*x^2 (ascending coeffs)
    // p2 = 3 + 1*x + 0*x^2
    // sum = 4 + 1*x + 2*x^2
    std::vector<double> p1 = {1.0, 0.0, 2.0};
    std::vector<double> p2 = {3.0, 1.0, 0.0};
    auto sum = poly_add(p1, p2);
    ASSERT_GE(sum.size(), 3u);
    EXPECT_NEAR(sum[0], 4.0, 1e-10);
    EXPECT_NEAR(sum[1], 1.0, 1e-10);
    EXPECT_NEAR(sum[2], 2.0, 1e-10);
}

TEST(PolyPipeline, Sub_Produces_Correct_Coefficients) {
    std::vector<double> p1 = {5.0, 3.0, 1.0};  // 5 + 3x + x^2
    std::vector<double> p2 = {2.0, 1.0, 1.0};  // 2 + x + x^2
    auto diff = poly_sub(p1, p2);
    ASSERT_GE(diff.size(), 2u);
    // Expected: [3, 2, ...]
    EXPECT_NEAR(diff[0], 3.0, 1e-10);
    EXPECT_NEAR(diff[1], 2.0, 1e-10);
}

TEST(PolyPipeline, AddThenSub_IsIdentity) {
    std::vector<double> p = {1.0, -2.0, 3.0, -4.0};
    std::vector<double> q = {0.5, 1.0, -0.5, 2.0};
    auto sum = poly_add(p, q);
    auto back = poly_sub(sum, q);
    ASSERT_GE(back.size(), p.size());
    for (size_t i = 0; i < p.size(); ++i)
        EXPECT_NEAR(back[i], p[i], 1e-10);
}

// ---------------------------------------------------------------------------
// Domain: combinatorics pipeline
// ---------------------------------------------------------------------------

TEST(DomainPipeline, BinomialCoefficients_Values) {
    EXPECT_EQ(nchoosek(5, 2), 10u);
    EXPECT_EQ(nchoosek(6, 3), 20u);
    EXPECT_EQ(nchoosek(10, 5), 252u);
}

TEST(DomainPipeline, Factorial_Sequence) {
    EXPECT_EQ(factorial(0), 1u);
    EXPECT_EQ(factorial(1), 1u);
    EXPECT_EQ(factorial(5), 120u);
    EXPECT_EQ(factorial(7), 5040u);
}

TEST(DomainPipeline, GCD_Values) {
    EXPECT_EQ(gcd(12u, 8u), 4u);
    EXPECT_EQ(gcd(100u, 75u), 25u);
    EXPECT_EQ(gcd(7u, 13u), 1u);
}

TEST(DomainPipeline, Combinatorics_Consistency) {
    unsigned long n = 6, k = 2;
    EXPECT_EQ(nchoosek(n, k) * factorial(k) * factorial(n - k), factorial(n));
}

// ---------------------------------------------------------------------------
// Probability + Domain integration
// ---------------------------------------------------------------------------

TEST(ProbDomainPipeline, BinomCDF_Matches_Manual_Sum) {
    int n = 10;
    double p = 0.4;
    double manual_cdf = 0.0;
    for (int i = 0; i <= 3; ++i)
        manual_cdf += binom_pdf(i, n, p);
    double cdf_val = binom_cdf(3, n, p);
    EXPECT_NEAR(cdf_val, manual_cdf, 1e-8);
}

TEST(ProbDomainPipeline, NormPPF_Is_InverseOf_CDF) {
    // norm_ppf(norm_cdf(x)) should give back x
    for (double x : {-2.0, -1.0, 0.0, 0.5, 1.0, 1.5}) {
        double cdf_val = norm_cdf(x, 0.0, 1.0);
        double ppf_val = norm_ppf(cdf_val, 0.0, 1.0);
        EXPECT_NEAR(ppf_val, x, 1e-6);
    }
}

TEST(ProbDomainPipeline, GammaDistribution_IntegratesToOne) {
    double alpha = 2.0, beta = 1.0;
    double integral = 0.0;
    const int N = 10000;
    const double dx = 20.0 / N;
    for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * dx;
        integral += gamma_pdf(x, alpha, beta) * dx;
    }
    EXPECT_NEAR(integral, 1.0, 0.01);
}

// ---------------------------------------------------------------------------
// Symbolic simplification
// ---------------------------------------------------------------------------

TEST(SymbolicPipeline, Simplify_DoesNotCrash) {
    EXPECT_NO_THROW({
        auto simplified = sym_simplify(sym_add(sym_var("x"), sym_const(0.0)));
        (void)simplified;
    });
}

TEST(SymbolicPipeline, Eval_Polynomial_Expression) {
    // p(x) = x^2 + 2x + 1, evaluate at x=3 → 16
    auto poly = sym_add(
        sym_add(
            sym_pow(sym_var("x"), sym_const(2.0)),
            sym_mul(sym_const(2.0), sym_var("x"))
        ),
        sym_const(1.0)
    );
    double val = sym_eval(poly, {{"x", 3.0}});
    EXPECT_NEAR(val, 16.0, 1e-8);
}

TEST(SymbolicPipeline, ToString_NotEmpty) {
    std::string s = sym_to_string(sym_add(sym_var("x"), sym_const(1.0)));
    EXPECT_GT(s.size(), 0u);
}
