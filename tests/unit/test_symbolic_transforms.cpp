#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <numbers>
#include <string>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

namespace {

SymExpr clone_expr(const SymExpr& expr) {
    SymExpr copy;
    copy.op = expr.op;
    copy.value = expr.value;
    copy.name = expr.name;
    if (expr.left) {
        copy.left = std::make_unique<SymExpr>(clone_expr(*expr.left));
    }
    if (expr.right) {
        copy.right = std::make_unique<SymExpr>(clone_expr(*expr.right));
    }
    return copy;
}

bool is_deriv_sentinel(const SymExpr& original, const SymExpr& result, const std::string& var) {
    const SymExpr expected = sym_deriv(clone_expr(original), var);
    return sym_to_string(result) == sym_to_string(expected);
}

} // namespace

TEST(SymbolicTransformsTest, fourier_exponential_decay) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_var("t"))));
    const SymExpr spectrum = sym_simplify(sym_fourier(time, "t", "omega"));

    EXPECT_NE(sym_to_string(spectrum).find("omega"), std::string::npos);
    const double value = sym_eval(spectrum, {{"omega", 1.0}});
    EXPECT_NEAR(value, 4.0 / (4.0 + 1.0), 1e-9);
}

TEST(SymbolicTransformsTest, ifourier_exponential_decay) {
    const SymExpr spectrum = sym_div(
        sym_const(4.0),
        sym_add(sym_pow(sym_const(2.0), sym_const(2.0)), sym_pow(sym_var("omega"), sym_const(2.0))));
    const SymExpr time = sym_simplify(sym_ifourier(spectrum, "omega", "t"));

    EXPECT_NE(sym_to_string(time).find("exp"), std::string::npos);
    EXPECT_NEAR(sym_eval(time, {{"t", 0.5}}), std::exp(-1.0), 1e-9);
}

TEST(SymbolicTransformsTest, fourier_ifourier_roundtrip_decay) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(3.0), sym_var("t"))));
    const SymExpr roundtrip = sym_simplify(sym_ifourier(sym_fourier(time, "t", "omega"), "omega", "t"));
    EXPECT_EQ(sym_to_string(roundtrip), sym_to_string(time));
}

TEST(SymbolicTransformsTest, fourier_gaussian_pair) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(1.5), sym_pow(sym_var("t"), sym_const(2.0)))));
    const SymExpr spectrum = sym_simplify(sym_fourier(time, "t", "omega"));
    const double expected_scale = std::sqrt(std::numbers::pi / 1.5);
    const double expected = expected_scale * std::exp(-1.0 / (4.0 * 1.5));

    EXPECT_NEAR(sym_eval(spectrum, {{"omega", 1.0}}), expected, 1e-9);
}

TEST(SymbolicTransformsTest, ifourier_gaussian_roundtrip) {
    const SymExpr time = sym_exp(sym_neg(sym_mul(sym_const(2.0), sym_pow(sym_var("t"), sym_const(2.0)))));
    const SymExpr roundtrip = sym_simplify(sym_ifourier(sym_fourier(time, "t", "omega"), "omega", "t"));
    EXPECT_EQ(sym_to_string(roundtrip), sym_to_string(time));
}

TEST(SymbolicTransformsTest, fourier_unsupported_returns_deriv_sentinel) {
    const SymExpr expr = sym_sin(sym_var("t"));
    const SymExpr result = sym_fourier(expr, "t", "omega");
    EXPECT_TRUE(is_deriv_sentinel(expr, result, "t"));
}

TEST(SymbolicTransformsTest, ztransform_geometric_sequence) {
    const SymExpr seq = sym_pow(sym_const(0.5), sym_var("n"));
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));

    EXPECT_NE(sym_to_string(zdomain).find("z"), std::string::npos);
    const double value = sym_eval(zdomain, {{"z", 2.0}});
    EXPECT_NEAR(value, 2.0 / (2.0 - 0.5), 1e-9);
}

TEST(SymbolicTransformsTest, ztransform_scaled_geometric_sequence) {
    const SymExpr seq = sym_mul(sym_const(3.0), sym_pow(sym_const(0.25), sym_var("n")));
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));
    const double value = sym_eval(zdomain, {{"z", 1.0}});
    EXPECT_NEAR(value, 3.0 * 1.0 / (1.0 - 0.25), 1e-9);
}

TEST(SymbolicTransformsTest, iztransform_geometric_sequence) {
    const SymExpr zdomain = sym_div(sym_var("z"), sym_sub(sym_var("z"), sym_const(0.5)));
    const SymExpr seq = sym_simplify(sym_iztransform(zdomain, "z", "n"));

    EXPECT_NE(sym_to_string(seq).find("n"), std::string::npos);
    EXPECT_NEAR(sym_eval(seq, {{"n", 2.0}}), std::pow(0.5, 2.0), 1e-9);
}

TEST(SymbolicTransformsTest, ztransform_iztransform_roundtrip) {
    const SymExpr seq = sym_pow(sym_const(0.75), sym_var("n"));
    const SymExpr roundtrip = sym_simplify(sym_iztransform(sym_ztransform(seq, "n", "z"), "z", "n"));
    EXPECT_EQ(sym_to_string(roundtrip), sym_to_string(seq));
}

TEST(SymbolicTransformsTest, ztransform_constant_sequence) {
    const SymExpr seq = sym_const(4.0);
    const SymExpr zdomain = sym_simplify(sym_ztransform(seq, "n", "z"));
    const double value = sym_eval(zdomain, {{"z", 3.0}});
    EXPECT_NEAR(value, 4.0 * 3.0 / (3.0 - 1.0), 1e-9);
}

TEST(SymbolicTransformsTest, iztransform_constant_sequence) {
    const SymExpr zdomain = sym_div(sym_var("z"), sym_sub(sym_var("z"), sym_const(1.0)));
    const SymExpr seq = sym_simplify(sym_iztransform(zdomain, "z", "n"));
    EXPECT_NEAR(sym_eval(seq, {{"n", 5.0}}), 1.0, 1e-9);
}
