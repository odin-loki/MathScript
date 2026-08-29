
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationPoly,  ComboPoly) {
    Interpreter interp;

    expect_contains(interp, "help", "combo_prev_perm");
    expect_contains(interp, "help", "poly_partial_fractions");
    expect_contains(interp, "help", "poly_cheb_expand");

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    EXPECT_EQ(interp.state().matrices.at("pp").rows(), 3u);

    expect_ok(interp, "num = [1; 0]");
    expect_ok(interp, "den = [1; -1]");
    expect_ok(interp, "pf = poly_partial_fractions(num, den)");
    EXPECT_GT(interp.state().matrices.at("pf").rows(), 0u);

    expect_ok(interp, "p = [0; 0; 1]");
    expect_ok(interp, "cheb = poly_cheb_expand(p, 3)");
    EXPECT_GT(interp.state().matrices.at("cheb").rows(), 0u);
}

TEST(IntegrationPoly,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "pg = special_polygamma(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pg"), ms::polygamma(1, 1.0), 1e-9);
}
