// MathScript Integration Tests: REPL Interpreter – Wave 297 Pipeline
//
// Wave 297 REPL smoke: numthy/poly tail11 extensions, special_pochhammer scalar.

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

TEST(ReplWave297Pipeline, NumthyPoly) {
    Interpreter interp;

    expect_contains(interp, "help", "numthy_factor_exp");
    expect_contains(interp, "help", "poly_lcm");

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_GT(interp.state().matrices.at("fe").rows(), 0u);

    expect_ok(interp, "f = numthy_farey(5)");
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    EXPECT_GT(interp.state().matrices.at("l").rows(), 0u);
}

TEST(ReplWave297Pipeline, SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "ph = special_pochhammer(2.5, 3)");
    EXPECT_NEAR(interp.state().scalars.at("ph"), ms::pochhammer(2.5, 3), 1e-9);
}
