// MathScript Integration Tests: REPL Interpreter – Wave 262 Pipeline
//
// poly_resultant / poly_discriminant REPL scalar bindings on coefficient columns.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave262Pipeline, PolyResultantSharedRoot) {
    Interpreter interp;

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "q = [10; -7; 1]");
    expect_ok(interp, "r = poly_resultant(p, q)");
    ASSERT_GT(interp.state().scalars.count("r"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("r"), 0.0, 1e-6);
    expect_contains(interp, "help", "poly_resultant(p,q)");
}

TEST(ReplWave262Pipeline, PolyDiscriminantRepeatedRoot) {
    Interpreter interp;

    expect_ok(interp, "sq = [1; -2; 1]");
    expect_ok(interp, "d = poly_discriminant(sq)");
    ASSERT_GT(interp.state().scalars.count("d"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("d"), 0.0, 1e-6);
    expect_contains(interp, "help", "poly_discriminant(p)");
}
