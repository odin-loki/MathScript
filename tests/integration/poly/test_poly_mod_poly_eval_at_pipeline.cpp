
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

TEST(IntegrationPoly,  PolyModEvalAtTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_mod");
    expect_contains(interp, "help", "poly_eval_at");

    expect_ok(interp, "rm = poly_mod([1; 0; 1], [1; 1])");
    EXPECT_NEAR(interp.state().matrices.at("rm")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "vals = poly_eval_at([1; 2; 3], [0; 1; 2])");
    EXPECT_EQ(interp.state().matrices.at("vals").rows(), 3u);
}

TEST(IntegrationPoly,  KelvinBeiScalar) {
    Interpreter interp;
    expect_ok(interp, "kb = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kb"), ms::kelvin_bei(0, 1), 1e-8);
}
