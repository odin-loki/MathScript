
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

TEST(IntegrationLinalg,  SolveIter) {
    Interpreter interp;
    expect_contains(interp, "help", "solve");
    expect_contains(interp, "help", "bicgstab");
    expect_contains(interp, "help", "qmr");

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(IntegrationLinalg,  GeoVec2dLengthScalar) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}
