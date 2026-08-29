
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

TEST(IntegrationSpecial,  Hull3dLinregTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_convex_hull_3d");
    expect_contains(interp, "help", "stats_linear_regression");

    expect_ok(interp, "H = geo_convex_hull_3d([1,1,1; 1,-1,-1; -1,1,-1; -1,-1,1])");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "lr = stats_linear_regression(x, y)");
    EXPECT_NEAR(interp.state().matrices.at("lr")(0, 0), 2.0, 1e-9);
}

TEST(IntegrationSpecial,  JacobiSnScalar) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sn(0, 0.5), 1e-8);
}
