
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

TEST(IntegrationStats,  ArfitMultiregTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_arfit");
    expect_contains(interp, "help", "stats_multiple_regression");

    expect_ok(interp, "phi = stats_arfit([1; 1.5; 2; 1.8; 2.1; 2.5; 2.3], 2)");
    EXPECT_EQ(interp.state().matrices.at("phi").rows(), 2u);

    expect_ok(interp, "X = [1, 0, 0; 1, 1, 0; 1, 0, 1; 1, 1, 1; 1, 2, 1]");
    expect_ok(interp, "y = [1; 3; 4; 6; 8]");
    expect_ok(interp, "beta = stats_multiple_regression(X, y)");
    EXPECT_NEAR(interp.state().matrices.at("beta")(0, 0), 1.0, 0.01);
}

TEST(IntegrationStats,  EllipEScalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}
