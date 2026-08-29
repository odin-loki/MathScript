
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

TEST(IntegrationControl,  C2dParallelTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "control_c2d");
    expect_contains(interp, "help", "control_parallel");

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);
}

TEST(IntegrationControl,  Hypergeo1f1Scalar) {
    Interpreter interp;
    expect_ok(interp, "h = hypergeo_1f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::hypergeo_1f1(1, 0.5), 1e-8);
}
