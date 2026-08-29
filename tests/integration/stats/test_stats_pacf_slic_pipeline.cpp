
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

TEST(IntegrationStats,  PacfSlicTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_pacf(x,max_lag)");
    expect_contains(interp, "help", "slic(M,K");

    expect_ok(interp, "p = stats_pacf([1; 2; 3; 4; 5], 2)");
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 3u);

    expect_ok(interp, "RGB = [0, 0, 0, 1; 0, 0, 0, 1; 1, 1, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "S = slic(RGB, 4)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 4u);
}

TEST(IntegrationStats,  JacobiNsScalar) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}
