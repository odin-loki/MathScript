
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

TEST(IntegrationImage,  PrewittScharrTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "prewitt(M)");
    expect_contains(interp, "help", "scharr(M)");

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(IntegrationImage,  Hypergeo1f1Scalar) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}
