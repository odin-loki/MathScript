
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

TEST(IntegrationSpecial,  SobelXYTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "sobel_x");
    expect_contains(interp, "help", "sobel_y");

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "Sx = sobel_x(G)");
    expect_ok(interp, "Sy = sobel_y(G)");
    EXPECT_EQ(interp.state().matrices.at("Sx").rows(), 2u);
}

TEST(IntegrationSpecial,  ChebyshevWScalar) {
    Interpreter interp;
    expect_ok(interp, "cw = chebyshev_w(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cw"), ms::chebyshev_w(2, 0.5), 1e-8);
}
