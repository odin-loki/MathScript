
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

TEST(IntegrationLinalg,  TriuHessTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "triu");
    expect_contains(interp, "help", "hess");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(IntegrationLinalg,  Theta2Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta2(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta2(0.2, 0.1), 1e-8);
}
