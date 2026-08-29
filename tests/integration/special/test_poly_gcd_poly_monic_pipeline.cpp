
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

TEST(IntegrationSpecial,  PolyGcdMonicTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_gcd");
    expect_contains(interp, "help", "poly_monic");

    expect_ok(interp, "a = [6; -5; 1]");
    expect_ok(interp, "b = [10; -7; 1]");
    expect_ok(interp, "g = poly_gcd(a, b)");
    EXPECT_NEAR(interp.state().matrices.at("g")(1, 0), 1.0, 1e-5);

    expect_ok(interp, "m = poly_monic([6; -5; 2])");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
}

TEST(IntegrationSpecial,  StruveHScalar) {
    Interpreter interp;
    expect_ok(interp, "sh = struve_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), ms::struve_h(0, 1), 1e-8);
}
