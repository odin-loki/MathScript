
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

TEST(IntegrationPoly,  PolyLcmDivTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_lcm");
    expect_contains(interp, "help", "poly_div_quot");

    expect_ok(interp, "l = poly_lcm([-1; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("l"), 0u);
    expect_ok(interp, "q = poly_div_quot([1; 0; 1], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("q"), 0u);
}

TEST(IntegrationPoly,  WeberEScalar) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}
