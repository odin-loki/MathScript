
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

TEST(IntegrationGeo,  SchurBezierTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "schur");
    expect_contains(interp, "help", "geo_bezier_eval");

    expect_ok(interp, "T = schur([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "pt = geo_bezier_eval(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 1.0, 1e-9);
    expect_ok(interp, "d = geo_bezier_deriv(ctrl, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("d")(0, 0), 2.0, 1e-9);
}

TEST(IntegrationGeo,  Theta3Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta3(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta3(0, 0.1), 1e-8);
}
