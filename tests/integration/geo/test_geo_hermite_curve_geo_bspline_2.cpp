
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

TEST(IntegrationGeo,  HermiteBsplineTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_hermite_curve");
    expect_contains(interp, "help", "geo_bspline_eval");

    expect_ok(interp, "pt = geo_hermite_curve(0, 0, 0, 1, 1, 0, 0, -1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("pt")(0, 0), 0.5, 1e-9);

    expect_ok(interp, "ctrl = [0, 0; 2, 0]");
    expect_ok(interp, "knots = [0, 0, 1, 1]");
    expect_ok(interp, "bs = geo_bspline_eval(ctrl, knots, 1, 0.5)");
    EXPECT_NEAR(interp.state().matrices.at("bs")(0, 0), 1.0, 1e-9);
}

TEST(IntegrationGeo,  Theta4Scalar) {
    Interpreter interp;
    expect_ok(interp, "th = theta4(0, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th"), ms::theta4(0, 0.1), 1e-8);
}
