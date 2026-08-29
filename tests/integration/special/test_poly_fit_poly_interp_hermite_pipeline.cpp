
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

TEST(IntegrationSpecial,  PolyFitHermiteTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_fit");
    expect_contains(interp, "help", "poly_interp_hermite");

    expect_ok(interp, "xs = [0; 1; 2; 3]");
    expect_ok(interp, "ys = [1; 3; 5; 7]");
    expect_ok(interp, "c = poly_fit(xs, ys, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c")(0, 0), 1.0, 1e-6);

    expect_ok(interp, "hxs = [2]");
    expect_ok(interp, "hys = [5]");
    expect_ok(interp, "hdys = [3]");
    expect_ok(interp, "ph = poly_interp_hermite(hxs, hys, hdys)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
}

TEST(IntegrationSpecial,  BesselLScalar) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(0, 1), 1e-8);
}
