
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"

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

TEST(IntegrationLinalg,  RepmatLinspaceRgb2grayRgb2hsvTail21) {
    Interpreter interp;
    expect_contains(interp, "help", "repmat");
    expect_contains(interp, "help", "linspace");
    expect_contains(interp, "help", "rgb2gray");
    expect_contains(interp, "help", "rgb2hsv");

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    const auto& r = interp.state().matrices.at("R");
    EXPECT_EQ(r.rows(), 4u);
    EXPECT_EQ(r.cols(), 4u);
    EXPECT_NEAR(r(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(r(1, 1), 4.0, 1e-8);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    const auto& v = interp.state().matrices.at("V");
    EXPECT_EQ(v.rows(), 5u);
    EXPECT_EQ(v.cols(), 1u);
    EXPECT_NEAR(v(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(v(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    const auto& g = interp.state().matrices.at("G");
    EXPECT_EQ(g.rows(), 2u);
    EXPECT_EQ(g.cols(), 1u);
    EXPECT_NEAR(g(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(g(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1,0,0; 0,1,0; 0,0,1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    const auto& hsv = interp.state().matrices.at("HSV");
    EXPECT_EQ(hsv.rows(), 3u);
    EXPECT_EQ(hsv.cols(), 3u);
    EXPECT_NEAR(hsv(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(hsv(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(IntegrationLinalg,  ProbExpCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "ec = prob_exp_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ec"), ms::exp_cdf(1, 1), 1e-8);
}
