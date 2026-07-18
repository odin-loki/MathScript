// MathScript Integration Tests: REPL Interpreter – Wave 258 Pipeline
//
// Smoke: Wave 257 APIs already on main (stats Friedman/Levene, image
// label_components/hough_lines, prob Weibull CDF). Deterministic; Wave 258
// features belong in separate tests once merged.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave258Pipeline, StatsFriedman) {
    Interpreter interp;

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    ASSERT_GT(interp.state().matrices.count("fr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("fr").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 2), std::exp(-0.75), 1e-9);
    expect_contains(interp, "help", "stats_friedman(data)");
}

TEST(ReplWave258Pipeline, StatsLevene) {
    Interpreter interp;

    expect_ok(interp, "G = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(G)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lv").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("lv").cols(), 4u);
    EXPECT_GT(interp.state().matrices.at("lv")(0, 0), 0.0);
    EXPECT_LT(interp.state().matrices.at("lv")(0, 1), 0.05);
    EXPECT_NEAR(interp.state().matrices.at("lv")(0, 2), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("lv")(0, 3), 9.0, 1e-12);
    expect_contains(interp, "help", "stats_levene(G)");
}

TEST(ReplWave258Pipeline, LabelComponents) {
    Interpreter interp;

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 4u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("L")(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("L")(0, 1), interp.state().matrices.at("L")(1, 2));
    EXPECT_NE(interp.state().matrices.at("L")(0, 1), interp.state().matrices.at("L")(3, 0));
    expect_contains(interp, "help", "label_components(B)");
}

TEST(ReplWave258Pipeline, HoughLinesHorizontal) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "H = hough_lines(E, 0.5, 180, 200, 5)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("H").rows(), 0u);
    EXPECT_NEAR(interp.state().matrices.at("H")(0, 1), 1.5707963267948966, 0.05);
    EXPECT_NEAR(interp.state().matrices.at("H")(0, 0), 5.0, 1.5);
    expect_contains(interp, "help", "hough_lines(M[,edge])");
}

TEST(ReplWave258Pipeline, ProbWeibullCdf) {
    Interpreter interp;

    expect_ok(interp, "wc = prob_weibull_cdf(1, 1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("wc"), 1.0 - std::exp(-1.0), 1e-9);
    expect_contains(interp, "help", "prob_weibull_cdf(x,lambda,k)");
}
