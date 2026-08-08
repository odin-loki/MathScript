// MathScript Integration Tests: REPL Interpreter – Wave 291 Pipeline
//
// Wave 291 REPL smoke: image/stats tail11 extensions, legendre_pn/assoc_legendre_p scalar.

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

TEST(ReplWave291Pipeline, ImageStats) {
    Interpreter interp;

    expect_contains(interp, "help", "harris");
    expect_contains(interp, "help", "stats_kde");
    expect_contains(interp, "help", "gray2rgb");

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    ASSERT_GT(interp.state().matrices.at("H").rows(), 0u);

    expect_ok(interp, "k = stats_kde([0; 1; 2], [-1; 0; 1; 2], 1)");
    EXPECT_GT(interp.state().matrices.at("k").rows(), 0u);

    expect_ok(interp, "G = [10, 20, 30, 40; 50, 60, 70, 80; 90, 100, 110, 120]");
    expect_ok(interp, "RGB = gray2rgb(G)");
    EXPECT_EQ(interp.state().matrices.at("RGB").cols(), 3u);
}

TEST(ReplWave291Pipeline, SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "lp = legendre_pn(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lp"), ms::legendre_pn(2, 1, 0.5), 1e-9);

    expect_ok(interp, "ap = assoc_legendre_p(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ap"), ms::assoc_legendre_p(2, 1, 0.5), 1e-9);
}
