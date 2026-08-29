
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"
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

TEST(IntegrationGeo,  ConvexHullOtsuTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_convex_hull(P)");

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
}

TEST(IntegrationGeo,  ProbChi2CdfScalar) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}
