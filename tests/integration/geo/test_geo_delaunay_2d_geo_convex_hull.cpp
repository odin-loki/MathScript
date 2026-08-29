
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

TEST(IntegrationGeo,  DelaunayHullTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_delaunay_2d");
    expect_contains(interp, "help", "geo_convex_hull");

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
}

TEST(IntegrationGeo,  ProbChi2CdfScalar) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}
