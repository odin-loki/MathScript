
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

TEST(IntegrationStats,  KruskalShapiroTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "kruskal_wallis");
    expect_contains(interp, "help", "stats_shapiro_wilk");

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);
}

TEST(IntegrationStats,  ProbPoisCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}
