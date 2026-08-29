
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

TEST(IntegrationStats,  ShapiroMannWhitneyTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_shapiro_wilk(x)");
    expect_contains(interp, "help", "stats_mann_whitney_u(a,b)");

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "sw = stats_shapiro_wilk(x)");
    EXPECT_EQ(interp.state().matrices.at("sw").cols(), 2u);

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);
}

TEST(IntegrationStats,  ProbExpPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}
