
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

TEST(IntegrationStats,  FlignerWilcoxonTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_fligner(G)");
    expect_contains(interp, "help", "stats_wilcoxon_signed_rank(x,y)");

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);
}

TEST(IntegrationStats,  ProbRayleighCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}
