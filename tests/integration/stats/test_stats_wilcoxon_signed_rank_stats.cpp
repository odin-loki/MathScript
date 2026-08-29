
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

TEST(IntegrationStats,  WilcoxonFriedmanJarqueLjungTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_wilcoxon_signed_rank");
    expect_contains(interp, "help", "stats_friedman");
    expect_contains(interp, "help", "stats_jarque_bera");
    expect_contains(interp, "help", "stats_ljung_box");

    expect_ok(interp, "wx = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "wy = [3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "ws = stats_wilcoxon_signed_rank(wx, wy)");
    ASSERT_GT(interp.state().matrices.count("ws"), 0u);

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(IntegrationStats,  ProbRayleighCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}
