
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

TEST(IntegrationStats,  AnovaVarianceTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_one_way_anova");
    expect_contains(interp, "help", "stats_levene");

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    expect_ok(interp, "fk = stats_fligner(V)");
    ASSERT_GT(interp.state().matrices.count("fk"), 0u);
}

TEST(IntegrationStats,  ProbRayleighPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}
