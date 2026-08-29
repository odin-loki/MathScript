
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

TEST(IntegrationStats,  LeveneBartlettTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_levene(G)");
    expect_contains(interp, "help", "stats_bartlett(G)");

    expect_ok(interp, "V = [1, 2, 3, 4; 10, 20, 30, 40; 100, 200, 300, 400]");
    expect_ok(interp, "lv = stats_levene(V)");
    expect_ok(interp, "bt = stats_bartlett(V)");
    ASSERT_GT(interp.state().matrices.count("lv"), 0u);
    ASSERT_GT(interp.state().matrices.count("bt"), 0u);
}

TEST(IntegrationStats,  ProbRayleighPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}
