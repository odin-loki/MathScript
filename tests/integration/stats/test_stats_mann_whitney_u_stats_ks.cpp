
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

TEST(IntegrationStats,  MannWhitneyKsTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_mann_whitney_u");
    expect_contains(interp, "help", "stats_ks_2sample");

    expect_ok(interp, "a = [1; 2; 3; 4; 5]");
    expect_ok(interp, "b = [11; 12; 13; 14; 15]");
    expect_ok(interp, "mw = stats_mann_whitney_u(a, b)");
    EXPECT_EQ(interp.state().matrices.at("mw").cols(), 3u);

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);
}

TEST(IntegrationStats,  ProbExpPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}
