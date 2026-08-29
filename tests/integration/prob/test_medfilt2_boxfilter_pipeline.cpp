
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

TEST(IntegrationProb,  MedfiltBoxfilterTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "medfilt2");
    expect_contains(interp, "help", "boxfilter");

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
    expect_ok(interp, "B = boxfilter(M, 3)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
}

TEST(IntegrationProb,  ProbExpPpfScalar) {
    Interpreter interp;
    expect_ok(interp, "eq = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eq"), ms::exp_ppf(0.5, 1), 1e-8);
}
