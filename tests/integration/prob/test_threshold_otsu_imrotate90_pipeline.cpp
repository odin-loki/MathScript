
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

TEST(IntegrationProb,  OtsuRotateTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "threshold_otsu");
    expect_contains(interp, "help", "imrotate90");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "T = threshold_otsu(G)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "R = imrotate90(M)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
}

TEST(IntegrationProb,  ProbTCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}
