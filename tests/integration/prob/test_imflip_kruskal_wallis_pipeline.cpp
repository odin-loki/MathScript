
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

TEST(IntegrationProb,  ImflipKruskalWallisTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "imflip(M,horizontal)");
    expect_contains(interp, "help", "kruskal_wallis");

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);

    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    ASSERT_GT(interp.state().matrices.count("kw"), 0u);
}

TEST(IntegrationProb,  ProbPoisCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 1), 1e-8);
}
