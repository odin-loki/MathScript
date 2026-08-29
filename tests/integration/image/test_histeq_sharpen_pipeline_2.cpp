
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

TEST(IntegrationImage,  HisteqSharpenTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "histeq");
    expect_contains(interp, "help", "sharpen");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(IntegrationImage,  ProbExpPpfScalar) {
    Interpreter interp;
    expect_ok(interp, "eq = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eq"), ms::exp_ppf(0.5, 1), 1e-8);
}
