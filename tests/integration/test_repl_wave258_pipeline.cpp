// MathScript Integration Tests: REPL Interpreter – Wave 258 Pipeline
//
// info_permutation_entropy / info_transfer_entropy REPL bindings.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave258Pipeline, InfoPermutationEntropy) {
    Interpreter interp;

    // Monotonic series -> single ordinal pattern -> normalized PE = 0
    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "pe = info_permutation_entropy(x)");
    ASSERT_GT(interp.state().scalars.count("pe"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("pe"), 0.0, 1e-10);

    // Hand-computed: probs {0.5,0.5} -> H=1 bit; normalized = 1/log2(6)
    expect_ok(interp, "y = [1; 3; 2; 5; 4; 6]");
    expect_ok(interp, "pe2 = info_permutation_entropy(y, 3, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pe2"), 1.0 / std::log2(6.0), 1e-10);

    expect_ok(interp, "pe3 = info_permutation_entropy(y, 3)");
    EXPECT_NEAR(interp.state().scalars.at("pe3"), interp.state().scalars.at("pe2"), 1e-12);

    expect_contains(interp, "help", "info_permutation_entropy");
}

TEST(ReplWave258Pipeline, InfoTransferEntropy) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "te = info_transfer_entropy(x, y)");
    ASSERT_GT(interp.state().scalars.count("te"), 0u);
    EXPECT_GE(interp.state().scalars.at("te"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("te")));

    expect_ok(interp, "te2 = info_transfer_entropy(x, y, 4, 1)");
    EXPECT_GE(interp.state().scalars.at("te2"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("te2")));

    expect_contains(interp, "help", "info_transfer_entropy");
}
