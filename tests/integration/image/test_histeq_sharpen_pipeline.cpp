
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

TEST(IntegrationImage,  HisteqSharpenTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "histeq(M)");
    expect_contains(interp, "help", "sharpen(M)");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "H = histeq(G)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    expect_ok(interp, "S = sharpen(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(IntegrationImage,  ProbTPpfScalar) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}
