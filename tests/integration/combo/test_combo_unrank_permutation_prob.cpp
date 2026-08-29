
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

TEST(IntegrationCombo,  ComboUnrankPermutation) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_unrank_permutation(n,rank)");

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    const auto& p = interp.state().matrices.at("p");
    EXPECT_EQ(p.rows(), 3u);
    EXPECT_NEAR(p(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(p(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(p(2, 0), 0.0, 1e-9);
}

TEST(IntegrationCombo,  ProbExpPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_pdf(1, 1), 1e-8);
}
