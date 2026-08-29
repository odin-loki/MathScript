
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

TEST(IntegrationStats,  LabelAnovaTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "label_components(B)");
    expect_contains(interp, "help", "stats_one_way_anova(G)");

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").cols(), 4u);
}

TEST(IntegrationStats,  ProbChi2PdfScalar) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}
