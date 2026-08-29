
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

TEST(IntegrationProb,  BinaryLabelTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "threshold_binary");
    expect_contains(interp, "help", "label_components");

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "T = threshold_binary(M, 0.5)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("T")(0, 0), 0.0);

    expect_ok(interp, "B = [0, 1, 1, 0; 0, 1, 1, 0; 0, 0, 0, 0; 1, 1, 0, 0]");
    expect_ok(interp, "L = label_components(B)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 4u);
}

TEST(IntegrationProb,  ProbChi2PdfScalar) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}
