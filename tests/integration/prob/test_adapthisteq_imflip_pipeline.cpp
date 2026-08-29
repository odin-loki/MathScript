
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

TEST(IntegrationProb,  AdapthisteqImflipTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "adapthisteq");
    expect_contains(interp, "help", "imflip");

    expect_ok(interp, "M = [0, 1; 0.25, 0.75]");
    expect_ok(interp, "A = adapthisteq(M)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    expect_ok(interp, "F = imflip(M, 1)");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("F")(0, 0), 1.0);
}

TEST(IntegrationProb,  ProbPoisPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 1), 1e-8);
}
