
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

TEST(IntegrationNumthy,  NumthyContinuedFraction) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_continued_fraction(x,n)");

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(IntegrationNumthy,  ProbChi2PdfScalar) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_pdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_pdf(1, 2), 1e-8);
}
