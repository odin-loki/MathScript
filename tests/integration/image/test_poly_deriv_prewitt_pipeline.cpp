
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

TEST(IntegrationImage,  PolyPrewittScharrTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_deriv");
    expect_contains(interp, "help", "prewitt");
    expect_contains(interp, "help", "scharr");

    expect_ok(interp, "pd = poly_deriv([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "E = prewitt(M)");
    EXPECT_GT(interp.state().matrices.at("E")(2, 2), 0.0);
    expect_ok(interp, "sc = scharr(M)");
    EXPECT_GT(interp.state().matrices.at("sc")(2, 2), 0.0);
}

TEST(IntegrationImage,  ProbTPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}
