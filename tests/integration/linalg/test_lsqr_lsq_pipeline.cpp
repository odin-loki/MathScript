
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"
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

TEST(IntegrationLinalg,  LsqrLsqTfqmrLsmr) {
    Interpreter interp;
    expect_contains(interp, "help", "lsqr");
    expect_contains(interp, "help", "lsq");
    expect_contains(interp, "help", "tfqmr");
    expect_contains(interp, "help", "lsmr");

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(IntegrationLinalg,  ProbTPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}
