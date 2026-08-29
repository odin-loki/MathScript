
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

TEST(IntegrationDistributed,  DistBicgstabMinresQmrTfqmr) {
    Interpreter interp;
    expect_contains(interp, "help", "dist_bicgstab");
    expect_contains(interp, "help", "dist_minres");
    expect_contains(interp, "help", "dist_qmr");
    expect_contains(interp, "help", "dist_tfqmr");

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(IntegrationDistributed,  ProbExpPpfScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = prob_exp_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::exp_ppf(0.5, 1), 1e-8);
}
