
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

TEST(IntegrationDistributed,  DistLsmrLsqrMatmul) {
    Interpreter interp;
    expect_contains(interp, "help", "dist_lsmr");
    expect_contains(interp, "help", "dist_lsqr");
    expect_contains(interp, "help", "dist_matmul");

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(IntegrationDistributed,  ProbChi2PpfScalar) {
    Interpreter interp;
    expect_ok(interp, "cp = prob_chi2_ppf(0.5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cp"), ms::chi2_ppf(0.5, 2), 1e-8);
}
