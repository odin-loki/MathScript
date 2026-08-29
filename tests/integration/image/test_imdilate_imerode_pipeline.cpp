
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

TEST(IntegrationImage,  MorphBilateralTail19) {
    Interpreter interp;
    expect_contains(interp, "help", "imdilate(M,k)");
    expect_contains(interp, "help", "imerode(M,k)");
    expect_contains(interp, "help", "imopen(M,k)");
    expect_contains(interp, "help", "imclose(M,k)");
    expect_contains(interp, "help", "bilateral(M,sigma_s,sigma_r)");

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
}

TEST(IntegrationImage,  ProbRayleighPpfScalar) {
    Interpreter interp;
    expect_ok(interp, "rq = prob_rayleigh_ppf(0.5, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rq"), ms::rayleigh_ppf(0.5, 1), 1e-8);
}
