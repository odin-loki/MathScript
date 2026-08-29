
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

TEST(IntegrationImage,  RobertsLaplacianTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "roberts(M)");
    expect_contains(interp, "help", "laplacian(M)");

    expect_ok(interp, "M = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1; 0, 0, 0, 1, 1]");
    expect_ok(interp, "rb = roberts(M)");
    ASSERT_GT(interp.state().matrices.count("rb"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "L = laplacian(G)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
}

TEST(IntegrationImage,  ProbTPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}
