
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
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

TEST(IntegrationOde,  AdamsAutocorrTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_adams_bashforth2(");
    expect_contains(interp, "help", "signal_autocorr(x,max_lag)");

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "bd = ode_bdf2(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("bd").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(IntegrationOde,  JacobiCnScalar) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}
