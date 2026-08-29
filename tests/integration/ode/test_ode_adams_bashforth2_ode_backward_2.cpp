
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

TEST(IntegrationOde,  AdamsBdfTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_adams_bashforth2");
    expect_contains(interp, "help", "ode_backward_euler");
    expect_contains(interp, "help", "ode_bdf2");

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "bd = ode_bdf2(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("bd").rows(), 0u);
}

TEST(IntegrationOde,  EllipEIncScalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.5), 1e-8);
}
