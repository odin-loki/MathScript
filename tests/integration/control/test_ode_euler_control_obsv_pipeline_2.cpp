
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

TEST(IntegrationControl,  EulerObsvKalmanCovTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_euler");
    expect_contains(interp, "help", "control_obsv");
    expect_contains(interp, "help", "control_kalman_predict_cov");

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    EXPECT_GT(interp.state().matrices.at("Ob").rows(), 0u);

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);
}

TEST(IntegrationControl,  AngerJScalar) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-8);
}
