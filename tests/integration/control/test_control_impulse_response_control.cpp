
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

TEST(IntegrationControl,  ImpulseKalmanUpdateCovTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "control_impulse_response");
    expect_contains(interp, "help", "control_kalman_update_cov");

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
}

TEST(IntegrationControl,  KelvinBeiScalar) {
    Interpreter interp;
    expect_ok(interp, "bei = kelvin_bei(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("bei"), ms::kelvin_bei(0, 1.0), 1e-8);
}
