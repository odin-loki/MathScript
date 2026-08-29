
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

TEST(IntegrationControl,  KalmanCtrbTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "control_kalman_predict");
    expect_contains(interp, "help", "control_kalman_update");
    expect_contains(interp, "help", "control_ctrb");

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-6);

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    EXPECT_GT(interp.state().matrices.at("Co").rows(), 0u);
}

TEST(IntegrationControl,  StruveKScalar) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(1, 1.0), 1e-8);
}
