
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

TEST(IntegrationControl,  KalmanPredictCovStepResponse) {
    Interpreter interp;
    expect_contains(interp, "help", "control_step_response(num,den[,t_end[,n_pts]])");

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(IntegrationControl,  KelvinKerScalar) {
    Interpreter interp;
    expect_ok(interp, "ker = kelvin_ker(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ker"), ms::kelvin_ker(0, 1), 1e-8);
}
