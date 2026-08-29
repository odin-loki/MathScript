
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

TEST(IntegrationControl,  KalmanUpdateCovCtrbGram) {
    Interpreter interp;
    expect_contains(interp, "help", "control_ctrb_gram(A,B)");

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    EXPECT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(IntegrationControl,  ChebyshevUScalar) {
    Interpreter interp;
    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-8);
}
