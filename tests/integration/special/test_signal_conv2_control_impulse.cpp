
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

TEST(IntegrationSpecial,  SignalConv2ImpulseResponse) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_conv2(A,K)");
    expect_contains(interp, "help", "control_impulse_response(num,den[,t_end[,n_pts]])");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(C(2, 2), 4.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(IntegrationSpecial,  BesselYScalar) {
    Interpreter interp;
    expect_ok(interp, "by = bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("by"), ms::bessel_y(0, 1), 1e-8);
}
