
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

TEST(IntegrationControl,  FeedbackSs2tfTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "control_feedback");
    expect_contains(interp, "help", "control_ss2tf");

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
}

TEST(IntegrationControl,  EllipPiScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}
