
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

TEST(IntegrationControl,  Tf2ssSeries) {
    Interpreter interp;
    expect_contains(interp, "help", "control_tf2ss(num,den)");
    expect_contains(interp, "help", "control_series(num1,den1,num2,den2)");

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(IntegrationControl,  ChebyshevTScalar) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}
