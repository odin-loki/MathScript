
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

TEST(IntegrationSignal,  SignalResampleSavgolTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_resample");
    expect_contains(interp, "help", "signal_savgol");

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(IntegrationSignal,  LaguerreLaScalar) {
    Interpreter interp;
    expect_ok(interp, "la = laguerre_la(2, 0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("la"), ms::laguerre_la(2, 0.5, 0.5), 1e-8);
}
