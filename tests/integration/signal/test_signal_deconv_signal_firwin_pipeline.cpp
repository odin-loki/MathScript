
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

TEST(IntegrationSignal,  DeconvFirwin) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_deconv(y,b)");
    expect_contains(interp, "help", "signal_firwin(n_taps,cutoff[,window])");

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("fw").cols(), 1u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(IntegrationSignal,  StruveYnScalar) {
    Interpreter interp;
    expect_ok(interp, "syn = struve_yn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::struve_yn(1, 1), 1e-8);
}
