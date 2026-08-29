
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

TEST(IntegrationSignal,  DeconvFirwinXcorrTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_deconv");
    expect_contains(interp, "help", "signal_firwin");
    expect_contains(interp, "help", "signal_xcorr");

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 3u);

    expect_ok(interp, "h = signal_firwin(4, 0.3)");
    EXPECT_GT(interp.state().matrices.at("h").rows(), 0u);

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    EXPECT_EQ(interp.state().matrices.at("xc").rows(), 5u);
}

TEST(IntegrationSignal,  LambertWScalar) {
    Interpreter interp;
    expect_ok(interp, "lw = lambert_w(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lw"), ms::lambert_w(0, 0.5), 1e-8);
}
