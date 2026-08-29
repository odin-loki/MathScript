
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

TEST(IntegrationFft,  FftfreqRfftfreqTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "fftfreq(n[,d])");
    expect_contains(interp, "help", "rfftfreq(n[,d])");

    expect_ok(interp, "f = fftfreq(8)");
    const auto& freqs = interp.state().matrices.at("f");
    EXPECT_EQ(freqs.rows(), 8u);
    EXPECT_EQ(freqs.cols(), 1u);
    EXPECT_NEAR(freqs(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(freqs(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(freqs(2, 0), 0.25, 1e-12);
    EXPECT_NEAR(freqs(3, 0), 0.375, 1e-12);
    EXPECT_NEAR(freqs(4, 0), -0.5, 1e-12);
    EXPECT_NEAR(freqs(5, 0), -0.375, 1e-12);
    EXPECT_NEAR(freqs(6, 0), -0.25, 1e-12);
    EXPECT_NEAR(freqs(7, 0), -0.125, 1e-12);

    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    EXPECT_GT(interp.state().matrices.at("f2").rows(), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    EXPECT_GT(interp.state().matrices.at("rf").rows(), 0u);
}

TEST(IntegrationFft,  StruveLScalar) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(1, 1), 1e-8);
}
