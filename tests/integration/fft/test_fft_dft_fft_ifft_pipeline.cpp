
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

TEST(IntegrationFft,  DftIfftTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dft(x)");
    expect_contains(interp, "help", "fft_ifft(spectrum)");

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(IntegrationFft,  KelvinKeiScalar) {
    Interpreter interp;
    expect_ok(interp, "kk = kelvin_kei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kk"), ms::kelvin_kei(0, 1), 1e-8);
}
