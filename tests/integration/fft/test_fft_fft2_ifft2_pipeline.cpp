
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

TEST(IntegrationFft,  Fft2Ifft2Tail30) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_fft2(S)");
    const auto help = interp.execute("help");
    ASSERT_TRUE(help.has_value()) << "help";
    if (help->find("ifft2") != std::string::npos) {
        expect_contains(interp, "help", "ifft2(S)");
    }

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);
}

TEST(IntegrationFft,  BesselJScalar) {
    Interpreter interp;
    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1), 1e-8);
}
