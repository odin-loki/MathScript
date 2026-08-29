
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

TEST(IntegrationLinalg,  WaveletWave1dTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "wavelet_compress_vec");
    expect_contains(interp, "help", "pde_wave_1d");

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(IntegrationLinalg,  StruveYnScalar) {
    Interpreter interp;
    expect_ok(interp, "yn = struve_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("yn"), ms::struve_yn(1, 1.0), 1e-8);
}
