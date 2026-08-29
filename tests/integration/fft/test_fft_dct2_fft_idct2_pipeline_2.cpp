
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

TEST(IntegrationFft,  DctIdctDstTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dct2");
    expect_contains(interp, "help", "fft_idct2");
    expect_contains(interp, "help", "fft_dst2");

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    expect_ok(interp, "xr = fft_idct2(d)");
    EXPECT_EQ(interp.state().matrices.at("xr").rows(), 4u);

    expect_ok(interp, "s = fft_dst2([1; 2; 3; 4])");
    EXPECT_GT(interp.state().matrices.at("s").rows(), 0u);
}

TEST(IntegrationFft,  JacobiSdScalar) {
    Interpreter interp;
    expect_ok(interp, "sd = jacobi_sd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sd"), ms::jacobi_sd(0.5, 0.5), 1e-8);
}
