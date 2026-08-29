
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

TEST(IntegrationFft,  RfftDftIfftTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_rfft");
    expect_contains(interp, "help", "fft_dft");
    expect_contains(interp, "help", "fft_ifft");

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);

    expect_ok(interp, "S = fft_dft([1; 0; -1; 0])");
    expect_ok(interp, "x = fft_ifft(S)");
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 4u);
}

TEST(IntegrationFft,  JacobiAmScalar) {
    Interpreter interp;
    expect_ok(interp, "am = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("am"), ms::jacobi_am(0.5, 0.5), 1e-8);
}
