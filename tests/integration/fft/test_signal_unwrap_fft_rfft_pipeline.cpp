
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

TEST(IntegrationFft,  UnwrapRfftTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_rfft(x)");

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(IntegrationFft,  WeberEScalar) {
    Interpreter interp;
    expect_ok(interp, "we = weber_e(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("we"), ms::weber_e(0, 1), 1e-8);
}
