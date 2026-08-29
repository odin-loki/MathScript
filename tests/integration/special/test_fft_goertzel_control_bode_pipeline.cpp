
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

TEST(IntegrationSpecial,  GoertzelBodeTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_goertzel(x,f,fs)");
    expect_contains(interp, "help", "control_bode(num,den,w)");

    expect_ok(interp, "G = fft_goertzel([1; 1; 1; 1], 0, 4)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    EXPECT_EQ(interp.state().matrices.at("bode").rows(), 1u);
}

TEST(IntegrationSpecial,  JacobiNdScalar) {
    Interpreter interp;
    expect_ok(interp, "nd = jacobi_nd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nd"), ms::jacobi_nd(0.5, 0.5), 1e-8);
}
