
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

TEST(IntegrationFft,  Fft2Ifft2Idst2Tail15) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_fft2");
    expect_contains(interp, "help", "ifft2");
    expect_contains(interp, "help", "idst2");

    expect_ok(interp, "S = [1, 0; 2, 0; 3, 0; 4, 0]");
    expect_ok(interp, "F = fft_fft2(S)");
    expect_ok(interp, "back = ifft2(F)");
    EXPECT_EQ(interp.state().matrices.at("back").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);
}

TEST(IntegrationFft,  JacobiScScalar) {
    Interpreter interp;
    expect_ok(interp, "sc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("sc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}
