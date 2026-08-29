
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

TEST(IntegrationFft,  Idst2Dct2Tail30) {
    Interpreter interp;
    expect_contains(interp, "help", "fft_dct2(x)");
    expect_contains(interp, "help", "fft_dst2(x)");

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "c = fft_dst2(x)");
    expect_ok(interp, "ib = idst2(c)");
    EXPECT_EQ(interp.state().matrices.at("ib").rows(), 4u);

    expect_ok(interp, "d = fft_dct2([1; 2; 3; 4])");
    EXPECT_GT(interp.state().matrices.at("d").rows(), 0u);
}

TEST(IntegrationFft,  EllipEScalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e(0.5), 1e-8);
}
