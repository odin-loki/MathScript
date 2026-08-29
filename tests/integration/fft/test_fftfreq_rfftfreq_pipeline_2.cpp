
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

TEST(IntegrationFft,  FftfreqRfftfreqTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "fftfreq");
    expect_contains(interp, "help", "rfftfreq");

    expect_ok(interp, "f = fftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    expect_ok(interp, "f2 = fftfreq(8, 0.5)");
    ASSERT_GT(interp.state().matrices.count("f2"), 0u);
    expect_ok(interp, "rf = rfftfreq(8)");
    ASSERT_GT(interp.state().matrices.count("rf"), 0u);
}

TEST(IntegrationFft,  JacobiDcScalar) {
    Interpreter interp;
    expect_ok(interp, "dc = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dc"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}
