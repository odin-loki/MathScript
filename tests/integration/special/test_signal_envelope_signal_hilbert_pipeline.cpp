
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

TEST(IntegrationSpecial,  EnvelopeHilbertTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_envelope");
    expect_contains(interp, "help", "signal_hilbert");

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "env = signal_envelope(x)");
    expect_ok(interp, "h = signal_hilbert(x)");
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);
}

TEST(IntegrationSpecial,  JacobiCnScalar) {
    Interpreter interp;
    expect_ok(interp, "cn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}
