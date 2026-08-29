
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

TEST(IntegrationSignal,  AutocorrLmsTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_autocorr");
    expect_contains(interp, "help", "signal_lms");

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
}

TEST(IntegrationSignal,  Theta1PrimeScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = theta1_prime(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::theta1_prime(0.2, 0.1), 1e-8);
}
