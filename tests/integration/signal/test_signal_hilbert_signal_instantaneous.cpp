
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

TEST(IntegrationSignal,  HilbertPhaseTail30) {
    Interpreter interp;
    const auto help = interp.execute("help");
    ASSERT_TRUE(help.has_value()) << "help";
    if (help->find("signal_hilbert(x)") != std::string::npos) {
        expect_contains(interp, "help", "signal_hilbert(x)");
    }
    if (help->find("signal_instantaneous_phase(x)") != std::string::npos) {
        expect_contains(interp, "help", "signal_instantaneous_phase(x)");
    }

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(IntegrationSignal,  JacobiNcScalar) {
    Interpreter interp;
    expect_ok(interp, "nc = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("nc"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}
