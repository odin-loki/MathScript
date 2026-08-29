
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

TEST(IntegrationSignal,  LmsEnvelopeTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_lms(x,d,filter_length,mu)");
    expect_contains(interp, "help", "signal_lms_weights(x,d,filter_length,mu)");

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(IntegrationSignal,  JacobiDnScalar) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}
