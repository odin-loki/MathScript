
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

TEST(IntegrationSpecial,  CoherenceRosenbrockTail30) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_coherence(x,y,fs,nperseg)");
    expect_contains(interp, "help", "ode_rosenbrock23(\"formula\",t0,y0,t_end,steps)");

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(IntegrationSpecial,  JacobiCdScalar) {
    Interpreter interp;
    expect_ok(interp, "cd = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cd"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}
