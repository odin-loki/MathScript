
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

TEST(IntegrationSignal,  SosfiltRk4SparseCooTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_sosfilt");
    expect_contains(interp, "help", "ode_rk4");
    expect_contains(interp, "help", "sparse_from_coo");

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(IntegrationSignal,  StruveHnScalar) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1.0), 1e-8);
}
