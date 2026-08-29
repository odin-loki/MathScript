// MathScript Integration Tests: REPL Interpreter – Audit Wave 318 Final Sweep Pipeline
//
// Inventory: leftover C++ names are already bound as combo_catalan/motzkin/eulerian,
// finance_sharpe/sortino/treynor, special_erfinv/airy_*, control_pidtune_*,
// quantum_wigner/husimi. New bind: ode_rk2 (Heun) next to ode_rk4.

#include <cmath>
#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(ReplAuditW318FinalSweepPipeline, ComboCatalanThreeIsFive) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_catalan(n)");

    expect_ok(interp, "c = combo_catalan(3)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("c"), 5.0);
}

TEST(ReplAuditW318FinalSweepPipeline, OdeRk2Exponential) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_rk2(\"formula\",t0,y0,t_end,steps)");

    // y' = y, y(0) = 1, t_end = 1 → y(1) = e. Heun/RK2 with 200 steps is well inside 1e-3.
    expect_ok(interp, "traj = ode_rk2(\"y\", 0, 1, 1, 200)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    const auto& traj = interp.state().matrices.at("traj");
    ASSERT_GE(traj.rows(), 2u);
    ASSERT_GE(traj.cols(), 2u);
    EXPECT_NEAR(traj(traj.rows() - 1, 1), std::exp(1.0), 1e-3);
}
