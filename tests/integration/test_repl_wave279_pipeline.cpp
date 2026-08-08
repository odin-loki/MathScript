// MathScript Integration Tests: REPL Interpreter – Wave 279 Pipeline
//
// Wave 279 REPL smoke: CellMemory introspection, topo/quantum tail11 migrations,
// explicit cfd_upwind_step_2d, backtest scalar metrics, spherical_jn eval fix.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave279Pipeline, CellMemoryBacktestSpecial) {
    Interpreter interp;
    ms::izaac::clear_session();

    expect_contains(interp, "help", "cellmemory_input_dim");
    expect_contains(interp, "help", "run_backtest_sharpe");
    expect_contains(interp, "help", "cellmemory_long_term_state");

    expect_ok(interp, "cellmemory_new(cm279, 2, 3, [0.5, 2])");
    expect_ok(interp, "cellmemory_input_dim(cm279)");
    expect_ok(interp, "cellmemory_memory_dim(cm279)");
    expect_contains(interp, "cellmemory_time_scales(cm279)", "0.5");

    expect_ok(interp, "cellmemory_step(cm279, [1; 0])");
    expect_ok(interp, "cellmemory_consolidate(cm279)");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm279)");
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "prices = [100, 102, 101, 103]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "sh = run_backtest_sharpe(prices, pos, 10000)");
    expect_ok(interp, "dd = run_backtest_max_drawdown(prices, pos, 10000)");
    EXPECT_TRUE(interp.state().scalars.count("sh") > 0);
    EXPECT_TRUE(interp.state().scalars.count("dd") > 0);

    expect_ok(interp, "x = 1.5");
    expect_ok(interp, "sj = spherical_jn(2, x)");
    EXPECT_TRUE(interp.state().scalars.count("sj") > 0);
}

TEST(ReplWave279Pipeline, TopoQuantumCfd) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 1, 1]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.5)");
    EXPECT_GT(interp.state().matrices.at("ac").rows(), 0u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);

    expect_ok(interp, "u = [1, 1, 1; 1, 1, 1; 1, 1, 1]");
    expect_ok(interp, "u1 = cfd_upwind_step_2d(u, 1, 0, 0.01, 0.1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 3u);
}
