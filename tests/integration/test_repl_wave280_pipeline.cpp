// MathScript Integration Tests: REPL Interpreter – Wave 280 Pipeline
//
// Wave 280 REPL smoke: backtest total return, CellMemory recall assign,
// CellAI energy, GRIA CA scalars, tail11 migrations (quantum/GRIA/CellAI).

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

TEST(ReplWave280Pipeline, BacktestCellMemoryEnergy) {
    Interpreter interp;
    ms::izaac::clear_session();

    expect_contains(interp, "help", "run_backtest_total_return");
    expect_contains(interp, "help", "cellmemory_recall");

    expect_ok(interp, "prices = [100, 101, 102]");
    expect_ok(interp, "pos = [0, 1, 0]");
    expect_ok(interp, "tr = run_backtest_total_return(prices, pos, 1000)");
    EXPECT_TRUE(interp.state().scalars.count("tr") > 0);

    expect_ok(interp, "cellmemory_new(cm, 2, 2, [1, 2])");
    expect_ok(interp, "cellmemory_step(cm, [1; 0])");
    expect_ok(interp, "rec = cellmemory_recall(cm, 1.0)");
    EXPECT_EQ(interp.state().matrices.at("rec").rows(), 2u);

    expect_ok(interp, "W = [0, 1; 1, 0]");
    expect_ok(interp, "V = [1; 0]");
    expect_ok(interp, "H = [1; 0]");
    expect_ok(interp, "e = cellai_energy(W, V, H)");
    EXPECT_TRUE(interp.state().scalars.count("e") > 0);
}

TEST(ReplWave280Pipeline, GriaQuantumCellai) {
    Interpreter interp;

    expect_ok(interp, "s = [1; 0; 1; 0; 1; 0; 1; 0]");
    expect_ok(interp, "n = gria_ca_step(s, 30)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 8u);

    expect_ok(interp, "lam = gria_langton_lambda(30)");
    expect_ok(interp, "alpha = gria_alpha_ca(30, 4, 8)");
    EXPECT_TRUE(interp.state().scalars.count("lam") > 0);
    EXPECT_TRUE(interp.state().scalars.count("alpha") > 0);

    expect_ok(interp, "H = [1, 0; 0, 2]");
    expect_ok(interp, "spec = quantum_eigenspectrum(H)");
    expect_ok(interp, "gs = quantum_ground_state(H)");
    EXPECT_EQ(interp.state().matrices.at("spec").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("gs").rows(), 2u);

    expect_ok(interp, "E = [0, 1, 2]");
    expect_ok(interp, "bw = cellai_boltzmann_weights(E, 1)");
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 3u);
}
