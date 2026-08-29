
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

TEST(IntegrationFem,  TokenCryptoBwt) {
    Interpreter interp;
    ms::izaac::clear_session();

    expect_contains(interp, "help", "tokenbucket_capacity");
    expect_contains(interp, "help", "crypto_bytes_to_hex");
    expect_contains(interp, "help", "run_backtest_equity");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "tokenbucket_new(tb, 10, 2)");
    expect_ok(interp, "tokenbucket_capacity(tb)");
    expect_ok(interp, "tokenbucket_refill_rate(tb)");

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(IntegrationFem,  FemCfdQuantum2d) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 4, 4)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    EXPECT_GT(interp.state().matrices.at("u2").rows(), 0u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 8, 8)");
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.2, 0.2)");
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 8u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "traj = quantum_schrodinger(H, psi0, 0, 0.2, 4)");
    EXPECT_EQ(interp.state().matrices.at("traj").rows(), 5u);

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);
}
