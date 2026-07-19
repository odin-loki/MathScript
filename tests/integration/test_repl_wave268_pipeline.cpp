// MathScript Integration Tests: REPL Interpreter – Wave 268 Pipeline
//
// Wave 268 REPL smoke: pde_wave_2d, pde_advection_1d_lax_wendroff,
// pde_reaction_diffusion_1d.
// Regression: Wave 267 stats_max_value.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

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

TEST(ReplWave268Pipeline, PdeWave2d) {
    Interpreter interp;

    expect_ok(interp, "u0 = zeros(7, 7)");
    expect_ok(interp, "v0 = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0, v0, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
    expect_contains(interp, "help", "pde_wave_2d(");
}

TEST(ReplWave268Pipeline, PdeAdvectionLaxWendroff) {
    Interpreter interp;

    expect_ok(interp, "u0 = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);
    expect_contains(interp, "help", "pde_advection_1d_lax_wendroff(");
}

TEST(ReplWave268Pipeline, PdeReactionDiffusion1d) {
    Interpreter interp;

    expect_ok(interp, "u0 = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
    expect_contains(interp, "help", "pde_reaction_diffusion_1d(");
}

TEST(ReplWave268Pipeline, StatsMaxValueRegression) {
    Interpreter interp;

    expect_ok(interp, "mx = stats_max_value([3; 1; 4; 1; 5; 9; 2])");
    ASSERT_GT(interp.state().scalars.count("mx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("mx"), 9.0, 1e-9);
    expect_contains(interp, "help", "stats_max_value(");
}
