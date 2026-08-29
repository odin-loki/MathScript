// MathScript Integration Tests: REPL PDE bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(PdeReplPipeline, PdeBindingsPipeline) {
    Interpreter interp;

    // pde_heat_1d: valid 1D heat diffusion
    expect_ok(interp, "x0 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "h1 = pde_heat_1d(x0, 0.1, 0.1, 0.001, 10)");
    ASSERT_GT(interp.state().matrices.count("h1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h1").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h1").cols(), 1u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("h1")(i, 0)));
    }
    expect_error(interp, "pde_heat_1d(x0, 1.0, 0.1, 0.01, 10)");

    // pde_heat_2d: valid 2D heat on 5x5 grid (rows×cols)
    expect_ok(interp,
              "u0 = [0,0,0,0,0; 0,0,0,0,0; 0,0,10,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "h2 = pde_heat_2d(u0, 0.05, 0.1, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2").cols(), 5u);
    expect_error(interp, "pde_heat_2d(u0, 1.0, 0.1, 0.1, 0.1, 5)");

    // pde_wave_1d: valid wave evolution
    expect_ok(interp, "u0w = [0; 0; 0; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "v0w = zeros(9, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0w, v0w, 1.0, 0.1, 0.05, 10)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 9u);
    expect_error(interp, "pde_wave_1d(u0w, v0w, 2.0, 0.1, 0.1, 5)");

    // pde_advection_1d: valid periodic advection
    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "a1 = pde_advection_1d(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("a1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("a1").rows(), 10u);
    expect_error(interp, "pde_advection_1d(u0a, 2.0, 0.1, 0.1, 5)");

    // pde_poisson_2d: zero source converges to zero
    expect_ok(interp, "f0 = zeros(7, 7)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 100, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("p2").cols(), 7u);
    expect_error(interp, "pde_poisson_2d(zeros(2, 2), 0.1, 0.1, 10, 1e-6)");

    // pde_burgers_1d: valid viscous Burgers
    expect_ok(interp, "u0b = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 5u);
    expect_error(interp, "pde_burgers_1d([0; 1], 0.01, 0.1, 0.001, 5)");
}
