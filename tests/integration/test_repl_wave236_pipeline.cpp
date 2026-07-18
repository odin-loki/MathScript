// MathScript Integration Tests: REPL Interpreter – Wave 236 Bindings Pipeline
//
// Wired: fem_poisson3d (3D P1 Poisson on unit cube via mesh3d_box + assemble_stiffness_3d).
// Skipped (not on main): cfd_advection3d (no 3D CFD headers), crypto_x25519 (no X25519 API).

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave236Pipeline, FemPoisson3dBinding) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 27u);
    EXPECT_GT(interp.state().matrices.at("u3")(13, 0), 0.0);

    expect_error(interp, "fem_poisson3d(0, 2, 2)");
    expect_error(interp, "fem_poisson3d(2, 2)");

    expect_contains(interp, "help", "fem_poisson3d");
}
