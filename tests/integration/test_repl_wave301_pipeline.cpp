// MathScript Integration Tests: REPL Interpreter – Wave 301 Pipeline
//
// Wave 301 REPL smoke: ml/linalg/graph tail11 extensions, spherical/bessel-zero scalar validation.

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

TEST(ReplWave301Pipeline, MlLinalgGraph) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_mat_transpose");
    expect_contains(interp, "help", "funm");
    expect_contains(interp, "help", "graph_min_arborescence");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    EXPECT_EQ(interp.state().matrices.at("At").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("At").cols(), 2u);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);

    expect_ok(interp, "Pd = precond_diag(A)");
    EXPECT_GT(interp.state().matrices.at("Pd").rows(), 0u);

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);
}

TEST(ReplWave301Pipeline, SphericalBesselZeroScalar) {
    Interpreter interp;

    expect_ok(interp, "in0 = spherical_in(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("in0"), ms::spherical_in(0, 1.0), 1e-9);

    expect_ok(interp, "jz = bessel_zero_jnu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("jz"), ms::bessel_zero_jnu(0, 1), 1e-9);
}
