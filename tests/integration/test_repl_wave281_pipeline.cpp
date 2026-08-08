// MathScript Integration Tests: REPL Interpreter – Wave 281 Pipeline
//
// Wave 281 REPL smoke: quantum density/algebra tail11, CFD/FEM pipelines,
// signal resample/savgol assign, GRIA hamming scalar, cellmemory_reset, polylog.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/frameworks/izaac/izaac.hpp"
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

TEST(ReplWave281Pipeline, QuantumFemCfd) {
    Interpreter interp;

    expect_contains(interp, "help", "quantum_density_matrix");
    expect_contains(interp, "help", "cellmemory_reset");

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psi_n = quantum_ket_normalise(psi)");
    expect_ok(interp, "rho = quantum_density_matrix(psi_n)");
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "H = [1, 0; 0, 2]");
    expect_ok(interp, "gs = quantum_ground_state(H)");
    expect_ok(interp, "out = quantum_op_apply(H, gs)");
    EXPECT_EQ(interp.state().matrices.at("out").rows(), 2u);

    expect_ok(interp, "u = fem_poisson2d(4, 4)");
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(16, 1, 0.1, 0.01)");
    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("g").rows(), 8u);
}

TEST(ReplWave281Pipeline, SignalGriaCell) {
    Interpreter interp;
    ms::izaac::clear_session();

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6]");
    expect_ok(interp, "y = signal_resample(x, 2, 3)");
    expect_ok(interp, "z = signal_savgol(x, 5, 2)");
    EXPECT_GT(interp.state().matrices.at("y").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("z").rows(), 6u);

    expect_ok(interp, "a = [1; 0; 1; 0]");
    expect_ok(interp, "b = [1; 1; 0; 0]");
    expect_ok(interp, "d = gria_hamming_distance(a, b)");
    EXPECT_TRUE(interp.state().scalars.count("d") > 0);

    expect_ok(interp, "p = polylog(2, 0.5)");
    EXPECT_TRUE(interp.state().scalars.count("p") > 0);

    expect_ok(interp, "cellmemory_new(cm, 2, 2)");
    expect_ok(interp, "cellmemory_step(cm, [1; 0])");
    expect_ok(interp, "cellmemory_reset(cm)");
}
