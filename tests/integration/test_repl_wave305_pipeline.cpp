// MathScript Integration Tests: REPL Interpreter – Wave 305 Pipeline
//
// Wave 305 REPL smoke: diffgeo/quantum/FEM/CFD tail11 extensions, Kelvin/Struve scalar validation.

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

TEST(ReplWave305Pipeline, DiffgeoQuantumFemCfd) {
    Interpreter interp;

    expect_contains(interp, "help", "diffgeo_surface_normal_sphere");
    expect_contains(interp, "help", "quantum_ket_superposition");
    expect_contains(interp, "help", "fem_poisson3d");

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    EXPECT_GT(interp.state().matrices.at("sup").rows(), 0u);

    expect_ok(interp, "kb = quantum_ket_basis(2, 0)");
    EXPECT_GT(interp.state().matrices.at("kb").rows(), 0u);

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplWave305Pipeline, KelvinStruveScalar) {
    Interpreter interp;

    expect_ok(interp, "bei = kelvin_bei(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bei"), ms::kelvin_bei(0, 1.0), 1e-9);

    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1.0), 1e-9);
}
