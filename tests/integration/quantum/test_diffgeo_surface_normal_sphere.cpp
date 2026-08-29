
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

TEST(IntegrationQuantum,  SurfaceNormalKetSuperpositionTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_surface_normal_sphere");
    expect_contains(interp, "help", "quantum_ket_superposition");

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 1u);

    expect_ok(interp, "sup = quantum_ket_superposition([1; 1])");
    ASSERT_GT(interp.state().matrices.count("sup"), 0u);
}

TEST(IntegrationQuantum,  SphericalYnScalar) {
    Interpreter interp;
    expect_ok(interp, "syn = spherical_yn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("syn"), ms::spherical_yn(1, 1.0), 1e-8);
}
