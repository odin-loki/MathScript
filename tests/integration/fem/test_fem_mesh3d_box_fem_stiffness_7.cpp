
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

TEST(IntegrationFem,  Fem3dMeshStiffnessLoadTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_mesh3d_box");
    expect_contains(interp, "help", "fem_stiffness_3d");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("f3").rows(), interp.state().matrices.at("K3").rows());
}

TEST(IntegrationFem,  SphericalKnScalar) {
    Interpreter interp;
    expect_ok(interp, "sk = spherical_kn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::spherical_kn(1, 1.0), 1e-8);
}
