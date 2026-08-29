
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

TEST(IntegrationSpecial,  IradonFemMeshTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "iradon");
    expect_contains(interp, "help", "fem_mesh2d");

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(IntegrationSpecial,  JacobiDcScalar) {
    Interpreter interp;
    expect_ok(interp, "jdc = jacobi_dc(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jdc"), ms::jacobi_dc(0.2, 0.3), 1e-8);
}
