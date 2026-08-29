
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

TEST(IntegrationFem,  FemLoad3dCfdGrid3d) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_load_3d");
    expect_contains(interp, "help", "cfd_grid3d");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(IntegrationFem,  KelvinBerScalar) {
    Interpreter interp;
    expect_ok(interp, "kbr = kelvin_ber(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("kbr"), ms::kelvin_ber(0, 1.0), 1e-8);
}
