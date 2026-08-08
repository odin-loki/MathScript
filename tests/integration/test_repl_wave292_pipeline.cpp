// MathScript Integration Tests: REPL Interpreter – Wave 292 Pipeline
//
// Wave 292 REPL smoke: image/FEM/CFD tail11 extensions, gegenbauer_c scalar.

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

TEST(ReplWave292Pipeline, ImageFemCfd) {
    Interpreter interp;

    expect_contains(interp, "help", "impad");
    expect_contains(interp, "help", "fem_mesh2d_rectangular");
    expect_contains(interp, "help", "cfd_grid2d");

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("P").cols(), 4u);

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);

    expect_ok(interp, "g = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    EXPECT_GT(interp.state().matrices.at("g").rows(), 0u);
}

TEST(ReplWave292Pipeline, SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "gc = gegenbauer_c(2, 1.0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("gc"), ms::gegenbauer_c(2, 1.0, 0.5), 1e-9);
}
