
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

TEST(IntegrationFem,  FemMesh2dTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_mesh2d");

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(IntegrationFem,  AiryBipScalar) {
    Interpreter interp;
    expect_ok(interp, "ab = airy_bip(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ab"), ms::airy_bip(0.5), 1e-8);
}
