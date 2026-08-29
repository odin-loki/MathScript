
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

TEST(IntegrationFrameworks,  CellmemoryGriaTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cellmemory_recall");
    expect_contains(interp, "help", "gria_ca_step");
    expect_contains(interp, "help", "gria_gf2n_generate_field");

    expect_ok(interp, "cellmemory_new(cm332, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm332, 1)");
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(IntegrationFrameworks,  BesselLScalar) {
    Interpreter interp;
    expect_ok(interp, "bl = bessel_l(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("bl"), ms::bessel_l(1, 1.0), 1e-8);
}
