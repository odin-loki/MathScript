
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

TEST(IntegrationSpecial,  GraphInfoTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_connected_components");
    expect_contains(interp, "help", "info_channel_capacity_input");

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(IntegrationSpecial,  BesselLuScalar) {
    Interpreter interp;
    expect_ok(interp, "blu = bessel_lu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("blu"), ms::bessel_lu(0, 1), 1e-8);
}
