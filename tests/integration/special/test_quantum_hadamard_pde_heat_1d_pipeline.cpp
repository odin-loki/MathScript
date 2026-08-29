
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

TEST(IntegrationSpecial,  HadamardHeat) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_hadamard");
    expect_contains(interp, "help", "pde_heat_1d");
    expect_contains(interp, "help", "pde_heat_1d_cn");

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(IntegrationSpecial,  BesselHScalar) {
    Interpreter interp;
    expect_ok(interp, "bh = bessel_h(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bh"), ms::bessel_h(0, 1), 1e-8);
}
