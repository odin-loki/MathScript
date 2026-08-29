
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

TEST(IntegrationPde,  LaplaceHelmholtzTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_laplace_2d");
    expect_contains(interp, "help", "pde_helmholtz_2d");

    expect_ok(interp, "B = [1,1,1,1,1; 0,0,0,0,0; 0,0,0,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
}

TEST(IntegrationPde,  KelvinKeiScalar) {
    Interpreter interp;
    expect_ok(interp, "kei = kelvin_kei(0, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("kei"), ms::kelvin_kei(0, 1.0), 1e-8);
}
