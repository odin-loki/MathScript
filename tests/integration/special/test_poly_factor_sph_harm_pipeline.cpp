
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

TEST(IntegrationSpecial,  PolyFactorSphHarm) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_factor");
    expect_contains(interp, "help", "sph_harm");

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "fac = poly_factor(p2)");
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(IntegrationSpecial,  LaguerreLScalar) {
    Interpreter interp;
    expect_ok(interp, "lg = laguerre_l(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("lg"), ms::laguerre_l(1, 0.5), 1e-8);
}
