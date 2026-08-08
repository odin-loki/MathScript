// MathScript Integration Tests: REPL Interpreter – Wave 300 Pipeline
//
// Wave 300 REPL smoke: poly/sph tail11 extensions, bessel scalar validation.

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
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

TEST(ReplWave300Pipeline, PolySph) {
    Interpreter interp;

    expect_contains(interp, "help", "poly_lagrange");
    expect_contains(interp, "help", "poly_interp_newton");
    expect_contains(interp, "help", "sph_harm");

    expect_ok(interp, "xs = [0; 1; 2]");
    expect_ok(interp, "ys = [1; 3; 5]");
    expect_ok(interp, "p = poly_lagrange(xs, ys)");
    EXPECT_GT(interp.state().matrices.at("p").rows(), 0u);

    expect_ok(interp, "pn = poly_interp_newton(xs, ys)");
    EXPECT_GT(interp.state().matrices.at("pn").rows(), 0u);

    expect_ok(interp, "p2 = [6; -5; 1]");
    expect_ok(interp, "rts = poly_roots(p2)");
    EXPECT_EQ(interp.state().matrices.at("rts").rows(), 2u);

    expect_ok(interp, "fac = poly_factor(p2)");
    EXPECT_EQ(interp.state().matrices.at("fac").rows(), 2u);

    expect_ok(interp, "Y = sph_harm(1, 1, 0.5, 1)");
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);
}

TEST(ReplWave300Pipeline, BesselScalar) {
    Interpreter interp;

    expect_ok(interp, "bj = bessel_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bj"), ms::bessel_j(0, 1.0), 1e-9);

    expect_ok(interp, "sj = sph_bessel_j(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::sph_bessel_j(2, 1.0), 1e-9);
}
