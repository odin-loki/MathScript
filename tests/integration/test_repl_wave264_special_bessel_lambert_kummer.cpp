// MathScript Integration Tests: REPL Interpreter – Wave 264 Special Functions
//
// Wave 264 REPL smoke: bessel_y/i, lambert_w, kummer_u, special_airy_bi.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave264SpecialBesselLambertKummer, BesselY) {
    Interpreter interp;
    expect_ok(interp, "y = bessel_y(0, 1)");
    ASSERT_GT(interp.state().scalars.count("y"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("y"), ms::bessel_y(0, 1.0), 1e-9);
    expect_contains(interp, "help", "bessel_y(nu,x)");
    expect_ok(interp, "ys = special_bessel_y(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ys"), ms::bessel_y(0, 1.0), 1e-9);
}

TEST(ReplWave264SpecialBesselLambertKummer, BesselI) {
    Interpreter interp;
    expect_ok(interp, "i = bessel_i(0, 1)");
    ASSERT_GT(interp.state().scalars.count("i"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("i"), ms::bessel_i(0, 1.0), 1e-9);
    expect_contains(interp, "help", "bessel_i(nu,x)");
    expect_ok(interp, "is = special_bessel_i(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("is"), ms::bessel_i(0, 1.0), 1e-9);
}

TEST(ReplWave264SpecialBesselLambertKummer, LambertW) {
    Interpreter interp;
    expect_ok(interp, "w = lambert_w(0, 1)");
    ASSERT_GT(interp.state().scalars.count("w"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("w"), ms::lambert_w(0, 1.0), 1e-9);
    expect_contains(interp, "help", "lambert_w(branch,z)");
    expect_ok(interp, "ws = special_lambert_w(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("ws"), ms::lambert_w(0, 1.0), 1e-9);
}

TEST(ReplWave264SpecialBesselLambertKummer, KummerU) {
    Interpreter interp;
    expect_ok(interp, "u = kummer_u(1, 2, 0.5)");
    ASSERT_GT(interp.state().scalars.count("u"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("u"), ms::kummer_u(1.0, 2.0, 0.5), 1e-9);
    expect_contains(interp, "help", "kummer_u(a,b,z)");
}

TEST(ReplWave264SpecialBesselLambertKummer, SpecialAiryBi) {
    Interpreter interp;
    expect_ok(interp, "bi = special_airy_bi(0)");
    ASSERT_GT(interp.state().scalars.count("bi"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bi"), ms::airy_bi(0.0), 1e-9);
    expect_contains(interp, "help", "special_airy_bi(x)");
}
