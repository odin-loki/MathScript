// MathScript Integration Tests: REPL Interpreter – Audit Wave 314 Bignum/Cplx Pipeline
//
// Leftover user-facing binds: bigint_lcm / is_even / is_odd / pow / isqrt / is_prime
// and cplx_cauchy_integral (Cauchy integral of f(z)=z^2+1 on the unit circle).
// cplx_argument_principle skipped (CFunc + contour, no clean numeric smoke).

#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(ReplAuditW314BignumCplxPipeline, BigintLcmAndParity) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_lcm(a,b)");
    expect_contains(interp, "help", "bigint_is_even(n)");
    expect_contains(interp, "help", "bigint_is_odd(n)");

    expect_ok(interp, "l = bigint_lcm(48, 18)");
    expect_ok(interp, "ev = bigint_is_even(4)");
    expect_ok(interp, "od = bigint_is_odd(4)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("l"), 144.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("ev"), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("od"), 0.0);
}

TEST(ReplAuditW314BignumCplxPipeline, BigintIsqrtPowPrime) {
    Interpreter interp;
    expect_contains(interp, "help", "bigint_isqrt(n)");
    expect_contains(interp, "help", "bigint_pow(base,exp)");
    expect_contains(interp, "help", "bigint_is_prime(n)");

    expect_ok(interp, "r = bigint_isqrt(16)");
    expect_ok(interp, "p = bigint_pow(2, 10)");
    expect_ok(interp, "pr = bigint_is_prime(7)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("r"), 4.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("p"), 1024.0);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("pr"), 1.0);
}

TEST(ReplAuditW314BignumCplxPipeline, CplxCauchyIntegralAtOrigin) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_cauchy_integral(z0re,z0im)");

    // Cauchy integral formula: f(z0) = (1/2πi) ∮ f(z)/(z-z0) dz.
    // Smoke uses f(z)=z^2+1 on the unit circle, so f(0)=1.
    expect_ok(interp, "c0 = cplx_cauchy_integral(0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("c0"), 1.0, 0.2);
}
