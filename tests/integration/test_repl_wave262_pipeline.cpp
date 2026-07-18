// MathScript Integration Tests: REPL Interpreter – Wave 262 Pipeline
//
// finance_bachelier_call/put, finance_vasicek_bond_price, finance_cir_bond_price,
// finance_trinomial_option REPL bindings.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(ReplWave262Pipeline, BachelierCall) {
    Interpreter interp;

    expect_ok(interp, "bc = finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)");
    ASSERT_GT(interp.state().scalars.count("bc"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bc"), 0.11503712918258642, 1e-6);
    expect_contains(interp, "help", "finance_bachelier_call(F,K,T,r,sigma)");
}

TEST(ReplWave262Pipeline, BachelierPut) {
    Interpreter interp;

    expect_ok(interp, "bp = finance_bachelier_put(100, 100, 1.5, 0.04, 0.25)");
    ASSERT_GT(interp.state().scalars.count("bp"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bp"), 0.11503712918258642, 1e-6);
    expect_contains(interp, "help", "finance_bachelier_put(F,K,T,r,sigma)");
}

TEST(ReplWave262Pipeline, VasicekBondPrice) {
    Interpreter interp;

    expect_ok(interp, "vz = finance_vasicek_bond_price(0.05, 0.5, 0.05, 0.02, 1.0)");
    ASSERT_GT(interp.state().scalars.count("vz"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("vz"), 0.9512737476480422, 1e-6);
    expect_contains(interp, "help", "finance_vasicek_bond_price(r,a,b,sigma,tau)");
}

TEST(ReplWave262Pipeline, CirBondPrice) {
    Interpreter interp;

    expect_ok(interp, "cr = finance_cir_bond_price(0.05, 0.5, 0.05, 0.05, 1.0)");
    ASSERT_GT(interp.state().scalars.count("cr"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("cr"), 0.951243269844748, 1e-6);
    expect_contains(interp, "help", "finance_cir_bond_price(r,a,b,sigma,tau)");
}

TEST(ReplWave262Pipeline, TrinomialOption) {
    Interpreter interp;

    expect_ok(interp, "tri = finance_trinomial_option(100, 100, 1, 0.05, 0.2, 200, 1, 0)");
    ASSERT_GT(interp.state().scalars.count("tri"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("tri"), 10.450583572185565, 0.5);
    expect_contains(interp, "help",
                    "finance_trinomial_option(S,K,T,r,sigma,n_steps,is_call,is_american)");
}
