// MathScript Integration Tests: REPL Interpreter – Wave 262 Pipeline
//
// combo_motzkin_paths / combo_set_partitions REPL bindings.
// Smoke: Wave 261 APIs on main (combo_dyck_paths, combo_gray_code).

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

TEST(ReplWave262Pipeline, ComboDyckPathsSmoke) {
    Interpreter interp;

    expect_ok(interp, "d2 = combo_dyck_paths(2)");
    ASSERT_GT(interp.state().matrices.count("d2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d2").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("d2").cols(), 4u);
    expect_contains(interp, "help", "combo_dyck_paths(n)");
}

TEST(ReplWave262Pipeline, ComboGrayCodeSmoke) {
    Interpreter interp;

    expect_ok(interp, "gc = combo_gray_code(2)");
    ASSERT_GT(interp.state().matrices.count("gc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gc").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("gc").cols(), 2u);
    expect_contains(interp, "help", "combo_gray_code(n)");
}

TEST(ReplWave262Pipeline, ComboMotzkinPaths) {
    Interpreter interp;

    expect_ok(interp, "mp = combo_motzkin_paths(3)");
    ASSERT_GT(interp.state().matrices.count("mp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mp").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("mp").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("mp")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp")(0, 2), -1.0, 1e-9);
    expect_contains(interp, "help", "combo_motzkin_paths(n)");
}

TEST(ReplWave262Pipeline, ComboSetPartitions) {
    Interpreter interp;

    expect_ok(interp, "sp = combo_set_partitions(3)");
    ASSERT_GT(interp.state().matrices.count("sp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("sp").cols(), 3u);
    expect_contains(interp, "help", "combo_set_partitions(n)");
}

TEST(ReplWave262Pipeline, Lsq) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1; 1, 1; 2, 1; 3, 1]");
    expect_ok(interp, "b = [1; 3; 5; 7]");
    expect_contains(interp, "lsq(A, b)", "x =");
    expect_ok(interp, "x = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 2.0, 1e-5);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 1.0, 1e-5);
    expect_contains(interp, "help", "lsq(A");
}

TEST(ReplWave262Pipeline, Diag) {
    Interpreter interp;

    expect_ok(interp, "v = [2; 3; 5]");
    expect_contains(interp, "diag(v)", "D =");
    expect_ok(interp, "D = diag(v)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(2, 2), 5.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 0.0, 1e-12);
    expect_contains(interp, "help", "diag(v)");
}

TEST(ReplWave262Pipeline, EigSmoke) {
    Interpreter interp;

    expect_ok(interp, "Ag = [4, 1; 2, 3]");
    expect_ok(interp, "Dg, Vg = eig(Ag)");
    ASSERT_GT(interp.state().matrices.count("Dg"), 0u);
    expect_contains(interp, "help", "eig(A)");
}

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
