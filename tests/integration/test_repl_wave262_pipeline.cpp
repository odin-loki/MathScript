// MathScript Integration Tests: REPL Interpreter – Wave 262 Pipeline
//
// combo_necklaces, numthy_stern_brocot, lsq/diag, finance_bachelier_call,
// poly_resultant REPL bindings.
// Smoke: Wave 261 APIs on main (eig, fftfreq) and Wave 260 (minres).
// Deterministic; Wave 262 features merge from sibling branches before this
// pipeline lands last.

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

TEST(ReplWave262Pipeline, Minres) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "b = [1; 1; 1]");
    expect_ok(interp, "x = minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("x").cols(), 1u);
    expect_contains(interp, "help", "minres(A");
}

TEST(ReplWave262Pipeline, Eig) {
    Interpreter interp;

    expect_ok(interp, "Ag = [4, 1; 2, 3]");
    expect_ok(interp, "Dg, Vg = eig(Ag)");
    ASSERT_GT(interp.state().matrices.count("Dg"), 0u);
    ASSERT_GT(interp.state().matrices.count("Vg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dg").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Vg").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Vg").cols(), 2u);
    const double trace =
        interp.state().matrices.at("Dg")(0, 0) + interp.state().matrices.at("Dg")(1, 0);
    EXPECT_NEAR(trace, 7.0, 1e-4);
    expect_contains(interp, "help", "eig(A)");
}

TEST(ReplWave262Pipeline, Fftfreq) {
    Interpreter interp;

    expect_ok(interp, "f8 = fftfreq(8, 1)");
    ASSERT_GT(interp.state().matrices.count("f8"), 0u);
    EXPECT_EQ(interp.state().matrices.at("f8").rows(), 8u);
    EXPECT_EQ(interp.state().matrices.at("f8").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("f8")(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("f8")(1, 0), 0.125, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("f8")(4, 0), -0.5, 1e-12);
    expect_contains(interp, "help", "fftfreq(n[,d])");
}

TEST(ReplWave262Pipeline, ComboNecklaces) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    ASSERT_GT(interp.state().matrices.count("neck"), 0u);
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("neck").cols(), 2u);
    expect_contains(interp, "help", "combo_necklaces(n,k)");
}

TEST(ReplWave262Pipeline, NumthySternBrocot) {
    Interpreter interp;

    expect_ok(interp, "sb = numthy_stern_brocot(7)");
    ASSERT_GT(interp.state().matrices.count("sb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sb").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("sb").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("sb")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sb")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sb")(6, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sb")(6, 1), 1.0, 1e-9);
    expect_contains(interp, "help", "numthy_stern_brocot(n)");
}

TEST(ReplWave262Pipeline, Lsq) {
    Interpreter interp;

    expect_ok(interp, "Al = [1, 1; 2, 1; 3, 1]");
    expect_ok(interp, "bl = [2; 3; 4]");
    expect_ok(interp, "xl = lsq(Al, bl)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("xl").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xl")(0, 0), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xl")(1, 0), 1.0, 1e-6);
    expect_contains(interp, "help", "lsq(A");
}

TEST(ReplWave262Pipeline, Diag) {
    Interpreter interp;

    expect_ok(interp, "D = diag([2; 3; 4])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(2, 2), 4.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 0), 0.0, 1e-9);
    expect_contains(interp, "help", "diag(");
}

TEST(ReplWave262Pipeline, FinanceBachelierCall) {
    Interpreter interp;

    const double F = 100.0;
    const double K = 100.0;
    const double T = 1.5;
    const double r = 0.04;
    const double sigma = 0.25;
    const double expected =
        std::exp(-r * T) * sigma * std::sqrt(T) / std::sqrt(2.0 * M_PI);

    expect_ok(interp, "bc = finance_bachelier_call(100, 100, 1.5, 0.04, 0.25)");
    ASSERT_GT(interp.state().scalars.count("bc"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bc"), expected, 1e-6);
    expect_contains(interp, "help", "finance_bachelier_call(F,K,T,r,sigma)");
}

TEST(ReplWave262Pipeline, PolyResultant) {
    Interpreter interp;

    expect_ok(interp, "p = [3; 2; 1]");
    expect_ok(interp, "q = [5; 1]");
    expect_ok(interp, "res = poly_resultant(p, q)");
    ASSERT_GT(interp.state().scalars.count("res"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("res"), 18.0, 1e-6);
    expect_contains(interp, "help", "poly_resultant(p,q)");
}
