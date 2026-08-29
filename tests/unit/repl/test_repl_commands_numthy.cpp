#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, combo_numthy_finance_scalars) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_factorial(n)");
    expect_contains(interp, "help", "numthy_partition(n)");
    expect_contains(interp, "help", "finance_bond_price(c,y,n,fv)");

    expect_ok(interp, "f = combo_factorial(5)");
    EXPECT_NEAR(interp.state().scalars.at("f"), 120.0, 1e-9);
    expect_contains(interp, "combo_factorial(5)", "120");

    expect_ok(interp, "p = numthy_partition(5)");
    EXPECT_NEAR(interp.state().scalars.at("p"), 7.0, 1e-9);
    expect_contains(interp, "numthy_partition(5)", "7");

    expect_ok(interp, "bp = finance_bond_price(0.05, 0.05, 10)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 100.0, 1e-6);
    expect_contains(interp, "finance_bond_price(0.05, 0.05, 10)", "100");
}

TEST(ReplCommandsTest, numthy_gcd_assignment) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-9);
    expect_contains(interp, "numthy_gcd(48, 18)", "6");
}

TEST(ReplCommandsTest, numthy_isprime) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_isprime(n)");

    expect_ok(interp, "ip = numthy_isprime(17)");
    EXPECT_NEAR(interp.state().scalars.at("ip"), 1.0, 1e-9);

    expect_contains(interp, "numthy_isprime(17)", "1");
    expect_contains(interp, "numthy_isprime(18)", "0");
}

TEST(ReplCommandsTest, numthy_euler_phi) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_euler_phi(n)");

    expect_ok(interp, "phi = numthy_euler_phi(12)");
    EXPECT_NEAR(interp.state().scalars.at("phi"), 4.0, 1e-9);

    expect_contains(interp, "numthy_euler_phi(12)", "4");
}

TEST(ReplCommandsTest, numthy_mobius) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_mobius(n)");

    expect_ok(interp, "mu = numthy_mobius(6)");
    EXPECT_NEAR(interp.state().scalars.at("mu"), 1.0, 1e-9);

    expect_contains(interp, "numthy_mobius(6)", "1");
    expect_contains(interp, "numthy_mobius(4)", "0");
}

TEST(ReplCommandsTest, numthy_num_divisors) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_num_divisors(n)");

    expect_ok(interp, "tau = numthy_num_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("tau"), 6.0, 1e-9);

    expect_contains(interp, "numthy_num_divisors(12)", "6");
}

TEST(ReplCommandsTest, numthy_lcm) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_lcm(a,b)");

    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-9);

    expect_contains(interp, "numthy_lcm(4, 6)", "12");
}

TEST(ReplCommandsTest, numthy_mod_pow) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_mod_pow(base,exp,mod)");

    expect_ok(interp, "mp = numthy_mod_pow(2, 10, 1000)");
    EXPECT_NEAR(interp.state().scalars.at("mp"), 24.0, 1e-9);

    expect_contains(interp, "numthy_mod_pow(2, 10, 1000)", "24");
    expect_contains(interp, "numthy_mod_pow(3, 12, 13)", "1");
}

TEST(ReplCommandsTest, numthy_sum_divisors) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_sum_divisors(n)");

    expect_ok(interp, "sigma = numthy_sum_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("sigma"), 28.0, 1e-9);

    expect_contains(interp, "numthy_sum_divisors(12)", "28");
}

TEST(ReplCommandsTest, numthy_nextprime) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_nextprime(n)");

    expect_ok(interp, "np = numthy_nextprime(10)");
    EXPECT_NEAR(interp.state().scalars.at("np"), 11.0, 1e-9);

    expect_contains(interp, "numthy_nextprime(10)", "11");
    expect_contains(interp, "numthy_nextprime(11)", "13");
}

TEST(ReplCommandsTest, numthy_liouville) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_liouville(n)");

    expect_ok(interp, "lam = numthy_liouville(12)");
    EXPECT_NEAR(interp.state().scalars.at("lam"), -1.0, 1e-9);

    expect_contains(interp, "numthy_liouville(12)", "-1");
    expect_contains(interp, "numthy_liouville(6)", "1");
}

TEST(ReplCommandsTest, numthy_prime_pi) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_prime_pi(n)");

    expect_ok(interp, "pi = numthy_prime_pi(100)");
    EXPECT_NEAR(interp.state().scalars.at("pi"), 25.0, 1e-9);

    expect_contains(interp, "numthy_prime_pi(10)", "4");
    expect_contains(interp, "numthy_prime_pi(100)", "25");
}

TEST(ReplCommandsTest, numthy_legendre_symbol) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_legendre_symbol(a,p)");

    expect_ok(interp, "ls = numthy_legendre_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ls"), 1.0, 1e-9);

    expect_contains(interp, "numthy_legendre_symbol(2, 7)", "1");
    expect_contains(interp, "numthy_legendre_symbol(3, 7)", "-1");
}

TEST(ReplCommandsTest, numthy_prevprime) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_prevprime(n)");

    expect_ok(interp, "pp = numthy_prevprime(10)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), 7.0, 1e-9);

    expect_contains(interp, "numthy_prevprime(10)", "7");
    expect_contains(interp, "numthy_prevprime(3)", "2");
}

TEST(ReplCommandsTest, numthy_jacobi_symbol) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_jacobi_symbol(a,n)");

    expect_ok(interp, "js = numthy_jacobi_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("js"), 1.0, 1e-9);

    expect_contains(interp, "numthy_jacobi_symbol(2, 7)", "1");
    expect_contains(interp, "numthy_jacobi_symbol(3, 7)", "-1");
}

TEST(ReplCommandsTest, numthy_prime_nth) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_prime_nth(n)");

    expect_ok(interp, "pn = numthy_prime_nth(6)");
    EXPECT_NEAR(interp.state().scalars.at("pn"), 13.0, 1e-9);

    expect_contains(interp, "numthy_prime_nth(6)", "13");
    expect_contains(interp, "numthy_prime_nth(1)", "2");
}

TEST(ReplCommandsTest, numthy_kronecker_symbol) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_kronecker_symbol(a,n)");

    expect_ok(interp, "ks = numthy_kronecker_symbol(2, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ks"), 1.0, 1e-9);

    expect_contains(interp, "numthy_kronecker_symbol(2, 7)", "1");
    expect_contains(interp, "numthy_kronecker_symbol(3, 7)", "-1");
}

TEST(ReplCommandsTest, numthy_tonelli_shanks) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_tonelli_shanks(n,p)");

    expect_ok(interp, "x = numthy_tonelli_shanks(2, 7)");
    const int64_t root = static_cast<int64_t>(interp.state().scalars.at("x"));
    EXPECT_EQ((root * root) % 7, 2);

    expect_contains(interp, "numthy_tonelli_shanks(2, 7)", "4");
}

TEST(ReplCommandsTest, numthy_mod_inv) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_mod_inv(a,m)");

    expect_ok(interp, "inv = numthy_mod_inv(3, 7)");
    EXPECT_NEAR(interp.state().scalars.at("inv"), 5.0, 1e-9);

    expect_contains(interp, "numthy_mod_inv(3, 7)", "5");
}

TEST(ReplCommandsTest, numthy_factor_count) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor_count(n)");

    expect_ok(interp, "fc = numthy_factor_count(12)");
    EXPECT_NEAR(interp.state().scalars.at("fc"), 3.0, 1e-9);

    expect_contains(interp, "numthy_factor_count(12)", "3");
}

TEST(ReplCommandsTest, numthy_primitive_root) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_primitive_root(p)");

    expect_ok(interp, "proot = numthy_primitive_root(7)");
    EXPECT_NEAR(interp.state().scalars.at("proot"), 3.0, 1e-9);

    expect_contains(interp, "numthy_primitive_root(7)", "3");
}

TEST(ReplCommandsTest, numthy_extended_gcd) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_extended_gcd(a,b)");

    expect_ok(interp, "g = numthy_extended_gcd(35, 15)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 5.0, 1e-9);

    expect_contains(interp, "numthy_extended_gcd(35, 15)", "5");
}

TEST(ReplCommandsTest, numthy_crt) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_crt(r,m)");

    expect_ok(interp, "r = [2; 3; 2]");
    expect_ok(interp, "m = [3; 5; 7]");
    expect_ok(interp, "x = numthy_crt(r, m)");
    EXPECT_NEAR(interp.state().scalars.at("x"), 23.0, 1e-9);

    expect_contains(interp, "numthy_crt([2; 3; 2], [3; 5; 7])", "23");
}

TEST(ReplCommandsTest, numthy_divisors_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_divisors_vec(n)");

    expect_ok(interp, "d = numthy_divisors_vec(12)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    const auto& divs = interp.state().matrices.at("d");
    EXPECT_EQ(divs.rows(), 6u);
    EXPECT_EQ(divs.cols(), 1u);
    const double expected[] = {1, 2, 3, 4, 6, 12};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(divs(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, numthy_factor_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor_vec(n)");

    expect_ok(interp, "f = numthy_factor_vec(12)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& factors = interp.state().matrices.at("f");
    EXPECT_EQ(factors.rows(), 3u);
    EXPECT_EQ(factors.cols(), 1u);
    EXPECT_NEAR(factors(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(2, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_continued_fraction) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_continued_fraction(x,n)");

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);

    expect_ok(interp, "numthy_continued_fraction(3.14159, 5)");
}

TEST(ReplCommandsTest, numthy_convergents) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_convergents(cf)");

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "numthy_convergents(cf)");
}

TEST(ReplCommandsTest, numthy_primes) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_primes(lo,hi)");

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_is_primitive_root) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_is_primitive_root(g,p)");

    expect_ok(interp, "pr = numthy_is_primitive_root(2, 11)");
    EXPECT_NEAR(interp.state().scalars.at("pr"), 1.0, 1e-9);

    expect_contains(interp, "numthy_is_primitive_root(2, 11)", "1");
}

TEST(ReplCommandsTest, numthy_discrete_log) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_discrete_log(g,h,p)");

    expect_ok(interp, "dl = numthy_discrete_log(2, 8, 11)");
    EXPECT_NEAR(interp.state().scalars.at("dl"), 3.0, 1e-9);

    expect_contains(interp, "numthy_discrete_log(2, 8, 11)", "3");
}

TEST(ReplCommandsTest, numthy_von_mangoldt) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_von_mangoldt(n)");

    // 8 = 2^3, so von Mangoldt(8) = ln(2).
    expect_ok(interp, "vm = numthy_von_mangoldt(8)");
    EXPECT_NEAR(interp.state().scalars.at("vm"), std::log(2.0), 1e-9);

    expect_contains(interp, "numthy_von_mangoldt(8)", "0.693");
}

TEST(ReplCommandsTest, numthy_jordan_totient) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_jordan_totient(k,n)");

    // J_2(6) = 36*(1-1/2)*(1-1/3) = 12.
    expect_ok(interp, "jt = numthy_jordan_totient(2, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 12.0, 1e-9);

    expect_contains(interp, "numthy_jordan_totient(2, 6)", "12");
}

TEST(ReplCommandsTest, numthy_factor) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor(n)");

    expect_ok(interp, "f = numthy_factor(12)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& factors = interp.state().matrices.at("f");
    EXPECT_EQ(factors.rows(), 3u);
    EXPECT_EQ(factors.cols(), 1u);
    EXPECT_NEAR(factors(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(factors(2, 0), 3.0, 1e-9);

    expect_ok(interp, "p = numthy_factor(17)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 17.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_divisors) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_divisors(n)");
    expect_contains(interp, "help", "numthy_num_divisors(n)");
    expect_contains(interp, "help", "numthy_sum_divisors(n)");

    expect_ok(interp, "d = numthy_divisors(12)");
    ASSERT_GT(interp.state().matrices.count("d"), 0u);
    const auto& divs = interp.state().matrices.at("d");
    EXPECT_EQ(divs.rows(), 6u);
    EXPECT_EQ(divs.cols(), 1u);
    const double expected[] = {1, 2, 3, 4, 6, 12};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(divs(i, 0), expected[i], 1e-9);
    }

    expect_ok(interp, "tau = numthy_num_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("tau"), 6.0, 1e-9);

    expect_ok(interp, "sigma = numthy_sum_divisors(12)");
    EXPECT_NEAR(interp.state().scalars.at("sigma"), 28.0, 1e-9);

    expect_ok(interp, "p = numthy_divisors(17)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("p")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("p")(1, 0), 17.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_factor_exp_farey_carmichael) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_factor_exp(n)");
    expect_contains(interp, "help", "numthy_farey(n)");
    expect_contains(interp, "help", "numthy_is_carmichael(n)");

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    ASSERT_GT(interp.state().matrices.count("fe"), 0u);
    const auto& factor_exp = interp.state().matrices.at("fe");
    EXPECT_EQ(factor_exp.rows(), 2u);
    EXPECT_EQ(factor_exp.cols(), 2u);
    EXPECT_NEAR(factor_exp(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(factor_exp(0, 1), 2.0, 1e-9);
    EXPECT_NEAR(factor_exp(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(factor_exp(1, 1), 1.0, 1e-9);

    expect_ok(interp, "f = numthy_farey(4)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    const auto& farey = interp.state().matrices.at("f");
    EXPECT_EQ(farey.rows(), 7u);
    EXPECT_EQ(farey.cols(), 2u);
    EXPECT_NEAR(farey(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(farey(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(farey(6, 0), 1.0, 1e-9);
    EXPECT_NEAR(farey(6, 1), 1.0, 1e-9);

    expect_contains(interp, "numthy_is_carmichael(561)", "1");
    expect_contains(interp, "numthy_is_carmichael(97)", "0");
}

TEST(ReplCommandsTest, numthy_stern_lucas_order_lambda) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_stern_brocot(n)");
    expect_contains(interp, "help", "numthy_lucas_sequence(k,P,Q)");
    expect_contains(interp, "help", "numthy_multiplicative_order(a,n)");
    expect_contains(interp, "help", "numthy_carmichael_lambda(n)");

    expect_ok(interp, "sb = numthy_stern_brocot(7)");
    ASSERT_GT(interp.state().matrices.count("sb"), 0u);
    const auto& stern = interp.state().matrices.at("sb");
    EXPECT_EQ(stern.rows(), 7u);
    EXPECT_EQ(stern.cols(), 2u);
    EXPECT_NEAR(stern(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(stern(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(stern(6, 0), 3.0, 1e-9);
    EXPECT_NEAR(stern(6, 1), 1.0, 1e-9);

    expect_ok(interp, "lu = numthy_lucas_sequence(5, 1, -1)");
    ASSERT_GT(interp.state().matrices.count("lu"), 0u);
    const auto& lucas = interp.state().matrices.at("lu");
    EXPECT_EQ(lucas.rows(), 1u);
    EXPECT_EQ(lucas.cols(), 2u);
    EXPECT_NEAR(lucas(0, 0), 5.0, 1e-9);
    EXPECT_NEAR(lucas(0, 1), 11.0, 1e-9);

    expect_ok(interp, "ord = numthy_multiplicative_order(3, 7)");
    EXPECT_NEAR(interp.state().scalars.at("ord"), 6.0, 1e-9);
    expect_contains(interp, "numthy_multiplicative_order(3, 7)", "6");

    expect_ok(interp, "lam = numthy_carmichael_lambda(15)");
    EXPECT_NEAR(interp.state().scalars.at("lam"), 4.0, 1e-9);
    expect_contains(interp, "numthy_carmichael_lambda(15)", "4");
}

TEST(ReplCommandsTest, numthy_pell_quadratic) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_pell_solve(D)");
    expect_contains(interp, "help", "numthy_quadratic_residues(p)");

    expect_ok(interp, "sol = numthy_pell_solve(61)");
    ASSERT_GT(interp.state().matrices.count("sol"), 0u);
    const auto& pell = interp.state().matrices.at("sol");
    EXPECT_EQ(pell.rows(), 1u);
    EXPECT_EQ(pell.cols(), 2u);
    const double x = pell(0, 0);
    const double y = pell(0, 1);
    EXPECT_NEAR(x, 1766319049.0, 1e-3);
    EXPECT_NEAR(y, 226153980.0, 1e-3);
    EXPECT_NEAR(x * x - 61.0 * y * y, 1.0, 1e6);

    expect_ok(interp, "qr = numthy_quadratic_residues(7)");
    ASSERT_GT(interp.state().matrices.count("qr"), 0u);
    const auto& residues = interp.state().matrices.at("qr");
    EXPECT_EQ(residues.rows(), 3u);
    EXPECT_EQ(residues.cols(), 1u);
    EXPECT_NEAR(residues(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(residues(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(residues(2, 0), 4.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_cornacchia(d,p)");

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(sol(0, 0) * sol(0, 0) + 1.0 * sol(0, 1) * sol(0, 1), 5.0, 1e-9);

    expect_ok(interp, "xy13 = numthy_cornacchia(1, 13)");
    ASSERT_GT(interp.state().matrices.count("xy13"), 0u);
    const auto& s13 = interp.state().matrices.at("xy13");
    EXPECT_EQ(s13.rows(), 1u);
    EXPECT_EQ(s13.cols(), 2u);
    EXPECT_NEAR(s13(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(s13(0, 1), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, combo_convergents) {
    Interpreter interp;

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);
}

TEST(ReplCommandsTest, reverse_numthy) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_2) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_2) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, combo_convergents_2) {
    Interpreter interp;

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);
}

TEST(ReplCommandsTest, reverse_numthy_2) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_2) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_3) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_3) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_3) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_4) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_4) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_2) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_2) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_4) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_5) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_5) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_3) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_3) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_5) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_6) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_6) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_4) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_4) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_6) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_7) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_7) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_5) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_5) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_7) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_8) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_8) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_6) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_6) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_8) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_9) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_9) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_7) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_7) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_9) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_10) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_10) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_8) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_8) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_10) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_11) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_11) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_9) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_9) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_11) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_12) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_12) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_10) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_10) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_12) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_13) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_13) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_11) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_11) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_13) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_14) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_14) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_12) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_12) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_14) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_15) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_15) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_13) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_13) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_15) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_16) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_16) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_14) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_14) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_16) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_17) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_17) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_15) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_15) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_34) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_17) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_18) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_18) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_35) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_16) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_16) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_36) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_18) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_19) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_19) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_37) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_17) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_17) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_38) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_19) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_20) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_20) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_39) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_18) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_18) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_40) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_20) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_21) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_21) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_41) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_19) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_19) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_42) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_21) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_22) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_22) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_43) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_20) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_20) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_44) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_22) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_23) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_23) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_45) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_66) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_21) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_21) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_46) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_67) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_68) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_23) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_24) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_24) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_47) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_69) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_22) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_22) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_48) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_70) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_71) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_24) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_25) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_25) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_49) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_72) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_23) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_23) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_50) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_73) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_74) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_25) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_26) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_26) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_51) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_75) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_24) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_24) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_52) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_76) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_77) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_26) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_27) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_27) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_53) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_78) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_25) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_25) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_54) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_79) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_80) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_27) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_28) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_28) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_55) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_81) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_26) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_26) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_56) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_82) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_83) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_28) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_29) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_29) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_57) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_84) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_27) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_27) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_58) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_85) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_86) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_29) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_30) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_30) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_59) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_87) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_28) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_28) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_60) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_88) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_89) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_30) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_31) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_31) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_61) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_90) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_29) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_29) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_62) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_91) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_92) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_31) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_32) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_32) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_63) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_93) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_30) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_30) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_64) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(12, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_94) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_95) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_cf_32) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 5)");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 5u);
}

TEST(ReplCommandsTest, numthy_primes_33) {
    Interpreter interp;

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    ASSERT_GT(interp.state().matrices.count("Pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pr").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("Pr")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_cornacchia_33) {
    Interpreter interp;

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    ASSERT_GT(interp.state().matrices.count("xy"), 0u);
    const auto& sol = interp.state().matrices.at("xy");
    EXPECT_EQ(sol.rows(), 1u);
    EXPECT_EQ(sol.cols(), 2u);
    EXPECT_NEAR(sol(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(sol(0, 1), 1.0, 1e-9);
}

TEST(ReplCommandsTest, numthy_gcd_scalar_65) {
    Interpreter interp;
    expect_ok(interp, "g = numthy_gcd(48, 18)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 6.0, 1e-8);
}

TEST(ReplCommandsTest, numthy_lcm_scalar_96) {
    Interpreter interp;
    expect_ok(interp, "l = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("l"), 12.0, 1e-8);
}

TEST(ReplCommandsTest, jordan_totient_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jt = numthy_jordan_totient(1, 6)");
    EXPECT_NEAR(interp.state().scalars.at("jt"), 2.0, 1e-8);
}

TEST(ReplCommandsTest, convergents_imcrop_31) {
    Interpreter interp;

    expect_ok(interp, "cf = numthy_continued_fraction(3.14159, 4)");
    expect_ok(interp, "cv = numthy_convergents(cf)");
    EXPECT_EQ(interp.state().matrices.at("cv").cols(), 2u);

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
}

TEST(ReplCommandsTest, poly_reverse_numthy_factor_exp_numthy_farey_31) {
    Interpreter interp;

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}
