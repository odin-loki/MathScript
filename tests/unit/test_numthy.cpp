#include "ms/numthy/numthy.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

using namespace ms::numthy;

TEST(NumthyPrimality, BasicPrimes) {
    EXPECT_FALSE(isprime(0));
    EXPECT_FALSE(isprime(1));
    EXPECT_TRUE(isprime(2));
    EXPECT_TRUE(isprime(3));
    EXPECT_FALSE(isprime(4));
    EXPECT_TRUE(isprime(97));
    EXPECT_FALSE(isprime(100));
    EXPECT_TRUE(isprime(7919));
    EXPECT_TRUE(isprime(104729));
}

TEST(NumthyPrimality, LargePrimes) {
    EXPECT_TRUE(isprime(1000000007ULL));
    EXPECT_TRUE(isprime(1000000009ULL));
    EXPECT_FALSE(isprime(1000000008ULL));
}

TEST(NumthyPrimality, NextPrevPrime) {
    EXPECT_EQ(nextprime(10), 11u);
    EXPECT_EQ(nextprime(11), 13u);
    EXPECT_EQ(prevprime(10), 7u);
    EXPECT_EQ(prevprime(3), 2u);
    EXPECT_EQ(prevprime(2), 0u);
}

TEST(NumthySieve, PrimesInRange) {
    auto ps = primes(1, 20);
    std::vector<uint64_t> expected = {2, 3, 5, 7, 11, 13, 17, 19};
    EXPECT_EQ(ps, expected);
}

TEST(NumthySieve, PrimePi) {
    EXPECT_EQ(prime_pi(10), 4u);
    EXPECT_EQ(prime_pi(100), 25u);
}

TEST(NumthySieve, PrimeNth) {
    EXPECT_EQ(prime_nth(1), 2u);
    EXPECT_EQ(prime_nth(6), 13u);
    EXPECT_EQ(prime_nth(25), 97u);
}

TEST(NumthyFactor, SmallNumbers) {
    auto f = factor(12);
    EXPECT_EQ(f, (std::vector<uint64_t>{2, 2, 3}));
    f = factor(30);
    EXPECT_EQ(f, (std::vector<uint64_t>{2, 3, 5}));
    f = factor(1);
    EXPECT_TRUE(f.empty());
    f = factor(7);
    EXPECT_EQ(f, (std::vector<uint64_t>{7}));
}

TEST(NumthyFactor, FactorExp) {
    auto fe = factor_exp(12);
    EXPECT_EQ(fe.size(), 2u);
    EXPECT_EQ(fe[0].first, 2u); EXPECT_EQ(fe[0].second, 2);
    EXPECT_EQ(fe[1].first, 3u); EXPECT_EQ(fe[1].second, 1);
}

TEST(NumthyDivisors, Basic) {
    auto d = divisors(12);
    EXPECT_EQ(d, (std::vector<uint64_t>{1, 2, 3, 4, 6, 12}));
    EXPECT_EQ(num_divisors(12), 6u);
    EXPECT_EQ(sum_divisors(12), 28u);
}

TEST(NumthyMult, EulerPhi) {
    EXPECT_EQ(euler_phi(1), 1u);
    EXPECT_EQ(euler_phi(7), 6u);   // prime
    EXPECT_EQ(euler_phi(12), 4u);  // phi(4*3) = phi(4)*phi(3) = 2*2
    EXPECT_EQ(euler_phi(36), 12u);
}

TEST(NumthyMult, Mobius) {
    EXPECT_EQ(mobius(1), 1);
    EXPECT_EQ(mobius(2), -1);
    EXPECT_EQ(mobius(6), 1);   // 2*3, squarefree, 2 factors
    EXPECT_EQ(mobius(4), 0);   // 2^2, not squarefree
}

TEST(NumthyMult, Liouville) {
    EXPECT_EQ(liouville(1), 1);
    EXPECT_EQ(liouville(6), 1);   // 2*3, 2 prime factors
    EXPECT_EQ(liouville(12), -1); // 2*2*3, 3 prime factors
}

TEST(NumthyArith, GcdLcm) {
    EXPECT_EQ(gcd(12, 8), 4u);
    EXPECT_EQ(lcm(4, 6), 12u);
    EXPECT_EQ(gcd(7, 13), 1u);
}

TEST(NumthyArith, ExtGcd) {
    auto [g, x, y] = extended_gcd(35, 15);
    EXPECT_EQ(g, 5);
    EXPECT_EQ(35 * x + 15 * y, 5);
}

TEST(NumthyArith, ModInv) {
    auto inv = mod_inv(3, 7);
    ASSERT_TRUE(inv.has_value());
    EXPECT_EQ((3 * inv.value()) % 7, 1u);
}

TEST(NumthyArith, ModPow) {
    EXPECT_EQ(mod_pow(2, 10, 1000), 24u);   // 1024 mod 1000
    // 3^12 ≡ 1 mod 13 (Fermat), 100 = 8*12 + 4, so 3^100 ≡ 3^4 = 81 ≡ 3 mod 13
    EXPECT_EQ(mod_pow(3, 100, 13), 3u);
    EXPECT_EQ(mod_pow(3, 12, 13), 1u);     // Fermat: 3^12 ≡ 1 mod 13
}

TEST(NumthyArith, CRT) {
    auto x = crt({2, 3, 2}, {3, 5, 7});
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x.value() % 3, 2u);
    EXPECT_EQ(x.value() % 5, 3u);
    EXPECT_EQ(x.value() % 7, 2u);
}

TEST(NumthyModular, LegendreSymbol) {
    EXPECT_EQ(legendre_symbol(2, 7), 1);   // 2 is QR mod 7 (3^2=9≡2)
    EXPECT_EQ(legendre_symbol(3, 7), -1);  // 3 is not QR mod 7
    EXPECT_EQ(legendre_symbol(0, 7), 0);
}

TEST(NumthyModular, JacobiSymbol) {
    EXPECT_EQ(jacobi_symbol(2, 7), 1);
    EXPECT_EQ(jacobi_symbol(3, 7), -1);
    EXPECT_EQ(jacobi_symbol(0, 7), 0);
}

TEST(NumthyModular, KroneckerSymbol) {
    EXPECT_EQ(kronecker_symbol(2, 7), 1);
    EXPECT_EQ(kronecker_symbol(3, 7), -1);
}

TEST(NumthyModular, TonelliShanks) {
    auto r = tonelli_shanks(2, 7);
    ASSERT_TRUE(r.has_value());
    uint64_t x = r.value();
    EXPECT_EQ((x * x) % 7, 2u);
}

TEST(NumthyModular, PrimitiveRoot) {
    EXPECT_TRUE(is_primitive_root(2, 11));
    EXPECT_FALSE(is_primitive_root(4, 11));
}

TEST(NumthyModular, PrimitiveRootValue) {
    EXPECT_EQ(primitive_root(7), 3u);
}

TEST(NumthyModular, DiscreteLog) {
    auto x = discrete_log(2, 8, 11);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x.value(), 3u);
    EXPECT_EQ(mod_pow(2, x.value(), 11), 8u);
}

TEST(NumthyCF, ContinuedFraction) {
    auto cf = continued_fraction(3.14159265358979, 5);
    EXPECT_EQ(cf[0], 3);  // [3; 7, 15, 1, ...]
    EXPECT_EQ(cf[1], 7);
}

TEST(NumthyCF, Convergents) {
    auto cf = continued_fraction(3.14159265358979, 4);
    auto conv = convergents(cf);
    EXPECT_FALSE(conv.empty());
    // 22/7 is a well-known convergent of π
    bool found = false;
    for (auto& [p, q] : conv)
        if (p == 22 && q == 7) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST(NumthyPartition, Small) {
    EXPECT_EQ(partition(0), 1u);
    EXPECT_EQ(partition(1), 1u);
    EXPECT_EQ(partition(4), 5u);
    EXPECT_EQ(partition(5), 7u);
    EXPECT_EQ(partition(10), 42u);
}

TEST(NumthyMult, JordanTotientMatchesEulerPhi) {
    for (uint64_t n = 1; n <= 500; ++n)
        EXPECT_EQ(jordan_totient(1, n), euler_phi(n)) << "n=" << n;
    EXPECT_EQ(jordan_totient(1, 720720), euler_phi(720720));
    EXPECT_EQ(jordan_totient(1, 1000003), euler_phi(1000003));
}

TEST(NumthyMult, JordanTotientK2) {
    EXPECT_EQ(jordan_totient(2, 1), 1u);
    EXPECT_EQ(jordan_totient(2, 6), 12u);   // 36*(1-1/2)*(1-1/3) = 12
    EXPECT_EQ(jordan_totient(2, 12), 48u);  // 144*(1-1/2)*(1-1/3) = 48
    EXPECT_EQ(jordan_totient(2, 7), 42u);   // 7^2 - 7 = 42
    EXPECT_EQ(jordan_totient(3, 6), 72u);   // 216*(1-1/2)*(1-1/3) = 72
}

TEST(NumthyMult, VonMangoldtPrimePowers) {
    EXPECT_DOUBLE_EQ(von_mangoldt(1), 0.0);
    EXPECT_NEAR(von_mangoldt(2), std::log(2.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(3), std::log(3.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(5), std::log(5.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(4), std::log(2.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(8), std::log(2.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(9), std::log(3.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(16), std::log(2.0), 1e-12);
    EXPECT_NEAR(von_mangoldt(27), std::log(3.0), 1e-12);
}

TEST(NumthyMult, VonMangoldtNonPrimePowers) {
    EXPECT_DOUBLE_EQ(von_mangoldt(6), 0.0);
    EXPECT_DOUBLE_EQ(von_mangoldt(10), 0.0);
    EXPECT_DOUBLE_EQ(von_mangoldt(12), 0.0);
    EXPECT_DOUBLE_EQ(von_mangoldt(15), 0.0);
    EXPECT_DOUBLE_EQ(von_mangoldt(30), 0.0);
}

TEST(NumthyModular, QuadraticResiduesMod7) {
    auto qr = quadratic_residues(7);
    EXPECT_EQ(qr, (std::vector<uint64_t>{1, 2, 4}));
    EXPECT_EQ(qr.size(), 3u);
}

TEST(NumthyModular, QuadraticResiduesCount) {
    for (uint64_t p : {3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u}) {
        auto qr = quadratic_residues(p);
        EXPECT_EQ(qr.size(), (p - 1) / 2) << "p=" << p;
    }
}

TEST(NumthyModular, QuadraticResiduesInvalid) {
    EXPECT_TRUE(quadratic_residues(2).empty());
    EXPECT_TRUE(quadratic_residues(4).empty());
    EXPECT_TRUE(quadratic_residues(15).empty());
}

TEST(NumthyFarey, Order4Exact) {
    auto f = farey(4);
    std::vector<std::pair<uint64_t,uint64_t>> expected = {
        {0, 1}, {1, 4}, {1, 3}, {1, 2}, {2, 3}, {3, 4}, {1, 1}
    };
    EXPECT_EQ(f, expected);
}

TEST(NumthyFarey, CountFormula) {
    for (uint32_t n = 1; n <= 20; ++n) {
        uint64_t expected = 1;
        for (uint32_t k = 1; k <= n; ++k)
            expected += euler_phi(k);
        EXPECT_EQ(farey(n).size(), expected) << "n=" << n;
    }
}

TEST(NumthyPell, FundamentalSolutions) {
    struct Case { uint64_t D; uint64_t x; uint64_t y; };
    for (auto [D, x, y] : {Case{2, 3, 2}, Case{3, 2, 1}, Case{5, 9, 4},
                          Case{7, 8, 3}, Case{13, 649, 180}}) {
        auto sol = pell_solve(D);
        ASSERT_TRUE(sol.has_value()) << "D=" << D;
        EXPECT_EQ(sol->first, x) << "D=" << D;
        EXPECT_EQ(sol->second, y) << "D=" << D;
        EXPECT_EQ(sol->first * sol->first - D * sol->second * sol->second, 1u);
    }
}

TEST(NumthyPell, PerfectSquareError) {
    EXPECT_FALSE(pell_solve(4).has_value());
    EXPECT_FALSE(pell_solve(9).has_value());
    EXPECT_FALSE(pell_solve(16).has_value());
}

TEST(NumthyPell, HarderCase) {
    auto sol = pell_solve(61);
    ASSERT_TRUE(sol.has_value());
    const uint64_t x = sol->first;
    const uint64_t y = sol->second;
    EXPECT_EQ(x, 1766319049u);
    EXPECT_EQ(y, 226153980u);
#if defined(__SIZEOF_INT128__)
    const __int128_t norm = static_cast<__int128_t>(x) * x
                          - static_cast<__int128_t>(61) * y * y;
    EXPECT_EQ(norm, 1);
#endif
}
