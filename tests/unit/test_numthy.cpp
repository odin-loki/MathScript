#include "ms/numthy/numthy.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <set>

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

TEST(NumthyCornacchia, KnownSolutionsIdentity) {
    // Verify x^2 + d*y^2 == p and non-negativity, rather than hard-coding
    // "the" solution (either root ordering is a valid answer).
    struct Case { uint64_t d, p; };
    for (auto [d, p] : {Case{1, 5}, Case{1, 13}, Case{1, 2}, Case{3, 7},
                        Case{2, 11}, Case{4, 29}}) {
        auto sol = cornacchia(d, p);
        ASSERT_TRUE(sol.has_value()) << "d=" << d << " p=" << p;
        const uint64_t x = sol->first, y = sol->second;
        EXPECT_EQ(x * x + d * y * y, p) << "d=" << d << " p=" << p;
    }
}

TEST(NumthyCornacchia, HandVerifiedLiterals) {
    // Manually traced through the algorithm for these specific inputs.
    auto s1 = cornacchia(1, 5);
    ASSERT_TRUE(s1.has_value());
    EXPECT_EQ(s1->first, 2u);
    EXPECT_EQ(s1->second, 1u);

    auto s2 = cornacchia(1, 13);
    ASSERT_TRUE(s2.has_value());
    EXPECT_EQ(s2->first, 3u);
    EXPECT_EQ(s2->second, 2u);

    auto s3 = cornacchia(1, 2);
    ASSERT_TRUE(s3.has_value());
    EXPECT_EQ(s3->first, 1u);
    EXPECT_EQ(s3->second, 1u);

    auto s4 = cornacchia(3, 7);
    ASSERT_TRUE(s4.has_value());
    EXPECT_EQ(s4->first, 2u);
    EXPECT_EQ(s4->second, 1u);
}

TEST(NumthyCornacchia, NonPrimeError) {
    EXPECT_FALSE(cornacchia(1, 4).has_value());
    EXPECT_FALSE(cornacchia(1, 15).has_value());
    EXPECT_FALSE(cornacchia(1, 1).has_value());
}

TEST(NumthyCornacchia, ZeroDError) {
    EXPECT_FALSE(cornacchia(0, 5).has_value());
    EXPECT_FALSE(cornacchia(0, 13).has_value());
}

TEST(NumthyCornacchia, DTooLargeError) {
    EXPECT_FALSE(cornacchia(5, 5).has_value());   // d == p
    EXPECT_FALSE(cornacchia(7, 5).has_value());   // d > p
}

TEST(NumthyCornacchia, NoSolutionCase) {
    // Sum of two squares theorem: p = x^2+y^2 solvable iff p == 2 or p ≡ 1 (mod 4).
    // 7 ≡ 3 (mod 4), so no representation exists for d=1.
    EXPECT_FALSE(cornacchia(1, 7).has_value());
    EXPECT_FALSE(cornacchia(1, 11).has_value());
    EXPECT_FALSE(cornacchia(1, 19).has_value());
}

TEST(NumthyCornacchia, ValidSolutionsAcrossPrimesCongruentTo1Mod4) {
    for (uint64_t p : {5u, 13u, 17u, 29u, 37u, 41u}) {
        auto sol = cornacchia(1, p);
        ASSERT_TRUE(sol.has_value()) << "p=" << p;
        EXPECT_EQ(sol->first * sol->first + sol->second * sol->second, p) << "p=" << p;
    }
}

TEST(NumthySternBrocot, EmptyForZero) {
    EXPECT_TRUE(stern_brocot(0).empty());
}

TEST(NumthySternBrocot, RootOnly) {
    auto sb = stern_brocot(1);
    ASSERT_EQ(sb.size(), 1u);
    EXPECT_EQ(sb[0], (std::pair<uint64_t,uint64_t>{1, 1}));
}

TEST(NumthySternBrocot, FirstElementIsAlwaysRoot) {
    for (uint64_t n : {1u, 2u, 3u, 5u, 10u, 20u}) {
        auto sb = stern_brocot(n);
        ASSERT_FALSE(sb.empty());
        EXPECT_EQ(sb[0], (std::pair<uint64_t,uint64_t>{1, 1}));
    }
}

TEST(NumthySternBrocot, BfsOrderSmall) {
    // Root's children: left = 1/2, right = 2/1. Their children come next in BFS order.
    std::vector<std::pair<uint64_t,uint64_t>> expected = {
        {1, 1}, {1, 2}, {2, 1}, {1, 3}, {3, 2}, {2, 3}, {3, 1}
    };
    EXPECT_EQ(stern_brocot(7), expected);
}

TEST(NumthySternBrocot, CountMatchesN) {
    for (uint64_t n : {0u, 1u, 2u, 3u, 7u, 15u, 31u}) {
        EXPECT_EQ(stern_brocot(n).size(), n);
    }
}

TEST(NumthySternBrocot, AllReducedForm) {
    auto sb = stern_brocot(50);
    for (auto& [p, q] : sb)
        EXPECT_EQ(gcd(p, q), 1u) << p << "/" << q;
}

TEST(NumthySternBrocot, AllDistinct) {
    auto sb = stern_brocot(50);
    std::set<std::pair<uint64_t,uint64_t>> unique(sb.begin(), sb.end());
    EXPECT_EQ(unique.size(), sb.size());
}

TEST(NumthyCarmichael, KnownCarmichaelNumbers) {
    EXPECT_TRUE(is_carmichael(561));   // 3*11*17
    EXPECT_TRUE(is_carmichael(1105));  // 5*13*17
    EXPECT_TRUE(is_carmichael(1729));  // 7*13*19
    EXPECT_TRUE(is_carmichael(2465));  // 5*17*29
    EXPECT_TRUE(is_carmichael(2821));  // 7*13*31
}

TEST(NumthyCarmichael, KnownPrimesAreNotCarmichael) {
    EXPECT_FALSE(is_carmichael(7));
    EXPECT_FALSE(is_carmichael(97));
    // Individual prime factors of 561 must not themselves be Carmichael numbers.
    EXPECT_FALSE(is_carmichael(3));
    EXPECT_FALSE(is_carmichael(11));
    EXPECT_FALSE(is_carmichael(17));
}

TEST(NumthyCarmichael, OrdinaryCompositesAreNotCarmichael) {
    for (uint64_t n : {4u, 6u, 8u, 9u, 15u, 100u}) {
        EXPECT_FALSE(is_carmichael(n)) << "n=" << n;
    }
}

TEST(NumthyCarmichael, NonSquarefreeCompositeFails) {
    EXPECT_FALSE(is_carmichael(12)); // 2^2 * 3, not squarefree
    EXPECT_FALSE(is_carmichael(18)); // 2 * 3^2, not squarefree
}

TEST(NumthyCarmichael, SmallInputsBelowThree) {
    EXPECT_FALSE(is_carmichael(0));
    EXPECT_FALSE(is_carmichael(1));
    EXPECT_FALSE(is_carmichael(2));
}

TEST(NumthyCarmichael, LargerKnownCarmichaelNumber) {
    EXPECT_TRUE(is_carmichael(6601));   // 7*23*41
    EXPECT_TRUE(is_carmichael(10585));  // 5*29*73
}

TEST(NumthyCarmichael, KorseltCriterionHoldsForEachFactor) {
    for (uint64_t n : {561u, 1105u, 1729u}) {
        ASSERT_TRUE(is_carmichael(n)) << "n=" << n;
        for (auto& [p, e] : factor_exp(n)) {
            EXPECT_EQ(e, 1) << "n=" << n << " p=" << p;
            EXPECT_EQ((n - 1) % (p - 1), 0u) << "n=" << n << " p=" << p;
        }
    }
}

TEST(NumthyLucas, FibonacciCase) {
    // P=1, Q=-1 gives U_k = Fibonacci numbers.
    std::vector<int64_t> fib = {0, 1, 1, 2, 3, 5, 8, 13};
    for (size_t k = 0; k < fib.size(); ++k) {
        auto [u, v] = lucas_sequence(static_cast<int64_t>(k), 1, -1);
        EXPECT_EQ(u, fib[k]) << "k=" << k;
    }
}

TEST(NumthyLucas, LucasNumberCase) {
    // P=1, Q=-1 gives V_k = Lucas numbers.
    std::vector<int64_t> lucas = {2, 1, 3, 4, 7, 11, 18};
    for (size_t k = 0; k < lucas.size(); ++k) {
        auto [u, v] = lucas_sequence(static_cast<int64_t>(k), 1, -1);
        EXPECT_EQ(v, lucas[k]) << "k=" << k;
    }
}

TEST(NumthyLucas, BaseCasesAnyPQ) {
    for (auto [P, Q] : std::vector<std::pair<int64_t,int64_t>>{{1, -1}, {2, -1}, {3, 2}, {-2, 5}}) {
        auto [u0, v0] = lucas_sequence(0, P, Q);
        EXPECT_EQ(u0, 0) << "P=" << P << " Q=" << Q;
        EXPECT_EQ(v0, 2) << "P=" << P << " Q=" << Q;
        auto [u1, v1] = lucas_sequence(1, P, Q);
        EXPECT_EQ(u1, 1) << "P=" << P << " Q=" << Q;
        EXPECT_EQ(v1, P) << "P=" << P << " Q=" << Q;
    }
}

TEST(NumthyLucas, RecurrenceIdentityHoldsPellCase) {
    // P=2, Q=-1: Pell-number-related sequence. Cross-check against a naive
    // linear recurrence computed independently in the test.
    const int64_t P = 2, Q = -1;
    std::vector<int64_t> u_naive = {0, 1}, v_naive = {2, P};
    for (int k = 2; k <= 15; ++k) {
        u_naive.push_back(P * u_naive[k - 1] - Q * u_naive[k - 2]);
        v_naive.push_back(P * v_naive[k - 1] - Q * v_naive[k - 2]);
    }
    for (int k = 0; k <= 15; ++k) {
        auto [u, v] = lucas_sequence(k, P, Q);
        EXPECT_EQ(u, u_naive[static_cast<size_t>(k)]) << "k=" << k;
        EXPECT_EQ(v, v_naive[static_cast<size_t>(k)]) << "k=" << k;
    }
}

TEST(NumthyLucas, RecurrenceIdentityHoldsGeneralCase) {
    // P=3, Q=2: arbitrary pair, cross-checked against a naive reference recurrence.
    const int64_t P = 3, Q = 2;
    std::vector<int64_t> u_naive = {0, 1}, v_naive = {2, P};
    for (int k = 2; k <= 12; ++k) {
        u_naive.push_back(P * u_naive[k - 1] - Q * u_naive[k - 2]);
        v_naive.push_back(P * v_naive[k - 1] - Q * v_naive[k - 2]);
    }
    for (int k = 0; k <= 12; ++k) {
        auto [u, v] = lucas_sequence(k, P, Q);
        EXPECT_EQ(u, u_naive[static_cast<size_t>(k)]) << "k=" << k;
        EXPECT_EQ(v, v_naive[static_cast<size_t>(k)]) << "k=" << k;
    }
}

TEST(NumthyLucas, LargerKMatchesNaiveIteration) {
    // Exercise the O(log k) path (k=30, k=40) against an independently computed
    // naive iterative reference, using small P,Q to avoid overflow.
    for (int64_t kmax : {30, 40}) {
        const int64_t P = 1, Q = -1;
        std::vector<int64_t> u_naive = {0, 1}, v_naive = {2, P};
        for (int64_t k = 2; k <= kmax; ++k) {
            u_naive.push_back(P * u_naive[static_cast<size_t>(k - 1)] - Q * u_naive[static_cast<size_t>(k - 2)]);
            v_naive.push_back(P * v_naive[static_cast<size_t>(k - 1)] - Q * v_naive[static_cast<size_t>(k - 2)]);
        }
        auto [u, v] = lucas_sequence(kmax, P, Q);
        EXPECT_EQ(u, u_naive[static_cast<size_t>(kmax)]) << "kmax=" << kmax;
        EXPECT_EQ(v, v_naive[static_cast<size_t>(kmax)]) << "kmax=" << kmax;
    }
}

TEST(NumthyLucas, NegativeKClampedToZero) {
    auto [u, v] = lucas_sequence(-5, 1, -1);
    EXPECT_EQ(u, 0);
    EXPECT_EQ(v, 2);
}
