// Regressions for the crashes an oversized argument used to cause.
//
// A sweep that calls every REPL builtin with 3e9 and 1e18 in each argument position found
// one out-of-bounds read, five std::bad_alloc / std::length_error aborts (this library is
// built with -fno-exceptions, so an allocation failure calls std::terminate), and several
// unbounded computations. The counting functions were also returning silently wrapped
// values well before that, since their results stop fitting in uint64_t in the twenties.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "ms/combo/combo.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/stats/stats.hpp"

using namespace ms::interp;

TEST(ReplResourceGuards, PercentileOutsideZeroToOneHundred) {
    // idx = (p/100)*(n-1) was cast to size_t without a clamp: p = 3e9 indexed about 3e7
    // elements past the end and segfaulted, and a negative p is undefined as a size_t.
    const std::vector<double> v{4.0, 1.0, 3.0, 2.0};

    EXPECT_DOUBLE_EQ(ms::percentile(v, 3e9), 4.0) << "p above 100 gives the maximum";
    EXPECT_DOUBLE_EQ(ms::percentile(v, 100.0), 4.0);
    EXPECT_DOUBLE_EQ(ms::percentile(v, 1e18), 4.0);
    EXPECT_DOUBLE_EQ(ms::percentile(v, -3e9), 1.0) << "p below 0 gives the minimum";
    EXPECT_DOUBLE_EQ(ms::percentile(v, -1.0), 1.0);
    EXPECT_DOUBLE_EQ(ms::percentile(v, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(ms::percentile(v, std::numeric_limits<double>::quiet_NaN()), 1.0);

    // The ordinary range is unchanged: nearest-rank on the sorted sample.
    EXPECT_DOUBLE_EQ(ms::percentile(v, 50.0), 2.0);
    EXPECT_DOUBLE_EQ(ms::percentile(v, 75.0), 3.0);

    // trimmed_mean had the same unclamped conversion.
    EXPECT_TRUE(std::isfinite(ms::trimmed_mean(v, -1.0)));
    EXPECT_TRUE(std::isfinite(ms::trimmed_mean(v, 3e9)));
    EXPECT_DOUBLE_EQ(ms::trimmed_mean(v, 0.0), 2.5);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("V = [4; 1; 3; 2]").has_value());
    EXPECT_TRUE(interp.execute("stats_percentile(V, 3000000000)").has_value());
    EXPECT_TRUE(interp.execute("stats_percentile(V, -3000000000)").has_value());
    EXPECT_TRUE(interp.execute("stats_trimmed_mean(V, -1)").has_value());
}

TEST(ReplResourceGuards, CountingFunctionsReportOverflowInsteadOfWrapping) {
    using namespace ms::combo;
    // factorial already used UINT64_MAX as its overflow sentinel; its siblings now do too,
    // which also bounds the O(n) and O(n^2) recurrences behind them. bell_num(3e9) built a
    // Bell triangle with three billion rows and never returned.
    EXPECT_EQ(factorial(21), UINT64_MAX);
    EXPECT_EQ(subfactorial(21), UINT64_MAX);
    EXPECT_EQ(double_factorial(34), UINT64_MAX);
    EXPECT_EQ(catalan_num(37), UINT64_MAX);
    EXPECT_EQ(stirling1(22, 3), UINT64_MAX);
    EXPECT_EQ(stirling2(27, 3), UINT64_MAX);
    EXPECT_EQ(eulerian_number(22, 3), UINT64_MAX);
    EXPECT_EQ(bell_num(26), UINT64_MAX);
    EXPECT_EQ(motzkin_num(46), UINT64_MAX);
    EXPECT_EQ(involutions(32), UINT64_MAX);

    // The last representable value of each is still computed, so the limits are not off
    // by one in the conservative direction.
    EXPECT_NE(factorial(20), UINT64_MAX);
    EXPECT_NE(subfactorial(20), UINT64_MAX);
    EXPECT_NE(double_factorial(33), UINT64_MAX);
    EXPECT_NE(catalan_num(36), UINT64_MAX);
    EXPECT_NE(stirling1(21, 3), UINT64_MAX);
    EXPECT_NE(stirling2(26, 3), UINT64_MAX);
    EXPECT_NE(eulerian_number(21, 3), UINT64_MAX);
    EXPECT_NE(bell_num(25), UINT64_MAX);
    EXPECT_NE(motzkin_num(45), UINT64_MAX);
    EXPECT_NE(involutions(31), UINT64_MAX);

    // Known small values are unchanged.
    EXPECT_EQ(factorial(10), 3628800u);
    EXPECT_EQ(catalan_num(5), 42u);
    EXPECT_EQ(bell_num(5), 52u);
    EXPECT_EQ(motzkin_num(6), 51u);
    EXPECT_EQ(involutions(4), 10u);
    EXPECT_EQ(subfactorial(4), 9u);
    EXPECT_EQ(double_factorial(7), 105u);
    EXPECT_EQ(stirling2(4, 2), 7u);
    EXPECT_EQ(stirling1(4, 2), 11u);

    // And the REPL forms return rather than hanging.
    Interpreter interp;
    for (const char* cmd : {"combo_bell(3000000000)", "combo_bell_num(1e18)",
                            "combo_motzkin(3000000000)", "combo_subfactorial(1e18)"}) {
        EXPECT_TRUE(interp.execute(cmd).has_value()) << cmd;
    }
}

TEST(ReplResourceGuards, SieveAndPartitionRefuseUnsatisfiableSizes) {
    // primes(2, 1e18) asked for a 1e18-bit array and aborted on the allocation failure;
    // partition(3e9) asked for a 24 GB table.
    EXPECT_TRUE(ms::numthy::primes(2, 1'000'000'000'000ull).empty());
    EXPECT_EQ(ms::numthy::prime_pi(1'000'000'000'000ull), UINT64_MAX);
    EXPECT_EQ(ms::numthy::partition(417u), UINT64_MAX);

    // Ordinary ranges are unchanged.
    const auto small = ms::numthy::primes(2, 30);
    ASSERT_EQ(small.size(), 10u);
    EXPECT_EQ(small.front(), 2u);
    EXPECT_EQ(small.back(), 29u);
    EXPECT_EQ(ms::numthy::prime_pi(100), 25u);
    EXPECT_EQ(ms::numthy::partition(0u), 1u);
    EXPECT_EQ(ms::numthy::partition(5u), 7u);
    EXPECT_NE(ms::numthy::partition(416u), UINT64_MAX);

    Interpreter interp;
    EXPECT_TRUE(interp.execute("numthy_partition(3000000000)").has_value());
    EXPECT_TRUE(interp.execute("numthy_prime_pi(1e18)").has_value());
}

TEST(ReplResourceGuards, RandomBytesAndTensorRankAreBounded) {
    Interpreter interp;
    // The result is printed as hex, so a huge count is an allocation with nowhere to go.
    EXPECT_FALSE(interp.execute("crypto_random_bytes(3000000000)").has_value());
    EXPECT_FALSE(interp.execute("crypto_random_bytes(1e18)").has_value());
    const auto ok = interp.execute("crypto_random_bytes(16)");
    ASSERT_TRUE(ok.has_value());
    EXPECT_EQ(ok->size(), 33u) << "32 hex characters and a newline";

    // Each CP factor matrix is (dimension x rank), so an unbounded rank is an unbounded
    // allocation; it threw std::length_error, which aborts under -fno-exceptions.
    ASSERT_TRUE(interp.execute("T = [1, 2; 3, 4]").has_value());
    EXPECT_FALSE(interp.execute("tensorops_decompose_cp(cp1, T, 3000000000)").has_value());
    EXPECT_TRUE(interp.execute("tensorops_decompose_cp(cp2, T, 2)").has_value());
}
