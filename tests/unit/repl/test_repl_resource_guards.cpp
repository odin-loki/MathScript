// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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
#include "ms/quantum/quantum.hpp"
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

TEST(ReplResourceGuards, QubitCountsAreBounded) {
    // A qubit count sets the dimension to 2^n, so an unvalidated n is an unbounded
    // allocation. quantum_qft_gate(24) asked for a 2^24 x 2^24 dense matrix and reached
    // 10 GB of resident memory before the OOM killer took the process -- found by fuzzing
    // the REPL, not by any hand-written case. `1 << n` is also undefined past 30.
    EXPECT_TRUE(ms::quantum::qft_gate(24).empty());
    EXPECT_TRUE(ms::quantum::qft_gate(13).empty());
    EXPECT_TRUE(ms::quantum::qft_gate(64).empty());
    EXPECT_TRUE(ms::quantum::qft_gate(0).empty());
    EXPECT_TRUE(ms::quantum::qft_gate(-1).empty());

    EXPECT_TRUE(ms::quantum::ghz_state(21).empty());
    EXPECT_TRUE(ms::quantum::w_state(31).empty());
    EXPECT_TRUE(ms::quantum::grover_search(13, {0}, 1).empty());

    // The usable range is unchanged. QFT on 2 qubits is 4x4 with every entry of modulus
    // 1/2, and its first row is all 1/2.
    const auto q2 = ms::quantum::qft_gate(2);
    ASSERT_EQ(q2.size(), 4u);
    ASSERT_EQ(q2[0].size(), 4u);
    for (const auto& row : q2) {
        for (const auto& z : row) {
            EXPECT_NEAR(std::abs(z), 0.5, 1e-12);
        }
    }
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(q2[0][k].real(), 0.5, 1e-12);
        EXPECT_NEAR(q2[0][k].imag(), 0.0, 1e-12);
    }

    // GHZ on 3 qubits is (|000> + |111>)/sqrt(2).
    const auto ghz = ms::quantum::ghz_state(3);
    ASSERT_EQ(ghz.size(), 8u);
    EXPECT_NEAR(ghz[0].real(), 1.0 / std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(ghz[7].real(), 1.0 / std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(ghz[3].real(), 0.0, 1e-12);

    // W on 3 qubits puts equal amplitude on the three one-hot basis states.
    const auto w = ms::quantum::w_state(3);
    ASSERT_EQ(w.size(), 8u);
    for (const int i : {1, 2, 4}) {
        EXPECT_NEAR(w[static_cast<std::size_t>(i)].real(), 1.0 / std::sqrt(3.0), 1e-12);
    }
    EXPECT_NEAR(w[0].real(), 0.0, 1e-12);

    Interpreter interp;
    EXPECT_TRUE(interp.execute("quantum_qft_gate(24)").has_value());
    EXPECT_TRUE(interp.execute("quantum_qft_gate(3)").has_value());
    EXPECT_TRUE(interp.execute("det([1, 2; 3, 4])").has_value()) << "session still usable";
}

// A tree ensemble's size arrived from the command line unbounded:
// ml_random_forest_fit(X, y, 3000000000) grew trees until the process was
// killed, and ml_isolation_forest_fit(X, 1e18) asked for an allocation that
// aborts rather than reports under -fno-exceptions.
TEST(ReplResourceGuards, TreeEnsembleSizesAreBounded) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("V4 = [1; 2; 3; 4]").has_value());
    ASSERT_TRUE(interp.execute("Y4 = [0; 1; 0; 1]").has_value());

    for (const char* cmd : {"m = ml_random_forest_fit(V4, Y4, 3000000000)",
                            "m = ml_adaboost_fit(V4, Y4, 3000000000)",
                            "m = ml_gradient_boosting_fit(V4, Y4, 3000000000)",
                            "m = ml_isolation_forest_fit(V4, 3000000000)",
                            "m = ml_isolation_forest_fit(V4, 1e18)",
                            "m = ml_decision_tree_fit(V4, Y4, 3000000000)"}) {
        EXPECT_FALSE(interp.execute(cmd).has_value()) << cmd;
    }

    // Ordinary sizes still fit.
    EXPECT_TRUE(interp.execute("ok = ml_random_forest_fit(V4, Y4, 4, 2)").has_value());
    EXPECT_TRUE(interp.execute("ok2 = ml_isolation_forest_fit(V4, 4, 2, 42)").has_value());
}
