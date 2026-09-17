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
#include <string>
#include <vector>

#include "ms/combo/combo.hpp"
#include "ms/error/error_types.hpp"
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

    // And the REPL forms return rather than hanging. They used to return the sentinel
    // itself, printed as 18446744073709551615; now they say what it means. Either way
    // the point of this assertion is that the call comes back at all.
    Interpreter interp;
    for (const char* cmd : {"combo_bell(3000000000)", "combo_bell_num(1e18)",
                            "combo_motzkin(3000000000)", "combo_subfactorial(1e18)"}) {
        const auto result = interp.execute(cmd);
        ASSERT_FALSE(result.has_value()) << cmd;
        const std::string message = ms::format_error(result.error());
        EXPECT_NE(message.find("does not fit in 64 bits"), std::string::npos)
            << cmd << " error: " << message;
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

    // The REPL forms return rather than hanging, and now say what the marker means
    // instead of printing it as 18446744073709551615.
    //
    // The two sentinels look identical from the outside and mean different things, so
    // they say different things. p(3000000000) really is past 64 bits -- p(417) already
    // is. pi(1e18) is not: it is about 2.4e16 and fits with three digits to spare, and
    // what stops it is the span a sieve can hold. Reporting that as "does not fit in 64
    // bits" was a true-sounding sentence about the wrong quantity.
    Interpreter interp;
    const auto overflowed = interp.execute("numthy_partition(3000000000)");
    ASSERT_FALSE(overflowed.has_value());
    EXPECT_NE(ms::format_error(overflowed.error()).find("does not fit in 64 bits"),
              std::string::npos)
        << ms::format_error(overflowed.error());

    const auto unsievable = interp.execute("numthy_prime_pi(1e18)");
    ASSERT_FALSE(unsievable.has_value());
    EXPECT_NE(ms::format_error(unsievable.error()).find("past what this can sieve"),
              std::string::npos)
        << ms::format_error(unsievable.error());
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

TEST(ReplResourceGuards, SessionObjectDimensionsAreBounded) {
    // Found by the 24-hour fuzz marathon, and only once its corpus was actually
    // being loaded: `cellmemory_new(cm, 2, 4555555555555555, [0.1,31, 107])`
    // asked `new[]` for 36 petabytes and ended the process. Both constructors
    // checked that their dimensions were positive integers and stopped there.
    //
    // `CellMemory` allocates two `Matrix<double>(memory_dim, 1)`; `DifModel`
    // allocates `weights_(output_dim, input_dim)`, so what has to fit there is
    // the PRODUCT -- 1000 by 1000 is a million elements and is refused, which
    // two independent caps of 262144 would have let through.
    Interpreter interp;

    for (const char* cmd : {"cellmemory_new(a, 2, 4555555555555555, [0.1, 31, 107])",
                            "cellmemory_new(b, 4555555555555555, 2, [1])",
                            "cellmemory_new(c, 2, 1e18)",
                            "difmodel_new(d, 4555555555555555, 2, 4, 0.1)",
                            "difmodel_new(e, 2, 4555555555555555, 4, 0.1)",
                            "difmodel_new(f, 1000, 1000, 4, 0.1)",
                            "difmodel_new(g, 2, 2, 1e18, 0.1)"}) {
        EXPECT_FALSE(interp.execute(cmd).has_value()) << cmd;
    }

    // The sizes anybody would actually ask for still work, and the object that
    // comes back remembers what it was given.
    ASSERT_TRUE(interp.execute("cellmemory_new(ok, 2, 8, [1])").has_value());
    EXPECT_TRUE(interp.execute("difmodel_new(okd, 3, 2, 4, 0.1)").has_value());
    const auto dim = interp.execute("cellmemory_memory_dim(ok)");
    ASSERT_TRUE(dim.has_value());
    EXPECT_NE(dim->find('8'), std::string::npos) << *dim;
}

TEST(ReplResourceGuards, TensorRankVectorsAreBoundedToIntRange) {
    // Found by the fuzz marathon, second crash of the same rehearsal:
    // `tensorops_decompose_hosvd(jk2, [1, 0; 0, 1], [1, 1555...555])` -- 39 digits
    // -- ended the process with a `std::length_error` escaping a library built
    // with `-fno-exceptions`.
    //
    // The handler DID guard the rank: `rank > tensor->shape[mode]` rejects a rank
    // larger than the mode it truncates. What reached that guard was not the rank
    // that was written. `parse_bracket_int_vector_literal` converted with a bare
    // `static_cast<int>`, which is undefined behaviour for a double this far past
    // `int`'s range, and on x86-64 leaves INT_MIN behind. INT_MIN is not greater
    // than 2, so an upper-bound guard is exactly the shape of guard such a value
    // walks through -- and then `truncated_svd_left` asked for a
    // `std::vector<double>` of 2147483648 elements.
    //
    // The conversion is the bug, so the fix is in the parser and not in the three
    // handlers that call it. The message is pinned because "rejected" is not the
    // property under test: the old code rejected these too, by crashing.
    Interpreter interp;

    const auto hosvd = interp.execute(
        "tensorops_decompose_hosvd(h1, [1, 0; 0, 1], "
        "[1, 155555555555555555555555555555555555555])");
    ASSERT_FALSE(hosvd.has_value());
    EXPECT_NE(ms::format_error(hosvd.error()).find("vector element"), std::string::npos)
        << ms::format_error(hosvd.error());

    // The same parser feeds the other two, and `decompose_tucker` has the same
    // `truncated_svd_left` underneath it.
    for (const char* cmd :
         {"tensorops_decompose_tucker(h2, [1, 0; 0, 1], "
          "[1, 155555555555555555555555555555555555555], 10, 1e-6)",
          "tensorops_decompose_tt(h3, [1, 2, 3, 4, 5, 6, 7, 8], "
          "[2, 2, 155555555555555555555555555555555555555], 1e-9)",
          // Each dimension is now inside `int`, so the overflow moves to the
          // product: `numel` is a signed `long` and this is 1e21. The bound it
          // is held to is the one it has to satisfy anyway -- the product must
          // equal the matrix's element count, and that is capped well below this.
          //
          // Read this one for what it is. Deleting that bound does not make this
          // line fail: the wrapped product is 3875520019714212735, which is not 4
          // either, so the command is still refused and the only difference is
          // that the refusal came from undefined behaviour. UBSan sees it --
          // "signed integer overflow: 99999980000001 * 9999999 cannot be
          // represented in type 'long int'" -- and UBSan's default is to print and
          // return 0, so the sanitizer job went green over the top of it. This line
          // pins the behaviour; what pins the BOUND is
          // `UBSAN_OPTIONS: halt_on_error=1` on the sanitizer job in
          // .github/workflows/ci.yml, which is what makes that diagnostic a
          // failure. Delete the bound and this test fails there, not here.
          "tensorops_decompose_tt(h4, [1, 2; 3, 4], [9999999, 9999999, 9999999], 1e-9)"}) {
        EXPECT_FALSE(interp.execute(cmd).has_value()) << cmd;
    }

    // A rank vector anyone would actually write is untouched.
    EXPECT_TRUE(interp.execute("tensorops_decompose_hosvd(ok, [1, 0; 0, 1], [1, 1])").has_value());
    EXPECT_TRUE(
        interp.execute("tensorops_decompose_tt(okt, [1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2], 1e-9)")
            .has_value());
}

TEST(ReplResourceGuards, IntegerVectorEntriesAreBoundedToIntRange) {
    // The `tensorops_decompose_hosvd` crash was the loud member of a family, and
    // the rest of the family was quiet. Eleven more sites checked that a vector
    // entry was integral -- some also that it was non-negative -- and then wrote
    // `static_cast<int>(entry)`, which is undefined behaviour for a double past
    // `int`'s range. Nothing downstream was in a position to notice, so nothing
    // crashed. `combo_next_perm([0, 1555...555])` -- 39 digits -- PRINTED
    //
    //     perm =
    //       [-2147483648.000000]
    //       [0.000000]
    //
    // and `combo_rank_permutation` on the same input answered 0. That is the same
    // defect as the combo counting functions returning a wrapped value as though
    // it were the answer, reached by a different route: an answer was given, and
    // it was not an answer to the question asked.
    //
    // The messages are pinned because "refused" was never the problem -- these
    // did not refuse, they replied.
    Interpreter interp;

    const char* big = "155555555555555555555555555555555555555";
    for (const std::string cmd :
         {std::string("combo_next_perm([0, ") + big + "])",
          std::string("combo_prev_perm([") + big + ", 0])",
          std::string("combo_next_comb([0, ") + big + "], 5)",
          std::string("combo_prev_comb([") + big + ", 0], 5)",
          std::string("combo_rank_permutation([0, ") + big + "])",
          std::string("combo_rank_combination([0, ") + big + "], 5)",
          std::string("quantum_grover_search(3, [") + big + "])",
          std::string("info_lz_complexity([1, ") + big + ", 0])"}) {
        const auto result = interp.execute(cmd);
        ASSERT_FALSE(result.has_value()) << cmd << " => " << result.value_or("");
        EXPECT_NE(ms::format_error(result.error()).find("is too large"), std::string::npos)
            << cmd << " => " << ms::format_error(result.error());
    }

    // A position is long or short, so that one is bounded on its magnitude and
    // the sign has to keep working.
    ASSERT_TRUE(interp.execute("P = [1; 2; 3]").has_value());
    ASSERT_TRUE(interp.execute(std::string("Q = [1; ") + big + "; 0]").has_value());
    const auto backtest = interp.execute("run_backtest(P, Q, 1000)");
    ASSERT_FALSE(backtest.has_value());
    EXPECT_NE(ms::format_error(backtest.error()).find("position"), std::string::npos)
        << ms::format_error(backtest.error());
    ASSERT_TRUE(interp.execute("R = [1; -1; 0]").has_value());
    EXPECT_TRUE(interp.execute("run_backtest(P, R, 1000)").has_value());

    // And the vectors anyone would actually write are untouched. `next_perm` of
    // [0, 1, 2] is [0, 2, 1] and the rank of [2, 0, 1] is 4; a bound that changed
    // either of those would be a worse bug than the one it replaced.
    const auto perm = interp.execute("combo_next_perm([0, 1, 2])");
    ASSERT_TRUE(perm.has_value());
    EXPECT_NE(perm->find("2.000000"), std::string::npos) << *perm;
    const auto rank = interp.execute("combo_rank_permutation([2, 0, 1])");
    ASSERT_TRUE(rank.has_value());
    EXPECT_NE(rank->find("4"), std::string::npos) << *rank;
}

TEST(ReplResourceGuards, ContinuedFractionCoefficientsMustFitInInt64) {
    // The int64 member of the same family, and the one that looks unreachable
    // because the range is so wide. `matrix_to_int64_coeff_vector` guarded only
    // `std::floor(entry) != entry` and then wrote `static_cast<int64_t>(entry)`,
    // which is undefined for a double past 2^63.
    //
    // What makes this one worth a test of its own is that the code underneath is
    // careful: `numthy::convergents` runs its recurrence through `checked_mul`
    // and `checked_add` and stops the moment either overflows. So the bad value
    // was never going to be caught downstream -- it was the FIRST convergent,
    // h_0 = cf[0], reported before the recurrence starts.
    Interpreter interp;

    const auto bad = interp.execute(
        "numthy_convergents([155555555555555555555555555555555555555; 7])");
    ASSERT_FALSE(bad.has_value());
    EXPECT_NE(ms::format_error(bad.error()).find("does not fit in 64 bits"), std::string::npos)
        << ms::format_error(bad.error());

    // 2^63 itself is one past the end and has to go; 2^62 is an ordinary value.
    EXPECT_FALSE(interp.execute("numthy_convergents([9223372036854775808; 7])").has_value());
    EXPECT_TRUE(interp.execute("numthy_convergents([4611686018427387904; 7])").has_value());

    // And the coefficients anybody would actually write still give the answer.
    // [3; 7; 15; 1] is the start of pi's continued fraction, so the convergents
    // are 3/1, 22/7, 333/106, 355/113.
    const auto pi = interp.execute("numthy_convergents([3; 7; 15; 1])");
    ASSERT_TRUE(pi.has_value());
    for (const char* expected : {"22", "7", "333", "106", "355", "113"}) {
        EXPECT_NE(pi->find(expected), std::string::npos) << expected << " in:\n" << *pi;
    }
}
