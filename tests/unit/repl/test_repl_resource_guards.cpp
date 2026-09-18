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
#include <complex>
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
    //
    // Which of the two things it says depends on which bound the argument crosses
    // first, and that distinction is worth keeping. Below 2^32 the index converts
    // cleanly and what fails is the arithmetic, so the message is about the RESULT.
    // Above it the index itself has no `uint32_t` to become -- `static_cast` there is
    // undefined, and until 2026-09-17 it was performed anyway -- so the message is
    // about the INDEX. 1e18 used to report the result's overflow, which was true and
    // beside the point.
    Interpreter interp;
    for (const char* cmd : {"combo_bell(3000000000)", "combo_motzkin(3000000000)"}) {
        const auto result = interp.execute(cmd);
        ASSERT_FALSE(result.has_value()) << cmd;
        const std::string message = ms::format_error(result.error());
        EXPECT_NE(message.find("does not fit in 64 bits"), std::string::npos)
            << cmd << " error: " << message;
    }
    for (const char* cmd : {"combo_bell_num(1e18)", "combo_subfactorial(1e18)"}) {
        const auto result = interp.execute(cmd);
        ASSERT_FALSE(result.has_value()) << cmd;
        const std::string message = ms::format_error(result.error());
        EXPECT_NE(message.find("32-bit count"), std::string::npos)
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

TEST(ReplResourceGuards, DiscreteLogIsBoundedByItsBabyStepTable) {
    // The fuzz marathon's finding, and the first of these that was not a crash.
    // `numthy_discrete_log(0, 8, 015532559262904483840)` spent 1,625 seconds inside
    // one call before libFuzzer's own timeout ended the process. Nothing was wrong
    // with the arithmetic: baby-step giant-step fills a table of about sqrt(p)
    // entries, and sqrt(1.55e19) is 3.9e9 of them -- about 62 GB, so it would have
    // died of memory had it not died of time first.
    //
    // All three arguments were bounded at `kMaxU64AsDouble`, which is the largest
    // value that IS a uint64 and says nothing about whether the command can answer
    // for it. `p` is now bounded by what the table costs; `g` and `h` are not,
    // because they only matter modulo p and buy no work.
    //
    // The order below is deliberate. The first assertion uses a modulus just past
    // the cap, so reverting the bound fails it in about a third of a second and
    // returns; putting the 39-digit case first would instead make the mutant run
    // for hours.
    Interpreter interp;

    const auto over = interp.execute("numthy_discrete_log(2, 5, 200000000000)");
    ASSERT_FALSE(over.has_value());
    ASSERT_NE(ms::format_error(over.error()).find("bounded at"), std::string::npos)
        << ms::format_error(over.error());

    const auto huge = interp.execute("numthy_discrete_log(0, 8, 15532559262904483840)");
    ASSERT_FALSE(huge.has_value());
    EXPECT_NE(ms::format_error(huge.error()).find("sqrt(p)"), std::string::npos)
        << ms::format_error(huge.error());

    // A modulus anyone would actually use still answers, and answers correctly:
    // 2^292379 = 5 (mod 1000003), which the guard cannot fake.
    const auto ok = interp.execute("numthy_discrete_log(2, 5, 1000003)");
    ASSERT_TRUE(ok.has_value()) << ms::format_error(ok.error());
    EXPECT_NE(ok->find("292379"), std::string::npos) << *ok;
    const auto back = interp.execute("numthy_mod_pow(2, 292379, 1000003)");
    ASSERT_TRUE(back.has_value());
    EXPECT_NE(back->find("5"), std::string::npos) << *back;
}

TEST(ReplResourceGuards, AxiomEvolveIsBoundedByPopulationTimesGenerations) {
    // The marathon's second finding. A 34-digit population_size went through
    // `static_cast<size_t>`, which has nothing to convert at 8.16e33, and reached
    // `population_.resize` in `Axiom::Axiom`:
    //
    //     terminate called after throwing an instance of 'std::length_error'
    //       what():  vector::_M_default_append
    //
    // `axiom_evolve` is neither a matrix call nor a session-object constructor -- it
    // is a special case inside `execute_assignment` -- which is why the sweeps that
    // found the other fourteen went past it.
    //
    // The two arguments MULTIPLY: one random tree per individual at construction,
    // then the whole population once per generation. So this is a `WorkBudget` rather
    // than two independent caps, and the assertion below that matters most is the
    // second one -- a population of 100,000 is refused not on its own account but
    // because the DEFAULT 25 generations no longer fit beside it.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("D = [0, 1, 0; 1, 0, 1]").has_value());

    // First, and deliberately: with the budget reverted this one SUCCEEDS in about
    // half a second, so the mutant fails here and returns rather than going on to
    // the 34-digit case, which would abort the test binary instead.
    const auto product = interp.execute("p = axiom_evolve(D, 100000)");
    ASSERT_FALSE(product.has_value());
    ASSERT_NE(ms::format_error(product.error()).find("max_generations"), std::string::npos)
        << ms::format_error(product.error());

    const auto huge = interp.execute("q = axiom_evolve(D, 8155555555555555555555555555555550)");
    ASSERT_FALSE(huge.has_value());
    EXPECT_NE(ms::format_error(huge.error()).find("population_size"), std::string::npos)
        << ms::format_error(huge.error());

    // The defaults and the sizes anyone would write are untouched, and the command
    // still answers rather than merely declining to crash.
    const auto plain = interp.execute("a = axiom_evolve(D)");
    ASSERT_TRUE(plain.has_value()) << ms::format_error(plain.error());
    EXPECT_TRUE(interp.execute("b = axiom_evolve(D, 20, 25)").has_value());
    EXPECT_TRUE(interp.execute("c = axiom_evolve(D, 1000, 100)").has_value());
    EXPECT_GT(interp.state().scalars.count("a"), 0u);
}

TEST(ReplResourceGuards, FloatToIntegerConversionsAreBoundedBeforeTheCast) {
    // Twelve sites found by sweeping the REPL under `-fsanitize=float-cast-overflow`,
    // which is the point: GCC's `-fsanitize=undefined` does NOT include that check, so
    // the sanitizer job could not see any of them. Measured on the same file:
    //
    //   gcc-13 -fsanitize=undefined                    prints 0, silently
    //   gcc-13 -fsanitize=undefined,float-cast-overflow diagnostic, then 0
    //   clang  -fsanitize=undefined                     diagnostic, then 9223372036854775808
    //
    // Two of the twelve were answering rather than refusing, and those are the
    // assertions below that can fail on an ordinary build. The rest were "rejected by
    // accident" -- the cast came first, INT_MIN came out, and `n < 1` turned it down --
    // which is the right outcome by undefined means and reads exactly like a guard.
    Interpreter interp;
    const char* big = "8155555555555555555555555555555550";

    // `combo_factorial` ANSWERED 1. The conversion landed on 0 and 0! is 1. Both
    // dispatch paths had it: the assigned form and the bare form are separate code.
    const auto assigned = interp.execute(std::string("f = combo_factorial(") + big + ")");
    ASSERT_FALSE(assigned.has_value());
    EXPECT_NE(ms::format_error(assigned.error()).find("32-bit count"), std::string::npos)
        << ms::format_error(assigned.error());
    const auto bare = interp.execute(std::string("combo_factorial(") + big + ")");
    ASSERT_FALSE(bare.has_value()) << bare.value_or("");
    const auto partition = interp.execute(std::string("numthy_partition(") + big + ")");
    ASSERT_FALSE(partition.has_value()) << partition.value_or("");

    // The counting sequences anyone would ask for are untouched: 10! and p(10) = 42.
    const auto fact = interp.execute("combo_factorial(10)");
    ASSERT_TRUE(fact.has_value());
    EXPECT_NE(fact->find("3628800"), std::string::npos) << *fact;
    const auto p10 = interp.execute("numthy_partition(10)");
    ASSERT_TRUE(p10.has_value());
    EXPECT_NE(p10->find("42"), std::string::npos) << *p10;

    // `stats_vif` ANSWERED for column 0 when asked for a 34-digit column, and answered
    // 0 for any column the matrix does not have. An index has to name a column.
    ASSERT_TRUE(interp.execute("X = [1, 2; 3, 4; 5, 7]").has_value());
    const auto vif0 = interp.execute("stats_vif(X, 0)");
    ASSERT_TRUE(vif0.has_value());
    EXPECT_NE(vif0->find("76"), std::string::npos) << *vif0;
    for (const std::string cmd : {std::string("stats_vif(X, 5)"),
                                  std::string("stats_vif(X, ") + big + ")"}) {
        const auto out = interp.execute(cmd);
        ASSERT_FALSE(out.has_value()) << cmd << " => " << out.value_or("");
        EXPECT_NE(ms::format_error(out.error()).find("out of range"), std::string::npos)
            << ms::format_error(out.error());
    }

    // The signal factors were the "rejected by accident" kind, so these two lines pass
    // either way on x86-64 and are here for the boundary, not as a mutant check.
    EXPECT_FALSE(interp.execute("signal_downsample([1;2;3;4], 3000000000)").has_value());
    EXPECT_TRUE(interp.execute("signal_downsample([1;2;3;4], 2)").has_value());
    EXPECT_TRUE(interp.execute("signal_median_filter([1;5;2;8;3], 3)").has_value());
}

TEST(ReplResourceGuards, GriaEntropyBinsAreBoundedByTheHistogramTheyAllocate) {
    // Found by the REPL-only fuzz session in 57 seconds:
    //
    //     gria_entropy([,22,3], 66666666666666664)
    //     AddressSanitizer: requested allocation size 0x766c7d748355540
    //       #8 ms::gria::entropy(...)  gria.cpp:31
    //
    // `bins` is the LENGTH of the histogram `gria::entropy` allocates. It was checked
    // for sign and integrality -- which 6.7e16 passes, being a positive integer -- and
    // then cast to `size_t` and handed to `std::vector<double>(bins)`.
    //
    // Same family as `axiom_evolve` and `cellmemory_new`: a size argument reaching an
    // allocation with only its sign checked. `gria_entropy` is a special case inside
    // `execute_impl`, which is why the registry sweeps went past it.
    Interpreter interp;

    // First, and cheap: just past the cap. With the bound reverted this allocates
    // 2.4 MB and SUCCEEDS, so the mutant fails here in microseconds rather than
    // asking the allocator for 5e17 doubles.
    const auto over = interp.execute("gria_entropy([1,2,3], 300000)");
    ASSERT_FALSE(over.has_value()) << over.value_or("");
    ASSERT_NE(ms::format_error(over.error()).find("is too large"), std::string::npos)
        << ms::format_error(over.error());

    const auto huge = interp.execute("gria_entropy([,22,3], 66666666666666664)");
    EXPECT_FALSE(huge.has_value()) << huge.value_or("");

    // The boundary is inclusive, and the entropy is still the entropy: three distinct
    // values in three occupied bins is log2(3), whatever the bin count above them.
    const auto at_cap = interp.execute("gria_entropy([1,2,3], 262144)");
    ASSERT_TRUE(at_cap.has_value()) << ms::format_error(at_cap.error());
    EXPECT_NE(at_cap->find("1.58496"), std::string::npos) << *at_cap;

    // The documented example and the default both still answer.
    const auto four = interp.execute("gria_entropy([1,2,2,3,3,3], 4)");
    ASSERT_TRUE(four.has_value());
    EXPECT_NE(four->find("1.459148"), std::string::npos) << *four;
    EXPECT_TRUE(interp.execute("gria_entropy([1,2,2,3,3,3])").has_value());
}

TEST(ReplResourceGuards, RiccatiWeightsMustBeSizedAgainstTheInputsTheyWeigh) {
    // Found by the 24-hour REPL-only fuzz session, 99 minutes in:
    //
    //     control_lqr([-4,0;0,-3],[1;1],eye(2),[1<NUL...>,1;0,1<NUL...>])
    //     AddressSanitizer: heap-buffer-overflow, READ of size 8
    //       #0 ms::control::matmul(...)   control.cpp:56
    //       #1 ms::control::q_from_k(...) control.cpp:1101
    //       #2 ms::control::riccati(...)  control.cpp:1178
    //       #3 ms::control::lqr(...)      control.cpp:1269
    //     0x... is located 0 bytes after 8-byte region allocated by
    //       ms::control::transpose(...)  control.cpp:65
    //
    // A is 2x2 and B is 2x1, so there is ONE input and R has to be 1x1. R was given
    // as 2x2 and nothing checked it: `q_from_k` transposes the 1x2 gain into a 2x1,
    // then multiplies it by a 2x2 R, and `matmul` takes its inner extent from R --
    // so it reads two doubles out of rows that hold one. The REPL layer checked B's
    // rows and Q's size against A and left R alone, and the library re-derived the
    // input count from B[0] without ever comparing it to R.
    //
    // Valgrind on the release build: two invalid reads of size 8 before the guard,
    // none after. The release build does not fault -- it ANSWERS, with a 2x2 gain
    // whose second row is zeros, where a 2x2 A and a 2x1 B admit only a 1x2 gain.
    // That is why the shape is asserted below and not merely the survival: a "does
    // not crash" check passes against the defect.
    Interpreter interp;

    // Cheap, and the mutant check: with the guard reverted each of these succeeds
    // (over-reading as it goes) instead of being refused.
    const std::pair<std::string, std::string> rejected[] = {
        {"control_lqr([-4,0;0,-3],[1;1],[1,0;0,1],[1,0;0,1])", "per input"},
        {"control_riccati([-4,0;0,-3],[1;1],[1,0;0,1],[1,0;0,1])", "per input"},
        {"control_dare([0.5,0;0,0.5],[1;1],[1,0;0,1],[1,0;0,1])", "per input"},
        {"control_lqe([-4,0;0,-3],[1,0],[1,0;0,1],[1,0;0,1])", "per measurement"},
    };
    for (const auto& [cmd, want] : rejected) {
        const auto out = interp.execute(cmd);
        ASSERT_FALSE(out.has_value()) << cmd << " => " << out.value_or("");
        EXPECT_NE(ms::format_error(out.error()).find(want), std::string::npos)
            << cmd << " => " << ms::format_error(out.error());
    }

    // The fuzzer's own line, with the NULs it carried folded to spaces the way the
    // fuzz target folds them.
    EXPECT_FALSE(
        interp.execute("control_lqr([-4,0;0,-3],[1;1],eye(2),[1   ,1;0,1   ])").has_value());

    // One input, so the gain is 1x2 -- not the 2x2 the defect returned.
    ASSERT_TRUE(interp.execute("K = control_lqr([-4,0;0,-3],[1;1],[1,0;0,1],[2])").has_value());
    const auto k_rows = interp.execute("mat_rows(K)");
    ASSERT_TRUE(k_rows.has_value()) << ms::format_error(k_rows.error());
    EXPECT_NE(k_rows->find("1"), std::string::npos) << *k_rows;
    const auto k_cols = interp.execute("mat_cols(K)");
    ASSERT_TRUE(k_cols.has_value()) << ms::format_error(k_cols.error());
    EXPECT_NE(k_cols->find("2"), std::string::npos) << *k_cols;

    // One measurement, so the estimator gain is 2x1 -- the dual shape.
    ASSERT_TRUE(interp.execute("L = control_lqe([-4,0;0,-3],[1,0],[1,0;0,1],[2])").has_value());
    const auto l_rows = interp.execute("mat_rows(L)");
    ASSERT_TRUE(l_rows.has_value()) << ms::format_error(l_rows.error());
    EXPECT_NE(l_rows->find("2"), std::string::npos) << *l_rows;
    const auto l_cols = interp.execute("mat_cols(L)");
    ASSERT_TRUE(l_cols.has_value()) << ms::format_error(l_cols.error());
    EXPECT_NE(l_cols->find("1"), std::string::npos) << *l_cols;

    // Two inputs with a 2x2 R is the case the guard must not refuse, cross terms
    // included -- non-diagonal R is ordinary in LQR.
    EXPECT_TRUE(
        interp.execute("control_lqr([-4,0;0,-3],[1,0;0,1],[1,0;0,1],[2,0.5;0.5,2])").has_value());
    EXPECT_TRUE(
        interp.execute("control_dare([0.5,0;0,0.5],[1,0;0,1],[1,0;0,1],[2,0.5;0.5,2])")
            .has_value());
}

TEST(ReplResourceGuards, VectorOdeTrajectoriesAreBoundedByTheRowsTheyStore) {
    // Found by the 24-hour REPL-only fuzz session, 2h47m in:
    //
    //     ode_backward_euler_vec("-y0", 0, [1],45  ,447555554)
    //     libFuzzer: out-of-memory (used: 2062Mb; limit: 2048Mb)
    //       805306368 bytes (59%) in 1 allocation
    //         ms::ode_backward_euler_vec(...)  ode.cpp:1114
    //       268435456 bytes (19%) in 1 allocation
    //         ms::ode_backward_euler_vec(...)  ode.cpp:1113
    //       218358160 bytes in 27294770 allocations
    //         ms::ode_backward_euler_vec(...)  ode.cpp:1114
    //
    // `steps` is the ROW COUNT of the trajectory the solver accumulates -- one t and one
    // state vector per step. `checked_ode_trajectory_steps` already bounded it at twelve
    // call sites; `eval_ode_vec_fixed_step_call` was the thirteenth and had its own
    // inline check, which tested the sign and the integrality and stopped. 447555554 is
    // a non-negative integer, so it passed, and the four commands behind that helper --
    // ode_euler_vec, ode_rk4_vec, ode_backward_euler_vec, ode_adams_bashforth2_vec --
    // had no upper bound at all. The cast was unbounded too: `static_cast<int>` of a
    // double past INT_MAX is undefined and `steps_d != steps_i` rejected it by accident.
    //
    // The guard also now takes the width of one row instead of assuming 2. A vector
    // solver stores y0.size() + 1 doubles per step, so a fifty-component state was
    // bounded at twenty-five times the elements every other REPL allocation answers to.
    Interpreter interp;

    // Cheap, and the mutant check: with the guard reverted this is a 447-million-row
    // trajectory, which is the out-of-memory itself and not something to run in a test.
    // One step past the cap is refused for the same reason and returns immediately.
    const auto over = interp.execute("ode_euler_vec(\"-y0\", 0, [1], 1, 131072)");
    ASSERT_FALSE(over.has_value()) << over.value_or("");
    ASSERT_NE(ms::format_error(over.error()).find("is too large"), std::string::npos)
        << ms::format_error(over.error());

    // The row width is y0's, so a wider state gets fewer steps -- 262144/3 - 1.
    EXPECT_FALSE(interp.execute("ode_euler_vec(\"-y0;-y1\", 0, [1,2], 1, 87381)").has_value());
    EXPECT_TRUE(interp.execute("ode_euler_vec(\"-y0;-y1\", 0, [1,2], 1, 4)").has_value());

    // ode_verlet_vec stores t, q and v, so 2 * q0.size() + 1.
    EXPECT_FALSE(
        interp.execute("ode_verlet_vec(\"-q0\", 0, [1], [0], 1, 87381)").has_value());
    EXPECT_TRUE(interp.execute("ode_verlet_vec(\"-q0\", 0, [1], [0], 1, 4)").has_value());

    // All four commands behind the helper that had no bound, and the fuzzer's own line.
    for (const std::string fn : {std::string("ode_euler_vec"), std::string("ode_rk4_vec"),
                                 std::string("ode_backward_euler_vec"),
                                 std::string("ode_adams_bashforth2_vec")}) {
        const auto huge = interp.execute(fn + "(\"-y0\", 0, [1],45  ,447555554)");
        ASSERT_FALSE(huge.has_value()) << fn << " => " << huge.value_or("");
        EXPECT_NE(ms::format_error(huge.error()).find("is too large"), std::string::npos)
            << fn << " => " << ms::format_error(huge.error());

        // dy/dt = -y from y(0) = 1 over four steps of h = 0.25 still integrates.
        const auto four = interp.execute(fn + "(\"-y0\", 0, [1], 1, 4)");
        ASSERT_TRUE(four.has_value()) << fn << " => " << ms::format_error(four.error());
        EXPECT_NE(four->find("1.000000"), std::string::npos) << fn << " => " << *four;
    }

    // The scalar family keeps the default row width of 2, and its boundary is unchanged.
    EXPECT_TRUE(interp.execute("ode_euler(\"-y\", 0, 1, 1, 4)").has_value());
    EXPECT_FALSE(interp.execute("ode_euler(\"-y\", 0, 1, 1, 131072)").has_value());
}

TEST(ReplResourceGuards, AShortRfftSpectrumIsZeroPaddedNotReadPastItsEnd) {
    // Found by chunk 2 of the chained fuzz run, nine minutes in, from chunk 1's corpus:
    //
    //     fft_irfft([1,33], 5)
    //     AddressSanitizer: heap-buffer-overflow, READ of size 8
    //       #0 ms::hermitian_extend_rfft_spectrum(...) fft.cpp:232
    //       #1 ms::irfft_half_transform(...)           fft.cpp:245
    //       #2 ms::irfft(...)                          fft.cpp:570
    //       #3 ms::interp::detail::eval_fft_irfft(...) repl_engine_internal.cpp:8775
    //
    // The array being filled is the HALF spectrum, bins 0..full_len/2. The fill loop
    // used Hermitian symmetry -- bin i from bin full_len - i -- which for every index
    // in that array but the last names a bin in the UPPER half, which a half spectrum
    // does not store. n = 5 gives full_len = 8 and half = 4, so the array holds five
    // bins and the caller gave two; the loop then read indices 6 and 5 out of it. The
    // one index that was inside the array, i == half, read the element it was
    // assigning. So the loop was out of bounds for every i it did anything for, and
    // could only ever run when the spectrum was shorter than the transform length --
    // which is why `rfft` round trips never touched it.
    //
    // WHAT KILLS THE MUTANT, measured rather than assumed, because the obvious answers
    // do not:
    //
    //   valgrind, release build        3 invalid reads of size 8  -> none after
    //   clang-18 -fsanitize=address    heap-buffer-overflow       -> clean after
    //   gcc-13   -fsanitize=address    SILENT
    //
    // GCC's AddressSanitizer does not report it. Reduced to three lines --
    // `std::vector<std::complex<double>> v(5); v[5].real();` -- clang-18 says "0 bytes
    // after 80-byte region" at -O0 and -O1 and gcc-13 says nothing at either, while
    // both catch the same shape on a `vector<double>` and on `new double[10]`. It is
    // the 16-byte complex element GCC misses. CI's sanitizer job is GCC, so it is blind
    // to this class the way it is already blind to float-cast-overflow; the Clang fuzz
    // build is what found this.
    //
    // The value assertions below do detect the defect when this test runs in its suite,
    // which is how ctest and CI run it -- with the loop restored the whole binary fails
    // here, shuffled orders included. They do NOT detect it when the test is run on its
    // own with --gtest_filter: the bins read out of bounds then land on an untouched
    // zero page, which is exactly the value the fix supplies, so the defect and the fix
    // agree. Measured at n = 5 and at n = 1000, where the read goes 8 KB past the end,
    // and with 4096 same-size-class allocations deliberately dirtied and freed first --
    // in isolation the allocator still hands back a clean page. The defect is the read,
    // not the result, so the dependable detectors are the two above and not a value.
    Interpreter interp;

    const auto shortest = interp.execute("fft_irfft([1,33], 5)");
    ASSERT_TRUE(shortest.has_value()) << ms::format_error(shortest.error());

    // A caller who gives fewer bins than the transform length needs has said nothing
    // about the high-frequency bins, so they are zero -- numpy.fft.irfft's reading. The
    // proof is that naming them explicitly changes nothing.
    ASSERT_TRUE(interp.execute("padded = fft_irfft([36,0;-4,9.656854;0,0;0,0;0,0], 8)")
                    .has_value());
    ASSERT_TRUE(interp.execute("bare = fft_irfft([36,0;-4,9.656854], 8)").has_value());
    const auto padded = interp.execute("padded");
    const auto bare = interp.execute("bare");
    ASSERT_TRUE(padded.has_value());
    ASSERT_TRUE(bare.has_value());
    // Past the `name =` line, which is the only part that differs.
    ASSERT_NE(padded->find('\n'), std::string::npos) << *padded;
    ASSERT_NE(bare->find('\n'), std::string::npos) << *bare;
    EXPECT_EQ(padded->substr(padded->find('\n')), bare->substr(bare->find('\n')))
        << "a short spectrum must mean the zero-padded one\n"
        << *padded << *bare;

    // And the round trip, which the defect never reached, still returns the signal:
    // rfft gives half + 1 bins, so the loop had nothing to fill.
    ASSERT_TRUE(interp.execute("sig = [1;2;3;4;5]").has_value());
    ASSERT_TRUE(interp.execute("spec = fft_rfft(sig)").has_value());
    const auto back = interp.execute("fft_irfft(spec, 5)");
    ASSERT_TRUE(back.has_value()) << ms::format_error(back.error());
    for (const char* sample : {"1.000000", "2.000000", "3.000000", "4.000000", "5.000000"}) {
        EXPECT_NE(back->find(sample), std::string::npos) << sample << " in " << *back;
    }

    // A spectrum LONGER than the transform length is cropped, as it always was.
    EXPECT_TRUE(interp.execute("fft_irfft([1,0;2,0;3,0;4,0;5,0;6,0;7,0], 4)").has_value());
    EXPECT_TRUE(interp.execute("fft_irfft([1,0], 1)").has_value());
}

TEST(ReplResourceGuards, DividingAPolynomialByZeroIsRefusedRatherThanSpunOn) {
    // Found by chunk 1 of the first chained 24-hour run, 4h16m in, as a TIMEOUT rather
    // than a crash -- the first of the fifteen that was not a memory error:
    //
    //     poly_div_quot([0.1;],[0])
    //     ALARM: working on the last Unit for 1252 seconds
    //     ERROR: libFuzzer: timeout after 1252 seconds
    //       ms::poly::poly_div_quot(...)  poly.cpp:168
    //
    // `strip` removes trailing near-zero coefficients but never goes below one, so a
    // STRIPPED divisor can still be the zero polynomial: the one-element case whose
    // coefficient is zero. Long division then divided by that leading coefficient --
    // 0.1 / 0 = inf -- and subtracted inf * 0 from the remainder, giving NaN. The loop
    // ends when the remainder shrinks, and it shrinks when `|rem[i]| < 1e-14` pops the
    // cancelled leading term; `|NaN| < 1e-14` is false, so nothing was ever popped and
    // the loop ran until the fuzzer's alarm. `poly_mod` had the identical loop and the
    // identical hang. `poly_gcd` and `poly_lcm` were measured safe: gcd only calls
    // poly_mod with a divisor of two or more coefficients, which strip guarantees is
    // non-zero, and lcm already refused both zero cases.
    //
    // Two fixes, because they answer different questions. The library refuses the zero
    // divisor and also enforces the loop's termination invariant directly -- no
    // shrinkage, no next iteration -- so no input can spin it again whatever the
    // arithmetic does. The REPL refuses it with a reason, which is what a user who
    // typed it needs.
    Interpreter interp;

    // Cheap, and first: with the REPL check reverted this returns a quotient of zero
    // instead of an error, and with the library guard reverted as well it HANGS -- so
    // run this test under a timeout when checking the mutant.
    for (const std::string cmd : {std::string("poly_div_quot([0.1;],[0])"),
                                  std::string("poly_mod([0.1;],[0])"),
                                  std::string("poly_div_quot([1;2;3],[0;0])"),
                                  std::string("poly_mod([1;2;3],[0])")}) {
        const auto out = interp.execute(cmd);
        ASSERT_FALSE(out.has_value()) << cmd << " => " << out.value_or("");
        EXPECT_NE(ms::format_error(out.error()).find("non-zero divisor"), std::string::npos)
            << cmd << " => " << ms::format_error(out.error());
    }

    // Division by a real polynomial still divides. (3x^2 + 2x + 1) / (x + 1) is
    // 3x - 1 remainder 2, and (x^2 - 1) / (x + 1) is x - 1 exactly.
    const auto quot = interp.execute("poly_div_quot([1;2;3],[1;1])");
    ASSERT_TRUE(quot.has_value()) << ms::format_error(quot.error());
    EXPECT_NE(quot->find("-1.000000"), std::string::npos) << *quot;
    EXPECT_NE(quot->find("3.000000"), std::string::npos) << *quot;

    const auto rem = interp.execute("poly_mod([1;2;3],[1;1])");
    ASSERT_TRUE(rem.has_value()) << ms::format_error(rem.error());
    EXPECT_NE(rem->find("2.000000"), std::string::npos) << *rem;

    const auto exact = interp.execute("poly_div_quot([-1;0;1],[1;1])");
    ASSERT_TRUE(exact.has_value()) << ms::format_error(exact.error());
    EXPECT_NE(exact->find("-1.000000"), std::string::npos) << *exact;
    EXPECT_NE(exact->find("1.000000"), std::string::npos) << *exact;

    // gcd and lcm take a zero argument without complaint, as they did before.
    EXPECT_TRUE(interp.execute("poly_gcd([1;2],[0])").has_value());
    EXPECT_TRUE(interp.execute("poly_lcm([1;2],[0])").has_value());
}

TEST(ReplResourceGuards, KroneckerSymbolInt64MinReturns) {
    // n = -2^63 used to recurse forever inside kronecker_symbol because
    // -INT64_MIN is not representable. The library now takes the magnitude in
    // unsigned arithmetic, and the REPL still accepts the exact int64_t edge.
    Interpreter interp;
    const auto out = interp.execute("numthy_kronecker_symbol(1, -9223372036854775808)");
    ASSERT_TRUE(out.has_value()) << ms::format_error(out.error());
    EXPECT_NE(out->find("0"), std::string::npos) << *out;

    const auto neg = interp.execute("numthy_kronecker_symbol(-3, -7)");
    ASSERT_TRUE(neg.has_value()) << ms::format_error(neg.error());
}
