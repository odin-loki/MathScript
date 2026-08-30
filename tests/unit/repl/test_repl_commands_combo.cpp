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

TEST(ReplCommandsTest, combo_nchoosek_assignment) {
    Interpreter interp;
    expect_ok(interp, "c = combo_nchoosek(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 10.0, 1e-9);
    expect_contains(interp, "combo_nchoosek(5, 2)", "10");
}

TEST(ReplCommandsTest, combo_stirling2) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_stirling2(n,k)");

    expect_ok(interp, "s2 = combo_stirling2(4, 2)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), 7.0, 1e-9);

    expect_contains(interp, "combo_stirling2(4, 2)", "7");
}

TEST(ReplCommandsTest, combo_catalan) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_catalan(n)");

    expect_ok(interp, "cat = combo_catalan(4)");
    EXPECT_NEAR(interp.state().scalars.at("cat"), 14.0, 1e-9);

    expect_contains(interp, "combo_catalan(4)", "14");
}

TEST(ReplCommandsTest, combo_bell) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_bell(n)");

    expect_ok(interp, "bell = combo_bell(4)");
    EXPECT_NEAR(interp.state().scalars.at("bell"), 15.0, 1e-9);

    expect_contains(interp, "combo_bell(4)", "15");
}

TEST(ReplCommandsTest, combo_stirling1) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_stirling1(n,k)");

    expect_ok(interp, "s1 = combo_stirling1(4, 2)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), 11.0, 1e-9);

    expect_contains(interp, "combo_stirling1(4, 2)", "11");
}

TEST(ReplCommandsTest, combo_motzkin) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_motzkin(n)");

    expect_ok(interp, "motz = combo_motzkin(4)");
    EXPECT_NEAR(interp.state().scalars.at("motz"), 9.0, 1e-9);

    expect_contains(interp, "combo_motzkin(4)", "9");
}

TEST(ReplCommandsTest, combo_permutations) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_permutations(n,k)");

    expect_ok(interp, "perm = combo_permutations(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("perm"), 20.0, 1e-9);

    expect_contains(interp, "combo_permutations(5, 2)", "20");
}

TEST(ReplCommandsTest, combo_subfactorial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_subfactorial(n)");

    expect_ok(interp, "subf = combo_subfactorial(4)");
    EXPECT_NEAR(interp.state().scalars.at("subf"), 9.0, 1e-9);

    expect_contains(interp, "combo_subfactorial(4)", "9");
}

TEST(ReplCommandsTest, combo_combinations_with_rep) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_combinations_with_rep(n,k)");

    expect_ok(interp, "cwr = combo_combinations_with_rep(3, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cwr"), 6.0, 1e-9);

    expect_contains(interp, "combo_combinations_with_rep(3, 2)", "6");
    expect_contains(interp, "combo_combinations_with_rep(5, 3)", "35");
}

TEST(ReplCommandsTest, combo_double_factorial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_double_factorial(n)");

    expect_ok(interp, "df = combo_double_factorial(5)");
    EXPECT_NEAR(interp.state().scalars.at("df"), 15.0, 1e-9);

    expect_contains(interp, "combo_double_factorial(5)", "15");
    expect_contains(interp, "combo_double_factorial(6)", "48");
}

TEST(ReplCommandsTest, combo_multinomial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_multinomial(n,ks)");

    expect_ok(interp, "m = combo_multinomial(6, [2; 2; 2])");
    EXPECT_NEAR(interp.state().scalars.at("m"), 90.0, 1e-9);

    expect_contains(interp, "combo_multinomial(6, [2; 2; 2])", "90");
}

TEST(ReplCommandsTest, combo_rank_permutation) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_rank_permutation(v)");

    expect_ok(interp, "r0 = combo_rank_permutation([0; 1; 2])");
    EXPECT_NEAR(interp.state().scalars.at("r0"), 0.0, 1e-9);

    expect_ok(interp, "r5 = combo_rank_permutation([2; 1; 0])");
    EXPECT_NEAR(interp.state().scalars.at("r5"), 5.0, 1e-9);

    expect_contains(interp, "combo_rank_permutation([2; 1; 0])", "5");
}

TEST(ReplCommandsTest, combo_unrank_permutation) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_unrank_permutation(n,rank)");

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);

    expect_contains(interp, "combo_unrank_permutation(3, 5)", "perm =");
}

TEST(ReplCommandsTest, combo_rank_combination) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_rank_combination(v,n)");

    expect_ok(interp, "r0 = combo_rank_combination([0; 1], 4)");
    EXPECT_NEAR(interp.state().scalars.at("r0"), 0.0, 1e-9);

    expect_ok(interp, "r1 = combo_rank_combination([0; 2], 4)");
    EXPECT_NEAR(interp.state().scalars.at("r1"), 1.0, 1e-9);

    expect_contains(interp, "combo_rank_combination([0; 2], 4)", "1");
}

TEST(ReplCommandsTest, combo_unrank_combination) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_unrank_combination(n,k,rank)");

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_derangements) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_derangements(n)");

    expect_ok(interp, "d3 = combo_derangements(3)");
    ASSERT_GT(interp.state().matrices.count("d3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("d3").cols(), 3u);
}

TEST(ReplCommandsTest, combo_all_permutations) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_permutations(n)");

    expect_ok(interp, "p3 = combo_all_permutations(3)");
    ASSERT_GT(interp.state().matrices.count("p3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p3").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("p3").cols(), 3u);
}

TEST(ReplCommandsTest, combo_all_subsets) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_subsets(n)");

    expect_ok(interp, "subs = combo_all_subsets(3)");
    ASSERT_GT(interp.state().matrices.count("subs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("subs").rows(), 8u);
    EXPECT_EQ(interp.state().matrices.at("subs").cols(), 3u);
}

TEST(ReplCommandsTest, combo_all_compositions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_compositions(n)");

    expect_ok(interp, "comps = combo_all_compositions(3)");
    ASSERT_GT(interp.state().matrices.count("comps"), 0u);
    EXPECT_EQ(interp.state().matrices.at("comps").rows(), 4u);
}

TEST(ReplCommandsTest, combo_all_partitions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_all_partitions(n)");

    expect_ok(interp, "parts = combo_all_partitions(4)");
    ASSERT_GT(interp.state().matrices.count("parts"), 0u);
    EXPECT_EQ(interp.state().matrices.at("parts").rows(), 5u);
}

TEST(ReplCommandsTest, combo_next_perm) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_next_perm(v)");

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);

    expect_ok(interp, "combo_next_perm([1; 2; 3])");
}

TEST(ReplCommandsTest, combo_next_comb) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_next_comb(v,n)");

    expect_ok(interp, "v2 = combo_next_comb([0; 1], 4)");
    ASSERT_GT(interp.state().matrices.count("v2"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("v2")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("v2")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_prev) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_prev_perm(v)");
    expect_contains(interp, "help", "combo_prev_comb(v,n)");

    expect_ok(interp, "pp = combo_prev_perm([1; 3; 2])");
    ASSERT_GT(interp.state().matrices.count("pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pp")(1, 0), 2.0, 1e-9);

    expect_ok(interp, "combo_prev_perm([1; 3; 2])");

    expect_ok(interp, "vp = combo_prev_comb([0; 2], 4)");
    ASSERT_GT(interp.state().matrices.count("vp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("vp")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("vp")(1, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, combo_binomial) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_binomial(n,k)");

    expect_ok(interp, "bin = combo_binomial(5, 2)");
    EXPECT_NEAR(interp.state().scalars.at("bin"), 10.0, 1e-9);

    expect_contains(interp, "combo_binomial(5, 2)", "10");
}

TEST(ReplCommandsTest, combo_bell_num) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_bell_num(n)");

    expect_ok(interp, "bn = combo_bell_num(4)");
    EXPECT_NEAR(interp.state().scalars.at("bn"), 15.0, 1e-9);

    expect_contains(interp, "combo_bell_num(4)", "15");
}

TEST(ReplCommandsTest, combo_eulerian) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_eulerian(n,k)");

    expect_ok(interp, "eul = combo_eulerian(4, 1)");
    EXPECT_NEAR(interp.state().scalars.at("eul"), 11.0, 1e-9);

    expect_contains(interp, "combo_eulerian(4, 1)", "11");
}

TEST(ReplCommandsTest, combo_gray_code) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_gray_code(n)");

    expect_ok(interp, "g2 = combo_gray_code(2)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("g2").cols(), 2u);
}

TEST(ReplCommandsTest, combo_dyck_paths) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_dyck_paths(n)");

    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    ASSERT_GT(interp.state().matrices.count("d3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("d3").cols(), 6u);
}

TEST(ReplCommandsTest, combo_necklaces) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_necklaces(n,k)");

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    ASSERT_GT(interp.state().matrices.count("neck"), 0u);
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("neck").cols(), 2u);
}

TEST(ReplCommandsTest, combo_bracelets_lyndon) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_bracelets(n,k)");
    expect_contains(interp, "help", "combo_lyndon_words(n,k)");

    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
    EXPECT_EQ(interp.state().matrices.at("br").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("br").cols(), 3u);

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("lw").cols(), 3u);
}

TEST(ReplCommandsTest, combo_de_bruijn_sequence) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_de_bruijn_sequence(k,n)");

    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    ASSERT_GT(interp.state().matrices.count("db"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("db").cols(), 1u);
}

TEST(ReplCommandsTest, combo_motzkin_paths) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_motzkin_paths(n)");

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    ASSERT_GT(interp.state().matrices.count("mp3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("mp3").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("mp3")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp3")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("mp3")(0, 2), -1.0, 1e-9);
}

TEST(ReplCommandsTest, combo_set_partitions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_set_partitions(n)");

    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    ASSERT_GT(interp.state().matrices.count("sp2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("sp2").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("sp2")(1, 1), 0.0, 1e-9);

    expect_ok(interp, "sp3 = combo_set_partitions(3)");
    ASSERT_GT(interp.state().matrices.count("sp3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp3").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("sp3").cols(), 3u);
}

TEST(ReplCommandsTest, combo_restricted_involutions) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_restricted_partitions(n,k)");
    expect_contains(interp, "help", "combo_involutions(n)");

    expect_ok(interp, "rp = combo_restricted_partitions(5,2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rp").rows(), 2u);
    {
        const auto& m = interp.state().matrices.at("rp");
        for (size_t r = 0; r < m.rows(); ++r) {
            double sum = 0.0;
            for (size_t c = 0; c < m.cols(); ++c) {
                sum += m(r, c);
            }
            EXPECT_NEAR(sum, 5.0, 1e-9);
        }
    }

    expect_ok(interp, "inv = combo_involutions(4)");
    EXPECT_NEAR(interp.state().scalars.at("inv"), 10.0, 1e-9);
    expect_contains(interp, "combo_involutions(4)", "10");
}

TEST(ReplCommandsTest, linalg_combo) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "gc = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("gc").rows(), 4u);

    expect_ok(interp, "dp = combo_dyck_paths(2)");
    EXPECT_GT(interp.state().matrices.at("dp").rows(), 0u);
}

TEST(ReplCommandsTest, gray_dyck) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, necklaces_bracelets) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, lyndon_debruijn) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, motzkin_partitions) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, gray_dyck_2) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, necklaces_bracelets_2) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, lyndon_debruijn_2) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, motzkin_partitions_2) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_2) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_2) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_3) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_3) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_2) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_2) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_2) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_2) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_4) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_4) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_3) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_3) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_3) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_3) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_5) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_5) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_4) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_4) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_4) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_4) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_6) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_6) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_5) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_5) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_5) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_5) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_7) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_7) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_6) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_6) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_6) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_6) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_8) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_8) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_7) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_7) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_7) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_7) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_9) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_9) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_8) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_8) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_8) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_8) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_10) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_10) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_9) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_9) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_9) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_9) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_11) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_11) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_10) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_10) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_10) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_10) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_12) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_12) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_11) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_11) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_11) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_11) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_13) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_13) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_12) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_12) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_12) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_12) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_14) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_14) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_13) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_13) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_13) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_13) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_15) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_15) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_14) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_14) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_14) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_14) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_16) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_16) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_15) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_15) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_15) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_15) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_17) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_17) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_16) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_16) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_16) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_16) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_18) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_18) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_17) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_17) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_17) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_17) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_19) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_19) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_18) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_18) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_18) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_18) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_20) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_20) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_19) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_19) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_19) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_19) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_21) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_21) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_20) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_20) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_20) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_20) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_22) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_22) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_21) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_21) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_21) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_21) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_23) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_23) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_22) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_22) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_22) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_22) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_24) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_24) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_23) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_23) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_23) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_23) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_25) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_25) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_24) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_24) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_24) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_24) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_26) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_26) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_25) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_25) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_25) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_25) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_27) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_27) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_26) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_26) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_26) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_26) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_28) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_28) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_27) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_27) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_27) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_27) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_29) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_29) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_28) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_28) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_28) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_28) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_30) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_30) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_29) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_29) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_29) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_29) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_31) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_31) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_30) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_30) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_30) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_30) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_unrank_perm_32) {
    Interpreter interp;

    expect_ok(interp, "p = combo_unrank_permutation(3, 5)");
    ASSERT_GT(interp.state().matrices.count("p"), 0u);
    const auto& perm = interp.state().matrices.at("p");
    EXPECT_EQ(perm.rows(), 3u);
    EXPECT_NEAR(perm(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(perm(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(perm(2, 0), 0.0, 1e-9);
}

TEST(ReplCommandsTest, combo_unrank_comb_32) {
    Interpreter interp;

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("c0"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    ASSERT_GT(interp.state().matrices.count("c1"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, combo_gray_code_combo_dyck_paths_31) {
    Interpreter interp;

    expect_ok(interp, "g2 = combo_gray_code(2)");
    EXPECT_EQ(interp.state().matrices.at("g2").rows(), 4u);
    expect_ok(interp, "d3 = combo_dyck_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("d3").rows(), 5u);
}

TEST(ReplCommandsTest, combo_necklaces_combo_bracelets_31) {
    Interpreter interp;

    expect_ok(interp, "neck = combo_necklaces(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("neck").rows(), 3u);
    expect_ok(interp, "br = combo_bracelets(3, 2)");
    ASSERT_GT(interp.state().matrices.count("br"), 0u);
}

TEST(ReplCommandsTest, combo_lyndon_words_combo_de_bruijn_sequence_31) {
    Interpreter interp;

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(ReplCommandsTest, combo_motzkin_paths_combo_set_partitions_31) {
    Interpreter interp;

    expect_ok(interp, "mp3 = combo_motzkin_paths(3)");
    EXPECT_EQ(interp.state().matrices.at("mp3").rows(), 4u);
    expect_ok(interp, "sp2 = combo_set_partitions(2)");
    EXPECT_EQ(interp.state().matrices.at("sp2").rows(), 2u);
}

TEST(ReplCommandsTest, combo_next_perm_noassign) {
    Interpreter interp;
    expect_contains(interp, "combo_next_perm([1; 2; 3])", "perm");
    expect_error_contains(interp, "combo_next_perm([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, combo_prev_perm_noassign) {
    Interpreter interp;
    expect_contains(interp, "combo_prev_perm([1; 3; 2])", "perm");
    expect_error_contains(interp, "combo_prev_perm([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, combo_rank_permutation_noassign) {
    Interpreter interp;
    expect_contains(interp, "combo_rank_permutation([2; 1; 0])", "5");
    expect_error_contains(interp, "combo_rank_permutation(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, combo_unrank_combination_noassign) {
    Interpreter interp;
    expect_contains(interp, "combo_unrank_combination(4, 2, 0)", "comb =");
    expect_error_contains(interp, "combo_unrank_combination(4, 2, missing)",
                          "expected combo_unrank_combination");
}

TEST(ReplCommandsTest, combo_next_comb_noassign) {
    Interpreter interp;
    expect_contains(interp, "combo_next_comb([0; 1], 4)", "comb =");
    expect_error_contains(interp, "combo_next_comb([0; 1], 1.5)", "non-negative integer n");
}

TEST(ReplCommandsTest, combo_prev_comb_noassign) {
    Interpreter interp;
    expect_contains(interp, "combo_prev_comb([0; 2], 4)", "comb =");
    expect_error_contains(interp, "combo_prev_comb([0; 2], 1.5)", "non-negative integer n");
}

TEST(ReplCommandsTest, combo_derangements_too_large_noassign) {
    Interpreter interp;
    expect_error_contains(interp, "combo_derangements(11)", "too large");
    expect_error_contains(interp, "combo_all_permutations(9)", "too large");
}
