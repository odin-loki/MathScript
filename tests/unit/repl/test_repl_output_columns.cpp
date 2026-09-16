// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.4 over `src/interp/repl_engine_internal.cpp`: what is IN the matrix a
// command returns, and where a guard's boundary actually is.
//
// The REPL command tests are thorough about two things -- that a command
// succeeds, and that the matrix it returns has the right shape -- and the
// mutation run found the gap between them. Writing a result into the wrong
// COLUMN of an output matrix leaves the shape untouched, so
// `EXPECT_EQ(rows(), 4u)` passes either way. Validation guards have the same
// shape of gap: a guard with six comparisons on one line is exercised by
// arguments that are far from all six, so moving any one bound goes unnoticed.

#include "repl_test_helpers.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using ms::interp::Interpreter;

namespace {

// The rows of a `name =\n  [a, b, c]\n  [d, e, f]` block, as vectors of doubles.
std::vector<std::vector<double>> parse_labelled_rows(const std::string& text,
                                                     const std::string& label) {
    std::vector<std::vector<double>> rows;
    std::istringstream in(text);
    std::string line;
    bool inside = false;
    while (std::getline(in, line)) {
        const std::string trimmed = line.substr(line.find_first_not_of(" \t") == std::string::npos
                                                     ? 0
                                                     : line.find_first_not_of(" \t"));
        if (trimmed.rfind(label, 0) == 0) {
            inside = true;
            continue;
        }
        if (!inside) {
            continue;
        }
        if (trimmed.empty() || trimmed.front() != '[') {
            break;
        }
        std::vector<double> row;
        std::string body = trimmed.substr(1, trimmed.find(']') - 1);
        std::replace(body.begin(), body.end(), ',', ' ');
        std::istringstream cells(body);
        double v = 0.0;
        while (cells >> v) {
            row.push_back(v);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

}  // namespace

TEST(ReplOutputColumns, TheDaeTrajectoryCarriesItsTimeInBothBlocks) {
    // `format_dae_trajectory` writes `y_out(i, 0) = t` and `z_out(i, 0) = t`
    // and then fills the remaining columns from the state vectors. Writing the
    // z block's time into column 1 instead is invisible to a shape check: the
    // loop below overwrites column 1 anyway, so the only trace is a first
    // column of zeros where the times should be. Nothing had read either block.
    Interpreter interp;
    const std::string cmd =
        "ode_dae_index1(\"-y0\", \"z0 - 2*y0\", 0, [1], [2], 1, 8)";
    const auto out = interp.execute(cmd);
    ASSERT_TRUE(out.has_value()) << cmd;

    const auto y_rows = parse_labelled_rows(*out, "y_traj");
    const auto z_rows = parse_labelled_rows(*out, "z_traj");
    ASSERT_FALSE(y_rows.empty()) << *out;
    ASSERT_EQ(y_rows.size(), z_rows.size()) << *out;

    for (size_t i = 0; i < y_rows.size(); ++i) {
        ASSERT_GE(y_rows[i].size(), 2u) << *out;
        ASSERT_GE(z_rows[i].size(), 2u) << *out;
        EXPECT_NEAR(z_rows[i][0], y_rows[i][0], 1e-12)
            << "row " << i << ": the two blocks are the same trajectory, so they "
            << "carry the same time\n" << *out;
    }
    // The times advance from 0 to the horizon rather than all being zero.
    EXPECT_NEAR(z_rows.front()[0], 0.0, 1e-12) << *out;
    EXPECT_NEAR(z_rows.back()[0], 1.0, 1e-9) << *out;
    EXPECT_GT(z_rows.back()[0], z_rows.front()[0]) << *out;
}

TEST(ReplOutputColumns, DelaunayTrianglesAreIndicesIntoThePointsGivenToIt) {
    // `geo_delaunay_2d` writes the three vertex indices of each triangle into
    // columns 0, 1 and 2. The suite asserted the ROW COUNT -- one triangle for
    // three points, two for a square -- and never looked at an index, so
    // writing the first vertex into column 1 (where the second then lands on
    // top of it) left column 0 as a row of zeros and nothing said so.
    Interpreter interp;
    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    const auto& t3 = interp.state().matrices.at("T3");
    ASSERT_EQ(t3.rows(), 1u);
    ASSERT_EQ(t3.cols(), 3u);
    // Three points admit exactly one triangle, and it uses all three of them.
    std::set<int> verts;
    for (size_t j = 0; j < 3; ++j) {
        verts.insert(static_cast<int>(t3(0, j)));
    }
    EXPECT_EQ(verts, (std::set<int>{0, 1, 2})) << "vertices " << t3(0, 0) << " "
                                               << t3(0, 1) << " " << t3(0, 2);

    // A square triangulates into two triangles that between them use all four
    // points, and no triangle repeats a vertex.
    expect_ok(interp, "Tsq = geo_delaunay_2d([0, 0; 1, 0; 0, 1; 1, 1])");
    const auto& tsq = interp.state().matrices.at("Tsq");
    ASSERT_EQ(tsq.rows(), 2u);
    ASSERT_EQ(tsq.cols(), 3u);
    std::set<int> used;
    for (size_t i = 0; i < tsq.rows(); ++i) {
        std::set<int> tri;
        for (size_t j = 0; j < 3; ++j) {
            const int v = static_cast<int>(tsq(i, j));
            EXPECT_GE(v, 0);
            EXPECT_LT(v, 4);
            tri.insert(v);
            used.insert(v);
        }
        EXPECT_EQ(tri.size(), 3u) << "triangle " << i << " repeats a vertex";
    }
    EXPECT_EQ(used, (std::set<int>{0, 1, 2, 3}));

    // Neither of the two cases above has a triangle whose FIRST vertex is
    // anything but 0 or 2, and a square's triangles stay distinct even when
    // column 0 is forced to zero. A five-point set does not: its triangles
    // include [2, 0, 4] and [3, 2, 4], which collapse to [0, 0, 4] and
    // [0, 2, 4] when the first vertex never reaches column 0 -- a repeated
    // vertex, which is not a triangle.
    expect_ok(interp, "T5 = geo_delaunay_2d([0, 0; 2, 0; 1, 2; 3, 2; 1.5, 0.5])");
    const auto& t5 = interp.state().matrices.at("T5");
    ASSERT_EQ(t5.cols(), 3u);
    ASSERT_GT(t5.rows(), 1u);
    std::set<int> used5;
    int nonzero_first = 0;
    for (size_t i = 0; i < t5.rows(); ++i) {
        std::set<int> tri;
        for (size_t j = 0; j < 3; ++j) {
            const int v = static_cast<int>(t5(i, j));
            EXPECT_GE(v, 0);
            EXPECT_LT(v, 5);
            tri.insert(v);
            used5.insert(v);
        }
        EXPECT_EQ(tri.size(), 3u)
            << "triangle " << i << " is [" << t5(i, 0) << ", " << t5(i, 1) << ", "
            << t5(i, 2) << "], which repeats a vertex";
        if (static_cast<int>(t5(i, 0)) != 0) {
            ++nonzero_first;
        }
    }
    EXPECT_EQ(used5, (std::set<int>{0, 1, 2, 3, 4}));
    EXPECT_GT(nonzero_first, 0)
        << "every triangle starts at vertex 0, which no triangulation of these "
        << "five points does -- column 0 is not being written";
}

TEST(ReplOutputColumns, AstarRejectsEachOutOfRangeEndpointOnItsOwn) {
    // The bounds guard is `source < 0 || target < 0 || source >= n || target >= n`,
    // four comparisons on one line, and every test passed endpoints that satisfy
    // all four. Turning any `||` into `&&` leaves one endpoint unchecked, and
    // an unchecked index is an out-of-bounds read. Each of the four is driven
    // here on its own, with the other three in range.
    Interpreter interp;
    expect_ok(interp, "A = [0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1; 0, 0, 0, 0]");
    expect_ok(interp, "h = [3; 2; 1; 0]");
    expect_ok(interp, "ok = graph_astar(A, 0, 3, h)");

    expect_error(interp, "bad1 = graph_astar(A, -1, 3, h)");
    expect_error(interp, "bad2 = graph_astar(A, 0, -1, h)");
    expect_error(interp, "bad3 = graph_astar(A, 4, 3, h)");
    expect_error(interp, "bad4 = graph_astar(A, 0, 4, h)");
    // The last in-range index on each side is still accepted.
    expect_ok(interp, "edge1 = graph_astar(A, 3, 3, h)");
    expect_ok(interp, "edge2 = graph_astar(A, 0, 0, h)");
}

TEST(ReplOutputColumns, TheLz77LengthBoundIsInclusive) {
    // `lz77_decode_vec` rejects a token whose length exceeds 65535, because the
    // decoder stores it in a uint16_t and a larger value would wrap to a
    // different, valid length. 65535 itself is representable and must be
    // accepted; written as `>=` the bound would reject the largest legal token
    // and no test had ever offered one.
    Interpreter interp;
    expect_ok(interp, "TokA = [0, 0, 97]");
    expect_ok(interp, "OutA = lz77_decode_vec(TokA)");
    EXPECT_EQ(interp.state().matrices.at("OutA").rows(), 1u);

    // One literal, then a run of the maximum length looking one byte back.
    expect_ok(interp, "TokMax = [0, 0, 97; 1, 65535, 98]");
    expect_ok(interp, "OutMax = lz77_decode_vec(TokMax)");
    const auto& out_max = interp.state().matrices.at("OutMax");
    EXPECT_EQ(out_max.rows(), 65537u) << "1 literal + 65535 copies + 1 literal";
    EXPECT_NEAR(out_max(0, 0), 97.0, 1e-12);
    EXPECT_NEAR(out_max(1, 0), 97.0, 1e-12);
    EXPECT_NEAR(out_max(65535, 0), 97.0, 1e-12);
    EXPECT_NEAR(out_max(65536, 0), 98.0, 1e-12);

    // One past the bound is refused, which is what the bound is for.
    expect_error_contains(interp, "OutOver = lz77_decode_vec([0, 0, 97; 1, 65536, 98])",
                          "65535");
    expect_error_contains(interp, "OutOff = lz77_decode_vec([65536, 1, 97])", "65535");
}

TEST(ReplOutputColumns, AnEmptySampleIsRejectedOnEitherSideOfATwoSampleTest) {
    // `eval_stats_two_sample_ttest` ends with `if (a->empty() || b->empty())`,
    // and nothing had passed an empty sample on either side. Measured while
    // writing this: that guard cannot be reached from the REPL, because
    // `matrix_to_coeff_vector` above it rejects the empty matrix first, with
    // "expected 1xN or Nx1 coefficient vector". The `||` there is defence
    // behind a door that is already locked -- recorded rather than asserted
    // into a shape it does not have.
    //
    // What IS the contract, and what nothing had checked, is that an empty
    // sample on EITHER side is refused rather than silently treated as a
    // sample of zero observations.
    Interpreter interp;
    expect_ok(interp, "E = []");
    expect_ok(interp, "S = [1; 2; 3; 4]");
    expect_ok(interp, "t_ok = stats_two_sample_ttest(S, S)");

    expect_error(interp, "t_a = stats_two_sample_ttest(E, S)");
    expect_error(interp, "t_b = stats_two_sample_ttest(S, E)");
    expect_error(interp, "t_c = stats_two_sample_ttest(E, E)");
}

TEST(ReplOutputColumns, AParenthesisedRightHandSideIsTheSameExpression) {
    // `strip_outer_parens` walks the string counting bracket depth and gives up
    // if the opening paren closes before the end. Starting that walk at index 1
    // -- skipping the very paren whose match it is looking for -- leaves the
    // depth permanently one too low, so it never sees the depth return to zero
    // and concludes the parens wrap everything: `(1)+(2)` becomes `1)+(2`.
    //
    // Chasing that mutant turned up why nothing had ever noticed: the
    // assignment path rejected a FULLY parenthesised right-hand side outright.
    // `x = (a+b)+0`, `x = (a)+(b)` and a bare `(a+b)` all worked, but
    // `x = (a+b)` answered "parse_matrix: expected [ ... ]", because
    // `is_scalar_expression_rhs` looks for operators at the TOP level and in
    // `(a+b)` the `+` is not one. It strips the parentheses and asks again now.
    Interpreter interp;
    expect_ok(interp, "a = 4");
    expect_ok(interp, "b = 3");

    // Two separately parenthesised operands: the outer parens are NOT outer,
    // and stripping them would leave `1)+(2`.
    expect_ok(interp, "s1 = (1)+(2)");
    EXPECT_NEAR(interp.state().scalars.at("s1"), 3.0, 1e-12);
    expect_ok(interp, "s2 = (a)*(b)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), 12.0, 1e-12);
    expect_ok(interp, "s3 = (a+1)/(b-1)");
    EXPECT_NEAR(interp.state().scalars.at("s3"), 2.5, 1e-12);
    expect_ok(interp, "s4 = (a)-(b)-(1)");
    EXPECT_NEAR(interp.state().scalars.at("s4"), 0.0, 1e-12);

    // Parentheses that DO wrap the whole right-hand side, which used to be the
    // one form that did not work.
    expect_ok(interp, "s5 = (a+b)");
    EXPECT_NEAR(interp.state().scalars.at("s5"), 7.0, 1e-12);
    expect_ok(interp, "s6 = ((a+b))");
    EXPECT_NEAR(interp.state().scalars.at("s6"), 7.0, 1e-12);
    expect_ok(interp, "s7 = ((a)+(b))");
    EXPECT_NEAR(interp.state().scalars.at("s7"), 7.0, 1e-12);
    expect_ok(interp, "s8 = -(a+b)");
    EXPECT_NEAR(interp.state().scalars.at("s8"), -7.0, 1e-12);
    expect_ok(interp, "s9 = (sqrt(4))");
    EXPECT_NEAR(interp.state().scalars.at("s9"), 2.0, 1e-12);
    expect_ok(interp, "s10 = (7)");
    EXPECT_NEAR(interp.state().scalars.at("s10"), 7.0, 1e-12);
    expect_ok(interp, "s11 = (4+3)");
    EXPECT_NEAR(interp.state().scalars.at("s11"), 7.0, 1e-12);

    // The unparenthesised forms are unchanged.
    expect_ok(interp, "s12 = a+b");
    EXPECT_NEAR(interp.state().scalars.at("s12"), 7.0, 1e-12);
    expect_ok(interp, "M = [1, 2; 3, 4]");
    ASSERT_GT(interp.state().matrices.count("M"), 0u);
    EXPECT_EQ(interp.state().matrices.at("M").rows(), 2u);

    // Two forms remain unsupported, and are pinned here so that the change
    // above is understood as exactly as wide as it is. A parenthesised MATRIX
    // literal and a parenthesised bare NAME both still go to the matrix path
    // and fail there: the first because a matrix literal in parentheses is not
    // a matrix literal, the second because whether `(y)` is a scalar
    // assignment depends on what y holds, which this classifier answers from
    // the text alone.
    expect_error(interp, "Mp = ([1, 2; 3, 4])");
    expect_error(interp, "ap = (a)");
}

TEST(ReplOutputColumns, TheSmallestAdmissibleSizesAreStillAdmissible) {
    // Two guards whose lower bound is zero rather than one, and nothing had
    // ever passed zero: `combo_bracelets`/`combo_necklaces` accept n = 0 (the
    // empty necklace), and moving the bound to `n < 1` rejects it.
    Interpreter interp;
    expect_ok(interp, "br3 = combo_bracelets(3, 2)");
    expect_ok(interp, "nk3 = combo_necklaces(3, 2)");

    expect_ok(interp, "br0 = combo_bracelets(0, 2)");
    expect_ok(interp, "nk0 = combo_necklaces(0, 2)");
    // Above the cap is still refused, so the bound is a bound.
    expect_error(interp, "brmax = combo_bracelets(11, 2)");
    expect_error(interp, "nkmax = combo_necklaces(11, 2)");
    // And a negative count is refused on the other side.
    expect_error(interp, "brneg = combo_bracelets(-1, 2)");
}

TEST(ReplOutputColumns, QubitCountsAreCheckedForBothRangeAndIntegrality) {
    // `if (n_qubits < 1 || arg != n_qubits)` is two checks on one line: the
    // count must be at least one, AND the argument must have been an integer in
    // the first place, since it arrives as a double. As `&&` neither check
    // fires on its own -- zero qubits passes because 0.0 IS an integer, and 2.5
    // passes because 2 is at least one -- and 2.5 silently becomes 2.
    Interpreter interp;
    expect_ok(interp, "w2 = quantum_w_state(2)");
    expect_ok(interp, "g2 = quantum_ghz_state(2)");

    expect_error_contains(interp, "w0 = quantum_w_state(0)", "n_qubits");
    expect_error_contains(interp, "g0 = quantum_ghz_state(0)", "n_qubits");
    expect_error_contains(interp, "wf = quantum_w_state(2.5)", "n_qubits");
    expect_error_contains(interp, "gf = quantum_ghz_state(2.5)", "n_qubits");
    expect_error(interp, "wn = quantum_w_state(-1)");
    expect_error(interp, "gn = quantum_ghz_state(-1)");
}

TEST(ReplOutputColumns, KruskalWallisReportsAStatisticADegreeCountAndAProbability) {
    // `eval_kruskal_wallis` returns a 3x1 column of (H, df, p). The existing
    // coverage is `expect_ok` on the call, so writing the degrees of freedom
    // into row 2 -- on top of the p-value -- leaves a 3x1 matrix of three
    // finite numbers and passes. Three groups of three with no ties has a
    // closed form: rank sums 6, 15 and 24 over N = 9 give
    // H = 12/(9*10) * (36 + 225 + 576)/3 - 3*10 = 7.2, df = k - 1 = 2, and the
    // chi-squared survival function at two degrees of freedom is exp(-H/2).
    Interpreter interp;
    expect_ok(interp, "kw = kruskal_wallis([10, 11, 12; 20, 21, 22; 30, 31, 32])");
    expect_ok(interp, "h = mat_at(kw, 0, 0)");
    expect_ok(interp, "df = mat_at(kw, 1, 0)");
    expect_ok(interp, "p = mat_at(kw, 2, 0)");

    EXPECT_NEAR(interp.state().scalars.at("h"), 7.2, 1e-9);
    EXPECT_EQ(interp.state().scalars.at("df"), 2.0);
    EXPECT_NEAR(interp.state().scalars.at("p"), std::exp(-3.6), 1e-6);
}

TEST(ReplOutputColumns, AZoomFftOfASingleBinIsASingleBin) {
    // `signal_czt_zoom` refuses `m < 1`, and one bin is the smallest thing that
    // is not less than one. Nothing had asked for it, so the bound could move
    // to `m <= 1` and only the case nobody ran would notice. The value is
    // worth pinning too: the chirp-z transform of a unit impulse is
    // sum_n x[n] A^-n W^(nk) = x[0] = 1 at every bin, whatever the zoom band is.
    Interpreter interp;
    expect_ok(interp, "x = [1, 0, 0, 0]");
    expect_ok(interp, "z = signal_czt_zoom(x, 0, 1, 1, 4)");
    expect_ok(interp, "zr = mat_rows(z)");
    expect_ok(interp, "zc = mat_cols(z)");
    EXPECT_EQ(interp.state().scalars.at("zr"), 1.0);
    EXPECT_EQ(interp.state().scalars.at("zc"), 2.0);

    expect_ok(interp, "re = mat_at(z, 0, 0)");
    expect_ok(interp, "im = mat_at(z, 0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("re"), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().scalars.at("im"), 0.0, 1e-12);

    expect_error_contains(interp, "bad = signal_czt_zoom(x, 0, 1, 0, 4)", "positive integer m");

    // Recorded, not asserted: the no-assignment form of this command splits its
    // arguments with a comma regex rather than the bracket-aware splitter the
    // assignment form uses, so `signal_czt_zoom([1, 0, 0, 0], 0, 1, 1, 4)`
    // without an `x =` in front of it reports `unknown matrix: [1`. That lives
    // in `repl_engine.cpp`, which is the next file this exercise measures.
}

TEST(ReplOutputColumns, FourBytesIsTheShortestStreamBzip2WillLookAt) {
    // The header of a bzip2-like stream is the four-byte primary index, so
    // `bytes.size() < 4` is a read-past-the-end guard and not a judgement about
    // the content. Four bytes passes it and is then refused by the content
    // check underneath, which is the only thing that distinguishes the two: at
    // `<= 4` the four-byte input never reaches the content check and comes back
    // with the wrong reason. A test that only asserted "this fails" would agree
    // with both.
    Interpreter interp;
    expect_error_contains(interp, "bzip2_decompress_vec([0; 0; 0])", "at least 4-byte");
    expect_error_contains(interp, "bzip2_decompress_vec([0; 0; 0; 0])",
                          "not a stream produced by bzip2_compress_vec");

    // And the stream the compressor does produce decompresses back to its input.
    expect_ok(interp, "c = bzip2_compress_vec([7; 7; 9; 7])");
    expect_ok(interp, "d = bzip2_decompress_vec(c)");
    expect_ok(interp, "dn = mat_rows(d)");
    ASSERT_EQ(interp.state().scalars.at("dn"), 4.0);
    const double expected[4] = {7.0, 7.0, 9.0, 7.0};
    for (int i = 0; i < 4; ++i) {
        const std::string cmd = "v = mat_at(d, " + std::to_string(i) + ", 0)";
        expect_ok(interp, cmd);
        EXPECT_EQ(interp.state().scalars.at("v"), expected[i]) << cmd;
    }
}

TEST(ReplOutputColumns, TheAdaBoostModelMatrixRecordsHowItWasFitted) {
    // Row 0 of the serialised model is (n_estimators, max_depth, seed,
    // estimator count), and `ml_adaboost_from_matrix` reads all four back. Only
    // two of them reach a prediction, so writing the seed one row down -- over
    // the first weak learner's depth, which prediction does not read either --
    // changes nothing any fit-then-predict test can see. The header is the
    // record of how the model was produced; a saved model that cannot say which
    // draw it came from is not reproducible, so the header is what to assert.
    Interpreter interp;
    expect_ok(interp, "X = [0, 0; 1, 1; 0, 1; 1, 0]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab = ml_adaboost_fit(X, y, 3, 2)");

    expect_ok(interp, "n_est = mat_at(ab, 0, 0)");
    expect_ok(interp, "depth = mat_at(ab, 0, 1)");
    expect_ok(interp, "sd = mat_at(ab, 0, 2)");
    expect_ok(interp, "n_tree = mat_at(ab, 0, 3)");

    EXPECT_EQ(interp.state().scalars.at("n_est"), 3.0);
    EXPECT_EQ(interp.state().scalars.at("depth"), 2.0);
    EXPECT_EQ(interp.state().scalars.at("sd"), 42.0);
    const double n_tree = interp.state().scalars.at("n_tree");
    EXPECT_GE(n_tree, 1.0);
    EXPECT_LE(n_tree, 3.0);

    // The rows under the header are the learners, and they still round-trip.
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab)");
    expect_ok(interp, "pn = mat_rows(ab_p)");
    EXPECT_EQ(interp.state().scalars.at("pn"), 4.0);
}

TEST(ReplOutputColumns, AGradientBoostingModelIsCheckedForLayoutSeparatelyFromShape) {
    // Two guards, two messages. The first asks whether the matrix is big enough
    // for a header row to exist at all; the second walks the layout the header
    // describes. A two-row, five-column matrix is the smallest input that gets
    // past the first and is refused by the second, so it is the input that
    // tells the two bounds apart -- move either `<` to `<=` and the answer
    // comes back with the other message. Nothing had read either message.
    Interpreter interp;
    expect_ok(interp, "X = [0, 0; 1, 1]");

    expect_error_contains(interp, "ml_gradient_boosting_predict(X, [0, 0, 0, 0, 0; 0, 0, 0, 0, 0])",
                          "invalid GradientBoosting model layout");
    expect_error_contains(interp, "ml_gradient_boosting_predict(X, [0, 0, 0, 0, 0])",
                          "expected GradientBoosting model matrix");
    expect_error_contains(interp, "ml_gradient_boosting_predict(X, [0, 0, 0, 0; 0, 0, 0, 0])",
                          "expected GradientBoosting model matrix");

    // And the header a real fit writes says what the fit was asked for.
    expect_ok(interp, "y = [0.5; 1.5]");
    expect_ok(interp, "gb = ml_gradient_boosting_fit(X, y, 2, 0.25, 1)");
    expect_ok(interp, "gt = mat_at(gb, 0, 0)");
    expect_ok(interp, "gd = mat_at(gb, 0, 1)");
    expect_ok(interp, "gl = mat_at(gb, 0, 2)");
    expect_ok(interp, "gs = mat_at(gb, 0, 3)");
    EXPECT_EQ(interp.state().scalars.at("gt"), 2.0);
    EXPECT_EQ(interp.state().scalars.at("gd"), 1.0);
    EXPECT_EQ(interp.state().scalars.at("gl"), 0.25);
    EXPECT_EQ(interp.state().scalars.at("gs"), 42.0);

    // Column 4 is the constant the boosting starts from, which is the mean of
    // the targets: (0.5 + 1.5) / 2.
    expect_ok(interp, "gi = mat_at(gb, 0, 4)");
    EXPECT_EQ(interp.state().scalars.at("gi"), 1.0);
}

TEST(ReplOutputColumns, ASupportVectorMachineIsDeserialisedWithTheKernelItWasSavedWith) {
    // Column 0 of the header is the kernel code and column 1 is C. Reading the
    // kernel out of column 1 still passes the layout check, because a fitted
    // model's C is 1 and 1 is a legal kernel code -- so a linear model comes
    // back as an RBF one and every prediction is drawn from a different
    // function. Nothing had asserted an SVM prediction at all.
    //
    // A single support vector AT THE ORIGIN separates the two kernels by hand:
    // the linear kernel is dot(sv, x), which is zero for every x, so the
    // decision function is the bias everywhere and every point is class -1.
    // The RBF kernel is exp(-gamma*||sv - x||^2), which is 1 at the origin, so
    // the same model would call the origin +1.
    Interpreter interp;
    //           kernel  C  gamma    b    tol  iter  p  n
    expect_ok(interp, "M = [0, 1, 1, -0.5, 0.001, 100, 1, 1; 0, 1, 1, 0, 0, 0, 0, 0]");
    expect_ok(interp, "yp = ml_svm_predict([0; 3], M)");
    expect_ok(interp, "p0 = mat_at(yp, 0, 0)");
    expect_ok(interp, "p1 = mat_at(yp, 1, 0)");
    EXPECT_EQ(interp.state().scalars.at("p0"), -1.0);
    EXPECT_EQ(interp.state().scalars.at("p1"), -1.0);

    // C and tol are written into the header and read back out of it, and the
    // only thing `ml_svm_from_matrix` feeds is `predict`, which reads neither.
    // Recorded rather than asserted: mutating either cell is not observable
    // through any command the REPL offers.
    expect_ok(interp, "F = ml_svm_fit([0, 0; 1, 1; 0, 1; 1, 0], [-1; 1; -1; 1])");
    expect_ok(interp, "k = mat_at(F, 0, 0)");
    EXPECT_EQ(interp.state().scalars.at("k"), 0.0) << "a fit with no gamma is the linear kernel";
}
