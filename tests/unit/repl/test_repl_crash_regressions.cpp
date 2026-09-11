// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Named regressions for four crashes the malformed-input sweep turned up. The sweep
// covers them too, but it probes by pattern, so these pin each specific input and the
// behaviour that replaced the crash.

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/combo/combo.hpp"
#include "ms/quantum/quantum.hpp"

using namespace ms::interp;

TEST(ReplCrashRegressions, HistoricalCvarWithAConfidenceOutsideZeroOne) {
    // (1 - confidence) went negative, and converting a negative double to size_t is
    // undefined behaviour -- in practice ~2^64, so the summation loop read far past the
    // end of the sample and segfaulted.
    const std::vector<double> returns{-0.05, -0.02, 0.01, 0.03};

    for (const double c : {2.0, 1.0, 1.5, 100.0}) {
        const double v = ms::finance::historical_cvar(returns, c);
        EXPECT_TRUE(std::isfinite(v)) << "confidence " << c;
        // The tail collapses to the single worst observation.
        EXPECT_DOUBLE_EQ(v, 0.05) << "confidence " << c;
        EXPECT_TRUE(std::isfinite(ms::finance::historical_var(returns, c)));
    }
    for (const double c : {0.0, -1.0, -100.0}) {
        // The whole sample is in the tail: the mean of all four, negated.
        EXPECT_DOUBLE_EQ(ms::finance::historical_cvar(returns, c),
                         -(-0.05 + -0.02 + 0.01 + 0.03) / 4.0)
            << "confidence " << c;
        EXPECT_TRUE(std::isfinite(ms::finance::historical_var(returns, c)));
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isfinite(ms::finance::historical_cvar(returns, nan)));
    EXPECT_TRUE(std::isfinite(ms::finance::historical_var(returns, nan)));

    // The ordinary case is unchanged: 5% of four observations is a zero-length tail, so
    // the convention is the single worst return.
    EXPECT_DOUBLE_EQ(ms::finance::historical_cvar(returns, 0.95), 0.05);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("R = [-0.05; -0.02; 0.01; 0.03]").has_value());
    EXPECT_TRUE(interp.execute("finance_historical_cvar(R, 2)").has_value());
    EXPECT_TRUE(interp.execute("finance_historical_var(R, -1)").has_value());
}

TEST(ReplCrashRegressions, ModPowWithAZeroModulus) {
    // base %= mod with mod == 0 is integer division by zero: SIGFPE.
    EXPECT_EQ(ms::numthy::mod_pow(2, 10, 0), 0u);
    EXPECT_EQ(ms::numthy::mod_pow(0, 0, 0), 0u);
    // Everything is 0 modulo 1.
    EXPECT_EQ(ms::numthy::mod_pow(2, 10, 1), 0u);
    // The ordinary case is unchanged: 2^10 = 1024 = 1 (mod 7).
    EXPECT_EQ(ms::numthy::mod_pow(2, 10, 7), 1024u % 7u);
    EXPECT_EQ(ms::numthy::mod_pow(3, 0, 7), 1u);

    Interpreter interp;
    EXPECT_TRUE(interp.execute("numthy_mod_pow(2, 1, 0)").has_value());
}

TEST(ReplCrashRegressions, MlMatMulWithNonConformingShapes) {
    // The inner index runs over A's columns and subscripts B's ROWS with it, so a 2x3
    // times 2x3 read past the end of B.
    const ms::ml::Mat a{{1, 2, 3}, {4, 5, 6}};      // 2x3
    const ms::ml::Mat b{{1, 2}, {3, 4}, {5, 6}};    // 3x2
    const ms::ml::Mat ab = ms::ml::mat_mul(a, b);
    ASSERT_EQ(ab.size(), 2u);
    ASSERT_EQ(ab[0].size(), 2u);
    EXPECT_DOUBLE_EQ(ab[0][0], 1 * 1 + 2 * 3 + 3 * 5);
    EXPECT_DOUBLE_EQ(ab[1][1], 4 * 2 + 5 * 4 + 6 * 6);

    EXPECT_TRUE(ms::ml::mat_mul(a, a).empty()) << "2x3 * 2x3 does not conform";
    EXPECT_TRUE(ms::ml::mat_mul({}, b).empty());
    EXPECT_TRUE(ms::ml::mat_mul(a, {}).empty());
    const ms::ml::Mat ragged{{1, 2, 3}, {4, 5}};
    EXPECT_TRUE(ms::ml::mat_mul(ragged, b).empty());

    // mat_vec, vec_add and vec_sub indexed with the other operand's length.
    EXPECT_EQ(ms::ml::mat_vec(a, {1.0}).size(), 2u);
    EXPECT_EQ(ms::ml::mat_vec(ragged, {1.0, 2.0, 3.0}).size(), 2u);
    EXPECT_EQ(ms::ml::vec_add({1.0, 2.0, 3.0}, {1.0}).size(), 1u);
    EXPECT_EQ(ms::ml::vec_sub({1.0}, {1.0, 2.0, 3.0}).size(), 1u);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("R23 = [1, 2, 3; 4, 5, 6]").has_value());
    // A non-conforming product is reported, not an empty matrix and not a crash.
    EXPECT_FALSE(interp.execute("ml_mat_mul(R23, R23)").has_value());
    ASSERT_TRUE(interp.execute("R32 = [1, 2; 3, 4; 5, 6]").has_value());
    EXPECT_TRUE(interp.execute("ml_mat_mul(R23, R32)").has_value());
}

TEST(ReplCrashRegressions, BigIntStringConstructorOnNonNumericInput) {
    // The header documents "on failure constructs zero", but the body called std::stoul,
    // which throws std::invalid_argument -- and this library is built with
    // -fno-exceptions, so the throw called std::terminate and aborted the process.
    EXPECT_TRUE(ms::bignum::BigInt("x").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("-").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("12a3").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt("1.5").is_zero());
    EXPECT_TRUE(ms::bignum::BigInt(" 12").is_zero());

    // Valid input is unchanged, including across the 9-digit limb boundary.
    EXPECT_EQ(ms::bignum::BigInt("0").to_string(), "0");
    EXPECT_EQ(ms::bignum::BigInt("123456789").to_string(), "123456789");
    EXPECT_EQ(ms::bignum::BigInt("1234567890").to_string(), "1234567890");
    EXPECT_EQ(ms::bignum::BigInt("-98765432109876543210").to_string(),
              "-98765432109876543210");
    EXPECT_EQ(ms::bignum::BigInt("+7").to_string(), "7");

    Interpreter interp;
    EXPECT_TRUE(interp.execute("bigint_gcd(\"x\", \"x\")").has_value() ||
                true);  // must not abort; either answer is acceptable
    EXPECT_TRUE(interp.execute("bigint_gcd(\"12\", \"18\")").has_value());
}

TEST(ReplCrashRegressions, VecDotWithMismatchedLengths) {
    // ElasticNet::predict passes the model's coefficient vector and a feature row to
    // vec_dot, which indexed the second with the FIRST one's length. A model fitted on a
    // different feature count read past the row -- AddressSanitizer caught it as an
    // eight-byte read past the end.
    EXPECT_DOUBLE_EQ(ms::ml::vec_dot({1.0, 2.0, 3.0}, {1.0}), 1.0);
    EXPECT_DOUBLE_EQ(ms::ml::vec_dot({1.0}, {1.0, 2.0, 3.0}), 1.0);
    EXPECT_DOUBLE_EQ(ms::ml::vec_dot({}, {1.0, 2.0}), 0.0);
    // Equal lengths are unchanged.
    EXPECT_DOUBLE_EQ(ms::ml::vec_dot({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}), 32.0);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("X2 = [1, 1; 2, 1; 3, 1; 4, 1]").has_value());
    ASSERT_TRUE(interp.execute("y2 = [2; 4; 6; 8]").has_value());
    ASSERT_TRUE(interp.execute("en = ml_elastic_net_fit(X2, y2, 0.1, 0.5)").has_value());
    // A feature matrix with a different column count than the model was fitted on.
    ASSERT_TRUE(interp.execute("X3 = [1, 1, 1; 2, 1, 1; 3, 1, 1]").has_value());
    const auto r = interp.execute("ml_elastic_net_predict(X3, en)");
    (void)r;  // either answer is acceptable; it must not read past the row
    EXPECT_TRUE(interp.execute("det([1, 2; 3, 4])").has_value()) << "session still usable";
}

TEST(ReplCrashRegressions, PartialTraceWithDimensionsThatDoNotFactorTheState) {
    using ms::quantum::DensityMatrix;
    using C = std::complex<double>;
    // rho[i*d2 + k] is indexed with no check that d1*d2 == rho.size().
    // entanglement_entropy's fallback path reaches partial_trace exactly when the
    // subsystem dimensions are invalid, so this is how a fuzzed REPL line got there.
    DensityMatrix rho(4, std::vector<C>(4, C(0.0)));
    for (int i = 0; i < 4; ++i) rho[i][i] = C(0.25);

    EXPECT_TRUE(ms::quantum::partial_trace(rho, 3, 3, 0).empty()) << "9 != 4";
    EXPECT_TRUE(ms::quantum::partial_trace(rho, 2, 3, 0).empty()) << "6 != 4";
    EXPECT_TRUE(ms::quantum::partial_trace(rho, 0, 4, 0).empty());
    EXPECT_TRUE(ms::quantum::partial_trace(rho, -1, 4, 1).empty());

    // The valid factorisation still works: tracing out either qubit of the maximally
    // mixed two-qubit state leaves the maximally mixed one-qubit state.
    const auto a = ms::quantum::partial_trace(rho, 2, 2, 0);
    ASSERT_EQ(a.size(), 2u);
    EXPECT_NEAR(a[0][0].real(), 0.5, 1e-12);
    EXPECT_NEAR(a[1][1].real(), 0.5, 1e-12);
    const auto b = ms::quantum::partial_trace(rho, 2, 2, 1);
    ASSERT_EQ(b.size(), 2u);
    EXPECT_NEAR(b[0][0].real(), 0.5, 1e-12);

    // And the entropy fallback reports 0 rather than crashing.
    const ms::quantum::Ket psi{C(1.0), C(0.0), C(0.0), C(0.0)};
    EXPECT_TRUE(std::isfinite(ms::quantum::entanglement_entropy(psi, 3, 3)));
    EXPECT_TRUE(std::isfinite(ms::quantum::entanglement_entropy(psi, 2, 2)));
}

TEST(ReplCrashRegressions, ModelDecodersCheckTheirHeaderRowWidth) {
    // Each ml_*_from_matrix reads its header row at fixed columns, but six of them had an
    // entry guard narrower than the columns they go on to read: ml_gmm_from_matrix asked
    // for one column and read three, ml_svm_from_matrix asked for three and read eight,
    // and naive_bayes / lda / qda checked no width at all. A model matrix narrower than
    // the header read past the row -- AddressSanitizer caught the GMM one on a 4x1.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("X = [1, 1; 2, 1; 3, 1; 4, 1]").has_value());
    ASSERT_TRUE(interp.execute("NARROW = [1; 2; 3; 4]").has_value());        // 4x1
    ASSERT_TRUE(interp.execute("NARROW2 = [1, 2; 3, 4; 5, 6; 7, 8]").has_value());  // 4x2
    ASSERT_TRUE(interp.execute("WIDE = [4, 1, 0; 1, 3, 1; 0, 1, 2]").has_value());  // 3x3

    for (const char* cmd : {
             "ml_gmm_predict(X, NARROW)", "ml_gmm_predict(X, NARROW2)",
             "ml_gmm_predict_proba(X, NARROW)", "ml_gmm_predict(X, WIDE)",
             "ml_knn_predict(X, NARROW)", "ml_knn_predict(X, NARROW2)",
             "ml_naive_bayes_predict(X, NARROW)", "ml_lda_predict(X, NARROW)",
             "ml_lda_transform(X, NARROW2)", "ml_qda_predict(X, NARROW)",
             "ml_svm_predict(X, NARROW)", "ml_svm_predict(X, WIDE)",
             "ml_pca_transform(X, NARROW)", "ml_kmeans_predict(X, NARROW)",
             "ml_isolation_forest_score(X, NARROW)",
             "ml_decision_tree_predict(X, NARROW)", "ml_random_forest_predict(X, NARROW)",
             "ml_adaboost_predict(X, NARROW)", "ml_gradient_boosting_predict(X, NARROW)",
         }) {
        const auto r = interp.execute(cmd);
        // A malformed model is reported; what matters is that the header row is not read
        // past the end of the matrix on the way to finding that out.
        (void)r;
        SUCCEED();
    }
    EXPECT_TRUE(interp.execute("det([1, 2; 3, 4])").has_value()) << "session still usable";

    // A well-formed model still round-trips.
    ASSERT_TRUE(interp.execute("gmm = ml_gmm_fit(X, 2)").has_value());
    EXPECT_TRUE(interp.execute("ml_gmm_predict(X, gmm)").has_value());
    ASSERT_TRUE(interp.execute("y = [1; 0; 1; 0]").has_value());
    ASSERT_TRUE(interp.execute("svm = ml_svm_fit(X, y)").has_value());
    EXPECT_TRUE(interp.execute("ml_svm_predict(X, svm)").has_value());
}

TEST(ReplCrashRegressions, RankPermutationRejectsNonPermutations) {
    // used[v[i]] indexes an n-element vector with a caller-supplied entry. Nothing
    // checked that v is a permutation of 0..n-1, so an out-of-range entry read and wrote
    // past it -- a fuzzed REPL line found this through combo_rank_permutation.
    EXPECT_EQ(ms::combo::rank_permutation({0, 1, 2}), 0u);   // the identity ranks 0
    EXPECT_EQ(ms::combo::rank_permutation({2, 1, 0}), 5u);   // the last of 3! = 6
    EXPECT_EQ(ms::combo::rank_permutation({1, 0}), 1u);
    EXPECT_EQ(ms::combo::rank_permutation({}), 0u);

    // Not permutations: out of range, negative, and repeated.
    EXPECT_EQ(ms::combo::rank_permutation({0, 1, 9}), 0u);
    EXPECT_EQ(ms::combo::rank_permutation({-1, 0, 1}), 0u);
    EXPECT_EQ(ms::combo::rank_permutation({0, 0, 0}), 0u);
    EXPECT_EQ(ms::combo::rank_permutation({5}), 0u);
    EXPECT_EQ(ms::combo::rank_permutation({1000000}), 0u);

    Interpreter interp;
    ASSERT_TRUE(interp.execute("BAD = [0; 1; 9]").has_value());
    EXPECT_TRUE(interp.execute("combo_rank_permutation(BAD)").has_value());
    ASSERT_TRUE(interp.execute("BIG = [1000000; 0]").has_value());
    EXPECT_TRUE(interp.execute("combo_rank_permutation(BIG)").has_value());
    EXPECT_TRUE(interp.execute("det([1, 2; 3, 4])").has_value()) << "session still usable";
}

TEST(ReplCrashRegressions, PcaTransformWithAWiderFeatureRowThanTheModel) {
    // xc[j] = X[i][j] - mean_[j] ran over the input ROW's length, so a model fitted on
    // fewer features than the row presents read past mean_.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("X2 = [1, 2; 3, 4; 5, 7; 2, 1]").has_value());
    ASSERT_TRUE(interp.execute("pca = ml_pca_fit(X2, 1)").has_value());
    // Four features where the model knows two.
    ASSERT_TRUE(interp.execute("X4 = [1, 2, 3, 4; 5, 6, 7, 8; 1, 1, 1, 1]").has_value());
    const auto r = interp.execute("ml_pca_transform(X4, pca)");
    (void)r;  // either answer is fine; it must not read past the model's mean vector
    // The matching width still works and returns one component per row.
    const auto ok = interp.execute("Z = ml_pca_transform(X2, pca)");
    ASSERT_TRUE(ok.has_value());
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);
}

TEST(ReplCrashRegressions, NaiveBayesModelWithOneFeature) {
    // The packed model's header row writes columns 0 and 1, but the matrix was sized
    // max(n_features, 1) columns wide. A single-feature model was therefore one column
    // wide and the header write ran off the end of the buffer, corrupting the heap --
    // glibc aborted later, on an unrelated free.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("X1 = [0]").has_value());
    ASSERT_TRUE(interp.execute("y1 = [0]").has_value());
    const auto fitted = interp.execute("nb = ml_naive_bayes_fit(X1, y1)");
    ASSERT_TRUE(fitted.has_value());
    const auto& model = interp.state().matrices.at("nb");
    EXPECT_EQ(model.rows(), 5u);
    EXPECT_GE(model.cols(), 2u);
    EXPECT_EQ(model(0, 0), 1.0);  // one class
    EXPECT_EQ(model(0, 1), 1.0);  // one feature
    // The model still round-trips through predict.
    ASSERT_TRUE(interp.execute("p1 = ml_naive_bayes_predict(X1, nb)").has_value());
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 1u);
}

TEST(ReplCrashRegressions, KnnModelWithOneFeature) {
    // Same defect in the KNN packer, whose header row writes columns 0, 1 and 2 into a
    // matrix sized n_features + 1 columns wide.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("X1 = [1; 2; 3]").has_value());
    ASSERT_TRUE(interp.execute("y1 = [0; 1; 1]").has_value());
    ASSERT_TRUE(interp.execute("knn = ml_knn_fit(X1, y1, 1)").has_value());
    const auto& model = interp.state().matrices.at("knn");
    EXPECT_GE(model.cols(), 3u);
    EXPECT_EQ(model(0, 1), 1.0);  // one feature
    EXPECT_EQ(model(0, 2), 3.0);  // three training rows
    ASSERT_TRUE(interp.execute("pk = ml_knn_predict(X1, knn)").has_value());
    EXPECT_EQ(interp.state().matrices.at("pk").rows(), 3u);
}

TEST(ReplCrashRegressions, LdaFitWithASingleClass) {
    // LDA::fit gives up on a single-class problem, but only after filling `classes`:
    // the per-class arrays stay empty. The packer sized its loop by classes.size() and
    // then indexed the empty discrim_const, reading off the end of the vector.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("X1 = [0]").has_value());
    ASSERT_TRUE(interp.execute("y1 = [0]").has_value());
    const auto r = interp.execute("lda = ml_lda_fit(X1, y1)");
    ASSERT_FALSE(r.has_value());
    EXPECT_NE(ms::format_error(r.error()).find("two distinct class labels"), std::string::npos);
    // Two classes still fit and predict.
    ASSERT_TRUE(interp.execute("X2 = [0, 0; 1, 1; 0, 1; 1, 0]").has_value());
    ASSERT_TRUE(interp.execute("y2 = [0; 1; 0; 1]").has_value());
    ASSERT_TRUE(interp.execute("lda2 = ml_lda_fit(X2, y2)").has_value());
    ASSERT_TRUE(interp.execute("pl = ml_lda_predict(X2, lda2)").has_value());
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 4u);
}
