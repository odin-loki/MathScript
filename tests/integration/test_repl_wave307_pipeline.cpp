// MathScript Integration Tests: REPL Interpreter – Wave 307 Pipeline
//
// Wave 307 REPL smoke: naive Bayes/LDA/PCA tail11 extensions, Mathieu scalar validation.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave307Pipeline, MlNaiveBayesLdaPcaTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_naive_bayes_fit(X,y)");
    expect_contains(interp, "help", "ml_lda_fit(X,y");
    expect_contains(interp, "help", "ml_pca_fit(X,n_components)");

    expect_ok(interp, "Nbx = [1,1; 2,2; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0; 0; 1; 1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    EXPECT_GT(interp.state().matrices.at("nb_p").rows(), 0u);

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    expect_ok(interp, "lda_p = ml_lda_predict([0,0], lda_m)");
    EXPECT_GT(interp.state().matrices.at("lda_p").rows(), 0u);

    expect_ok(interp, "X = [1, 0; 2, 0; 3, 0; 4, 0; 5, 0]");
    expect_ok(interp, "pca_m = ml_pca_fit(X, 1)");
    expect_ok(interp, "Z = ml_pca_transform(X, pca_m)");
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplWave307Pipeline, MathieuScalar) {
    Interpreter interp;

    const double a_ref = ms::mathieu_a(1, 0.1);
    expect_ok(interp, "a = mathieu_a(1, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("a"), a_ref, 1e-3);

    expect_ok(interp, "b = mathieu_b(1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("b"), ms::mathieu_b(1, 0.0), 1e-8);
}
