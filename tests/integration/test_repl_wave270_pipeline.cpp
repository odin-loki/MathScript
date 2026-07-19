// MathScript Integration Tests: REPL Interpreter – Wave 270 Pipeline
//
// Wave 270 REPL smoke (pipeline-only; no feature bindings in this wave):
// cross-check stable Wave 269 APIs that Wave 270 feature branches will extend.
//
// Covered: ml_confusion_matrix, sparse_from_coo, golomb_rice_encode_vec,
// fem_poisson1d, brentq.
//
// Wave 270 APIs (deferred — add tests when bindings land):
//   ml_gradient_boosting_fit, ml_isolation_forest_fit, ml_agglomerative_fit, ml_tsne_fit,
//   ml_precision_recall_curve, ml_roc_curve, wavelet_compress_vec, sparse_spmv,
//   tensorops_decompose_nmf, topo_alpha_complex, quantum_wigner,
//   ode_adams_bashforth2, cfd_advection1d, diffgeo_helix_torsion,
//   ml_standard_scaler_transform, ml_minmax_scaler_fit, ml_train_test_split, ml_roc_auc.

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

double parse_scalar_x_opt(const std::string& text) {
    const auto pos = text.find("x_opt = ");
    if (pos == std::string::npos) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(text.substr(pos + 8));
}

void expect_near_in_output(Interpreter& interp, const std::string& cmd, double expected,
                           double tol) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    const double x_opt = parse_scalar_x_opt(*result);
    ASSERT_FALSE(std::isnan(x_opt)) << cmd << " output: " << *result;
    EXPECT_NEAR(x_opt, expected, tol) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave270Pipeline, MlConfusionMatrix) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 0), 2.0, 1e-9);
    expect_contains(interp, "help", "ml_confusion_matrix(");
}

TEST(ReplWave270Pipeline, SparseFromCoo) {
    Interpreter interp;

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("A").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("A")(0, 0), 3.0, 1e-9);
    expect_contains(interp, "help", "sparse_from_coo(");
}

TEST(ReplWave270Pipeline, GolombRiceEncodeVec) {
    Interpreter interp;

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    ASSERT_GT(interp.state().matrices.count("GE"), 0u);
    EXPECT_GE(interp.state().matrices.at("GE").rows(), 1u);
    expect_contains(interp, "help", "golomb_rice_encode_vec(");
}

TEST(ReplWave270Pipeline, FemPoisson1d) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(8)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GE(interp.state().matrices.at("u1").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("u1").cols(), 1u);
    EXPECT_GT(interp.state().matrices.at("u1")(4, 0), 0.0);
    expect_contains(interp, "help", "fem_poisson1d(");
}

TEST(ReplWave270Pipeline, Brentq) {
    Interpreter interp;

    // (x-2)(x+1) has root x=2 on [0, 5]
    expect_near_in_output(interp, R"cmd(brentq("(x0-2)*(x0+1)", 0, 5))cmd", 2.0, 1e-6);
    expect_contains(interp, "help", "brentq(");
}
