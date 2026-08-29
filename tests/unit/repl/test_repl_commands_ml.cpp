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

TEST(ReplCommandsTest, ml_metrics) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_accuracy(p,t)");
    expect_contains(interp, "help", "ml_rmse(p,t)");

    expect_ok(interp, "yp = [1; 0; 1; 1]");
    expect_ok(interp, "yt = [1; 0; 0; 1]");
    expect_ok(interp, "acc = ml_accuracy(yp, yt)");
    expect_ok(interp, "rmse = ml_rmse([1; 2; 3], [1; 2; 4])");
    expect_ok(interp, "mse = ml_mse([1; 2; 3], [1; 2; 4])");
    expect_ok(interp, "r2 = ml_r2([1; 2; 3], [1; 2; 4])");
    expect_ok(interp, "f1 = ml_f1([1; 0; 1], [1; 0; 0])");

    EXPECT_DOUBLE_EQ(interp.state().scalars.at("acc"), 0.75);
    EXPECT_NEAR(interp.state().scalars.at("rmse"), 1.0 / std::sqrt(3.0), 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("mse"), 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("f1"), 2.0 / 3.0, 1e-9);

    expect_contains(interp, "ml_accuracy([1,0,1,1], [1,0,0,1])", "0.75");
}

TEST(ReplCommandsTest, ml_precision) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_precision(p,t)");

    expect_ok(interp, "prec = ml_precision([1; 1; 0; 0], [1; 0; 0; 1])");
    EXPECT_NEAR(interp.state().scalars.at("prec"), 0.5, 1e-9);

    expect_contains(interp, "ml_precision([1; 1; 0; 0], [1; 0; 0; 1])", "0.5");
}

TEST(ReplCommandsTest, ml_recall) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_recall(p,t)");

    expect_ok(interp, "rec = ml_recall([1; 1; 0; 0], [1; 0; 0; 1])");
    EXPECT_NEAR(interp.state().scalars.at("rec"), 0.5, 1e-9);

    expect_contains(interp, "ml_recall([1; 1; 0; 0], [1; 0; 0; 1])", "0.5");
}

TEST(ReplCommandsTest, ml_mae) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mae(p,t)");

    expect_ok(interp, "mae = ml_mae([1; 2; 3], [1; 3; 3])");
    EXPECT_NEAR(interp.state().scalars.at("mae"), 1.0 / 3.0, 1e-9);

    expect_contains(interp, "ml_mae([1; 2; 3], [1; 3; 3])", "0.333333");
}

TEST(ReplCommandsTest, ml_hinge) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_hinge(p,t)");

    expect_ok(interp, "hinge = ml_hinge([1.5; -0.5; 0.8], [1; -1; 1])");
    EXPECT_NEAR(interp.state().scalars.at("hinge"), 0.7 / 3.0, 1e-9);

    expect_contains(interp, "ml_hinge([1.5; -0.5; 0.8], [1; -1; 1])", "0.233333");
}

TEST(ReplCommandsTest, ml_huber) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_huber(p,t)");

    expect_ok(interp, "huber = ml_huber([0; 1; 5], [0; 0; 0])");
    EXPECT_GT(interp.state().scalars.at("huber"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("huber")));

    expect_contains(interp, "ml_huber([0; 1; 5], [0; 0; 0])", "1.666667");
}

TEST(ReplCommandsTest, ml_vec_norm) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_vec_norm(v)");

    expect_ok(interp, "vn = ml_vec_norm([3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("vn"), 5.0, 1e-9);

    expect_contains(interp, "ml_vec_norm([3; 4])", "5");
}

TEST(ReplCommandsTest, ml_vec_dot) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_vec_dot(a,b)");

    expect_ok(interp, "vd = ml_vec_dot([1; 2], [3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("vd"), 11.0, 1e-9);

    expect_contains(interp, "ml_vec_dot([1; 2], [3; 4])", "11");
}

TEST(ReplCommandsTest, ml_mat_transpose) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mat_transpose(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_EQ(interp.state().matrices.at("At").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 4.0, 1e-9);

    expect_ok(interp, "ml_mat_transpose(A)");
}

TEST(ReplCommandsTest, ml_mat_mul) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_mat_mul(A,B)");

    expect_ok(interp, "I = eye(3)");
    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "C = ml_mat_mul(I, A)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("C")(2, 2), 9.0, 1e-9);
}

TEST(ReplCommandsTest, ml_categorical_crossentropy) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_categorical_crossentropy(p,t)");

    expect_ok(interp, "cce = ml_categorical_crossentropy([0.7, 0.2, 0.1; 0.1, 0.8, 0.1], [1, 0, 0; 0, 1, 0])");
    EXPECT_GT(interp.state().scalars.at("cce"), 0.0);
    EXPECT_LT(interp.state().scalars.at("cce"), 1.0);

    expect_contains(interp,
                    "ml_categorical_crossentropy([0.7, 0.2, 0.1; 0.1, 0.8, 0.1], [1, 0, 0; 0, 1, 0])",
                    "0.289909");
}

TEST(ReplCommandsTest, ml_binary_crossentropy) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_binary_crossentropy(p,t)");

    expect_ok(interp, "bce = ml_binary_crossentropy([0.9; 0.1], [1; 0])");
    EXPECT_LT(interp.state().scalars.at("bce"), 0.15);

    expect_contains(interp, "ml_binary_crossentropy([0.9; 0.1], [1; 0])", "0.10536");
}

TEST(ReplCommandsTest, ml_supervised_regularized) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_lasso_fit(X,y,alpha)");
    expect_contains(interp, "help", "ml_lasso_predict(X,model)");
    expect_contains(interp, "help", "ml_elastic_net_fit(X,y,alpha,l1_ratio)");
    expect_contains(interp, "help", "ml_elastic_net_predict(X,model)");
    expect_contains(interp, "help", "ml_knn_fit(X,y,k)");
    expect_contains(interp, "help", "ml_knn_predict(X,model)");
    expect_contains(interp, "help", "ml_naive_bayes_fit(X,y)");
    expect_contains(interp, "help", "ml_naive_bayes_predict(X,model)");
    expect_contains(interp, "help", "ml_lda_fit(X,y");
    expect_contains(interp, "help", "ml_lda_predict(X,model)");
    expect_contains(interp, "help", "ml_lda_transform(X,model)");

    expect_ok(interp, "X = [0;1;2;3;4]");
    expect_ok(interp, "y = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(X, y, 0.001)");
    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);

    expect_ok(interp, "enet_m = ml_elastic_net_fit(X, y, 0.001, 0.5)");
    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    EXPECT_NEAR(interp.state().matrices.at("knn_p")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("knn_p")(1, 0), 1.0, 1e-6);

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0; 0; 0; 1; 1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);
    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
    EXPECT_GE(interp.state().matrices.at("lda_z").cols(), 1u);

    ms::ml::Mat X_ref = {{0}, {1}, {2}, {3}, {4}};
    ms::ml::Vec y_ref = {1, 3, 5, 7, 9};
    ms::ml::LassoRegression lasso_ref(0.001);
    lasso_ref.fit(X_ref, y_ref);
    EXPECT_NEAR(lasso_ref.predict({{5}})[0], interp.state().matrices.at("lasso_p")(0, 0), 0.5);
}

TEST(ReplCommandsTest, ml_unsupervised_pca_kmeans) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_pca_fit(X,n_components)");
    expect_contains(interp, "help", "ml_pca_transform(X,model)");
    expect_contains(interp, "help", "ml_pca_fit_transform(X,n_components)");
    expect_contains(interp, "help", "ml_kmeans_fit(X,k)");
    expect_contains(interp, "help", "ml_kmeans_predict(X,model)");
    expect_contains(interp, "help", "ml_kmeans_inertia(X,model)");

    expect_ok(interp, "X = [1, 0; 2, 0; 3, 0; 4, 0; 5, 0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("model").cols(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("km").cols(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
    const double l0 = interp.state().matrices.at("labels")(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("labels")(i, 0), l0, 1e-9);
    }
    const double l1 = interp.state().matrices.at("labels")(3, 0);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("labels")(i, 0), l1, 1e-9);
    }
    EXPECT_NE(l0, l1);

    expect_ok(interp, "inert = ml_kmeans_inertia(X2, km)");
    ASSERT_GT(interp.state().scalars.count("inert"), 0u);
    EXPECT_GT(interp.state().scalars.at("inert"), 0.0);
    EXPECT_LT(interp.state().scalars.at("inert"), 50.0);
}

TEST(ReplCommandsTest, ml_qda_svm) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_qda_fit(X,y)");
    expect_contains(interp, "help", "ml_qda_predict(X,model)");
    expect_contains(interp, "help", "ml_svm_fit(X,y");
    expect_contains(interp, "help", "ml_svm_predict(X,model)");

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0,0; 3,1], qda_m)");
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2,0; -1,0; 1,0; 2,0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5,0; 1.5,0], svm_m)");
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);

    ms::ml::Mat X_ref = {{-2, -1}, {-1, -0.5}, {2, 1}, {3, 1.5}};
    ms::ml::Vec y_ref = {0, 0, 1, 1};
    ms::ml::QDA qda_ref;
    qda_ref.fit(X_ref, y_ref);
    EXPECT_NEAR(qda_ref.predict({{0, 0}})[0], interp.state().matrices.at("qda_p")(0, 0), 1e-6);
}

TEST(ReplCommandsTest, ml_trees_ensemble) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_decision_tree_fit(X,y");
    expect_contains(interp, "help", "ml_decision_tree_predict(X,model)");
    expect_contains(interp, "help", "ml_random_forest_fit(X,y");
    expect_contains(interp, "help", "ml_random_forest_predict(X,model)");
    expect_contains(interp, "help", "ml_adaboost_fit(X,y");
    expect_contains(interp, "help", "ml_adaboost_predict(X,model)");

    expect_ok(interp, "X = [0,0; 0,1; 1,0; 1,1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    const auto& dt_p = interp.state().matrices.at("dt_p");
    int dt_correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(dt_p(i, 0) - interp.state().matrices.at("y")(i, 0)) < 0.5) {
            ++dt_correct;
        }
    }
    EXPECT_GE(dt_correct, 3);

    expect_ok(interp, "Rfx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    int rf_correct = 0;
    for (size_t i = 0; i < 6; ++i) {
        if (std::abs(interp.state().matrices.at("rf_p")(i, 0) -
                     interp.state().matrices.at("Rfy")(i, 0)) < 0.5) {
            ++rf_correct;
        }
    }
    EXPECT_GE(rf_correct, 5);

    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    int ab_correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(interp.state().matrices.at("ab_p")(i, 0) -
                     interp.state().matrices.at("y")(i, 0)) < 0.5) {
            ++ab_correct;
        }
    }
    EXPECT_GE(ab_correct, 3);

    ms::ml::Mat X_ref = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    ms::ml::Vec y_ref = {0, 1, 1, 0};
    ms::ml::DecisionTree dt_ref(5);
    dt_ref.fit(X_ref, y_ref);
    const auto dt_ref_pred = dt_ref.predict(X_ref);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(dt_p(i, 0), dt_ref_pred[i], 1e-9);
    }
}

TEST(ReplCommandsTest, ml_gradient_boosting) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_gradient_boosting_fit(X,y");
    expect_contains(interp, "help", "ml_gradient_boosting_predict(X,model)");

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    const auto& gb_p = interp.state().matrices.at("gb_p");
    EXPECT_EQ(gb_p.rows(), 5u);

    ms::ml::Mat X_ref = {{0}, {1}, {2}, {3}, {4}};
    ms::ml::Vec y_ref = {0, 1, 2, 3, 4};
    ms::ml::GradientBoosting gb_ref;
    gb_ref.config.n_trees = 20;
    gb_ref.config.learning_rate = 0.1;
    gb_ref.config.max_depth = 3;
    gb_ref.fit(X_ref, y_ref);
    const auto gb_ref_pred = gb_ref.predict(X_ref);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(gb_p(i, 0), gb_ref_pred[i], 1e-6);
    }

    double mse = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        const double err = gb_p(i, 0) - y_ref[i];
        mse += err * err;
    }
    mse /= 5.0;
    EXPECT_LT(mse, 0.5);
}

TEST(ReplCommandsTest, ml_gmm_dbscan) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_gmm_fit(X[,n_components])");
    expect_contains(interp, "help", "ml_gmm_predict(X,model)");
    expect_contains(interp, "help", "ml_gmm_predict_proba(X,model)");
    expect_contains(interp, "help", "ml_dbscan_fit(X,eps,min_samples)");
    expect_contains(interp, "help", "ml_spectral_clustering(X,k");

    expect_ok(interp, "X = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(X, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(X, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);
    const double g0 = interp.state().matrices.at("gmm_l")(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("gmm_l")(i, 0), g0, 1e-9);
    }
    const double g1 = interp.state().matrices.at("gmm_l")(3, 0);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("gmm_l")(i, 0), g1, 1e-9);
    }
    EXPECT_NE(g0, g1);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(X, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
    EXPECT_GE(interp.state().matrices.at("gmm_p").cols(), 2u);
    for (size_t i = 0; i < 6; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < interp.state().matrices.at("gmm_p").cols(); ++j) {
            row_sum += interp.state().matrices.at("gmm_p")(i, j);
        }
        EXPECT_NEAR(row_sum, 1.0, 1e-6);
    }

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);
    std::set<int> db_clusters;
    for (size_t i = 0; i < 10; ++i) {
        const int label = static_cast<int>(interp.state().matrices.at("db_l")(i, 0));
        if (label >= 0) {
            db_clusters.insert(label);
        }
    }
    EXPECT_EQ(db_clusters.size(), 2u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);
    std::set<int> sp_labels;
    for (size_t i = 0; i < 8; ++i) {
        sp_labels.insert(static_cast<int>(interp.state().matrices.at("sp_l")(i, 0)));
    }
    EXPECT_GE(sp_labels.size(), 2u);
}

TEST(ReplCommandsTest, ml_unsupervised_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_isolation_forest_fit(X[,n_trees[,sample_size[,seed]]])");
    expect_contains(interp, "help", "ml_isolation_forest_score(X,model)");
    expect_contains(interp, "help", "ml_agglomerative_fit(X[,n_clusters[,linkage]])");
    expect_contains(interp, "help", "ml_tsne_fit(X[,perplexity[,n_iter[,seed]]])");

    expect_ok(interp, "X = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(X, 50, 32, 42)");
    ASSERT_GT(interp.state().matrices.count("iso_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("iso_m").rows(), 2u);

    expect_ok(interp, "iso_s = ml_isolation_forest_score(X, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);
    double cluster_max = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        cluster_max = std::max(cluster_max, interp.state().matrices.at("iso_s")(i, 0));
    }
    EXPECT_GT(interp.state().matrices.at("iso_s")(5, 0), cluster_max);

    ms::ml::Mat X_ref;
    for (int i = 0; i < 5; ++i) {
        X_ref.push_back({0.1 * static_cast<double>(i), 0.0});
    }
    X_ref.push_back({10.0, 10.0});
    ms::ml::IsolationForest iso_ref(50, 32, 42);
    iso_ref.fit(X_ref);
    const auto ref_scores = iso_ref.anomaly_scores(X_ref);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("iso_s")(i, 0), ref_scores[i], 1e-9);
    }

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
    const double ac0 = interp.state().matrices.at("ac_l")(0, 0);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ac_l")(i, 0), ac0, 1e-9);
    }
    const double ac1 = interp.state().matrices.at("ac_l")(5, 0);
    for (size_t i = 5; i < 10; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ac_l")(i, 0), ac1, 1e-9);
    }
    EXPECT_NE(ac0, ac1);

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    ms::ml::Mat T_ref;
    T_ref.push_back({0, 0});
    T_ref.push_back({0.1, 0});
    T_ref.push_back({0.2, 0});
    T_ref.push_back({10, 10});
    T_ref.push_back({10.1, 10});
    T_ref.push_back({10.2, 10});
    ms::ml::TSNE tsne_ref(2, 5.0, 200.0, 10, 42);
    const auto Y_ref = tsne_ref.fit_transform(T_ref);
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(interp.state().matrices.at("Y")(i, j), Y_ref[i][j], 1e-9);
        }
    }
}

TEST(ReplCommandsTest, ml_scaler_roc) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_standard_scaler_fit(X)");
    expect_contains(interp, "help", "ml_standard_scaler_transform(X,model)");
    expect_contains(interp, "help", "ml_minmax_scaler_fit(X)");
    expect_contains(interp, "help", "ml_minmax_scaler_transform(X,model)");
    expect_contains(interp, "help", "ml_train_test_split(X,y");
    expect_contains(interp, "help", "ml_roc_auc(p,t)");
    expect_contains(interp, "help", "ml_average_precision(p,t)");

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    ASSERT_GT(interp.state().matrices.count("ss_m"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ss_m").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("ss_m").cols(), 2u);

    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    double col0_mean = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        col0_mean += interp.state().matrices.at("Z")(i, 0);
    }
    col0_mean /= 3.0;
    EXPECT_NEAR(col0_mean, 0.0, 1e-9);

    ms::ml::Mat X_cpp = {{1, 2}, {3, 4}, {5, 6}};
    ms::ml::StandardScaler sc_cpp;
    sc_cpp.fit(X_cpp);
    EXPECT_NEAR(interp.state().matrices.at("ss_m")(0, 0), sc_cpp.mean_[0], 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("ss_m")(1, 0), sc_cpp.std_[0], 1e-9);

    expect_ok(interp, "Xmm = [0, 10; 5, 20; 10, 30]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(Xmm)");
    expect_ok(interp, "Zmm = ml_minmax_scaler_transform(Xmm, mm_m)");
    EXPECT_NEAR(interp.state().matrices.at("Zmm")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Zmm")(2, 0), 1.0, 1e-9);

    expect_ok(interp, "y = [0; 1; 0; 1; 0; 1; 0; 1; 0; 1]");
    expect_ok(interp, "X10 = [0,0;1,1;2,2;3,3;4,4;5,5;6,6;7,7;8,8;9,9]");
    expect_ok(interp, "Xtr, ytr, Xte, yte = ml_train_test_split(X10, y, 0.2, 42)");
    ASSERT_GT(interp.state().matrices.count("Xtr"), 0u);
    ASSERT_GT(interp.state().matrices.count("Xte"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Xtr").rows() +
                  interp.state().matrices.at("Xte").rows(),
              10u);

    expect_ok(interp, "pred = [0.9; 0.8; 0.2; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0]");
    expect_ok(interp, "auc = ml_roc_auc(pred, true_l)");
    expect_ok(interp, "ap = ml_average_precision(pred, true_l)");
    ASSERT_GT(interp.state().scalars.count("auc"), 0u);
    ASSERT_GT(interp.state().scalars.count("ap"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("auc"), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("ap"), 1.0, 1e-9);
}

TEST(ReplCommandsTest, ml_metrics_2) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_confusion_matrix(p,t");
    expect_contains(interp, "help", "ml_roc_curve(p,t)");
    expect_contains(interp, "help", "ml_precision_recall_curve(p,t)");
    expect_contains(interp, "help", "[[TP,FP],[FN,TN]]");

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 1), 2.0, 1e-9);

    expect_ok(interp, "cm9 = ml_confusion_matrix(pred, true_l, 0.9)");
    EXPECT_NEAR(interp.state().matrices.at("cm9")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm9")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm9")(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm9")(1, 1), 3.0, 1e-9);

    expect_ok(interp, "pred_p = [1;1;1;1;1;0;0;0;0;0]");
    expect_ok(interp, "true_p = [1;1;1;1;1;0;0;0;0;0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);
    EXPECT_GE(interp.state().matrices.at("roc").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("roc")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("roc")(0, 2), 0.0, 1e-9);
    const size_t roc_last = interp.state().matrices.at("roc").rows() - 1;
    EXPECT_NEAR(interp.state().matrices.at("roc")(roc_last, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("roc")(roc_last, 2), 1.0, 1e-9);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);
    EXPECT_GE(interp.state().matrices.at("pr").rows(), 2u);
    ms::ml::Vec pred_p_vec = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    ms::ml::Vec true_p_vec = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
    const auto pr_cpp = ms::ml::precision_recall_curve(pred_p_vec, true_p_vec);
    ASSERT_FALSE(pr_cpp.empty());
    EXPECT_NEAR(interp.state().matrices.at("pr")(0, 1), pr_cpp.front().precision, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pr")(0, 2), pr_cpp.front().recall, 1e-9);
    const size_t pr_last = interp.state().matrices.at("pr").rows() - 1;
    EXPECT_NEAR(interp.state().matrices.at("pr")(pr_last, 2), pr_cpp.back().recall, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("pr")(pr_last, 1), pr_cpp.back().precision, 1e-9);

    ms::ml::Vec pred_cpp = {0.9, 0.4, 0.6, 0.3, 0.5, 0.1};
    ms::ml::Vec true_cpp = {1, 1, 0, 0, 1, 0};
    const auto cm_cpp = ms::ml::confusion_matrix(pred_cpp, true_cpp, 0.5);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 0), static_cast<double>(cm_cpp.tp), 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 1), static_cast<double>(cm_cpp.fp), 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 0), static_cast<double>(cm_cpp.fn), 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 1), static_cast<double>(cm_cpp.tn), 1e-9);
}

TEST(ReplCommandsTest, ml_finance) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "log_m = ml_logistic_fit(X, y)");
    expect_ok(interp, "log_p = ml_logistic_predict([2], log_m)");
    ASSERT_GT(interp.state().matrices.count("log_p"), 0u);

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);

    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "pi = [0.05; 0.07]");
    expect_ok(interp, "P = [1, 0]");
    expect_ok(interp, "Q = [0.10]");
    expect_ok(interp, "post = finance_bl_posterior_returns_default_omega(pi, cov2, P, Q, 0.05)");
    ASSERT_GT(interp.state().matrices.count("post"), 0u);

    expect_ok(interp, "merton = finance_merton_implied_asset_params(120, 0.3, 100, 0.05, 1)");
    EXPECT_EQ(interp.state().matrices.at("merton").cols(), 6u);
}

TEST(ReplCommandsTest, ml_naive_bayes_lda_pca) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0; 0; 0; 1; 1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
    EXPECT_GE(interp.state().matrices.at("lda_z").cols(), 1u);

    expect_ok(interp, "X = [1, 0; 2, 0; 3, 0; 4, 0; 5, 0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, ml_kmeans_qda_svm) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);
    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
    const double l0 = interp.state().matrices.at("labels")(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("labels")(i, 0), l0, 1e-9);
    }
    const double l1 = interp.state().matrices.at("labels")(3, 0);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("labels")(i, 0), l1, 1e-9);
    }
    EXPECT_NE(l0, l1);

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0,0; 3,1], qda_m)");
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2,0; -1,0; 1,0; 2,0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5,0; 1.5,0], svm_m)");
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, ml_tree_ensemble) {
    Interpreter interp;

    expect_ok(interp, "X = [0,0; 0,1; 1,0; 1,1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);
    int dt_correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(interp.state().matrices.at("dt_p")(i, 0) -
                     interp.state().matrices.at("y")(i, 0)) < 0.5) {
            ++dt_correct;
        }
    }
    EXPECT_GE(dt_correct, 3);

    expect_ok(interp, "Rfx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
    int rf_correct = 0;
    for (size_t i = 0; i < 6; ++i) {
        if (std::abs(interp.state().matrices.at("rf_p")(i, 0) -
                     interp.state().matrices.at("Rfy")(i, 0)) < 0.5) {
            ++rf_correct;
        }
    }
    EXPECT_GE(rf_correct, 5);

    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);
}

TEST(ReplCommandsTest, ml_gmm_dbscan_spectral) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(X, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(X, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);
    const double g0 = interp.state().matrices.at("gmm_l")(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("gmm_l")(i, 0), g0, 1e-9);
    }
    const double g1 = interp.state().matrices.at("gmm_l")(3, 0);
    for (size_t i = 3; i < 6; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("gmm_l")(i, 0), g1, 1e-9);
    }
    EXPECT_NE(g0, g1);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(X, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
    EXPECT_GE(interp.state().matrices.at("gmm_p").cols(), 2u);
    for (size_t i = 0; i < 6; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < interp.state().matrices.at("gmm_p").cols(); ++j) {
            row_sum += interp.state().matrices.at("gmm_p")(i, j);
        }
        EXPECT_NEAR(row_sum, 1.0, 1e-6);
    }

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);
    std::set<int> db_clusters;
    for (size_t i = 0; i < 10; ++i) {
        const int label = static_cast<int>(interp.state().matrices.at("db_l")(i, 0));
        if (label >= 0) {
            db_clusters.insert(label);
        }
    }
    EXPECT_EQ(db_clusters.size(), 2u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);
    std::set<int> sp_labels;
    for (size_t i = 0; i < 8; ++i) {
        sp_labels.insert(static_cast<int>(interp.state().matrices.at("sp_l")(i, 0)));
    }
    EXPECT_GE(sp_labels.size(), 2u);
}

TEST(ReplCommandsTest, ml_scaler_compress) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    double col0_mean = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        col0_mean += interp.state().matrices.at("Z")(i, 0);
    }
    col0_mean /= 3.0;
    EXPECT_NEAR(col0_mean, 0.0, 1e-9);

    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    EXPECT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    EXPECT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, ml_linear_ridge) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, ml_metrics_3) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("cm")(1, 1), 2.0, 1e-9);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("roc")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("roc")(0, 2), 0.0, 1e-9);
    const size_t roc_last = interp.state().matrices.at("roc").rows() - 1;
    EXPECT_NEAR(interp.state().matrices.at("roc")(roc_last, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("roc")(roc_last, 2), 1.0, 1e-9);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);
    EXPECT_GE(interp.state().matrices.at("pr").rows(), 2u);
}

TEST(ReplCommandsTest, ml_gboost_isolation) {
    Interpreter interp;

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);
    double cluster_max = 0.0;
    for (size_t i = 0; i < 5; ++i) {
        cluster_max = std::max(cluster_max, interp.state().matrices.at("iso_s")(i, 0));
    }
    EXPECT_GT(interp.state().matrices.at("iso_s")(5, 0), cluster_max);
}

TEST(ReplCommandsTest, ml_agglo_tsne_golomb) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
    const double ac0 = interp.state().matrices.at("ac_l")(0, 0);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ac_l")(i, 0), ac0, 1e-9);
    }
    const double ac1 = interp.state().matrices.at("ac_l")(5, 0);
    for (size_t i = 5; i < 10; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ac_l")(i, 0), ac1, 1e-9);
    }
    EXPECT_NE(ac0, ac1);

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
    const double golomb_expected[] = {0, 1, 2, 5, 10};
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("GR")(i, 0), golomb_expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, logistic) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
}

TEST(ReplCommandsTest, lasso) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, enet) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, knn) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, nb) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, lda) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, pca) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_iso_agglo) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, logistic_2) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
}

TEST(ReplCommandsTest, lasso_2) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, enet_2) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, knn_2) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, nb_2) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, lda_2) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, pca_2) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_2) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_2) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_2) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_2) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_2) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_2) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_2) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_3) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_3) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_3) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_3) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_3) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_3) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_3) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_2) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_2) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_2) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_2) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_2) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_2) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_2) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_2) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_2) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_2) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_4) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_4) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_4) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_4) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_4) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_4) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_4) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_3) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_3) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_3) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_3) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_3) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_3) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_3) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_3) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_3) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_3) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_5) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_5) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_5) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_5) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_5) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_5) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_5) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_4) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_4) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_4) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_4) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_4) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_4) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_4) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_4) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_4) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_4) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_6) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_6) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_6) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_6) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_6) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_6) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_6) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_5) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_5) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_5) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_5) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_5) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_5) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_5) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_5) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_5) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_5) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_7) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_7) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_7) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_7) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_7) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_7) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_7) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_6) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_6) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_6) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_6) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_6) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_6) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_6) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_6) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_6) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_6) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_8) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_8) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_8) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_8) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_8) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_8) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_8) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_7) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_7) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_7) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_7) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_7) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_7) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_7) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_7) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_7) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_7) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_9) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_9) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_9) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_9) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_9) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_9) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_9) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_8) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_8) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_8) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_8) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_8) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_8) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_8) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_8) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_8) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_8) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_10) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_10) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_10) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_10) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_10) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_10) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_10) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_9) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_9) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_9) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_9) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_9) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_9) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_9) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_9) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_9) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_9) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_11) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_11) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_11) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_11) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_11) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_11) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_11) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_10) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_10) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_10) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_10) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_10) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_10) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_10) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_10) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_10) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_10) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_12) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_12) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_12) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_12) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_12) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_12) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_12) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_11) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_11) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_11) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_11) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_11) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_11) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_11) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_11) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_11) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_11) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_13) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_13) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_13) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_13) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_13) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_13) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_13) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_12) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_12) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_12) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_12) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_12) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_12) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_12) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_12) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_12) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_12) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_14) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_14) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_14) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_14) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_14) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_14) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_14) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_13) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_13) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_13) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_13) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_13) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_13) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_13) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_13) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_13) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_13) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_15) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_15) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_15) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_15) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_15) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_15) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_15) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_14) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_14) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_14) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_14) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_14) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_14) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_14) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_14) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_14) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_14) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_16) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_16) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_16) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_16) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_16) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_16) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_16) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_15) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_15) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_15) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_15) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_15) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_15) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_15) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_15) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_15) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_15) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_17) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_17) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_17) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_17) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_17) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_17) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_17) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_16) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_16) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_16) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_16) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_16) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_16) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_16) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_16) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_16) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_16) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_18) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_18) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_18) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_18) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_18) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_18) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_18) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_17) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_17) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_17) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_17) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_17) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_17) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_17) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_17) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_17) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_17) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_19) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_19) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_19) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_19) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_19) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_19) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_19) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_18) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_18) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_18) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_18) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_18) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_18) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_18) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_18) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_18) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_18) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_20) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_20) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_20) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_20) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_20) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_20) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_20) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_19) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_19) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_19) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_19) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_19) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_19) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_19) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_19) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_19) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_19) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_21) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_21) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_21) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_21) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_21) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_21) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_21) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_20) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_20) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_20) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_20) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_20) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_20) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_20) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_20) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_20) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_20) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_22) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_22) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_22) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_22) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_22) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_22) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_22) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_21) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_21) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_21) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_21) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_21) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_21) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_21) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_21) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_21) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_21) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_23) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_23) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_23) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_23) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_23) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_23) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_23) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_22) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_22) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_22) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_22) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_22) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_22) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_22) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_22) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_22) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_22) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_24) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_24) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_24) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_24) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_24) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_24) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_24) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_23) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_23) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_23) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_23) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_23) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_23) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_23) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_23) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_23) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_23) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_25) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_25) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_25) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_25) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_25) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_25) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_25) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_24) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_24) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_24) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_24) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_24) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_24) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_24) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_24) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_24) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_24) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_26) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_26) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_26) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_26) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_26) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_26) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_26) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_25) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_25) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_25) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_25) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_25) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_25) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_25) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_25) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_25) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_25) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_27) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_27) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_27) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_27) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_27) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_27) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_27) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_26) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_26) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_26) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_26) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_26) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_26) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_26) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_26) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_26) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_26) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_28) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_28) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_28) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_28) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_28) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_28) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_28) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_27) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_27) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_27) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_27) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_27) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_27) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_27) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_27) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_27) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_27) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_29) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_29) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_29) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_29) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_29) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_29) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_29) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_28) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_28) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_28) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_28) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_28) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_28) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_28) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_28) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_28) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_28) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_30) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_30) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_30) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_30) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_30) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_30) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_30) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_29) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_29) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_29) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_29) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_29) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_29) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_29) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_29) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_29) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_29) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_31) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_31) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_31) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_31) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_31) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_31) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_31) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_30) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_30) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_30) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_logistic_fit_ml_logistic_predict_30) {
    Interpreter interp;

    expect_ok(interp, "X = [-2; -1; 1; 2]");
    expect_ok(interp, "y = [0; 0; 1; 1]");
    expect_ok(interp, "model = ml_logistic_fit(X, y)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);

    expect_ok(interp, "pred = ml_logistic_predict([2], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_GT(interp.state().matrices.at("pred").rows(), 0u);
}

TEST(ReplCommandsTest, ml_lasso_fit_ml_lasso_predict_30) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "lasso_m = ml_lasso_fit(Xr, yr, 0.001)");
    ASSERT_GT(interp.state().matrices.count("lasso_m"), 0u);

    expect_ok(interp, "lasso_p = ml_lasso_predict([5], lasso_m)");
    ASSERT_GT(interp.state().matrices.count("lasso_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("lasso_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_elastic_net_fit_ml_elastic_net_predict_30) {
    Interpreter interp;

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    ASSERT_GT(interp.state().matrices.count("enet_m"), 0u);

    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    ASSERT_GT(interp.state().matrices.count("enet_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(ReplCommandsTest, ml_knn_fit_ml_knn_predict_30) {
    Interpreter interp;

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0;0;0;1;1;1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    ASSERT_GT(interp.state().matrices.count("knn_m"), 0u);

    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(ReplCommandsTest, ml_naive_bayes_fit_ml_naive_bayes_predict_30) {
    Interpreter interp;

    expect_ok(interp, "Nbx = [1,1; 2,2; 1.5,1.5; -1,-1; -2,-2]");
    expect_ok(interp, "Nby = [0;0;0;1;1]");
    expect_ok(interp, "nb_m = ml_naive_bayes_fit(Nbx, Nby)");
    ASSERT_GT(interp.state().matrices.count("nb_m"), 0u);

    expect_ok(interp, "nb_p = ml_naive_bayes_predict([1,1], nb_m)");
    ASSERT_GT(interp.state().matrices.count("nb_p"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("nb_p")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, ml_lda_fit_ml_lda_predict_ml_lda_transform_30) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0;0;1;1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    ASSERT_GT(interp.state().matrices.count("lda_m"), 0u);

    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    ASSERT_GT(interp.state().matrices.count("lda_z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(ReplCommandsTest, ml_pca_fit_ml_pca_transform_ml_pca_fit_transform_30) {
    Interpreter interp;

    expect_ok(interp, "X = [1,0; 2,0; 3,0; 4,0; 5,0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    ASSERT_GT(interp.state().matrices.count("model"), 0u);
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    ASSERT_GT(interp.state().matrices.count("Z2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(ReplCommandsTest, linear_ridge_32) {
    Interpreter interp;

    expect_ok(interp, "X = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = [3; 5; 7; 9; 11]");
    expect_ok(interp, "model = ml_linear_fit(X, y)");
    expect_ok(interp, "pred = ml_linear_predict([6], model)");
    ASSERT_GT(interp.state().matrices.count("pred"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("pred")(0, 0), 13.0, 0.2);

    expect_ok(interp, "Xr = [0; 1; 2; 3]");
    expect_ok(interp, "yr = [1; 3; 5; 7]");
    expect_ok(interp, "rm = ml_ridge_fit(Xr, yr, 0.001)");
    expect_ok(interp, "rp = ml_ridge_predict([4], rm)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("rp")(0, 0), 9.0, 0.5);
}

TEST(ReplCommandsTest, kmeans_32) {
    Interpreter interp;

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    ASSERT_GT(interp.state().matrices.count("km"), 0u);
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    ASSERT_GT(interp.state().matrices.count("labels"), 0u);
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(ReplCommandsTest, qda_svm_32) {
    Interpreter interp;

    expect_ok(interp, "Lx = [-2, -1; -1, -0.5; 2, 1; 3, 1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "qda_m = ml_qda_fit(Lx, Ly)");
    expect_ok(interp, "qda_p = ml_qda_predict([0, 0; 3, 1], qda_m)");
    ASSERT_GT(interp.state().matrices.count("qda_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("qda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("qda_p")(1, 0), 0.5);

    expect_ok(interp, "Sx = [-2, 0; -1, 0; 1, 0; 2, 0]");
    expect_ok(interp, "Sy = [0; 0; 1; 1]");
    expect_ok(interp, "svm_m = ml_svm_fit(Sx, Sy)");
    expect_ok(interp, "svm_p = ml_svm_predict([-1.5, 0; 1.5, 0], svm_m)");
    ASSERT_GT(interp.state().matrices.count("svm_p"), 0u);
    EXPECT_LT(interp.state().matrices.at("svm_p")(0, 0), 0.0);
    EXPECT_GT(interp.state().matrices.at("svm_p")(1, 0), 0.0);
}

TEST(ReplCommandsTest, tree_forest_32) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "dt_m = ml_decision_tree_fit(X, y, 5)");
    expect_ok(interp, "dt_p = ml_decision_tree_predict(X, dt_m)");
    ASSERT_GT(interp.state().matrices.count("dt_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt_p").rows(), 4u);

    expect_ok(interp, "Rfx = [0, 0; 1, 0; 0, 1; 3, 3; 4, 3; 3, 4]");
    expect_ok(interp, "Rfy = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "rf_m = ml_random_forest_fit(Rfx, Rfy, 25, 4)");
    expect_ok(interp, "rf_p = ml_random_forest_predict(Rfx, rf_m)");
    ASSERT_GT(interp.state().matrices.count("rf_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rf_p").rows(), 6u);
}

TEST(ReplCommandsTest, adaboost_gmm_32) {
    Interpreter interp;

    expect_ok(interp, "X = [0, 0; 0, 1; 1, 0; 1, 1]");
    expect_ok(interp, "y = [0; 1; 1; 0]");
    expect_ok(interp, "ab_m = ml_adaboost_fit(X, y, 50, 3)");
    expect_ok(interp, "ab_p = ml_adaboost_predict(X, ab_m)");
    ASSERT_GT(interp.state().matrices.count("ab_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ab_p").rows(), 4u);

    expect_ok(interp, "Gx = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "gmm_m = ml_gmm_fit(Gx, 2)");
    ASSERT_GT(interp.state().matrices.count("gmm_m"), 0u);
    EXPECT_GE(interp.state().matrices.at("gmm_m").rows(), 4u);

    expect_ok(interp, "gmm_l = ml_gmm_predict(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_l").rows(), 6u);

    expect_ok(interp, "gmm_p = ml_gmm_predict_proba(Gx, gmm_m)");
    ASSERT_GT(interp.state().matrices.count("gmm_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gmm_p").rows(), 6u);
}

TEST(ReplCommandsTest, dbscan_spectral_scaler_32) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    ASSERT_GT(interp.state().matrices.count("db_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    ASSERT_GT(interp.state().matrices.count("sp_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(ReplCommandsTest, minmax_encode_32) {
    Interpreter interp;

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    ASSERT_GT(interp.state().matrices.count("Zm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    ASSERT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    ASSERT_GT(interp.state().matrices.count("ans"), 0u);
    ASSERT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(ReplCommandsTest, metrics_gboost_31) {
    Interpreter interp;

    expect_ok(interp, "pred = [0.9; 0.4; 0.6; 0.3; 0.5; 0.1]");
    expect_ok(interp, "true_l = [1; 1; 0; 0; 1; 0]");
    expect_ok(interp, "cm = ml_confusion_matrix(pred, true_l)");
    ASSERT_GT(interp.state().matrices.count("cm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cm").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("cm").cols(), 2u);

    expect_ok(interp, "pred_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "true_p = [1; 1; 1; 1; 1; 0; 0; 0; 0; 0]");
    expect_ok(interp, "roc = ml_roc_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("roc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("roc").cols(), 3u);

    expect_ok(interp, "pr = ml_precision_recall_curve(pred_p, true_p)");
    ASSERT_GT(interp.state().matrices.count("pr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pr").cols(), 3u);

    expect_ok(interp, "X = [0; 1; 2; 3; 4]");
    expect_ok(interp, "y = [0; 1; 2; 3; 4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    ASSERT_GT(interp.state().matrices.count("gb_p"), 0u);
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);
}

TEST(ReplCommandsTest, iso_agglo_31) {
    Interpreter interp;

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    ASSERT_GT(interp.state().matrices.count("iso_s"), 0u);
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    ASSERT_GT(interp.state().matrices.count("ac_l"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(ReplCommandsTest, tsne_golomb_31) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(ReplCommandsTest, ml_train_test_split_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "X10 = [0,0;1,1;2,2;3,3;4,4;5,5;6,6;7,7;8,8;9,9]");
    expect_ok(interp, "y = [0; 1; 0; 1; 0; 1; 0; 1; 0; 1]");
    expect_ok(interp, "Xtr, ytr, Xte, yte = ml_train_test_split(X10, y, 0.2, 42)");
    expect_error(interp, "ml_train_test_split(X10, y, 0.2, 42)");
    expect_error_contains(interp, "Xtr, ytr, Xte, yte = ml_train_test_split(X10, y, 0.2, 1.5)",
                          "integer seed");
}

TEST(ReplCommandsTest, ml_vec_norm_noassign) {
    Interpreter interp;
    expect_contains(interp, "ml_vec_norm([3; 4])", "5");
    expect_error_contains(interp, "ml_vec_norm(no_such_matrix)", "unknown matrix");
}
