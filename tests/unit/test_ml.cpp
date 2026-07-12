#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>

using namespace ms::ml;

// ---- Matrix utilities ----

TEST(MLMatUtils, EyeAndMul) {
    auto I=mat_eye(3);
    Mat A={{1,2,3},{4,5,6},{7,8,9}};
    auto C=mat_mul(I,A);
    for (size_t i=0;i<A.size();++i)
        for (size_t j=0;j<A[i].size();++j)
            EXPECT_NEAR(C[i][j], A[i][j], 1e-12);
}

TEST(MLMatUtils, TransposeAndVecOps) {
    Mat A={{1,2,3},{4,5,6}};
    auto At=mat_T(A);
    EXPECT_EQ(At.size(), 3u);
    EXPECT_EQ(At[0].size(), 2u);
    EXPECT_EQ(At[0][0], 1);
    EXPECT_EQ(At[2][1], 6);
    Vec u={1,0,0}, v={0,1,0};
    EXPECT_NEAR(vec_dot(u,v), 0.0, 1e-12);
    EXPECT_NEAR(vec_norm({3,4}), 5.0, 1e-12);
    auto y=mat_vec(A,{1,1,1});
    EXPECT_NEAR(y[0], 6.0, 1e-12);
    EXPECT_NEAR(y[1], 15.0, 1e-12);
}

TEST(MLMatUtils, MatZerosAndVecLinearOps) {
    auto Z=mat_zeros(2,3);
    EXPECT_EQ(Z.size(), 2u);
    EXPECT_EQ(Z[0].size(), 3u);
    for (auto& row:Z)
        for (double v:row)
            EXPECT_DOUBLE_EQ(v, 0.0);
    Vec a={1,2,3}, b={4,5,6};
    auto s=vec_add(a,b);
    EXPECT_NEAR(s[0], 5.0, 1e-12);
    EXPECT_NEAR(s[2], 9.0, 1e-12);
    auto d=vec_sub(b,a);
    EXPECT_NEAR(d[1], 3.0, 1e-12);
    auto sc=vec_scale(2.0,a);
    EXPECT_NEAR(sc[0], 2.0, 1e-12);
    EXPECT_NEAR(sc[2], 6.0, 1e-12);
}

// ---- Linear Regression ----

TEST(MLLinReg, PerfectFit) {
    // y = 2x + 1
    Mat X={{1},{2},{3},{4},{5}};
    Vec y={3,5,7,9,11};
    LinearRegression lr;
    lr.fit(X,y);
    EXPECT_NEAR(lr.coef[0], 2.0, 1e-6);
    EXPECT_NEAR(lr.intercept, 1.0, 1e-6);
    EXPECT_NEAR(lr.score(X,y), 1.0, 1e-6);
}

TEST(MLLinReg, Predict) {
    Mat X={{0},{1},{2}};
    Vec y={1,2,3};
    LinearRegression lr;
    lr.fit(X,y);
    auto p=lr.predict({{3}});
    EXPECT_NEAR(p[0], 4.0, 0.01);
}

// ---- Ridge Regression ----

TEST(MLRidge, SimpleRegression) {
    Mat X={{1},{2},{3},{4}};
    Vec y={2,4,6,8};
    RidgeRegression rr(0.001);
    rr.fit(X,y);
    EXPECT_NEAR(rr.coef[0], 2.0, 0.1);
    EXPECT_GT(rr.score(X,y), 0.99);
}

TEST(MLRidge, PredictWithIntercept) {
    Mat X={{0},{1},{2},{3}};
    Vec y={1,3,5,7};
    RidgeRegression rr(0.001);
    rr.fit(X,y);
    auto p=rr.predict({{4}});
    EXPECT_NEAR(p[0], 9.0, 0.2);
}

// ---- Lasso ----

TEST(MLLasso, SparsifyCoef) {
    Mat X={{1,0},{0,1},{1,1},{0,0}};
    Vec y={1,0,1,0};
    LassoRegression lasso(0.1);
    lasso.fit(X,y);
    EXPECT_EQ(lasso.coef.size(), 2u);
}

TEST(MLLasso, Predict) {
    Mat X={{1,0},{0,1},{1,1}};
    Vec y={1,0,1};
    LassoRegression lasso(0.01);
    lasso.fit(X,y);
    auto p=lasso.predict({{1,0}});
    EXPECT_GT(p[0], 0.5);
}

// ---- Elastic Net ----

// Small toy dataset with a couple of informative features and shared across
// the limiting-behaviour tests below so ElasticNet can be compared directly
// against LassoRegression/RidgeRegression on identical data.
static std::pair<Mat,Vec> elasticnet_toy_data() {
    Mat X={{1,0},{0,1},{1,1},{0,0},{2,0},{0,2},{2,2},{1,2},{2,1}};
    Vec y={1,0,1,0,2,0,2,1,1.5};
    return {X,y};
}

TEST(MLElasticNet, L1RatioOneMatchesLasso) {
    auto [X,y]=elasticnet_toy_data();
    double alpha=0.1;
    LassoRegression lasso(alpha);
    lasso.fit(X,y);
    ElasticNet enet(alpha, /*l1_ratio=*/1.0);
    enet.fit(X,y);
    ASSERT_EQ(enet.coef.size(), lasso.coef.size());
    for (size_t j=0;j<enet.coef.size();++j)
        EXPECT_NEAR(enet.coef[j], lasso.coef[j], 1e-6);
    EXPECT_NEAR(enet.intercept, lasso.intercept, 1e-6);
}

TEST(MLElasticNet, L1RatioZeroMatchesRidge) {
    // NOTE: RidgeRegression's closed-form solve minimizes the *unnormalized*
    // objective ||y-Xw||^2 + alpha*||w||^2, whereas ElasticNet (mirroring
    // LassoRegression's coordinate-descent convention) minimizes the
    // n-normalized (1/2n)*||y-Xw-b||^2 + ... objective. The two conventions
    // agree only in the small-alpha limit (where both approach the OLS
    // solution), so this test uses a small alpha to verify the l1_ratio=0.0
    // limiting behaviour without being dominated by that pre-existing
    // Ridge/Lasso scale-convention mismatch elsewhere in the module.
    auto [X,y]=elasticnet_toy_data();
    double alpha=0.001;
    RidgeRegression ridge(alpha);
    ridge.fit(X,y);
    ElasticNet enet(alpha, /*l1_ratio=*/0.0, /*max_iter=*/5000);
    enet.fit(X,y);
    ASSERT_EQ(enet.coef.size(), ridge.coef.size());
    for (size_t j=0;j<enet.coef.size();++j)
        EXPECT_NEAR(enet.coef[j], ridge.coef[j], 5e-3);
    EXPECT_NEAR(enet.intercept, ridge.intercept, 5e-3);
}

TEST(MLElasticNet, IntermediateRatioShrinksBetweenLassoAndRidge) {
    // Dataset with informative features (0,1) and pure-noise features (2,3).
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.05);
    Mat X; Vec y;
    for (int i=0;i<40;++i) {
        double x0=(i%5)-2.0, x1=((i/5)%5)-2.0;
        double noise_a=noise(rng), noise_b=noise(rng);
        X.push_back({x0, x1, noise_a, noise_b});
        y.push_back(3.0*x0 - 2.0*x1 + noise(rng));
    }
    double alpha=1.0;
    LassoRegression lasso(alpha);
    lasso.fit(X,y);
    RidgeRegression ridge(alpha);
    ridge.fit(X,y);
    ElasticNet enet(alpha, 0.5);
    enet.fit(X,y);

    ASSERT_EQ(enet.coef.size(), 4u);
    // Elastic net's noise coefficients should be shrunk toward zero, at
    // least as much as (typically more sparse than) Ridge's, since it still
    // carries an L1 component.
    EXPECT_LT(std::abs(enet.coef[2]), std::abs(ridge.coef[2]) + 1e-6);
    EXPECT_LT(std::abs(enet.coef[3]), std::abs(ridge.coef[3]) + 1e-6);

    // A strongly regularized elastic net on this data should zero out (or
    // nearly zero out) at least one of the pure-noise coefficients, unlike
    // Ridge's uniform (never-exactly-zero) shrinkage.
    ElasticNet enet_strong(5.0, 0.7);
    enet_strong.fit(X,y);
    bool some_noise_near_zero =
        std::abs(enet_strong.coef[2]) < 1e-3 || std::abs(enet_strong.coef[3]) < 1e-3;
    EXPECT_TRUE(some_noise_near_zero);
}

TEST(MLElasticNet, RecoversTrueSignalAndShrinksNoise) {
    // y = 3*x1 - 2*x2 + noise, plus a few pure-noise features.
    std::mt19937 rng(123);
    std::normal_distribution<double> feat_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, 0.1);
    Mat X; Vec y;
    for (int i=0;i<60;++i) {
        double x1=feat_dist(rng), x2=feat_dist(rng);
        double n1=feat_dist(rng), n2=feat_dist(rng), n3=feat_dist(rng);
        X.push_back({x1,x2,n1,n2,n3});
        y.push_back(3.0*x1 - 2.0*x2 + noise_dist(rng));
    }
    ElasticNet enet(0.05, 0.5);
    enet.fit(X,y);
    ASSERT_EQ(enet.coef.size(), 5u);
    // True features: correct sign and roughly right magnitude.
    EXPECT_GT(enet.coef[0], 1.5);
    EXPECT_LT(enet.coef[1], -1.0);
    // Noise features: shrunk well below the true features' magnitude.
    EXPECT_LT(std::abs(enet.coef[2]), 0.5);
    EXPECT_LT(std::abs(enet.coef[3]), 0.5);
    EXPECT_LT(std::abs(enet.coef[4]), 0.5);
}

TEST(MLElasticNet, PredictConsistentWithFit) {
    Mat X={{0},{1},{2},{3},{4}};
    Vec y={1,3,5,7,9};  // y = 2x + 1
    ElasticNet enet(0.001, 0.5);
    enet.fit(X,y);
    auto p=enet.predict(X);
    ASSERT_EQ(p.size(), y.size());
    for (size_t i=0;i<X.size();++i)
        EXPECT_NEAR(p[i], enet.intercept+enet.coef[0]*X[i][0], 1e-9);
    // Held-out point should extrapolate sensibly.
    auto p_new=enet.predict({{5},{6}});
    EXPECT_NEAR(p_new[0], 11.0, 0.5);
    EXPECT_NEAR(p_new[1], 13.0, 0.5);
    EXPECT_GT(p_new[1], p_new[0]);
}

TEST(MLElasticNet, EmptyInputNoCrash) {
    Mat X; Vec y;
    ElasticNet enet;
    enet.fit(X,y);
    EXPECT_TRUE(enet.coef.empty());
    EXPECT_DOUBLE_EQ(enet.intercept, 0.0);
    auto p=enet.predict({});
    EXPECT_TRUE(p.empty());
}

TEST(MLElasticNet, MismatchedDimensionsNoCrash) {
    Mat X={{1,2},{3,4},{5,6}};
    Vec y={1,2};  // fewer targets than rows
    ElasticNet enet;
    enet.fit(X,y);
    EXPECT_TRUE(enet.coef.empty());
}

// ---- Logistic Regression ----

TEST(MLLogReg, BinaryClassification) {
    // Linearly separable
    Mat X={{1,1},{2,2},{3,3},{-1,-1},{-2,-2},{-3,-3}};
    Vec y={1,1,1,0,0,0};
    LogisticRegression logr(1.0, 500);
    logr.fit(X,y);
    auto pred=logr.predict(X);
    double acc=accuracy(pred,y);
    EXPECT_GE(acc, 0.8);
}

TEST(MLLogReg, PredictProbaAndScore) {
    Mat X={{0},{1},{2},{3}};
    Vec y={0,0,1,1};
    LogisticRegression logr(1.0, 300);
    logr.fit(X,y);
    auto proba=logr.predict_proba({{0.5}});
    EXPECT_GT(proba[0], 0.0);
    EXPECT_LT(proba[0], 1.0);
    EXPECT_GE(logr.score(X,y), 0.75);
}

// ---- KNN ----

TEST(MLKNN, SimpleClassify) {
    Mat X={{0,0},{1,0},{0,1},{3,3},{4,3},{3,4}};
    Vec y={0,0,0,1,1,1};
    KNN knn(3);
    knn.fit(X,y);
    auto p=knn.predict({{0.5,0.5},{3.5,3.5}});
    EXPECT_NEAR(p[0],0.0,1e-10);
    EXPECT_NEAR(p[1],1.0,1e-10);
}

TEST(MLKNN, ScoreOnTrainSet) {
    Mat X={{0,0},{1,0},{0,1},{3,3},{4,3},{3,4}};
    Vec y={0,0,0,1,1,1};
    KNN knn(3);
    knn.fit(X,y);
    EXPECT_GE(knn.score(X,y), 0.99);
}

// ---- Naive Bayes ----

TEST(MLNaiveBayes, GaussianBasic) {
    Mat X={{1,1},{2,2},{1.5,1.5},{-1,-1},{-2,-2}};
    Vec y={0,0,0,1,1};
    NaiveBayes nb;
    nb.fit(X,y);
    auto pred=nb.predict({{1,1}});
    EXPECT_NEAR(pred[0],0.0,1e-10);
}

TEST(MLNaiveBayes, ScoreOnTrainSet) {
    Mat X={{1,1},{2,2},{1.5,1.5},{-1,-1},{-2,-2}};
    Vec y={0,0,0,1,1};
    NaiveBayes nb;
    nb.fit(X,y);
    EXPECT_GE(nb.score(X,y), 0.99);
}

// ---- LDA / QDA ----

static std::pair<Mat,Vec> lda_three_class_data() {
    Mat X; Vec y;
    for (int i=0;i<30;++i) {
        X.push_back({-3.0+0.1*(double)(i%6), 0.0+0.05*(double)(i/6)});
        y.push_back(0.0);
    }
    for (int i=0;i<30;++i) {
        X.push_back({0.0+0.05*(double)(i%6), 3.0+0.1*(double)(i/6)});
        y.push_back(1.0);
    }
    for (int i=0;i<30;++i) {
        X.push_back({3.0+0.1*(double)(i%6), 0.0+0.05*(double)(i/6)});
        y.push_back(2.0);
    }
    return {X,y};
}

static std::pair<Mat,Vec> qda_unequal_covariance_data() {
    Mat X; Vec y;
    std::mt19937 rng(123);
    std::normal_distribution<double> tight(0.0, 0.15);
    std::normal_distribution<double> wide_x(0.0, 2.5);
    std::normal_distribution<double> wide_y(0.0, 0.2);
    for (int i=0;i<80;++i) {
        X.push_back({-3.0+tight(rng), 0.0+tight(rng)});
        y.push_back(0.0);
    }
    for (int i=0;i<80;++i) {
        X.push_back({3.0+wide_x(rng), 0.0+wide_y(rng)});
        y.push_back(1.0);
    }
    return {X,y};
}

TEST(MLLDA, TwoClassLinearSeparableHighAccuracy) {
    Mat X; Vec y;
    for (int i=0;i<40;++i) {
        X.push_back({-2.0+0.1*(double)(i%8), -1.0+0.05*(double)(i/8)});
        y.push_back(0.0);
    }
    for (int i=0;i<40;++i) {
        X.push_back({2.0+0.1*(double)(i%8), 1.0+0.05*(double)(i/8)});
        y.push_back(1.0);
    }
    auto split=train_test_split(X,y,0.25,42);
    LDA lda;
    lda.fit(split.first.first, split.first.second);
    double acc=accuracy(lda.predict(split.second.first), split.second.second);
    RecordProperty("lda_two_class_accuracy", acc);
    EXPECT_GE(acc, 0.90);
}

TEST(MLLDA, ThreeClassHighAccuracy) {
    auto [X,y]=lda_three_class_data();
    auto split=train_test_split(X,y,0.25,7);
    LDA lda;
    lda.fit(split.first.first, split.first.second);
    double acc=accuracy(lda.predict(split.second.first), split.second.second);
    EXPECT_GE(acc, 0.85);
}

TEST(MLLDA, ScoreOnTrainSet) {
    Mat X; Vec y;
    for (int i=0;i<25;++i) { X.push_back({-1.0,0.0}); y.push_back(0.0); }
    for (int i=0;i<25;++i) { X.push_back({1.0,0.0}); y.push_back(1.0); }
    LDA lda;
    lda.fit(X,y);
    EXPECT_GE(lda.score(X,y), 0.95);
}

TEST(MLLDA, NonContiguousLabels) {
    Mat X; Vec y;
    for (int i=0;i<20;++i) { X.push_back({-2.0,0.0}); y.push_back(10.0); }
    for (int i=0;i<20;++i) { X.push_back({2.0,0.0}); y.push_back(20.0); }
    LDA lda;
    lda.fit(X,y);
    auto pred=lda.predict({{-2.5,0.0},{2.5,0.0}});
    EXPECT_NEAR(pred[0], 10.0, 1e-10);
    EXPECT_NEAR(pred[1], 20.0, 1e-10);
}

TEST(MLLDA, TransformReducesDimension) {
    auto [X,y]=lda_three_class_data();
    LDA lda(1e-6, 2);
    lda.fit(X,y);
    auto Z=lda.transform(X);
    EXPECT_EQ(Z.size(), X.size());
    EXPECT_EQ(Z[0].size(), 2u);
}

TEST(MLLDA, TransformPreservesClassSeparation) {
    auto [X,y]=lda_three_class_data();
    LDA lda(1e-6, 2);
    lda.fit(X,y);
    auto Z=lda.transform(X);
    Vec m0(2,0), m1(2,0), m2(2,0);
    int n0=0,n1=0,n2=0;
    for (size_t i=0;i<y.size();++i) {
        if (y[i]==0.0) { for (int d=0;d<2;++d) m0[d]+=Z[i][d]; ++n0; }
        else if (y[i]==1.0) { for (int d=0;d<2;++d) m1[d]+=Z[i][d]; ++n1; }
        else { for (int d=0;d<2;++d) m2[d]+=Z[i][d]; ++n2; }
    }
    for (int d=0;d<2;++d) { m0[d]/=n0; m1[d]/=n1; m2[d]/=n2; }
    double d01=0,d02=0;
    for (int d=0;d<2;++d) {
        double a=m0[d]-m1[d]; d01+=a*a;
        double b=m0[d]-m2[d]; d02+=b*b;
    }
    EXPECT_GT(std::sqrt(d01), 0.5);
    EXPECT_GT(std::sqrt(d02), 0.5);
}

TEST(MLLDA, SmallClassSampleNoCrash) {
    Mat X={{0,0},{0.1,0},{1,0},{1.1,0},{5,5}};
    Vec y={0,0,1,1,2};
    LDA lda(1e-4);
    lda.fit(X,y);
    auto pred=lda.predict(X);
    EXPECT_EQ(pred.size(), X.size());
    for (double v:pred) EXPECT_TRUE(std::isfinite(v));
}

TEST(MLLDA, PredictBeforeFitNoCrash) {
    Mat X={{1,2},{3,4}};
    LDA lda;
    auto pred=lda.predict(X);
    EXPECT_EQ(pred.size(), X.size());
}

TEST(MLQDA, UnequalCovarianceBeatsLDA) {
    auto [X,y]=qda_unequal_covariance_data();
    auto split=train_test_split(X,y,0.3,42);
    LDA lda;
    QDA qda;
    lda.fit(split.first.first, split.first.second);
    qda.fit(split.first.first, split.first.second);
    double acc_lda=accuracy(lda.predict(split.second.first), split.second.second);
    double acc_qda=accuracy(qda.predict(split.second.first), split.second.second);
    RecordProperty("lda_unequal_cov_accuracy", acc_lda);
    RecordProperty("qda_unequal_cov_accuracy", acc_qda);
    EXPECT_GE(acc_qda, acc_lda);
    EXPECT_GE(acc_qda, 0.75);
}

TEST(MLQDA, ThreeClassClassification) {
    auto [X,y]=lda_three_class_data();
    auto split=train_test_split(X,y,0.25,11);
    QDA qda;
    qda.fit(split.first.first, split.first.second);
    double acc=accuracy(qda.predict(split.second.first), split.second.second);
    EXPECT_GE(acc, 0.80);
}

TEST(MLQDA, ScoreOnTrainSet) {
    Mat X; Vec y;
    for (int i=0;i<30;++i) { X.push_back({-1.0,0.0}); y.push_back(0.0); }
    for (int i=0;i<30;++i) { X.push_back({1.0,0.0}); y.push_back(1.0); }
    QDA qda;
    qda.fit(X,y);
    EXPECT_GE(qda.score(X,y), 0.90);
}

TEST(MLQDA, NonContiguousLabels) {
    Mat X; Vec y;
    for (int i=0;i<15;++i) { X.push_back({-3.0,0.0}); y.push_back(5.0); }
    for (int i=0;i<15;++i) { X.push_back({3.0,0.0}); y.push_back(15.0); }
    QDA qda;
    qda.fit(X,y);
    auto pred=qda.predict({{-4.0,0.0},{4.0,0.0}});
    EXPECT_NEAR(pred[0], 5.0, 1e-10);
    EXPECT_NEAR(pred[1], 15.0, 1e-10);
}

TEST(MLQDA, SmallClassSampleNoCrash) {
    Mat X={{0,0},{1,0},{1.1,0},{5,5},{5.1,5}};
    Vec y={0,1,1,2,2};
    QDA qda(1e-4);
    qda.fit(X,y);
    auto pred=qda.predict(X);
    EXPECT_EQ(pred.size(), X.size());
    for (double v:pred) EXPECT_TRUE(std::isfinite(v));
}

TEST(MLQDA, PredictBeforeFitNoCrash) {
    Mat X={{1,2},{3,4}};
    QDA qda;
    auto pred=qda.predict(X);
    EXPECT_EQ(pred.size(), X.size());
}

TEST(MLQDA, EqualCovarianceStillWorks) {
    Mat X; Vec y;
    for (int i=0;i<30;++i) {
        X.push_back({-2.0+0.1*(double)(i%6), 0.0});
        y.push_back(0.0);
    }
    for (int i=0;i<30;++i) {
        X.push_back({2.0+0.1*(double)(i%6), 0.0});
        y.push_back(1.0);
    }
    auto split=train_test_split(X,y,0.25,3);
    QDA qda;
    qda.fit(split.first.first, split.first.second);
    EXPECT_GE(accuracy(qda.predict(split.second.first), split.second.second), 0.85);
}

// ---- Decision Tree ----

TEST(MLDecisionTree, XOR) {
    Mat X={{0,0},{0,1},{1,0},{1,1}};
    Vec y={0,1,1,0};
    DecisionTree dt(5);
    dt.fit(X,y);
    auto pred=dt.predict(X);
    EXPECT_GE(accuracy(pred,y), 0.75);
}

TEST(MLDecisionTree, EntropyCriterion) {
    Mat X={{0},{1},{2},{3},{4},{5}};
    Vec y={0,0,0,1,1,1};
    DecisionTree dt(3, "entropy");
    dt.fit(X,y);
    EXPECT_GE(dt.score(X,y), 0.8);
}

// ---- KMeans ----

TEST(MLKMeans, TwoClusters) {
    Mat X;
    for (int i=0;i<10;++i) X.push_back({(double)i,0});
    for (int i=0;i<10;++i) X.push_back({(double)i+100,0});
    KMeans km(2);
    km.fit(X);
    auto labels=km.predict(X);
    // Check that first 10 get same label, last 10 get same label
    double l0=labels[0];
    bool ok=true;
    for (int i=0;i<10;++i) if (labels[i]!=l0) {ok=false;break;}
    EXPECT_TRUE(ok);
}

TEST(MLKMeans, InertiaDecreases) {
    Mat X;
    for (int i=0;i<5;++i) X.push_back({0.1*(double)i, 0.0});
    for (int i=0;i<5;++i) X.push_back({10.0+0.1*(double)i, 10.0});
    KMeans km(2);
    km.fit(X);
    double inert=km.inertia(X);
    EXPECT_GT(inert, 0.0);
    EXPECT_LT(inert, 50.0);
}

// ---- GaussianMixture ----

static Mat gmm_two_blob_data() {
    Mat X;
    for (int i=0;i<15;++i)
        X.push_back({0.1*(double)(i%5), 0.1*(double)(i/5)});
    for (int i=0;i<15;++i)
        X.push_back({10.0+0.1*(double)(i%5), 10.0+0.1*(double)(i/5)});
    return X;
}

static Mat gmm_three_blob_data() {
    Mat X;
    for (int i=0;i<12;++i) X.push_back({0.1*(double)(i%4), 0.1*(double)(i/4)});
    for (int i=0;i<12;++i) X.push_back({10.0+0.1*(double)(i%4), 0.0+0.1*(double)(i/4)});
    for (int i=0;i<12;++i) X.push_back({0.0+0.1*(double)(i%4), 10.0+0.1*(double)(i/4)});
    return X;
}

static bool gmm_means_match(const Mat& means, const std::vector<Vec>& true_centers, double tol) {
    if (means.size()!=true_centers.size()) return false;
    std::vector<bool> used(means.size(), false);
    for (const auto& tc:true_centers) {
        bool found=false;
        for (size_t c=0;c<means.size();++c) {
            if (used[c]) continue;
            double d=0;
            for (size_t j=0;j<tc.size();++j) {
                double diff=means[c][j]-tc[j];
                d+=diff*diff;
            }
            if (std::sqrt(d)<tol) { used[c]=true; found=true; break; }
        }
        if (!found) return false;
    }
    return true;
}

static bool gmm_blob_labels_consistent(const Vec& labels, size_t start, size_t count) {
    if (count==0) return true;
    double l0=labels[start];
    for (size_t i=start;i<start+count;++i)
        if (labels[i]!=l0) return false;
    return true;
}

static int gmm_argmax_row(const Vec& row) {
    int bi=0;
    for (size_t c=1;c<row.size();++c)
        if (row[c]>row[bi]) bi=(int)c;
    return bi;
}

TEST(MLGMM, TwoBlobsMeansMatch) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.config.seed=42;
    gmm.fit(X);
    std::vector<Vec> centers={{0.2,0.2},{10.2,10.2}};
    EXPECT_TRUE(gmm_means_match(gmm.means, centers, 1.5));
}

TEST(MLGMM, ThreeBlobsMeansMatch) {
    auto X=gmm_three_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=3;
    gmm.config.seed=42;
    gmm.fit(X);
    std::vector<Vec> centers={{0.15,0.15},{10.15,0.15},{0.15,10.15}};
    EXPECT_TRUE(gmm_means_match(gmm.means, centers, 1.5));
}

TEST(MLGMM, WeightsSumToOne) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    double s=0;
    for (double w:gmm.weights) s+=w;
    EXPECT_NEAR(s, 1.0, 1e-9);
}

TEST(MLGMM, TwoBlobsHardAssignConsistent) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.config.seed=42;
    gmm.fit(X);
    auto labels=gmm.predict(X);
    EXPECT_TRUE(gmm_blob_labels_consistent(labels, 0, 15));
    EXPECT_TRUE(gmm_blob_labels_consistent(labels, 15, 15));
    EXPECT_NE(labels[0], labels[15]);
}

TEST(MLGMM, ThreeBlobsHardAssignConsistent) {
    auto X=gmm_three_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=3;
    gmm.config.seed=42;
    gmm.fit(X);
    auto labels=gmm.predict(X);
    EXPECT_TRUE(gmm_blob_labels_consistent(labels, 0, 12));
    EXPECT_TRUE(gmm_blob_labels_consistent(labels, 12, 12));
    EXPECT_TRUE(gmm_blob_labels_consistent(labels, 24, 12));
}

TEST(MLGMM, PredictProbaRowsSumToOne) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    auto proba=gmm.predict_proba(X);
    for (const auto& row:proba) {
        double s=0;
        for (double v:row) s+=v;
        EXPECT_NEAR(s, 1.0, 1e-9);
    }
}

TEST(MLGMM, PredictProbaNonNegative) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    auto proba=gmm.predict_proba(X);
    for (const auto& row:proba)
        for (double v:row)
            EXPECT_GE(v, 0.0);
}

TEST(MLGMM, PredictMatchesArgmaxProba) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    auto proba=gmm.predict_proba(X);
    auto labels=gmm.predict(X);
    for (size_t i=0;i<X.size();++i)
        EXPECT_EQ(labels[i], (double)gmm_argmax_row(proba[i]));
}

TEST(MLGMM, ScoreFinite) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    double sc=gmm.score(X);
    EXPECT_TRUE(std::isfinite(sc));
    EXPECT_GT(sc, -1e6);
}

TEST(MLGMM, LogLikelihoodRecorded) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    EXPECT_TRUE(std::isfinite(gmm.log_likelihood));
    EXPECT_GT(gmm.log_likelihood, gmm.score(X)*X.size()-1.0);
}

TEST(MLGMM, CorrectNComponentsScoresBetter) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm1, gmm2;
    gmm1.config.n_components=1;
    gmm2.config.n_components=2;
    gmm1.config.seed=42;
    gmm2.config.seed=42;
    gmm1.fit(X);
    gmm2.fit(X);
    EXPECT_GT(gmm2.score(X), gmm1.score(X));
}

TEST(MLGMM, VariancesAboveFloor) {
    auto X=gmm_two_blob_data();
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.fit(X);
    for (const auto& row:gmm.variances)
        for (double v:row)
            EXPECT_GE(v, 1e-6);
}

TEST(MLGMM, TinyDatasetNoCrash) {
    Mat X={{0,0},{10,10}};
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.config.seed=42;
    gmm.fit(X);
    auto labels=gmm.predict(X);
    EXPECT_EQ(labels.size(), 2u);
    auto proba=gmm.predict_proba(X);
    EXPECT_EQ(proba.size(), 2u);
    EXPECT_TRUE(std::isfinite(gmm.score(X)));
}

TEST(MLGMM, DuplicatePointsNoNaN) {
    Mat X;
    for (int i=0;i<10;++i) X.push_back({0.0, 0.0});
    for (int i=0;i<10;++i) X.push_back({10.0, 10.0});
    GaussianMixture gmm;
    gmm.config.n_components=2;
    gmm.config.seed=42;
    gmm.fit(X);
    for (const auto& row:gmm.variances)
        for (double v:row) {
            EXPECT_TRUE(std::isfinite(v));
            EXPECT_GE(v, 1e-6);
        }
    auto proba=gmm.predict_proba(X);
    for (const auto& row:proba)
        for (double v:row)
            EXPECT_TRUE(std::isfinite(v));
    EXPECT_TRUE(std::isfinite(gmm.score(X)));
}

TEST(MLGMM, PredictBeforeFitNoCrash) {
    Mat X={{1,2},{3,4}};
    GaussianMixture gmm;
    auto proba=gmm.predict_proba(X);
    EXPECT_TRUE(proba.empty());
    auto labels=gmm.predict(X);
    EXPECT_EQ(labels.size(), X.size());
}

TEST(MLGMM, EmptyFitNoCrash) {
    Mat X;
    GaussianMixture gmm;
    gmm.fit(X);
    EXPECT_TRUE(gmm.means.empty());
    EXPECT_TRUE(gmm.weights.empty());
}

// ---- DBSCAN ----

TEST(MLDBSCAN, TwoClusters) {
    Mat X;
    for (int i=0;i<5;++i) X.push_back({(double)i*0.1,0});
    for (int i=0;i<5;++i) X.push_back({(double)i*0.1+10,0});
    DBSCAN db(0.5, 2);
    db.fit(X);
    EXPECT_EQ(db.labels_.size(), 10u);
    // Should have 2 distinct non-noise clusters
    std::set<double> cls(db.labels_.begin(),db.labels_.end());
    cls.erase(-1);
    EXPECT_EQ(cls.size(), 2u);
}

TEST(MLAgglomerative, FitPredict) {
    Mat X;
    for (int i=0;i<5;++i) X.push_back({(double)i,0});
    for (int i=0;i<5;++i) X.push_back({(double)i+100,0});
    AgglomerativeClustering ac(2);
    ac.fit(X);
    EXPECT_EQ(ac.labels_.size(), 10u);
    double l0=ac.labels_[0];
    for (int i=0;i<5;++i) EXPECT_EQ(ac.labels_[i],l0);
    double l1=ac.labels_[5];
    for (int i=5;i<10;++i) EXPECT_EQ(ac.labels_[i],l1);
    EXPECT_NE(l0,l1);
}

// ---- PCA ----

TEST(MLPCA, ReducesDimension) {
    Mat X;
    for (int i=0;i<10;++i) X.push_back({(double)i, (double)i*2, (double)i*3});
    PCA pca(2);
    auto Z=pca.fit_transform(X);
    EXPECT_EQ(Z.size(), 10u);
    EXPECT_EQ(Z[0].size(), 2u);
}

TEST(MLPCA, ReconstructionError) {
    Mat X;
    for (int i=0;i<20;++i) X.push_back({(double)i, (double)i});
    PCA pca(1);
    auto Z=pca.fit_transform(X);
    auto Xr=pca.inverse_transform(Z);
    EXPECT_EQ(Xr.size(), 20u);
}

TEST(MLPCA, TransformAndVariance) {
    Mat X={{1,0},{2,0},{3,0},{4,0},{5,0}};
    PCA pca(1);
    pca.fit(X);
    EXPECT_EQ(pca.explained_variance.size(), 1u);
    EXPECT_GT(pca.explained_variance[0], 0.0);
    auto Z=pca.transform({{6,0}});
    EXPECT_EQ(Z.size(), 1u);
    EXPECT_EQ(Z[0].size(), 1u);
}

TEST(MLTSNE, FitTransformShape) {
    Mat X={{0,0},{1,0},{0,1},{1,1},{5,5}};
    TSNE tsne(2,5.0,100.0,10);
    Mat Y=tsne.fit_transform(X);
    EXPECT_EQ(Y.size(), 5u);
    EXPECT_EQ(Y[0].size(), 2u);
}

// ---- Autodiff ----

TEST(MLAutodiff, SimpleGrad) {
    // f(x) = x*x, grad = 2x
    auto f=[](const std::vector<Var>& v)->Var{ return v[0]*v[0]; };
    Vec g=grad(f,{3.0});
    EXPECT_NEAR(g[0], 6.0, 1e-6);
}

TEST(MLAutodiff, ChainRule) {
    // f(x) = exp(x), grad = exp(x)
    auto f=[](const std::vector<Var>& v)->Var{ return var_exp(v[0]); };
    Vec g=grad(f,{1.0});
    EXPECT_NEAR(g[0], std::exp(1.0), 1e-6);
}

TEST(MLAutodiff, ActivationGrads) {
    auto log_f=[](const std::vector<Var>& v)->Var{ return var_log(v[0]); };
    EXPECT_NEAR(grad(log_f,{2.0})[0], 0.5, 1e-6);
    auto tanh_f=[](const std::vector<Var>& v)->Var{ return var_tanh(v[0]); };
    EXPECT_NEAR(grad(tanh_f,{0.0})[0], 1.0, 1e-4);
    auto sig_f=[](const std::vector<Var>& v)->Var{ return var_sigmoid(v[0]); };
    EXPECT_NEAR(grad(sig_f,{0.0})[0], 0.25, 1e-4);
    auto relu_f=[](const std::vector<Var>& v)->Var{ return var_relu(v[0]); };
    EXPECT_NEAR(grad(relu_f,{2.0})[0], 1.0, 1e-6);
    EXPECT_NEAR(grad(relu_f,{-1.0})[0], 0.0, 1e-6);
    auto sqrt_f=[](const std::vector<Var>& v)->Var{ return var_sqrt(v[0]); };
    EXPECT_NEAR(grad(sqrt_f,{4.0})[0], 0.25, 1e-4);
    auto pow_f=[](const std::vector<Var>& v)->Var{ return var_pow(v[0], 3.0); };
    EXPECT_NEAR(grad(pow_f,{2.0})[0], 12.0, 1e-4);
}

TEST(MLAutodiff, MultiVarGrad) {
    // f(x,y) = x*y + y*y, ∂/∂x = y, ∂/∂y = x+2y
    auto f=[](const std::vector<Var>& v)->Var{ return v[0]*v[1] + v[1]*v[1]; };
    Vec g=grad(f,{2.0,3.0});
    EXPECT_NEAR(g[0], 3.0, 1e-6);  // df/dx = y = 3
    EXPECT_NEAR(g[1], 8.0, 1e-6);  // df/dy = x+2y = 2+6 = 8
}

TEST(MLAutodiff, Hessian) {
    // f(x,y) = x^2 + y^2, H = [[2,0],[0,2]]
    auto f=[](const std::vector<Var>& v)->Var{ return v[0]*v[0]+v[1]*v[1]; };
    auto H=hessian(f,{1.0,1.0});
    EXPECT_NEAR(H[0][0], 2.0, 0.01);
    EXPECT_NEAR(H[1][1], 2.0, 0.01);
    EXPECT_NEAR(H[0][1], 0.0, 0.01);
}

TEST(MLJacobian, OutputShape2x2) {
    auto F=[](const std::vector<Var>& v)->std::vector<Var>{
        return {v[0]*v[0], v[1]*v[1]};
    };
    Mat J=jacobian(F,{1.0,2.0});
    EXPECT_EQ(J.size(), 2u);
    EXPECT_EQ(J[0].size(), 2u);
    EXPECT_EQ(J[1].size(), 2u);
}

// ---- Loss functions ----

TEST(MLLoss, MSE) {
    Vec pred={1,2,3}, true_={1,2,3};
    EXPECT_NEAR(mse_loss(pred,true_), 0.0, 1e-12);
    Vec pred2={2,2,2};
    EXPECT_NEAR(mse_loss(pred2,true_), 2.0/3.0, 1e-6);
}

TEST(MLLoss, BCE) {
    Vec pred={0.9,0.1}, true_={1,0};
    double loss=binary_crossentropy(pred,true_);
    EXPECT_LT(loss, 0.15);
}

TEST(MLLoss, Huber) {
    Vec p={0,1,5}, t={0,0,0};
    double h=huber_loss(p,t,2.0);
    EXPECT_GT(h,0);
}

TEST(MLLoss, MAEAndHinge) {
    Vec pred={1,2,3}, true_={1,3,3};
    EXPECT_NEAR(mae_loss(pred,true_), 1.0/3.0, 1e-6);
    Vec scores={1.5,-0.5,0.8}, labels={1,-1,1};
    EXPECT_NEAR(hinge_loss(scores,labels), 0.7/3.0, 1e-6);
}

TEST(MLLoss, CategoricalCrossentropy) {
    Mat pred={{0.7,0.2,0.1},{0.1,0.8,0.1}};
    Mat true_={{1,0,0},{0,1,0}};
    double loss=categorical_crossentropy(pred,true_);
    EXPECT_GT(loss, 0.0);
    EXPECT_LT(loss, 1.0);
}

// ---- Metrics ----

TEST(MLMetrics, Accuracy) {
    Vec pred={1,0,1,1}, true_={1,0,0,1};
    EXPECT_NEAR(accuracy(pred,true_), 0.75, 1e-10);
}

TEST(MLMetrics, R2Score) {
    Vec pred={1,2,3}, true_={1,2,3};
    EXPECT_NEAR(r2_score(pred,true_), 1.0, 1e-10);
}

TEST(MLMetrics, RMSE) {
    Vec pred={1,2,3}, true_={1,3,3};
    EXPECT_NEAR(rmse(pred,true_), std::sqrt(1.0/3.0), 1e-6);
}

TEST(MLMetrics, PrecisionRecallF1) {
    Vec pred={1,1,0,0}, true_={1,0,0,1};
    EXPECT_NEAR(precision(pred,true_), 0.5, 1e-10);
    EXPECT_NEAR(recall(pred,true_), 0.5, 1e-10);
    EXPECT_NEAR(f1_score(pred,true_), 0.5, 1e-10);
}

// ---- Confusion Matrix / ROC / AUC ----

static std::pair<Vec,Vec> perfect_classifier_data() {
    // 5 true positives scored 1.0, 5 true negatives scored 0.0: perfectly separable.
    Vec pred={1,1,1,1,1, 0,0,0,0,0};
    Vec truth={1,1,1,1,1, 0,0,0,0,0};
    return {pred,truth};
}

static std::pair<Vec,Vec> anticorrelated_classifier_data() {
    // Positives score 0.0, negatives score 1.0: perfectly wrong.
    Vec pred={0,0,0,0,0, 1,1,1,1,1};
    Vec truth={1,1,1,1,1, 0,0,0,0,0};
    return {pred,truth};
}

static std::pair<Vec,Vec> decent_classifier_data() {
    // Positives generally score higher than negatives, but with overlap.
    Vec pred =  {0.9,0.8,0.75,0.6,0.55, 0.45,0.4,0.3,0.2,0.1};
    Vec truth = {1,  1,  1,   1,  0,    1,   0,  0,  0,  0};
    return {pred,truth};
}

TEST(MLConfusionMatrix, HandComputedExactCounts) {
    // threshold 0.5: predicted-positive iff score>=0.5, true-positive iff label>=0.5.
    Vec pred ={0.9, 0.4, 0.6, 0.3, 0.5, 0.1};
    Vec truth={1,   1,   0,   0,   1,   0};
    // idx0: pred_pos=T, true_pos=T -> tp
    // idx1: pred_pos=F, true_pos=T -> fn
    // idx2: pred_pos=T, true_pos=F -> fp
    // idx3: pred_pos=F, true_pos=F -> tn
    // idx4: pred_pos=T, true_pos=T -> tp
    // idx5: pred_pos=F, true_pos=F -> tn
    auto cm=confusion_matrix(pred, truth, 0.5);
    EXPECT_EQ(cm.tp, 2);
    EXPECT_EQ(cm.fp, 1);
    EXPECT_EQ(cm.tn, 2);
    EXPECT_EQ(cm.fn, 1);
}

TEST(MLConfusionMatrix, DifferentThresholdChangesCounts) {
    Vec pred ={0.9, 0.4, 0.6, 0.3, 0.5, 0.1};
    Vec truth={1,   1,   0,   0,   1,   0};
    // At threshold 0.9, only idx0 is predicted positive (true positive).
    auto cm=confusion_matrix(pred, truth, 0.9);
    EXPECT_EQ(cm.tp, 1);
    EXPECT_EQ(cm.fp, 0);
    EXPECT_EQ(cm.tn, 3);
    EXPECT_EQ(cm.fn, 2);
}

TEST(MLConfusionMatrix, MismatchedLengthReturnsZero) {
    Vec pred={0.1,0.9,0.5};
    Vec truth={1,0};
    auto cm=confusion_matrix(pred, truth);
    EXPECT_EQ(cm.tp, 0);
    EXPECT_EQ(cm.fp, 0);
    EXPECT_EQ(cm.tn, 0);
    EXPECT_EQ(cm.fn, 0);
}

TEST(MLConfusionMatrix, PerfectClassifierZeroErrorsAtDefaultThreshold) {
    auto [pred,truth]=perfect_classifier_data();
    auto cm=confusion_matrix(pred, truth, 0.5);
    EXPECT_EQ(cm.fp, 0);
    EXPECT_EQ(cm.fn, 0);
    EXPECT_EQ(cm.tp, 5);
    EXPECT_EQ(cm.tn, 5);
}

TEST(MLROC, PerfectClassifierAUCNearOne) {
    auto [pred,truth]=perfect_classifier_data();
    EXPECT_NEAR(roc_auc(pred,truth), 1.0, 1e-9);
}

TEST(MLROC, AntiCorrelatedClassifierAUCNearZero) {
    auto [pred,truth]=anticorrelated_classifier_data();
    EXPECT_NEAR(roc_auc(pred,truth), 0.0, 1e-9);
}

TEST(MLROC, ConstantScoreClassifierAUCIsHalf) {
    // All predictions identical (0.5) regardless of true label carries no
    // discriminating information. Sweeping thresholds above/below the
    // single distinct score only ever produces the (0,0) and (1,1)
    // endpoints (everyone-negative, then everyone-positive), so the curve
    // is a single straight diagonal segment -- its trapezoidal area works
    // out to exactly 0.5, matching the "no better than random" expectation.
    Vec pred(10, 0.5);
    Vec truth={1,1,1,1,1,0,0,0,0,0};
    auto curve=roc_curve(pred,truth);
    ASSERT_EQ(curve.size(), 2u);
    EXPECT_NEAR(curve[0].fpr, 0.0, 1e-12);
    EXPECT_NEAR(curve[0].tpr, 0.0, 1e-12);
    EXPECT_NEAR(curve[1].fpr, 1.0, 1e-12);
    EXPECT_NEAR(curve[1].tpr, 1.0, 1e-12);
    EXPECT_NEAR(roc_auc(pred,truth), 0.5, 1e-9);
}

TEST(MLROC, DecentClassifierAUCBetweenHalfAndOne) {
    auto [pred,truth]=decent_classifier_data();
    double auc=roc_auc(pred,truth);
    EXPECT_GT(auc, 0.5);
    EXPECT_LT(auc, 1.0);
}

TEST(MLROC, CurveStartsAtZeroFPREndsAtOneFPR) {
    auto check_endpoints=[](const Vec& pred, const Vec& truth){
        auto curve=roc_curve(pred,truth);
        ASSERT_FALSE(curve.empty());
        EXPECT_NEAR(curve.front().fpr, 0.0, 1e-12);
        EXPECT_NEAR(curve.back().fpr, 1.0, 1e-12);
    };
    {
        auto [p,t]=perfect_classifier_data();
        check_endpoints(p,t);
    }
    {
        auto [p,t]=anticorrelated_classifier_data();
        check_endpoints(p,t);
    }
    {
        auto [p,t]=decent_classifier_data();
        check_endpoints(p,t);
    }
}

TEST(MLROC, CurveIsMonotonicNonDecreasingFPR) {
    auto check_monotonic=[](const Vec& pred, const Vec& truth){
        auto curve=roc_curve(pred,truth);
        for (size_t i=1;i<curve.size();++i)
            EXPECT_GE(curve[i].fpr, curve[i-1].fpr);
    };
    {
        auto [p,t]=decent_classifier_data();
        check_monotonic(p,t);
    }
    {
        auto [p,t]=perfect_classifier_data();
        check_monotonic(p,t);
    }
    Vec pred(10,0.5), truth={1,1,1,1,1,0,0,0,0,0};
    check_monotonic(pred,truth);
}

TEST(MLROC, NoPositivesTPRZeroByConvention) {
    // All-negative true labels: tp+fn is always 0, so TPR is defined as 0.0
    // at every threshold (documented convention), and roc_auc degenerates
    // to 0.0 since every point has tpr==0.
    Vec pred={0.1,0.4,0.6,0.9};
    Vec truth={0,0,0,0};
    auto curve=roc_curve(pred,truth);
    for (auto& pt:curve) EXPECT_NEAR(pt.tpr, 0.0, 1e-12);
    EXPECT_NEAR(roc_auc(pred,truth), 0.0, 1e-9);
}

TEST(MLROC, NoNegativesFPRZeroByConvention) {
    // All-positive true labels: fp+tn is always 0, so FPR is defined as 0.0
    // at every threshold (documented convention), and roc_auc degenerates
    // to 0.0 since every point has fpr==0 (zero width, zero area).
    Vec pred={0.1,0.4,0.6,0.9};
    Vec truth={1,1,1,1};
    auto curve=roc_curve(pred,truth);
    for (auto& pt:curve) EXPECT_NEAR(pt.fpr, 0.0, 1e-12);
    EXPECT_NEAR(roc_auc(pred,truth), 0.0, 1e-9);
}

TEST(MLROC, MismatchedLengthReturnsEmptyCurveAndZeroAUC) {
    Vec pred={0.1,0.9,0.5};
    Vec truth={1,0};
    EXPECT_TRUE(roc_curve(pred,truth).empty());
    EXPECT_NEAR(roc_auc(pred,truth), 0.0, 1e-12);
}

TEST(MLROC, EmptyInputReturnsEmptyCurveAndZeroAUC) {
    Vec pred, truth;
    EXPECT_TRUE(roc_curve(pred,truth).empty());
    EXPECT_NEAR(roc_auc(pred,truth), 0.0, 1e-12);
}

TEST(MLROC, ThresholdSweepBoundedByCandidateCountAndFinite) {
    Vec pred={0.2,0.8};
    Vec truth={0,1};
    auto curve=roc_curve(pred,truth);
    // Distinct scores {0.2,0.8} plus one above max and one below min give at
    // most 4 threshold candidates (fewer after deduping identical (fpr,tpr)).
    EXPECT_LE(curve.size(), 4u);
    ASSERT_FALSE(curve.empty());
    for (auto& pt:curve) EXPECT_TRUE(std::isfinite(pt.threshold));
    // The lowest-threshold sweep must classify everything as positive.
    EXPECT_NEAR(curve.back().fpr, 1.0, 1e-12);
    EXPECT_NEAR(curve.back().tpr, 1.0, 1e-12);
}

TEST(MLROC, AUCConsistentWithManualTrapezoidalIntegration) {
    auto [pred,truth]=decent_classifier_data();
    auto curve=roc_curve(pred,truth);
    double manual=0.0;
    for (size_t i=1;i<curve.size();++i)
        manual += 0.5*(curve[i-1].tpr+curve[i].tpr)*(curve[i].fpr-curve[i-1].fpr);
    EXPECT_NEAR(roc_auc(pred,truth), manual, 1e-12);
}

TEST(MLROC, SingleSampleNoCrash) {
    Vec pred={0.7};
    Vec truth={1};
    auto curve=roc_curve(pred,truth);
    EXPECT_FALSE(curve.empty());
    double auc=roc_auc(pred,truth);
    EXPECT_TRUE(std::isfinite(auc));
}

// ---- Preprocessing ----

TEST(MLPreprocess, StandardScaler) {
    Mat X={{1,2},{3,4},{5,6}};
    StandardScaler sc;
    auto Z=sc.fit_transform(X);
    // Column means should be ~0, stds ~1
    double m0=0; for (auto& r:Z) m0+=r[0]; m0/=3;
    EXPECT_NEAR(m0, 0.0, 1e-10);
}

TEST(MLPreprocess, StandardScalerInverse) {
    Mat X={{1,2},{3,4},{5,6}};
    StandardScaler sc;
    auto Z=sc.fit_transform(X);
    auto Xr=sc.inverse_transform(Z);
    for (size_t i=0;i<X.size();++i)
        for (size_t j=0;j<X[i].size();++j)
            EXPECT_NEAR(Xr[i][j], X[i][j], 1e-10);
}

TEST(MLPreprocess, MinMaxScaler) {
    Mat X={{0,10},{5,20},{10,30}};
    MinMaxScaler sc;
    auto Z=sc.fit_transform(X);
    EXPECT_NEAR(Z[0][0], 0.0, 1e-10);
    EXPECT_NEAR(Z[2][0], 1.0, 1e-10);
}

// ---- Train/test split ----

TEST(MLSplit, TrainTestSplit) {
    Mat X; Vec y;
    for (int i=0;i<100;++i){X.push_back({(double)i});y.push_back(i);}
    auto result=train_test_split(X,y,0.2);
    auto& Xtr=result.first.first;
    auto& Xte=result.second.first;
    EXPECT_EQ(Xtr.size()+Xte.size(), 100u);
    EXPECT_EQ(Xte.size(), 20u);
}

TEST(MLCrossVal, BasicScore) {
    Mat X={{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}};
    Vec y={1,1,1,0,0,0,1,0,0,0};
    auto scorer=[](const Mat& Xtr,const Vec& ytr,const Mat& Xte,const Vec& yte)->double{
        LogisticRegression lr(1.0,100);
        lr.fit(Xtr,ytr);
        return accuracy(lr.predict(Xte),yte);
    };
    Vec scores=cross_val_score(scorer,X,y,5);
    EXPECT_GE(scores.size(), 1u);
    for (double s:scores) EXPECT_GE(s,0.0);
}

// ---- Dense layer ----

TEST(MLDenseLayer, ForwardShape) {
    DenseLayer layer(3, 2, "relu");
    Vec x={1,0,-1};
    auto z=layer.forward(x);
    EXPECT_EQ(z.size(), 2u);
    auto a=layer.forward_activate(z);
    EXPECT_EQ(a.size(), 2u);
    for (double v:a) EXPECT_GE(v, 0.0);
}

// ---- Neural Network (simple) ----

TEST(MLNeuralNet, XOR) {
    // XOR dataset
    Mat X={{0,0},{0,1},{1,0},{1,1}};
    Mat Y={{0},{1},{1},{0}};
    NeuralNet nn;
    nn.add(4,"relu").add(1,"sigmoid");
    nn.compile("mse",0.05);
    nn.fit(X,Y,200,4);
    auto pred=nn.predict(X);
    EXPECT_EQ(pred.size(), 4u);
    EXPECT_EQ(pred[0].size(), 1u);
}

// ---- Random Forest ----

static std::pair<Mat,Vec> threshold_classification_data() {
    Mat X={{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}};
    Vec y={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};
    return {X,y};
}

TEST(MLRandomForest, ThresholdClassification) {
    auto [X,y]=threshold_classification_data();
    RandomForest rf;
    rf.config.n_trees=25;
    rf.config.max_depth=4;
    rf.config.seed=42;
    rf.fit(X,y);
    auto pred=rf.predict(X);
    EXPECT_GE(accuracy(pred,y), 0.85);
}

TEST(MLRandomForest, BetterThanConstantBaseline) {
    auto [X,y]=threshold_classification_data();
    RandomForest rf;
    rf.config.n_trees=20;
    rf.config.max_depth=5;
    rf.fit(X,y);
    double const_acc=std::max(
        (double)std::count(y.begin(),y.end(),0.0)/y.size(),
        (double)std::count(y.begin(),y.end(),1.0)/y.size());
    EXPECT_GT(accuracy(rf.predict(X),y), const_acc);
}

TEST(MLRandomForest, TrainAccuracyHigh) {
    Mat X; Vec y;
    for (int i=0;i<40;++i) {
        X.push_back({(double)i, (double)(i%2)});
        y.push_back(i<20?0.0:1.0);
    }
    RandomForest rf;
    rf.config.n_trees=40;
    rf.config.max_depth=6;
    rf.config.feature_subsample_ratio=1.0;
    rf.fit(X,y);
    EXPECT_GE(accuracy(rf.predict(X),y), 0.85);
}

TEST(MLRandomForest, FewTreesSanePredictions) {
    auto [X,y]=threshold_classification_data();
    RandomForest rf;
    rf.config.n_trees=3;
    rf.config.max_depth=3;
    rf.fit(X,y);
    auto pred=rf.predict(X);
    for (double v:pred) {
        EXPECT_TRUE(v==0.0 || v==1.0);
        EXPECT_TRUE(std::isfinite(v));
    }
}

TEST(MLRandomForest, VaryingMaxDepth) {
    auto [X,y]=threshold_classification_data();
    RandomForest rf_shallow, rf_deep;
    rf_shallow.config.n_trees=15;
    rf_shallow.config.max_depth=2;
    rf_deep.config.n_trees=15;
    rf_deep.config.max_depth=8;
    rf_shallow.fit(X,y);
    rf_deep.fit(X,y);
    auto ps=rf_shallow.predict(X), pd=rf_deep.predict(X);
    for (size_t i=0;i<X.size();++i) {
        EXPECT_TRUE(std::isfinite(ps[i]));
        EXPECT_TRUE(std::isfinite(pd[i]));
    }
    EXPECT_GE(accuracy(pd,y), accuracy(ps,y)-0.2);
}

TEST(MLRandomForest, FeatureSubsamplePredicts) {
    Mat X; Vec y;
    for (int i=0;i<30;++i) {
        X.push_back({(double)i, (double)(i*13 % 97), (double)(i*7 % 53)});
        y.push_back(i<15?0.0:1.0);
    }
    RandomForest rf;
    rf.config.n_trees=25;
    rf.config.max_depth=5;
    rf.config.feature_subsample_ratio=0.34;
    rf.config.seed=7;
    rf.fit(X,y);
    auto pred=rf.predict(X);
    EXPECT_EQ(pred.size(), X.size());
    EXPECT_GE(accuracy(pred,y), 0.75);
}

TEST(MLRandomForest, FeatureSubsampleBeatsNoise) {
    Mat X; Vec y;
    for (int i=0;i<40;++i) {
        X.push_back({(double)i, std::sin((double)i), std::cos((double)i*3)});
        y.push_back(i<20?0.0:1.0);
    }
    RandomForest rf_full, rf_sub;
    rf_full.config.n_trees=20;
    rf_full.config.feature_subsample_ratio=1.0;
    rf_sub.config.n_trees=20;
    rf_sub.config.feature_subsample_ratio=0.34;
    rf_full.fit(X,y);
    rf_sub.fit(X,y);
    EXPECT_GE(accuracy(rf_sub.predict(X),y), 0.7);
    EXPECT_GE(accuracy(rf_full.predict(X),y), accuracy(rf_sub.predict(X),y)-0.15);
}

TEST(MLRandomForest, SampleSubsampleBootstrap) {
    auto [X,y]=threshold_classification_data();
    RandomForest rf;
    rf.config.n_trees=20;
    rf.config.sample_subsample_ratio=0.6;
    rf.config.seed=99;
    rf.fit(X,y);
    EXPECT_EQ(rf.trees.size(), 20u);
    EXPECT_GE(accuracy(rf.predict(X),y), 0.75);
}

TEST(MLRandomForest, StoresFeatureIndices) {
    Mat X={{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    Vec y={0,0,1,1};
    RandomForest rf;
    rf.config.n_trees=5;
    rf.config.feature_subsample_ratio=0.5;
    rf.fit(X,y);
    EXPECT_EQ(rf.trees.size(), 5u);
    EXPECT_EQ(rf.feature_indices.size(), 5u);
    for (auto& fi:rf.feature_indices) {
        EXPECT_GE(fi.size(), 1u);
        EXPECT_LE(fi.size(), 2u);
    }
}

// ---- Gradient Boosting ----

static std::pair<Mat,Vec> linear_regression_data() {
    Mat X; Vec y;
    for (int i=0;i<50;++i) {
        X.push_back({(double)i});
        y.push_back(2.0*(double)i + 1.0);
    }
    return {X,y};
}

TEST(MLGradientBoosting, LinearFit) {
    auto [X,y]=linear_regression_data();
    GradientBoosting gb;
    gb.config.n_trees=40;
    gb.config.max_depth=3;
    gb.config.learning_rate=0.1;
    gb.fit(X,y);
    auto pred=gb.predict(X);
    EXPECT_LT(mse_loss(pred,y), 5.0);
    EXPECT_GT(r2_score(pred,y), 0.95);
}

TEST(MLGradientBoosting, MoreTreesLowerMSE) {
    auto [X,y]=linear_regression_data();
    GradientBoosting gb1, gb50;
    gb1.config.n_trees=1;
    gb1.config.max_depth=3;
    gb50.config.n_trees=50;
    gb50.config.max_depth=3;
    gb1.fit(X,y);
    gb50.fit(X,y);
    double mse1=mse_loss(gb1.predict(X),y);
    double mse50=mse_loss(gb50.predict(X),y);
    EXPECT_LT(mse50, mse1);
    EXPECT_LT(mse50, mse1*0.5);
}

TEST(MLGradientBoosting, PredictionsFinite) {
    Mat X; Vec y;
    for (int i=0;i<25;++i) {
        X.push_back({(double)i, (double)(i*i)});
        y.push_back(std::sin((double)i));
    }
    GradientBoosting gb;
    gb.config.n_trees=20;
    gb.fit(X,y);
    auto pred=gb.predict(X);
    for (double v:pred) EXPECT_TRUE(std::isfinite(v));
}

TEST(MLGradientBoosting, NonlinearPattern) {
    Mat X; Vec y;
    for (int i=0;i<40;++i) {
        double x=(double)i/10.0;
        X.push_back({x});
        y.push_back(x*x);
    }
    GradientBoosting gb;
    gb.config.n_trees=60;
    gb.config.max_depth=4;
    gb.config.learning_rate=0.15;
    gb.fit(X,y);
    EXPECT_LT(mse_loss(gb.predict(X),y), 0.5);
}

TEST(MLGradientBoosting, InitPredictionIsMean) {
    Vec y={1,2,3,4,5};
    Mat X={{0},{1},{2},{3},{4}};
    GradientBoosting gb;
    gb.config.n_trees=0;
    gb.fit(X,y);
    EXPECT_NEAR(gb.init_prediction, 3.0, 1e-12);
}

TEST(MLGradientBoosting, ZeroLearningRateConstant) {
    auto [X,y]=linear_regression_data();
    GradientBoosting gb;
    gb.config.n_trees=10;
    gb.config.learning_rate=0.0;
    gb.fit(X,y);
    auto pred=gb.predict(X);
    for (double v:pred)
        EXPECT_NEAR(v, gb.init_prediction, 1e-12);
}

TEST(MLGradientBoosting, SmallLearningRateSane) {
    auto [X,y]=linear_regression_data();
    GradientBoosting gb;
    gb.config.n_trees=30;
    gb.config.learning_rate=0.01;
    gb.fit(X,y);
    auto pred=gb.predict(X);
    for (double v:pred) EXPECT_TRUE(std::isfinite(v));
    EXPECT_LT(mse_loss(pred,y), mse_loss(Vec(y.size(),gb.init_prediction),y));
}

TEST(MLGradientBoosting, ShallowTreesWork) {
    auto [X,y]=linear_regression_data();
    GradientBoosting gb;
    gb.config.n_trees=25;
    gb.config.max_depth=1;
    gb.config.learning_rate=0.2;
    gb.fit(X,y);
    EXPECT_LT(mse_loss(gb.predict(X),y), 50.0);
}

TEST(MLGradientBoosting, StoresTrees) {
    auto [X,y]=linear_regression_data();
    GradientBoosting gb;
    gb.config.n_trees=12;
    gb.fit(X,y);
    EXPECT_EQ(gb.trees.size(), 12u);
}

// ---- Support Vector Machine (SMO) ----

static std::pair<Mat,Vec> linear_separable_svm_data() {
    Mat X; Vec y;
    for (int i=0;i<24;++i) {
        double x0=(i<12)?-1.5-0.05*i:1.5+0.05*(i-12);
        double x1=0.3*std::sin((double)i);
        X.push_back({x0, x1});
        y.push_back(x0<0.0?-1.0:1.0);
    }
    return {X,y};
}

static std::pair<Mat,Vec> xor_svm_data() {
    Mat X={{-1,-1},{-1,1},{1,-1},{1,1},
           {-0.7,-0.7},{-0.7,0.7},{0.7,-0.7},{0.7,0.7},
           {-0.5,0},{0.5,0},{0,-0.5},{0,0.5}};
    Vec y={-1,1,1,-1, -1,1,1,-1, -1,1,-1,1};
    return {X,y};
}

static double svm_train_accuracy(const SVM& svm, const Mat& X, const Vec& y) {
    return accuracy(svm.predict(X), y);
}

TEST(MLSVM, LinearSeparableHighAccuracy) {
    auto [X,y]=linear_separable_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=10.0;
    svm.fit(X,y);
    EXPECT_GE(svm_train_accuracy(svm,X,y), 0.95);
}

TEST(MLSVM, LinearDecisionFunctionMatchesPredict) {
    auto [X,y]=linear_separable_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=5.0;
    svm.fit(X,y);
    auto df=svm.decision_function(X);
    auto pred=svm.predict(X);
    ASSERT_EQ(df.size(), pred.size());
    for (size_t i=0;i<df.size();++i) {
        double expected=df[i]>=0.0?1.0:-1.0;
        EXPECT_DOUBLE_EQ(pred[i], expected);
    }
}

TEST(MLSVM, LinearSupportVectorConsistency) {
    auto [X,y]=linear_separable_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=10.0;
    svm.fit(X,y);
    EXPECT_EQ(svm.support_vectors.size(), svm.alphas.size());
    EXPECT_EQ(svm.support_vectors.size(), svm.sv_labels.size());
    for (size_t i=0;i<svm.support_vectors.size();++i) {
        EXPECT_TRUE(svm.sv_labels[i]==1.0 || svm.sv_labels[i]==-1.0);
        EXPECT_GT(svm.alphas[i], 0.0);
    }
}

TEST(MLSVM, RBFNonlinearHighAccuracy) {
    auto [X,y]=xor_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::RBF;
    svm.config.C=100.0;
    svm.config.gamma=1.0;
    svm.fit(X,y);
    EXPECT_GE(svm_train_accuracy(svm,X,y), 0.85);
}

TEST(MLSVM, RBFBetterThanLinearOnNonlinear) {
    auto [X,y]=xor_svm_data();
    SVM svm_lin, svm_rbf;
    svm_lin.config.kernel=SVMKernel::Linear;
    svm_lin.config.C=100.0;
    svm_rbf.config.kernel=SVMKernel::RBF;
    svm_rbf.config.C=100.0;
    svm_rbf.config.gamma=1.0;
    svm_lin.fit(X,y);
    svm_rbf.fit(X,y);
    double acc_lin=svm_train_accuracy(svm_lin,X,y);
    double acc_rbf=svm_train_accuracy(svm_rbf,X,y);
    EXPECT_LT(acc_lin, 0.85);
    EXPECT_GE(acc_rbf, acc_lin);
    EXPECT_GE(acc_rbf, 0.85);
}

TEST(MLSVM, SmallCFiniteBehavior) {
    auto [X,y]=linear_separable_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=1e-4;
    svm.fit(X,y);
    auto pred=svm.predict(X);
    for (double v:pred) EXPECT_TRUE(std::isfinite(v));
    EXPECT_TRUE(std::isfinite(svm.b));
}

TEST(MLSVM, LargeCNearHardMargin) {
    auto [X,y]=linear_separable_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=1e6;
    svm.fit(X,y);
    EXPECT_GE(svm_train_accuracy(svm,X,y), 0.95);
    for (double a:svm.alphas) EXPECT_TRUE(std::isfinite(a));
}

TEST(MLSVM, TwoPointsMinimalCase) {
    Mat X={{-1,0},{1,0}};
    Vec y={-1,1};
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=1.0;
    svm.fit(X,y);
    EXPECT_GE(svm_train_accuracy(svm,X,y), 1.0);
    EXPECT_GE(svm.support_vectors.size(), 1u);
}

TEST(MLSVM, PredictBeforeFitNoCrash) {
    Mat X={{1,2},{3,4}};
    SVM svm;
    auto pred=svm.predict(X);
    EXPECT_EQ(pred.size(), X.size());
    for (double v:pred) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_TRUE(v==1.0 || v==-1.0);
    }
}

TEST(MLSVM, DecisionFunctionBeforeFitNoCrash) {
    Mat X={{1,2},{3,4}};
    SVM svm;
    auto df=svm.decision_function(X);
    EXPECT_EQ(df.size(), X.size());
    for (double v:df) EXPECT_DOUBLE_EQ(v, 0.0);
}

TEST(MLSVM, EmptyFitNoCrash) {
    Mat X; Vec y;
    SVM svm;
    svm.fit(X,y);
    EXPECT_TRUE(svm.support_vectors.empty());
    auto pred=svm.predict({{1,2}});
    EXPECT_EQ(pred.size(), 1u);
}

TEST(MLSVM, RemapZeroOneLabels) {
    Mat X={{-2,0},{-1,0},{1,0},{2,0}};
    Vec y={0,0,1,1};
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=10.0;
    svm.fit(X,y);
    Vec y_pm1={-1,-1,1,1};
    EXPECT_GE(accuracy(svm.predict(X), y_pm1), 0.95);
}

TEST(MLSVM, InvalidLabelsNoOp) {
    auto [X,y]=linear_separable_svm_data();
    y[0]=0.5;
    SVM svm;
    svm.fit(X,y);
    EXPECT_TRUE(svm.support_vectors.empty());
    EXPECT_TRUE(svm.alphas.empty());
}

TEST(MLSVM, MixedLabelEncodingsNoOp) {
    Mat X={{-1,0},{1,0}};
    Vec y={0,1};
    y[1]=-1.0;
    SVM svm;
    svm.fit(X,y);
    EXPECT_TRUE(svm.support_vectors.empty());
}

TEST(MLSVM, RBFKernelGammaUsed) {
    auto [X,y]=xor_svm_data();
    SVM svm_lo, svm_hi;
    svm_lo.config.kernel=SVMKernel::RBF;
    svm_lo.config.C=50.0;
    svm_lo.config.gamma=0.01;
    svm_hi.config.kernel=SVMKernel::RBF;
    svm_hi.config.C=50.0;
    svm_hi.config.gamma=5.0;
    svm_lo.fit(X,y);
    svm_hi.fit(X,y);
    EXPECT_GE(svm_train_accuracy(svm_hi,X,y), svm_train_accuracy(svm_lo,X,y));
}

TEST(MLSVM, LinearBiasTermFinite) {
    auto [X,y]=linear_separable_svm_data();
    SVM svm;
    svm.config.kernel=SVMKernel::Linear;
    svm.config.C=10.0;
    svm.fit(X,y);
    EXPECT_TRUE(std::isfinite(svm.b));
    EXPECT_FALSE(svm.support_vectors.empty());
}
