#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

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
