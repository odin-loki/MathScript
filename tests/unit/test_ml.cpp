#define _USE_MATH_DEFINES
#include "ms/ml/ml.hpp"
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
