#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ms {
namespace ml {

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

// ========================== Utilities ==========================
Mat mat_zeros(int r, int c);
Mat mat_eye(int n);
Mat mat_T(const Mat& A);
Mat mat_mul(const Mat& A, const Mat& B);
Vec mat_vec(const Mat& A, const Vec& x);
Vec vec_add(const Vec& a, const Vec& b);
Vec vec_sub(const Vec& a, const Vec& b);
Vec vec_scale(double s, const Vec& v);
double vec_dot(const Vec& a, const Vec& b);
double vec_norm(const Vec& v);

// ========================== Supervised Learning ==========================

struct LinearRegression {
    Vec coef;
    double intercept = 0.0;
    void fit(const Mat& X, const Vec& y, bool use_intercept = true);
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;  // R²
};

struct RidgeRegression {
    Vec coef; double intercept = 0.0; double alpha;
    explicit RidgeRegression(double a = 1.0) : alpha(a) {}
    void fit(const Mat& X, const Vec& y, bool use_intercept = true);
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;
};

struct LassoRegression {
    Vec coef; double intercept = 0.0; double alpha;
    int max_iter;
    explicit LassoRegression(double a = 1.0, int mi = 1000) : alpha(a), max_iter(mi) {}
    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
};

struct LogisticRegression {
    Vec coef; double intercept = 0.0; double C; int max_iter;
    explicit LogisticRegression(double c = 1.0, int mi = 200) : C(c), max_iter(mi) {}
    void fit(const Mat& X, const Vec& y);  // binary: y ∈ {0,1}
    Vec predict_proba(const Mat& X) const;
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;
};

struct KNN {
    int k; std::string metric;
    Mat X_train; Vec y_train;
    explicit KNN(int k = 5, std::string m = "euclidean") : k(k), metric(std::move(m)) {}
    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;
};

struct NaiveBayes {
    Vec class_prior; Mat mean, var;
    Vec classes;
    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;
};

struct DecisionTree {
    int max_depth; std::string criterion;
    struct Node {
        int feature = -1; double threshold = 0.0, value = 0.0;
        int left = -1, right = -1;
    };
    std::vector<Node> nodes;
    explicit DecisionTree(int md = 5, std::string crit = "gini") : max_depth(md), criterion(std::move(crit)) {}
    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;
private:
    int build(const Mat& X, const Vec& y, std::vector<int>& idx, int depth);
    double predict_one(const Vec& x, int node) const;
};

// ========================== Unsupervised Learning ==========================

struct KMeans {
    int k, max_iter; double tol;
    Mat centers; Vec labels_;
    explicit KMeans(int k = 3, int mi = 100, double t = 1e-4) : k(k), max_iter(mi), tol(t) {}
    void fit(const Mat& X);
    Vec predict(const Mat& X) const;
    double inertia(const Mat& X) const;
};

struct DBSCAN {
    double eps; int min_samples;
    Vec labels_;
    explicit DBSCAN(double e = 0.5, int ms = 5) : eps(e), min_samples(ms) {}
    void fit(const Mat& X);
};

struct AgglomerativeClustering {
    int n_clusters; std::string linkage;
    Vec labels_;
    explicit AgglomerativeClustering(int n = 2, std::string l = "ward")
        : n_clusters(n), linkage(std::move(l)) {}
    void fit(const Mat& X);
};

// ========================== Dimensionality Reduction ==========================

struct PCA {
    int n_components;
    Mat components;  // eigenvectors (n_components × n_features)
    Vec explained_variance;
    Vec mean_;
    explicit PCA(int n = 2) : n_components(n) {}
    void fit(const Mat& X);
    Mat transform(const Mat& X) const;
    Mat fit_transform(const Mat& X);
    Mat inverse_transform(const Mat& Z) const;
};

struct TSNE {
    int n_components, max_iter; double perplexity, lr;
    explicit TSNE(int n = 2, double p = 30.0, double l = 200.0, int mi = 250)
        : n_components(n), max_iter(mi), perplexity(p), lr(l) {}
    Mat fit_transform(const Mat& X) const;
};

// ========================== Autodiff (reverse mode) ==========================

struct ADNode {
    double val = 0.0, grad = 0.0;
    std::function<void()> backward_fn;
    std::vector<std::shared_ptr<ADNode>> prev;  // input nodes for graph traversal
};

// Variable wrapper for autodiff
class Var {
public:
    std::shared_ptr<ADNode> node;
    explicit Var(double v = 0.0);
    double val() const { return node->val; }
    double grad() const { return node->grad; }
    void backward();

    Var operator+(const Var& o) const;
    Var operator-(const Var& o) const;
    Var operator*(const Var& o) const;
    Var operator/(const Var& o) const;
    Var operator-() const;
};

Var var_exp(const Var& x);
Var var_log(const Var& x);
Var var_tanh(const Var& x);
Var var_sigmoid(const Var& x);
Var var_relu(const Var& x);
Var var_sqrt(const Var& x);
Var var_pow(const Var& x, double n);

// Gradient of scalar function
Vec grad(std::function<Var(const std::vector<Var>&)> f, const Vec& x);
Mat jacobian(std::function<std::vector<Var>(const std::vector<Var>&)> F, const Vec& x);
Mat hessian(std::function<Var(const std::vector<Var>&)> f, const Vec& x);

// ========================== Neural Network ==========================

struct Activation { std::string name; };

struct DenseLayer {
    Mat W;  // (out, in)
    Vec b;  // (out)
    std::string activation;
    // Adam state
    Mat mW, vW; Vec mb, vb;

    DenseLayer(int in_dim, int out_dim, std::string act = "relu");
    Vec forward(const Vec& x) const;
    Vec forward_activate(const Vec& z) const;
};

// Simple feedforward network
struct NeuralNet {
    std::vector<DenseLayer> layers;
    std::string loss_name;
    double lr;
    int t = 0;  // Adam step

    NeuralNet& add(int out_dim, std::string activation = "relu");
    void compile(std::string loss = "mse", double lr = 0.001);
    void fit(const Mat& X, const Mat& Y, int epochs = 100, int batch = 32);
    Mat predict(const Mat& X);

private:
    void backward(const Vec& x, const Vec& target);
    int input_dim = 0;
};

// ========================== Loss Functions ==========================

double mse_loss(const Vec& y_pred, const Vec& y_true);
double mae_loss(const Vec& y_pred, const Vec& y_true);
double binary_crossentropy(const Vec& y_pred, const Vec& y_true);
double categorical_crossentropy(const Mat& y_pred, const Mat& y_true);
double huber_loss(const Vec& y_pred, const Vec& y_true, double delta = 1.0);
double hinge_loss(const Vec& y_pred, const Vec& y_true);

// ========================== Metrics ==========================

double accuracy(const Vec& y_pred, const Vec& y_true);
double precision(const Vec& y_pred, const Vec& y_true, double threshold = 0.5);
double recall(const Vec& y_pred, const Vec& y_true, double threshold = 0.5);
double f1_score(const Vec& y_pred, const Vec& y_true, double threshold = 0.5);
double r2_score(const Vec& y_pred, const Vec& y_true);
double rmse(const Vec& y_pred, const Vec& y_true);

// ========================== Preprocessing ==========================

struct StandardScaler {
    Vec mean_, std_;
    void fit(const Mat& X);
    Mat transform(const Mat& X) const;
    Mat fit_transform(const Mat& X);
    Mat inverse_transform(const Mat& Z) const;
};

struct MinMaxScaler {
    Vec min_, max_;
    void fit(const Mat& X);
    Mat transform(const Mat& X) const;
    Mat fit_transform(const Mat& X);
};

// ========================== Cross-validation ==========================
// Returns vector of test scores
Vec cross_val_score(std::function<double(const Mat&, const Vec&,
                                          const Mat&, const Vec&)> scorer,
                    const Mat& X, const Vec& y, int cv = 5);

// Train/test split
std::pair<std::pair<Mat,Vec>, std::pair<Mat,Vec>>
train_test_split(const Mat& X, const Vec& y, double test_size = 0.2, int seed = 42);

} // namespace ml
} // namespace ms
