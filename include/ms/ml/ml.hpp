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

/// Elastic Net regression — combines the L1 (Lasso) and L2 (Ridge) penalties
/// via a single mixing parameter. Minimizes:
///   (1/2n)*||y - Xw - b||^2 + alpha*(l1_ratio*||w||_1 + 0.5*(1-l1_ratio)*||w||_2^2)
///
/// Fit via cyclic coordinate descent, directly generalizing
/// LassoRegression's per-coordinate soft-thresholding update: each
/// coordinate's partial-residual correlation `rho_j` is soft-thresholded at
/// `alpha*l1_ratio` (the L1 term) and then divided by the feature's
/// (empirical) second moment plus `alpha*(1-l1_ratio)` (the extra L2 term
/// in the denominator, which shrinks every coefficient uniformly on top of
/// Lasso's sparsifying threshold).
///
/// @param alpha    Overall regularization strength (>= 0).
/// @param l1_ratio Mixing parameter in [0, 1] between the L1 and L2
///                 penalties.
/// @note l1_ratio = 1.0 makes the update identical to LassoRegression's
///       (pure L1 penalty; the L2 term vanishes from the denominator).
/// @note l1_ratio = 0.0 reduces to a pure L2 (Ridge-style) penalty: the
///       soft-threshold collapses to the identity (threshold 0), leaving
///       coordinate-descent Ridge shrinkage, which converges to the same
///       solution as RidgeRegression's closed-form normal-equation solve.
struct ElasticNet {
    Vec coef; double intercept = 0.0; double alpha; double l1_ratio;
    int max_iter;
    explicit ElasticNet(double a = 1.0, double l1r = 0.5, int mi = 1000)
        : alpha(a), l1_ratio(l1r), max_iter(mi) {}
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

/// Linear Discriminant Analysis — Gaussian classifier with a shared
/// (pooled) within-class covariance matrix. Supports classification via
/// the standard linear discriminant function and supervised dimensionality
/// reduction via transform().
struct LDA {
    double reg_epsilon = 1e-6;
    int n_components = 0;  // 0 → min(n_classes-1, n_features)
    Vec classes;
    Vec class_prior;
    Mat mean;              // n_classes × n_features
    Mat pooled_cov_inv;    // n_features × n_features
    Vec discrim_const;     // per-class constant term in discriminant
    Mat discrim_coef;      // n_classes × n_features linear coefficients
    Mat projection;        // n_components × n_features discriminant directions
    Vec overall_mean;

    explicit LDA(double eps = 1e-6, int n_comp = 0) : reg_epsilon(eps), n_components(n_comp) {}
    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
    double score(const Mat& X, const Vec& y) const;
    Mat transform(const Mat& X) const;
};

/// Quadratic Discriminant Analysis — Gaussian classifier with per-class
/// covariance matrices (the key difference from LDA). Classification
/// evaluates each class's multivariate-Gaussian log-density plus log-prior.
struct QDA {
    double reg_epsilon = 1e-6;
    Vec classes;
    Vec class_prior;
    Mat mean;              // n_classes × n_features
    Mat quad_coef;         // n_classes × (n_features*n_features) flattened Σ_k^{-1}
    Mat linear_coef;       // n_classes × n_features  (Σ_k^{-1} μ_k)
    Vec discrim_const;     // per-class constant in the quadratic discriminant

    explicit QDA(double eps = 1e-6) : reg_epsilon(eps) {}
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

struct RandomForestConfig {
    size_t n_trees = 50;
    size_t max_depth = 5;
    double feature_subsample_ratio = 0.7;
    double sample_subsample_ratio = 1.0;
    unsigned seed = 42;
};

struct RandomForest {
    RandomForestConfig config;
    std::vector<DecisionTree> trees;
    std::vector<std::vector<int>> feature_indices;

    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
};

struct GradientBoostingConfig {
    size_t n_trees = 50;
    size_t max_depth = 3;
    double learning_rate = 0.1;
    unsigned seed = 42;
};

struct GradientBoosting {
    GradientBoostingConfig config;
    double init_prediction = 0.0;
    std::vector<DecisionTree> trees;

    void fit(const Mat& X, const Vec& y);
    Vec predict(const Mat& X) const;
};

/// Support Vector Machine (binary classifier) trained via Platt's Sequential
/// Minimal Optimization (SMO). Each iteration optimizes a pair of Lagrange
/// multipliers with clipping to [0, C] and linear-equality bookkeeping until
/// KKT violations fall below `tol` or `max_iter` full passes occur with no
/// alpha updates. Kernels: Linear (dot product) and RBF (Gaussian).
///
/// Labels must be exactly +1.0 or -1.0 per sample. If every label is in {0, 1},
/// 0 is remapped to -1 and 1 is kept as +1 (documented convenience); mixed
/// {0,1} and {-1,+1} encodings or any other values cause fit() to no-op.
enum class SVMKernel { Linear, RBF };

struct SVMConfig {
    SVMKernel kernel = SVMKernel::Linear;
    double C = 1.0;
    double gamma = 0.1;
    double tol = 1e-3;
    int max_iter = 1000;
};

struct SVM {
    SVMConfig config;
    Mat support_vectors;
    Vec alphas;
    Vec sv_labels;
    double b = 0.0;

    void fit(const Mat& X, const Vec& y);
    Vec decision_function(const Mat& X) const;
    Vec predict(const Mat& X) const;
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

struct GMMConfig {
    size_t n_components = 2;
    size_t max_iter = 100;
    double tol = 1e-4;   // convergence tolerance on log-likelihood change
    unsigned seed = 42;  // for random initialization of component means
};

/// Gaussian Mixture Model trained via Expectation-Maximization (EM).
/// Uses diagonal covariance matrices (independent per-feature variances)
/// rather than full covariances — a standard, tractable variant that avoids
/// matrix inversion/determinant machinery. A variance floor (1e-6) prevents
/// EM from collapsing components when duplicate or near-duplicate points would
/// otherwise drive a component variance toward zero.
struct GaussianMixture {
    GMMConfig config;
    Mat means;                   // n_components × n_features
    Mat variances;               // n_components × n_features (diagonal entries)
    Vec weights;                 // n_components mixing weights, sum to 1
    double log_likelihood = 0.0; // final training log-likelihood

    void fit(const Mat& X);
    Mat predict_proba(const Mat& X) const;  // n_samples × n_components responsibilities
    Vec predict(const Mat& X) const;          // hard assignment: argmax component per sample
    double score(const Mat& X) const;         // average log-likelihood under the mixture
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
