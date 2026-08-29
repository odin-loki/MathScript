#pragma once

#include <atomic>
#include "ms/distributed/dist_matrix.hpp"
#include <functional>
#include "ms/distributed/iterative.hpp"
#include "ms/distributed/matmul.hpp"
#include "ms/distributed/mpi_context.hpp"
#include "ms/distributed/solve.hpp"
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/version.hpp"
#include "ms/interp/plot_console.hpp"
#include "ms/core/operations.hpp"
#include "ms/core/sparse.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/cuda/nccl.hpp"
#include "ms/cuda/nvml.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/simd/simd.hpp"
#include "ms/special/special.hpp"
#include "ms/image/image.hpp"
#include "ms/compress/compress.hpp"
#include "ms/bignum/bignum.hpp"
#include "ms/ml/ml.hpp"
#include "ms/graph/graph.hpp"
#include "ms/geo/geo.hpp"
#include "ms/combo/combo.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/control/control.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/finance/finance.hpp"
#include "ms/info/info.hpp"
#include "ms/cplx/cplx.hpp"
#include "ms/tensorops/tensorops.hpp"
#include "ms/diffgeo/diffgeo.hpp"
#include "ms/topo/topo.hpp"
#include "ms/stats/stats.hpp"
#include "ms/prob/prob.hpp"
#include "ms/signal/signal.hpp"
#include "ms/poly/poly.hpp"
#include "ms/pde/pde.hpp"
#include "ms/symbolic/symbolic.hpp"
#include "ms/ode/ode.hpp"
#include "ms/optim/optim.hpp"
#include "ms/crypto/crypto.hpp"
#include "ms/fem/fem.hpp"
#include "ms/cfd/cfd.hpp"
#include "ms/cuda/elementwise.hpp"
#include "ms/cuda/solver.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <regex>
#include <span>
#include <span>
#include <sstream>
#include <string_view>
#include <type_traits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms::interp::detail {

extern thread_local std::atomic<bool>* g_repl_cancel_flag;

ms::distributed::MPIContext& repl_mpi_context();

Result<Matrix<double>> eval_dist_solve(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_cg(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_gmres(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_jacobi(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_bicgstab(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_minres(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_qmr(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_tfqmr(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_lsmr(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_lsqr(const Matrix<double>& A, const Matrix<double>& b);

Result<Matrix<double>> eval_dist_matmul(const Matrix<double>& A, const Matrix<double>& B);

std::string trim_copy(std::string s);

std::string lower(std::string s);

bool parse_number(const std::string& text, double& value);

ColMatrix<double> matrix_to_col_matrix(const Matrix<double>& matrix);

bool parse_uint64(const std::string& text, uint64_t& value);

std::string_view trim_view(std::string_view s);

bool iequals(std::string_view a, std::string_view b);

bool parse_number_view(std::string_view text, double& value);

bool is_literal_arith_char(char c);

bool is_literal_arith_expr(std::string_view expr);

std::string_view strip_outer_parens_view(std::string_view expr);

bool is_binary_minus_view(std::string_view expr, size_t index);

std::optional<std::pair<size_t, char>> find_top_level_op_view(std::string_view expr,
                                                              const char* ops);

std::optional<std::pair<size_t, char>> find_scalar_binop_view(std::string_view rhs);

Result<double> eval_literal_arith(std::string_view expr_text);

struct ScalarFnCache {
    static constexpr size_t kCapacity = 32;
    std::array<std::string, kCapacity> keys{};
    std::array<std::string, kCapacity> lowered{};
    size_t count = 0;

    const std::string& lookup_lowered(std::string_view name) {
        const size_t search_limit = count < kCapacity ? count : kCapacity;
        for (size_t i = 0; i < search_limit; ++i) {
            if (iequals(keys[i], name)) {
                return lowered[i];
            }
        }
        std::string key;
        key.reserve(name.size());
        key.assign(name);
        std::string value = lower(key);
        if (count < kCapacity) {
            keys[count] = std::move(key);
            lowered[count] = std::move(value);
            return lowered[count++];
        }
        const size_t slot = count % kCapacity;
        keys[slot] = std::move(key);
        lowered[slot] = std::move(value);
        ++count;
        return lowered[slot];
    }

    std::string_view lookup(std::string_view name) {
        return lookup_lowered(name);
    }
};

extern thread_local ScalarFnCache g_scalar_fn_cache;

extern thread_local std::vector<double> g_scalar_call_arg_buf;

double matrix_max_value(const Matrix<double>& m);

Result<image::Image> matrix_to_rgb_image(const Matrix<double>& m);

Matrix<double> gray_image_to_column(const image::Image& img);

Result<image::Image> matrix_to_gray_image(const Matrix<double>& m);

Matrix<double> gray_image_to_matrix(const image::Image& img);

Result<std::vector<std::vector<float>>> matrix_to_filter_kernel(const Matrix<double>& k,
                                                                const char* fn);

Matrix<double> rgb_image_to_matrix(const image::Image& img);

Result<int> parse_morph_ksize(double ksize_d, const char* fn);

compress::Bytes matrix_to_bytes(const Matrix<double>& m);

Matrix<double> bytes_to_matrix_col(const compress::Bytes& bytes);

Result<double> eval_bigint_string(const std::string& decimal);

Result<double> bigint_to_scalar(const bignum::BigInt& value, const char* fn);

Result<bignum::BigInt> bigint_from_scalar(double arg, const char* fn);

bool parse_quoted_string(const std::string& text, std::string& out);

Result<ml::Vec> matrix_to_ml_vec(const Matrix<double>& m, const char* fn);

Result<ml::Mat> matrix_to_ml_mat(const Matrix<double>& m, const char* fn);

Result<graph::Graph> graph_from_adjacency(const Matrix<double>& adj, const char* fn);

Result<graph::Graph> graph_from_adjacency_undirected(const Matrix<double>& adj, const char* fn);

Matrix<double> vector_to_column(const std::vector<double>& values);

Result<Matrix<double>> eval_cellai_boltzmann_weights(const std::vector<double>& energies,
                                                      double temperature);

Result<Matrix<double>> eval_cellai_hebbian_update(const Matrix<double>& w_m,
                                                  const Matrix<double>& x_m,
                                                  const Matrix<double>& y_m,
                                                  double learning_rate);

Result<Matrix<double>> eval_cellai_cell_to_cypha_features(
    const cellai::CellMemory& memory, const std::vector<double>& time_scales);

Matrix<double> psd_result_to_matrix(const PSDResult& psd);

Matrix<double> coherence_result_to_matrix(const CoherenceResult& coh);

Matrix<double> iir_coeffs_to_matrix(const IirCoeffs& coeffs);

Result<FilterType> parse_signal_filter_type(const std::string& text);

Result<FirWindow> parse_fir_window(const std::string& text, const char* fn);

Matrix<double> ml_model_to_matrix(const ml::Vec& coef, double intercept);

Result<std::pair<ml::Vec, double>> ml_model_from_matrix(const Matrix<double>& model,
                                                         const char* fn);

Result<Matrix<double>> eval_ml_linear_fit(const Matrix<double>& X_m, const Matrix<double>& y_m);

Result<Matrix<double>> eval_ml_linear_predict(const Matrix<double>& X_m,
                                              const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_ridge_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                         double alpha);

Result<Matrix<double>> eval_ml_ridge_predict(const Matrix<double>& X_m,
                                             const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_logistic_fit(const Matrix<double>& X_m, const Matrix<double>& y_m);

Result<Matrix<double>> eval_ml_logistic_predict(const Matrix<double>& X_m,
                                               const Matrix<double>& model_m);

Matrix<double> ml_pca_model_to_matrix(const ml::PCA& pca);

Result<ml::PCA> ml_pca_from_matrix(const Matrix<double>& model, const char* fn);

Result<ml::KMeans> ml_kmeans_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_pca_fit(const Matrix<double>& X_m, int n_components);

Result<Matrix<double>> eval_ml_pca_transform(const Matrix<double>& X_m,
                                             const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_pca_fit_transform(const Matrix<double>& X_m, int n_components);

Result<Matrix<double>> eval_ml_kmeans_fit(const Matrix<double>& X_m, int k);

Result<Matrix<double>> eval_ml_kmeans_predict(const Matrix<double>& X_m,
                                              const Matrix<double>& model_m);

Result<double> eval_ml_kmeans_inertia(const Matrix<double>& X_m,
                                      const Matrix<double>& model_m);

Matrix<double> ml_gmm_to_matrix(const ml::GaussianMixture& gmm);

Matrix<double> ml_standard_scaler_to_matrix(const ml::StandardScaler& sc);

Result<ml::GaussianMixture> ml_gmm_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_gmm_fit(const Matrix<double>& X_m, int n_components);

Result<Matrix<double>> eval_ml_gmm_predict(const Matrix<double>& X_m,
                                         const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_gmm_predict_proba(const Matrix<double>& X_m,
                                               const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_dbscan_fit(const Matrix<double>& X_m, double eps,
                                          int min_samples);

Result<Matrix<double>> eval_ml_spectral_clustering(const Matrix<double>& X_m, int k,
                                                  double sigma, int n_neighbors);

Matrix<double> ml_isolation_forest_to_matrix(const ml::IsolationForestState& state);

Result<ml::IsolationForest> ml_isolation_forest_from_matrix(const Matrix<double>& model,
                                                            const char* fn);

Result<std::string> parse_ml_linkage(const std::string& text, const char* fn);

Result<Matrix<double>> eval_ml_isolation_forest_fit(const Matrix<double>& X_m, size_t n_trees,
                                                     size_t sample_size, unsigned seed);

Result<Matrix<double>> eval_ml_isolation_forest_score(const Matrix<double>& X_m,
                                                      const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_agglomerative_fit(const Matrix<double>& X_m, int n_clusters,
                                                 const std::string& linkage);

Result<Matrix<double>> eval_ml_tsne_fit(const Matrix<double>& X_m, double perplexity, int n_iter,
                                        unsigned seed);

Result<ml::StandardScaler> ml_standard_scaler_from_matrix(const Matrix<double>& model,
                                                          const char* fn);

Matrix<double> ml_minmax_scaler_to_matrix(const ml::MinMaxScaler& sc);

Result<ml::MinMaxScaler> ml_minmax_scaler_from_matrix(const Matrix<double>& model,
                                                      const char* fn);

Result<Matrix<double>> eval_ml_standard_scaler_fit(const Matrix<double>& X_m);

Result<Matrix<double>> eval_ml_standard_scaler_transform(const Matrix<double>& X_m,
                                                         const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_minmax_scaler_fit(const Matrix<double>& X_m);

Result<Matrix<double>> eval_ml_minmax_scaler_transform(const Matrix<double>& X_m,
                                                       const Matrix<double>& model_m);

Matrix<double> ml_confusion_matrix_to_matrix(const ml::ConfusionMatrix& cm);

Matrix<double> ml_roc_curve_to_matrix(const std::vector<ml::ROCPoint>& curve);

Matrix<double> ml_precision_recall_curve_to_matrix(const std::vector<ml::PRPoint>& curve);

Result<Matrix<double>> eval_ml_confusion_matrix(const Matrix<double>& y_pred_m,
                                                 const Matrix<double>& y_true_m,
                                                 double threshold);

Result<Matrix<double>> eval_ml_roc_curve(const Matrix<double>& y_pred_m,
                                         const Matrix<double>& y_true_m);

Result<Matrix<double>> eval_ml_precision_recall_curve(const Matrix<double>& y_pred_m,
                                                      const Matrix<double>& y_true_m);

Result<Matrix<double>> eval_ml_lasso_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                        double alpha);

Result<Matrix<double>> eval_ml_lasso_predict(const Matrix<double>& X_m,
                                            const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_elastic_net_fit(const Matrix<double>& X_m,
                                               const Matrix<double>& y_m, double alpha,
                                               double l1_ratio);

Result<Matrix<double>> eval_ml_elastic_net_predict(const Matrix<double>& X_m,
                                                  const Matrix<double>& model_m);

Matrix<double> ml_knn_to_matrix(const ml::KNN& knn);

Result<ml::KNN> ml_knn_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_knn_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                       int k);

Result<Matrix<double>> eval_ml_knn_predict(const Matrix<double>& X_m,
                                          const Matrix<double>& model_m);

Matrix<double> ml_naive_bayes_to_matrix(const ml::NaiveBayes& nb);

Result<ml::NaiveBayes> ml_naive_bayes_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_naive_bayes_fit(const Matrix<double>& X_m,
                                               const Matrix<double>& y_m);

Result<Matrix<double>> eval_ml_naive_bayes_predict(const Matrix<double>& X_m,
                                                  const Matrix<double>& model_m);

Matrix<double> ml_lda_to_matrix(const ml::LDA& lda);

Result<ml::LDA> ml_lda_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_lda_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                       int n_components);

Result<Matrix<double>> eval_ml_lda_predict(const Matrix<double>& X_m,
                                          const Matrix<double>& model_m);

Matrix<double> int_vector_to_column(const std::vector<int>& values);

Matrix<double> grid_to_matrix(const std::vector<std::vector<double>>& grid);

Result<Matrix<double>> eval_ml_lda_transform(const Matrix<double>& X_m,
                                            const Matrix<double>& model_m);

Matrix<double> ml_qda_to_matrix(const ml::QDA& qda);

Result<ml::QDA> ml_qda_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_qda_fit(const Matrix<double>& X_m, const Matrix<double>& y_m);

Result<Matrix<double>> eval_ml_qda_predict(const Matrix<double>& X_m,
                                           const Matrix<double>& model_m);

Matrix<double> ml_svm_to_matrix(const ml::SVM& svm);

Result<ml::SVM> ml_svm_from_matrix(const Matrix<double>& model, const char* fn);

Result<Matrix<double>> eval_ml_svm_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                      double C, double gamma, bool use_rbf);

Result<Matrix<double>> eval_ml_svm_predict(const Matrix<double>& X_m,
                                          const Matrix<double>& model_m);

Matrix<double> ml_decision_tree_to_matrix(const ml::DecisionTree& tree);

Result<ml::DecisionTree> ml_decision_tree_from_matrix(const Matrix<double>& model, const char* fn);

Matrix<double> ml_random_forest_to_matrix(const ml::RandomForest& rf);

Result<ml::RandomForest> ml_random_forest_from_matrix(const Matrix<double>& model, const char* fn);

Matrix<double> ml_adaboost_to_matrix(const ml::AdaBoost& ab);

Result<ml::AdaBoost> ml_adaboost_from_matrix(const Matrix<double>& model, const char* fn);

Matrix<double> ml_gradient_boosting_to_matrix(const ml::GradientBoosting& gb);

Result<ml::GradientBoosting> ml_gradient_boosting_from_matrix(const Matrix<double>& model,
                                                               const char* fn);

Result<Matrix<double>> eval_ml_decision_tree_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                                 int max_depth);

Result<Matrix<double>> eval_ml_decision_tree_predict(const Matrix<double>& X_m,
                                                    const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_random_forest_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                                 size_t n_trees, size_t max_depth);

Result<Matrix<double>> eval_ml_random_forest_predict(const Matrix<double>& X_m,
                                                     const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_adaboost_fit(const Matrix<double>& X_m, const Matrix<double>& y_m,
                                            size_t n_estimators, size_t max_depth);

Result<Matrix<double>> eval_ml_adaboost_predict(const Matrix<double>& X_m,
                                                const Matrix<double>& model_m);

Result<Matrix<double>> eval_ml_gradient_boosting_fit(const Matrix<double>& X_m,
                                                     const Matrix<double>& y_m,
                                                     size_t n_estimators, double learning_rate,
                                                     size_t max_depth);

Result<Matrix<double>> eval_ml_gradient_boosting_predict(const Matrix<double>& X_m,
                                                        const Matrix<double>& model_m);

Matrix<double> grid3d_to_matrix(const std::vector<std::vector<std::vector<double>>>& grid);

Result<std::vector<std::vector<double>>> matrix_to_grid(const Matrix<double>& m, const char* fn);

Result<std::vector<geo::Point2D>> matrix_to_points2d(const Matrix<double>& m, const char* fn);

Result<std::vector<geo::Point3D>> matrix_to_points3d(const Matrix<double>& m, const char* fn);

Result<std::vector<std::complex<double>>> matrix_to_complex_spectrum(const Matrix<double>& m,
                                                                     const char* fn);

Result<std::vector<double>> matrix_to_coeff_vector(const Matrix<double>& m, const char* fn);

Result<quantum::Ket> matrix_to_ket(const Matrix<double>& m, const char* fn);

Result<quantum::Ket> matrix_to_ket2(const Matrix<double>& m, const char* fn);

Matrix<double> ket_to_column_matrix(const quantum::Ket& psi);

Result<quantum::DensityMatrix> matrix_to_density_matrix(const Matrix<double>& m, const char* fn);

Matrix<double> density_matrix_to_matrix(const quantum::DensityMatrix& rho);

Result<tensorops::Tensor> matrix_to_tensor(const Matrix<double>& m, const char* fn);

Result<tensorops::Tensor> matrix_to_tensor_shaped(const Matrix<double>& m,
                                                  const std::vector<int>& shape,
                                                  const char* fn);

Result<double> eval_finance_npv(double rate, const Matrix<double>& cashflows_m);

Result<double> eval_finance_sharpe(const Matrix<double>& returns_m);

Result<double> eval_finance_sortino(const Matrix<double>& returns_m);

Result<double> eval_finance_var(const Matrix<double>& returns_m);

Result<double> eval_finance_cvar(const Matrix<double>& returns_m);

Result<double> eval_finance_historical_var(const Matrix<double>& returns_m, double confidence);

Result<double> eval_finance_historical_cvar(const Matrix<double>& returns_m, double confidence);

Result<double> eval_finance_treynor(const Matrix<double>& returns_m, double risk_free,
                                    double beta);

Result<double> eval_finance_information_ratio(const Matrix<double>& returns_m,
                                              const Matrix<double>& benchmark_m);

Result<double> eval_finance_merton_distance_to_default(double asset_value, double asset_volatility,
                                                      double debt_face_value, double risk_free_rate,
                                                      double time_horizon);

Result<Matrix<double>> eval_finance_merton_implied_asset_params(double equity_value,
                                                                double equity_volatility,
                                                                double debt_face_value,
                                                                double risk_free_rate,
                                                                double time_horizon);

Result<double> eval_finance_max_drawdown(const Matrix<double>& equity_m);

Result<double> eval_finance_irr(const Matrix<double>& cashflows_m);

Result<double> eval_finance_bond_ytm(double price, double c, double n_d);

Result<double> eval_numthy_tonelli_shanks(double n_d, double p_d);

Result<double> eval_numthy_mod_inv(double a_d, double m_d);

Result<double> eval_numthy_discrete_log(double g_d, double h_d, double p_d);

Result<double> eval_finance_bs_implied_vol(double price, double S, double K, double T, double r,
                                           double call_d);

Result<std::vector<double>> matrix_to_row_major_flat(const Matrix<double>& m, const char* fn);

Result<double> eval_finance_portfolio_return(const Matrix<double>& w_m,
                                             const Matrix<double>& ret_m);

Result<double> eval_finance_portfolio_variance(const Matrix<double>& w_m,
                                               const Matrix<double>& cov_m);

Result<Matrix<double>> eval_finance_min_variance_portfolio(const Matrix<double>& cov_m);

Result<Matrix<double>> eval_finance_max_sharpe_portfolio(const Matrix<double>& cov_m,
                                                         const Matrix<double>& mu_m,
                                                         double risk_free);

Result<Matrix<double>> eval_finance_efficient_frontier(const Matrix<double>& cov_m,
                                                       const Matrix<double>& mu_m,
                                                       double target_return);

Result<Matrix<double>> eval_finance_max_sharpe(const Matrix<double>& cov_m,
                                               const Matrix<double>& mu_m, double risk_free);

Result<Matrix<double>> eval_finance_bl_implied_returns(const Matrix<double>& cov_m,
                                                       const Matrix<double>& w_mkt_m,
                                                       double delta);

Result<Matrix<double>> eval_finance_bl_posterior_returns(
    const Matrix<double>& pi_m, const Matrix<double>& cov_m, const Matrix<double>& P_m,
    const Matrix<double>& Q_m, double tau);

Result<Matrix<double>> eval_finance_bl_posterior_returns_default_omega(
    const Matrix<double>& pi_m, const Matrix<double>& cov_m, const Matrix<double>& P_m,
    const Matrix<double>& Q_m, double tau);

Result<double> eval_info_entropy(const Matrix<double>& prob_m);

Result<double> eval_info_lz_complexity(const Matrix<double>& seq_m);

Result<double> eval_info_redundancy(const Matrix<double>& p_m);

Result<double> eval_info_efficiency(const Matrix<double>& p_m);

Result<double> eval_info_mutual_info(const Matrix<double>& joint_m);

Result<double> eval_info_blahut_arimoto(const Matrix<double>& W_m);

Result<double> eval_info_channel_capacity(const Matrix<double>& W_m);

Result<Matrix<double>> eval_info_channel_capacity_input(const Matrix<double>& W_m);

Result<double> eval_info_normalized_entropy(const Matrix<double>& p_m);

Result<double> eval_info_joint_entropy(const Matrix<double>& joint_m, int rows, int cols);

Result<double> eval_info_conditional_entropy(const Matrix<double>& joint_m, int rows, int cols);

Result<double> eval_info_sample_entropy(const Matrix<double>& x_m, int m, double r);

Result<double> eval_info_permutation_entropy(const Matrix<double>& x_m, int order, int delay);

Result<double> eval_info_transfer_entropy(const Matrix<double>& x_m, const Matrix<double>& y_m,
                                          int bins, int lag);

Result<double> eval_info_kl_divergence(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<double> eval_info_js_divergence(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<double> eval_info_tv_distance(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<double> eval_info_hellinger_dist(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<double> eval_info_cross_entropy(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<double> eval_info_renyi_entropy(double alpha, const Matrix<double>& p_m);

Result<double> eval_info_source_coding_rate(const Matrix<double>& p_m);

Result<double> eval_info_tsallis_entropy(double q_param, const Matrix<double>& p_m);

Result<double> eval_quantum_entanglement_entropy(const Matrix<double>& psi_m, int dim_a,
                                                 int dim_b);

Result<double> eval_quantum_von_neumann_entropy(const Matrix<double>& rho_m);

Result<double> eval_quantum_concurrence(const Matrix<double>& rho_m);

Result<Matrix<double>> eval_quantum_ket_normalise_matrix(const Matrix<double>& psi_m);

Result<Matrix<double>> eval_quantum_ket_basis(int dim, int index);

Result<Matrix<double>> eval_quantum_fock_state(int n, int n_max);

Result<double> eval_quantum_fidelity(const Matrix<double>& rho_m, const Matrix<double>& sigma_m);

Result<double> eval_quantum_expectation_dm(const Matrix<double>& rho_m, const Matrix<double>& op_m);

Result<double> eval_quantum_expectation(const Matrix<double>& psi_m, const Matrix<double>& op_m);

Result<double> eval_quantum_inner(const Matrix<double>& bra_m, const Matrix<double>& ket_m);

Result<double> eval_quantum_trace_distance(const Matrix<double>& rho_m,
                                           const Matrix<double>& sigma_m);

Result<Matrix<double>> eval_quantum_partial_trace_matrix(const Matrix<double>& rho_m, int d1,
                                                         int d2, int subsystem);

Result<double> eval_tensorops_norm(const Matrix<double>& tensor_m);

Matrix<double> tensor_to_matrix(const tensorops::Tensor& t);

Result<Matrix<double>> eval_tensorops_matmul(const Matrix<double>& left_m,
                                             const Matrix<double>& right_m);

Result<Matrix<double>> eval_tensorops_einsum(const Matrix<double>& left_m,
                                             const Matrix<double>& right_m);

Result<double> eval_tensorops_inner(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<double> eval_geo_polygon_area(const Matrix<double>& points_m);

Result<double> eval_geo_polygon_perimeter(const Matrix<double>& points_m);

Result<double> eval_geo_signed_area(const Matrix<double>& points_m);

Result<double> eval_geo_moment_of_inertia(const Matrix<double>& points_m);

Result<double> eval_geo_bezier_eval_x(const Matrix<double>& ctrl_m, double t);

Result<double> eval_geo_bezier_eval_y(const Matrix<double>& ctrl_m, double t);

Result<Matrix<double>> eval_geo_bezier_eval(const Matrix<double>& ctrl_m, double t);

Result<Matrix<double>> eval_geo_bezier_deriv(const Matrix<double>& ctrl_m, double t);

Result<Matrix<double>> eval_geo_catmull_rom(const Matrix<double>& ctrl_m, double t);

Result<Matrix<double>> eval_geo_bspline_eval(const Matrix<double>& ctrl_m,
                                             const Matrix<double>& knots_m, double degree_d,
                                             double t);

Result<double> eval_geo_point_in_polygon(double px, double py, const Matrix<double>& points_m);

Result<double> eval_ml_categorical_crossentropy(const Matrix<double>& pred_m,
                                              const Matrix<double>& true_m);

Result<double> eval_ml_vec_norm(const Matrix<double>& vec_m);

Result<double> eval_control_step_final(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<double> eval_control_dcgain(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<double> eval_control_is_stable(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<std::vector<std::vector<double>>> matrix_to_nested(const Matrix<double>& m,
                                                           const char* fn);

Result<std::vector<std::vector<double>>> matrix_to_square_nested(const Matrix<double>& m,
                                                                   const char* fn);

Result<double> eval_control_is_controllable(const Matrix<double>& A_m,
                                            const Matrix<double>& B_m);

Result<double> eval_control_is_observable(const Matrix<double>& A_m,
                                          const Matrix<double>& C_m);

Result<double> eval_numthy_extended_gcd(double a_d, double b_d);

Result<double> eval_numthy_crt(const Matrix<double>& r_m, const Matrix<double>& m_m);

Result<double> eval_geo_centroid_x(const Matrix<double>& points_m);

Result<Matrix<double>> eval_quantum_ket_superposition_matrix(const Matrix<double>& amps_m);

Result<Matrix<double>> eval_quantum_ghz_state(int n_qubits);

Result<double> eval_geo_centroid_y(const Matrix<double>& points_m);

Result<Matrix<double>> eval_quantum_w_state(int n_qubits);

Result<Matrix<double>> eval_numthy_divisors_vec(int n);

Result<compress::Bytes> matrix_col_to_bytes(const Matrix<double>& m, const char* fn);

Result<double> eval_bwt_primary_index(const Matrix<double>& m);

Result<Matrix<double>> eval_bwt_decode_vec(const Matrix<double>& l_m, double primary_index);

Result<double> eval_control_impulse_final(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m);

Result<double> eval_combo_multinomial(double n_d, const Matrix<double>& ks_m);

Result<Matrix<double>> eval_numthy_factor_vec(int n);

Result<Matrix<double>> eval_numthy_factor_exp(int n);

Result<Matrix<double>> eval_numthy_farey(int n);

Result<Matrix<double>> eval_numthy_lucas_sequence(int64_t k, int64_t P, int64_t Q);

Result<double> eval_numthy_multiplicative_order(double a_d, double n_d);

Matrix<double> codes_to_matrix_col(const std::vector<uint32_t>& codes);

Result<Matrix<double>> eval_numthy_stern_brocot(int n);

Result<Matrix<double>> eval_numthy_pell_solve(int D);

Result<Matrix<double>> eval_numthy_cornacchia(uint64_t d, uint64_t p);

Result<Matrix<double>> eval_numthy_quadratic_residues(int p);

Result<Matrix<double>> eval_lzw_encode_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_lzw_decode_vec(const Matrix<double>& codes_m);

Result<Matrix<double>> eval_huffman_encode_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_huffman_decode_vec(const Matrix<double>& orig_m,
                                               const Matrix<double>& /*encoded_m*/);

Result<Matrix<double>> eval_arithmetic_encode_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_arithmetic_decode_vec(const Matrix<double>& orig_m,
                                                  const Matrix<double>& /*encoded_m*/);

Result<Matrix<double>> eval_ans_encode_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_ans_decode_vec(const Matrix<double>& orig_m,
                                           const Matrix<double>& /*encoded_m*/);

Result<std::vector<uint32_t>> matrix_col_to_u32(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> eval_golomb_rice_encode_vec(const Matrix<double>& values_m, int m_bits);

Result<Matrix<double>> eval_golomb_rice_decode_vec(const Matrix<double>& encoded_m, int m_bits,
                                                   size_t count);

Result<Matrix<double>> eval_gria_ca_step(const Matrix<double>& state_m, int rule);

Result<double> eval_gria_langton_lambda(int rule);

Result<double> eval_gria_alpha_ca(int rule, size_t steps, size_t width);

Result<double> eval_gria_hamming_distance(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_gria_divergence_trajectory(const Matrix<double>& a_m,
                                                         const Matrix<double>& b_m, int rule,
                                                         int n_steps);

Result<double> eval_gria_settling_time(const Matrix<double>& a_m, const Matrix<double>& b_m,
                                         int rule, int n_steps);

Result<Matrix<double>> eval_wavelet_compress_vec(const Matrix<double>& m,
                                                 double threshold = 0.0);

Result<Matrix<double>> eval_wavelet_decompress_vec(const Matrix<double>& compressed_m);

Result<Matrix<double>> eval_quantum_coherent_state(double alpha_re, double alpha_im, int n_max);

Matrix<double> nested_to_matrix(const std::vector<std::vector<double>>& nested);

Result<Matrix<double>> eval_control_lyap(const Matrix<double>& A_m,
                                        const Matrix<double>& Q_m);

Result<Matrix<double>> eval_control_ctrb(const Matrix<double>& A_m,
                                         const Matrix<double>& B_m);

Result<Matrix<double>> eval_control_obsv(const Matrix<double>& A_m,
                                         const Matrix<double>& C_m);

Result<Matrix<double>> eval_control_ctrb_gram(const Matrix<double>& A_m,
                                              const Matrix<double>& B_m);

Result<Matrix<double>> eval_control_obsv_gram(const Matrix<double>& A_m,
                                              const Matrix<double>& C_m);

Result<control::KalmanState> kalman_state_from_matrices(const Matrix<double>& x_m,
                                                          const Matrix<double>& P_m,
                                                          const char* fn);

Result<Matrix<double>> eval_control_kalman_predict(const Matrix<double>& x_m,
                                                   const Matrix<double>& P_m,
                                                   const Matrix<double>& A_m,
                                                   const Matrix<double>& Q_m);

Result<Matrix<double>> eval_control_kalman_predict_cov(const Matrix<double>& x_m,
                                                         const Matrix<double>& P_m,
                                                         const Matrix<double>& A_m,
                                                         const Matrix<double>& Q_m);

Result<Matrix<double>> eval_control_kalman_update(const Matrix<double>& x_m,
                                                  const Matrix<double>& P_m,
                                                  const Matrix<double>& z_m,
                                                  const Matrix<double>& H_m,
                                                  const Matrix<double>& R_m);

Result<Matrix<double>> eval_control_kalman_update_cov(const Matrix<double>& x_m,
                                                       const Matrix<double>& P_m,
                                                       const Matrix<double>& z_m,
                                                       const Matrix<double>& H_m,
                                                       const Matrix<double>& R_m);

Result<double> eval_combo_rank_permutation(const Matrix<double>& v_m);

Result<Matrix<double>> eval_combo_unrank_permutation(int n, uint64_t rank);

Result<Matrix<double>> eval_control_lqe(const Matrix<double>& A_m,
                                        const Matrix<double>& C_m,
                                        const Matrix<double>& Q_m,
                                        const Matrix<double>& R_m);

Result<Matrix<double>> eval_control_lqr(const Matrix<double>& A_m,
                                        const Matrix<double>& B_m,
                                        const Matrix<double>& Q_m,
                                        const Matrix<double>& R_m);

Result<double> eval_combo_rank_combination(const Matrix<double>& v_m, int n);

Result<Matrix<double>> eval_lz77_encode_vec(const Matrix<double>& m, int window = 255,
                                            int lookahead = 15);

Result<Matrix<double>> eval_lz77_decode_vec(const Matrix<double>& tokens_m);

Result<double> eval_control_pidtune_kp(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<double> eval_control_pidtune_ki(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<double> eval_control_pidtune_kd(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<Matrix<double>> eval_combo_unrank_combination(int n, int k, uint64_t rank);

Result<double> eval_cplx_power_series_eval(const Matrix<double>& coeffs_m, double zre, double zim);

Result<double> eval_cplx_winding_number(const Matrix<double>& gamma_m, double z0re, double z0im);

Result<Matrix<double>> eval_quantum_schrodinger_matrix(const Matrix<double>& H_m,
                                                       const Matrix<double>& psi0_m, double t0,
                                                       double t1, int n_steps);

Result<double> eval_topo_vietoris_rips_betti0(const Matrix<double>& dist_m, double r,
                                              int max_dim);

Result<Matrix<double>> eval_control_dlyap(const Matrix<double>& A_m, const Matrix<double>& Q_m);

Result<Matrix<double>> eval_control_riccati(const Matrix<double>& A_m,
                                            const Matrix<double>& B_m,
                                            const Matrix<double>& Q_m,
                                            const Matrix<double>& R_m);

Result<Matrix<double>> eval_control_dare(const Matrix<double>& A_m,
                                         const Matrix<double>& B_m,
                                         const Matrix<double>& Q_m,
                                         const Matrix<double>& R_m);

Result<double> eval_control_bode_mag_db(const Matrix<double>& num_m, const Matrix<double>& den_m,
                                        double w);

Result<double> eval_control_bode_phase(const Matrix<double>& num_m, const Matrix<double>& den_m,
                                       double w);

Result<double> eval_control_phase_margin(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<double> eval_control_gain_margin(const Matrix<double>& num_m, const Matrix<double>& den_m);

Result<Matrix<double>> eval_control_bode(const Matrix<double>& num_m, const Matrix<double>& den_m,
                                         double w);

Result<Matrix<double>> eval_combo_all_permutations(int n);

Result<Matrix<double>> eval_quantum_op_apply(const Matrix<double>& op_m,
                                             const Matrix<double>& psi_m);

Result<Matrix<double>> eval_topo_persistence_diagram(const Matrix<double>& simplices_m,
                                                     const Matrix<double>& births_m);

diffgeo::MetricFn euclidean_2d_metric_fn();

Result<Matrix<double>> eval_diffgeo_geodesic_euclidean(double x0, double y0, double vx, double vy,
                                                      double s_end);

Result<Matrix<double>> eval_compress_bits_to_bytes(const Matrix<double>& bits_m);

Result<double> eval_cplx_blaschke_product(double zre, double zim, const Matrix<double>& zeros_m);

Result<double> eval_graph_diameter(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_compress_bytes_to_bits(const Matrix<double>& bytes_m);

Result<double> eval_graph_radius(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_combo_all_subsets(int n);

Result<Matrix<double>> eval_control_margins(const Matrix<double>& num_m,
                                            const Matrix<double>& den_m);

Result<Matrix<double>> eval_control_poles(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m);

Result<Matrix<double>> eval_control_zeros(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m);

Result<Matrix<double>> eval_control_step_info(const Matrix<double>& num_m,
                                              const Matrix<double>& den_m);

Result<Matrix<double>> eval_control_nyquist(const Matrix<double>& num_m,
                                            const Matrix<double>& den_m);

Result<Matrix<double>> step_data_to_matrix(const control::StepData& data, const char* fn);

Result<Matrix<double>> eval_control_step_response(const Matrix<double>& num_m,
                                                  const Matrix<double>& den_m, double t_end,
                                                  int n_pts);

Result<Matrix<double>> eval_control_impulse_response(const Matrix<double>& num_m,
                                                     const Matrix<double>& den_m, double t_end,
                                                     int n_pts);

Result<Matrix<double>> pack_state_space(const control::StateSpace& sys, const char* fn);

Result<Matrix<double>> eval_control_tf2ss(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m);

Result<Matrix<double>> eval_control_c2d(const Matrix<double>& A_m, const Matrix<double>& B_m,
                                        const Matrix<double>& C_m, const Matrix<double>& D_m,
                                        double Ts, control::DiscretizationMethod method,
                                        const char* fn);

Result<Matrix<double>> eval_control_c2d_B(const Matrix<double>& A_m, const Matrix<double>& B_m,
                                          const Matrix<double>& C_m, const Matrix<double>& D_m,
                                          double Ts);

Matrix<double> pack_transfer_function(const control::TransferFunction& sys);

Result<control::TransferFunction> matrices_to_transfer_function(const Matrix<double>& num_m,
                                                                const Matrix<double>& den_m,
                                                                const char* fn);

Result<control::StateSpace> packed_ss_to_state_space(const Matrix<double>& ss_m,
                                                     const char* fn);

Result<Matrix<double>> eval_control_series(const Matrix<double>& num1_m,
                                           const Matrix<double>& den1_m,
                                           const Matrix<double>& num2_m,
                                           const Matrix<double>& den2_m);

Result<Matrix<double>> eval_control_parallel(const Matrix<double>& num1_m,
                                           const Matrix<double>& den1_m,
                                           const Matrix<double>& num2_m,
                                           const Matrix<double>& den2_m);

Result<Matrix<double>> eval_control_feedback(const Matrix<double>& numG_m,
                                             const Matrix<double>& denG_m,
                                             const Matrix<double>& numH_m,
                                             const Matrix<double>& denH_m, int sign);

Result<Matrix<double>> eval_control_ss2tf(const Matrix<double>& ss_m);

Result<Matrix<double>> eval_control_d2c(const Matrix<double>& A_m, const Matrix<double>& B_m,
                                        const Matrix<double>& C_m, const Matrix<double>& D_m,
                                        double Ts, control::DiscretizationMethod method,
                                        const char* fn);

Result<Matrix<double>> eval_control_c2d_tf(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m, double Ts,
                                          control::DiscretizationMethod method,
                                          const char* fn);

Result<Matrix<double>> eval_control_d2c_tf(const Matrix<double>& num_m,
                                          const Matrix<double>& den_m, double Ts,
                                          control::DiscretizationMethod method,
                                          const char* fn);

Result<double> eval_quantum_purity(const Matrix<double>& rho_m);

Result<double> eval_quantum_wigner(const Matrix<double>& rho_m, double x, double p);

Result<double> eval_quantum_husimi(const Matrix<double>& rho_m, double alpha_re, double alpha_im);

Result<Matrix<double>> eval_quantum_grover_search(int n_qubits, const Matrix<double>& marked_m,
                                                  int n_iterations);

Result<double> eval_quantum_schmidt_rank(const Matrix<double>& psi_m, int dim_a, int dim_b);

Result<double> eval_quantum_uncertainty(const Matrix<double>& psi_m, const Matrix<double>& A_m,
                                        const Matrix<double>& B_m);

Result<Matrix<double>> eval_quantum_schrodinger_final(const Matrix<double>& H_m,
                                                      const Matrix<double>& psi0_m, double t0,
                                                      double t1, int n_steps);

Result<Matrix<double>> eval_graph_betweenness(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_imcrop(const Matrix<double>& m, int r0, int c0, int r1, int c1);

Matrix<double> hough_lines_to_matrix(const std::vector<image::HoughLine>& lines);

Matrix<double> hough_circles_to_matrix(const std::vector<image::HoughCircle>& circles);

Matrix<double> keypoints_to_matrix(const std::vector<image::KeyPoint>& kps);

Result<Matrix<double>> eval_hough_lines(const Matrix<double>& m, double edge_threshold,
                                        int n_theta, int n_rho, int vote_threshold);

Result<Matrix<double>> eval_hough_circles(const Matrix<double>& m, double edge_threshold,
                                          double r_min, double r_max, int r_step,
                                          int vote_threshold);

Result<Matrix<double>> eval_harris(const Matrix<double>& m, float k, float threshold);

Result<Matrix<double>> eval_shi_tomasi(const Matrix<double>& m, int n, float quality_level);

Matrix<double> combo_enum_rows_to_matrix(const std::vector<std::vector<int>>& rows);

Result<Matrix<double>> eval_combo_all_compositions(int n);

Result<Matrix<double>> eval_combo_all_partitions(int n);

Result<Matrix<double>> eval_graph_closeness(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_degree_centrality(const Matrix<double>& adj_m);

Result<double> eval_graph_max_flow(const Matrix<double>& adj_m, int source, int sink);

Result<double> eval_graph_min_cut(const Matrix<double>& adj_m, int source, int sink);

Result<Matrix<double>> eval_graph_bridges(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_maximum_matching(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_transitive_closure(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_quantum_commutator(const Matrix<double>& A_m,
                                               const Matrix<double>& B_m);

Result<double> eval_stats_correlation(const Matrix<double>& x_m, const Matrix<double>& y_m);

Result<Matrix<double>> eval_signal_moving_average(const Matrix<double>& x_m, size_t window);

Result<Matrix<double>> eval_signal_upsample(const Matrix<double>& x_m, int n);

Result<Matrix<double>> eval_signal_downsample(const Matrix<double>& x_m, int n);

Result<int> require_positive_int_arg(double v, const char* fn, const char* arg_name);

Result<Matrix<double>> eval_signal_decimate(const Matrix<double>& x_m, int q);

Result<Matrix<double>> eval_signal_interpolate(const Matrix<double>& x_m, int p);

Result<Matrix<double>> eval_signal_resample(const Matrix<double>& x_m, int p, int q);

Result<Matrix<double>> eval_signal_integer_factor(const std::string& fn,
                                                 const Matrix<double>& x_m,
                                                 double factor);

std::string format_labeled_matrix(const std::string& label, const Matrix<double>& m);

Result<Matrix<double>> eval_signal_resample_pq(const Matrix<double>& x_m, double p_d,
                                              double q_d);

Result<Matrix<double>> eval_signal_savgol(const Matrix<double>& x_m, int window_length,
                                          int polyorder);

Result<Matrix<double>> eval_signal_savgol_wp(const Matrix<double>& x_m, double window_d,
                                            double poly_d);

Result<Matrix<double>> eval_signal_median_filter(const Matrix<double>& x_m, int window_length);

Result<Matrix<double>> eval_signal_median_filter_w(const Matrix<double>& x_m, double window_d);

Result<LMSResult> run_signal_lms(const Matrix<double>& x_m, const Matrix<double>& d_m,
                                 double filter_length_d, double mu, const char* fn);

Result<Matrix<double>> eval_signal_lms(const Matrix<double>& x_m, const Matrix<double>& d_m,
                                       double filter_length_d, double mu);

Result<Matrix<double>> eval_signal_lms_weights(const Matrix<double>& x_m,
                                               const Matrix<double>& d_m,
                                               double filter_length_d, double mu);

Result<Matrix<double>> eval_geo_delaunay_2d(const Matrix<double>& P_m);

Matrix<double> points2d_to_matrix(const std::vector<geo::Point2D>& pts);

Result<double> eval_geo_kdtree_nearest(const Matrix<double>& P_m, double qx, double qy);

Result<double> eval_geo_kdtree_3d_nearest(const Matrix<double>& P_m, double qx, double qy,
                                          double qz);

Result<Matrix<double>> eval_geo_kdtree_knn(const Matrix<double>& P_m, double qx, double qy,
                                           double k_d);

Result<Matrix<double>> eval_geo_kdtree_range(const Matrix<double>& P_m, double qx, double qy,
                                             double r);

Result<Matrix<double>> eval_topo_pairwise_distances(const Matrix<double>& P_m);

Result<Matrix<double>> eval_numthy_continued_fraction(double x, int max_terms);

Result<Matrix<double>> eval_combo_next_perm(const Matrix<double>& v_m);

Result<Matrix<double>> eval_combo_prev_perm(const Matrix<double>& v_m);

Result<double> eval_cplx_mobius_re(double a, double b, double c, double d, double z_re,
                                   double z_im);

Result<Matrix<double>> eval_geo_voronoi(const Matrix<double>& P_m);

Result<Matrix<double>> eval_geo_convex_hull(const Matrix<double>& P_m);

Result<Matrix<double>> eval_geo_upper_hull(const Matrix<double>& P_m);

Result<Matrix<double>> eval_geo_lower_hull(const Matrix<double>& P_m);

Result<Matrix<double>> eval_geo_bezier_subdivide(const Matrix<double>& ctrl_m, double t);

Result<Matrix<double>> eval_geo_kdtree_3d_knn(const Matrix<double>& P_m, double qx, double qy,
                                               double qz, double k_d);

Result<Matrix<double>> eval_geo_kdtree_3d_range(const Matrix<double>& P_m, double qx, double qy,
                                                double qz, double r);

Result<Matrix<double>> eval_geo_triangulate_polygon(const Matrix<double>& P_m);

Result<Matrix<double>> eval_geo_convex_hull_3d(const Matrix<double>& P_m);

Result<Matrix<double>> eval_geo_poly_boolean(const char* fn, const Matrix<double>& a_m,
                                             const Matrix<double>& b_m);

Result<Matrix<double>> eval_geo_minkowski_sum(const Matrix<double>& a_m,
                                              const Matrix<double>& b_m);

Result<Matrix<double>> eval_geo_clip_polygon(const Matrix<double>& subject_m,
                                             const Matrix<double>& window_m);

Result<Matrix<double>> eval_geo_min_bounding_rect(const Matrix<double>& P_m);

Result<std::vector<int64_t>> matrix_to_int64_coeff_vector(const Matrix<double>& m,
                                                          const char* fn);

Result<std::vector<int>> matrix_to_int_coeff_vector(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> eval_numthy_convergents(const Matrix<double>& cf_m);

Result<Matrix<double>> eval_ml_mat_transpose(const Matrix<double>& A_m);

Result<Matrix<double>> eval_combo_next_comb(const Matrix<double>& v_m, int n);

Result<Matrix<double>> eval_combo_prev_comb(const Matrix<double>& v_m, int n);

Result<Matrix<double>> eval_numthy_primes(uint64_t lo, uint64_t hi);

Result<Matrix<double>> eval_graph_scc(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_connected_components(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_louvain(const Matrix<double>& adj_m);

Matrix<double> graph_edge_components_to_matrix(
    const std::vector<std::vector<graph::Edge>>& components);

Result<double> eval_graph_bipartite_match(const Matrix<double>& adj_m, int left_size);

Result<Matrix<double>> eval_graph_biconnected_components(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_eulerian_path(const Matrix<double>& adj_m);

Result<double> eval_graph_is_isomorphic(const Matrix<double>& adj_a_m,
                                        const Matrix<double>& adj_b_m);

Result<Matrix<double>> eval_graph_hamiltonian_path(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_tsp_heuristic(const Matrix<double>& dist_m);

Result<Matrix<double>> eval_graph_eigenvector_centrality(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_katz_centrality(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_adjacency_spectrum(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_laplacian(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_normalised_laplacian(const Matrix<double>& adj_m);

Result<std::vector<std::vector<int>>> community_matrix_to_partition(const Matrix<double>& C,
                                                                     const char* fn);

Result<double> eval_graph_modularity(const Matrix<double>& adj_m, const Matrix<double>& C);

Result<Matrix<double>> eval_graph_eccentricity(const Matrix<double>& adj_m);

Result<double> eval_graph_is_strongly_connected(const Matrix<double>& adj_m);

Result<double> eval_graph_algebraic_connectivity(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_articulation_points(const Matrix<double>& adj_m);

Result<double> eval_geo_hermite_x(double p0x, double p0y, double m0x, double m0y, double p1x,
                                  double p1y, double m1x, double m1y, double t);

Result<Matrix<double>> eval_geo_hermite_curve(double p0x, double p0y, double m0x, double m0y,
                                              double p1x, double p1y, double m1x, double m1y,
                                              double t);

Result<Matrix<double>> eval_ml_mat_mul(const Matrix<double>& A_m, const Matrix<double>& B_m);

Result<double> eval_stats_min_value(const Matrix<double>& x_m);

Result<double> eval_stats_max_value(const Matrix<double>& x_m);

Result<double> eval_count_components(const Matrix<double>& bw_m);

Result<Matrix<double>> eval_prewitt(const Matrix<double>& m);

Result<Matrix<double>> eval_scharr(const Matrix<double>& m);

Result<Matrix<double>> eval_roberts(const Matrix<double>& m);

Result<Matrix<double>> eval_fftshift(const Matrix<double>& S_m);

Result<Matrix<double>> eval_ifftshift(const Matrix<double>& S_m);

Result<Matrix<double>> eval_fftfreq(size_t n, double d);

Result<Matrix<double>> eval_rfftfreq(size_t n, double d);

Result<Matrix<double>> eval_fft_rfft(const Matrix<double>& x_m);

Result<double> eval_graph_is_bipartite(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_poly_deriv(const Matrix<double>& coeffs_m);

Result<double> eval_poly_eval(const Matrix<double>& coeffs_m, double x);

Result<double> eval_poly_resultant(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<double> eval_poly_discriminant(const Matrix<double>& coeffs_m);

Result<double> eval_graph_is_dag(const Matrix<double>& adj_m);

Result<double> eval_stats_mean(const Matrix<double>& x_m);

Result<Matrix<double>> eval_fft_irfft(const Matrix<double>& spectrum_m, size_t n);

Result<Matrix<double>> eval_fft_ifft(const Matrix<double>& spectrum_m);

Result<Matrix<double>> eval_signal_convolve(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_signal_conv2(const Matrix<double>& A, const Matrix<double>& K);

Result<Matrix<double>> eval_signal_deconv(const Matrix<double>& y_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_pde_heat_1d(const Matrix<double>& x0_m, double alpha, double dx,
                                       double dt, std::size_t steps);

Result<Matrix<double>> eval_pde_heat_2d(const Matrix<double>& u0_m, double alpha, double dx,
                                       double dy, double dt, std::size_t steps);

Result<Matrix<double>> eval_pde_heat_1d_cn(const Matrix<double>& x0_m, double alpha, double dx,
                                            double dt, std::size_t steps);

Result<Matrix<double>> eval_pde_heat_2d_cn_adi(const Matrix<double>& u0_m, double alpha,
                                                double dx, double dy, double dt,
                                                std::size_t steps);

Result<Matrix<double>> eval_pde_wave_1d(const Matrix<double>& u0_m, const Matrix<double>& v0_m,
                                        double c, double dx, double dt, std::size_t steps);

Result<Matrix<double>> eval_pde_advection_1d(const Matrix<double>& u0_m, double v, double dx,
                                            double dt, std::size_t steps);

Result<Matrix<double>> eval_pde_poisson_2d(const Matrix<double>& f_m, double dx, double dy,
                                          std::size_t max_iterations, double tolerance);

Result<Matrix<double>> eval_pde_poisson_1d(const Matrix<double>& f_m, double dx, double ua,
                                          double ub);

Result<Matrix<double>> eval_pde_laplace_2d(int nx, int ny, const Matrix<double>& boundary_m);

Result<Matrix<double>> eval_pde_helmholtz_2d(const Matrix<double>& f_m, double k, double dx,
                                            double dy,
                                            const Matrix<double>* g_m = nullptr);

Result<Matrix<double>> eval_pde_burgers_1d(const Matrix<double>& u0_m, double nu, double dx,
                                           double dt, std::size_t steps);

Result<Matrix<double>> eval_pde_wave_2d(const Matrix<double>& u0_m, const Matrix<double>& v0_m,
                                        double c, double dx, double dy, double dt,
                                        std::size_t steps);

Result<Matrix<double>> eval_pde_advection_1d_lax_wendroff(const Matrix<double>& u0_m, double v,
                                                          double dx, double dt,
                                                          std::size_t steps);

Result<Matrix<double>> eval_pde_reaction_diffusion_1d(const Matrix<double>& u0_m, double D,
                                                    double r, double dx, double dt,
                                                    std::size_t steps);

int hex_nibble(char c);

Result<std::vector<uint8_t>> parse_hex_arg(const std::string& text, const char* fn,
                                           const char* arg_name);

Result<std::string> eval_crypto_aes128_encrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg);

Result<std::string> eval_crypto_aes128_decrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg);

Result<std::string> eval_crypto_aes256_encrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg);

Result<std::string> eval_crypto_aes256_decrypt_block(const std::string& key_arg,
                                                     const std::string& block_arg);

Result<std::string> eval_crypto_aes128_cbc_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& plain_arg);

Result<std::string> eval_crypto_aes128_cbc_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& cipher_arg);

Result<std::string> eval_crypto_aes256_cbc_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& plain_arg);

Result<std::string> eval_crypto_aes256_cbc_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& cipher_arg);

Result<std::string> eval_crypto_aes128_gcm_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& plain_arg);

Result<std::string> eval_crypto_aes128_gcm_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& cipher_arg,
                                                   const std::string& tag_arg);

Result<std::string> eval_crypto_aes256_gcm_encrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& plain_arg);

Result<std::string> eval_crypto_aes256_gcm_decrypt(const std::string& key_arg,
                                                   const std::string& iv_arg,
                                                   const std::string& aad_arg,
                                                   const std::string& cipher_arg,
                                                   const std::string& tag_arg);

Result<std::string> eval_crypto_chacha20(const std::string& key_arg, const std::string& nonce_arg,
                                         const std::string& counter_arg,
                                         const std::string& data_arg);

Result<std::string> eval_crypto_chacha20_poly1305_encrypt(const std::string& key_arg,
                                                          const std::string& nonce_arg,
                                                          const std::string& aad_arg,
                                                          const std::string& plain_arg);

Result<std::string> eval_crypto_chacha20_poly1305_decrypt(const std::string& key_arg,
                                                          const std::string& nonce_arg,
                                                          const std::string& aad_arg,
                                                          const std::string& cipher_arg,
                                                          const std::string& tag_arg);

Result<std::vector<uint8_t>> parse_hex_arg_allow_empty(const std::string& text, const char* fn,
                                                      const char* arg_name);

Result<std::string> eval_crypto_sha256(const std::string& data_arg);

Result<std::string> eval_crypto_to_hex(const std::string& data_arg);

Result<Matrix<double>> eval_crypto_from_hex(const std::string& hex_arg);

Result<std::string> eval_crypto_bytes_to_hex(const Matrix<double>& bytes_m);

Result<Matrix<double>> eval_crypto_bytes_to_hex_vec(const Matrix<double>& bytes_m);

Result<std::string> eval_crypto_sha512(const std::string& data_arg);

Result<std::string> eval_crypto_hmac_sha256(const std::string& key_arg,
                                            const std::string& data_arg);

Result<std::string> eval_crypto_hmac_sha512(const std::string& key_arg,
                                            const std::string& data_arg);

Result<std::string> eval_crypto_hkdf_sha256(const std::string& ikm_arg,
                                            const std::string& salt_arg,
                                            const std::string& info_arg,
                                            const std::string& len_arg);

Result<std::string> eval_crypto_hkdf_sha512(const std::string& ikm_arg,
                                            const std::string& salt_arg,
                                            const std::string& info_arg,
                                            const std::string& len_arg);

Result<std::string> eval_crypto_pbkdf2_sha256(const std::string& pass_arg,
                                              const std::string& salt_arg,
                                              const std::string& iter_arg,
                                              const std::string& dklen_arg);

Result<std::string> eval_crypto_pbkdf2_hmac_sha512(const std::string& pass_arg,
                                                     const std::string& salt_arg,
                                                     const std::string& iter_arg,
                                                     const std::string& dklen_arg);

Result<std::string> eval_crypto_x25519_keypair(const std::string& priv_arg);

Result<std::string> eval_crypto_x25519_shared(const std::string& priv_arg,
                                              const std::string& pub_arg);

Result<std::string> eval_crypto_ed25519_keypair(const std::string& seed_arg);

Result<std::string> eval_crypto_ed25519_sign(const std::string& secret_arg,
                                              const std::string& msg_arg);

Result<std::string> eval_crypto_ed25519_verify(const std::string& pub_arg,
                                               const std::string& msg_arg,
                                               const std::string& sig_arg);

Result<std::string> eval_crypto_constant_time_eq(const std::string& hex_a,
                                                 const std::string& hex_b);

Result<std::string> eval_crypto_random_bytes(const std::string& n_arg);

std::vector<std::size_t> fem_rectangular_boundary_nodes(std::size_t nx, std::size_t ny);

std::vector<std::size_t> fem_box_boundary_nodes(
    std::size_t nx, std::size_t ny, std::size_t nz);

Result<Matrix<double>> eval_fem_poisson1d(std::size_t n);

Result<Matrix<double>> eval_fem_poisson2d(std::size_t nx, std::size_t ny);

Result<Matrix<double>> eval_fem_poisson3d(
    std::size_t nx, std::size_t ny, std::size_t nz);

Matrix<double> pack_fem_mesh1d(const fem::Mesh1D& mesh);

Result<fem::Mesh1D> matrix_to_fem_mesh1d(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> eval_fem_mesh1d(double a, double b, std::size_t n_elements);

Result<Matrix<double>> eval_fem_stiffness_1d(const Matrix<double>& mesh_m);

Result<Matrix<double>> eval_fem_load_1d(const Matrix<double>& mesh_m, double f_const);

Result<Matrix<double>> eval_fem_lagrange_eval(double xi);

Matrix<double> pack_fem_mesh2d(const fem::Mesh2D& mesh);

Result<fem::Mesh2D> matrix_to_fem_mesh2d(const Matrix<double>& m, const char* fn);

Result<std::vector<std::size_t>> matrix_to_fem_node_indices(const Matrix<double>& m,
                                                            const char* fn);

Result<Matrix<double>> eval_fem_mesh2d_rectangular(
    double x0, double y0, double x1, double y1, std::size_t nx, std::size_t ny);

Result<Matrix<double>> eval_fem_stiffness_2d(const Matrix<double>& mesh_m);

Result<Matrix<double>> eval_fem_load_2d(const Matrix<double>& mesh_m, double f_const);

Result<Matrix<double>> eval_fem_apply_dirichlet(const Matrix<double>& K_m,
                                               const Matrix<double>& f_m,
                                               const Matrix<double>& nodes_m,
                                               const Matrix<double>& values_m);

Result<Matrix<double>> eval_fem_solve(const Matrix<double>& K_m, const Matrix<double>& f_m);

Result<Matrix<double>> eval_fem_solve_packed(const Matrix<double>& sys_m);

Result<Matrix<double>> eval_fem_solve_3d(const Matrix<double>& K_m, const Matrix<double>& f_m);

Result<Matrix<double>> eval_fem_lagrange_deriv(double xi);

Result<double> eval_cplx_green_function_disk(double zre, double zim, double z0re, double z0im,
                                              double radius);

Result<std::string> eval_cplx_cauchy_principal_value_call(const std::string& formula_arg,
                                                          const std::string& a_arg,
                                                          const std::string& c_arg,
                                                          const std::string& b_arg,
                                                          const std::string& n_pts_arg);

Matrix<double> sparse_to_packed_matrix(const Sparse<double>& S);

Result<Sparse<double>> sparse_from_packed_matrix(const Matrix<double>& m, const char* fn);

Result<ColMatrix<double>> sparse_vector_to_col_column(const Matrix<double>& m, size_t expected_len,
                                                      const char* fn);

Matrix<double> col_matrix_to_matrix(const ColMatrix<double>& m);

Result<Matrix<double>> eval_sparse_from_coo(std::size_t rows, std::size_t cols,
                                            const Matrix<double>& row_idx_m,
                                            const Matrix<double>& col_idx_m,
                                            const Matrix<double>& values_m);

Result<Matrix<double>> eval_sparse_spmv(const Matrix<double>& packed_m,
                                       const Matrix<double>& x_m);

Result<Matrix<double>> eval_sparse_to_dense(const Matrix<double>& packed_m);

Result<Matrix<double>> eval_sparse_add(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_cfd_advection1d(std::size_t nx, double vx, double t_end, double dt);

Result<Matrix<double>> eval_cfd_advection2d(std::size_t nx, std::size_t ny, double vx, double vy,
                                            double t_end, double dt);

Result<Matrix<double>> eval_cfd_advection3d(std::size_t nx, std::size_t ny, std::size_t nz,
                                            double vx, double vy, double vz, double t_end,
                                            double dt);

Matrix<double> pack_cfd_grid1d(const cfd::Grid1D& grid);

Result<cfd::Grid1D> cfd_grid1d_from_packed_matrix(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> eval_cfd_grid1d(double x0, double x1, std::size_t n);

Result<Matrix<double>> eval_cfd_square_pulse(
    const Matrix<double>& grid_m, double xc, double width, double amplitude);

Result<Matrix<double>> eval_cfd_run_advection(const Matrix<double>& grid_m,
                                              const Matrix<double>& u_m, double v,
                                              double t_end, double dt);

Result<Matrix<double>> eval_cfd_upwind_step_1d(const Matrix<double>& grid_m,
                                               const Matrix<double>& u_m, double v, double dt,
                                               cfd::BoundaryCondition bc);

Matrix<double> pack_cfd_grid2d(const cfd::Grid2D& grid);

Result<cfd::Grid2D> cfd_grid2d_from_packed_matrix(const Matrix<double>& m, const char* fn);

Result<cfd::BoundaryCondition> parse_cfd_bc(double value, const char* fn);

Result<Matrix<double>> eval_cfd_grid2d(
    double x0, double x1, double y0, double y1, std::size_t nx, std::size_t ny);

Result<Matrix<double>> eval_cfd_square_pulse_2d(
    const Matrix<double>& grid_m,
    double xc,
    double yc,
    double width_x,
    double width_y,
    double amplitude);

Result<Matrix<double>> eval_cfd_upwind_step_2d(
    const Matrix<double>& u_m,
    double vx,
    double vy,
    double dt,
    double dx,
    double dy,
    cfd::BoundaryCondition bc_x,
    cfd::BoundaryCondition bc_y);

Result<double> eval_cfd_integrated_mass_2d(
    const Matrix<double>& u_m, double dx, double dy);

Result<double> eval_cfd_integrated_mass_1d(const Matrix<double>& grid_m,
                                           const Matrix<double>& u_m);

Result<double> eval_cfd_integrated_mass_2d_from_grid(const Matrix<double>& grid_m,
                                                     const Matrix<double>& u_m);

Result<Matrix<double>> eval_cfd_constant_velocity(std::size_t n, double v);

Result<Matrix<double>> eval_cfd_upwind_step_2d_from_grid(
    const Matrix<double>& grid_m,
    const Matrix<double>& u_m,
    double vx,
    double vy,
    double dt,
    cfd::BoundaryCondition bc_x,
    cfd::BoundaryCondition bc_y);

Result<Matrix<double>> eval_cfd_run_advection_2d(const Matrix<double>& grid_m,
                                                 const Matrix<double>& u_m, double vx, double vy,
                                                 double t_end, double dt);

Matrix<double> pack_fem_mesh3d(const fem::Mesh3D& mesh);

Result<fem::Mesh3D> matrix_to_fem_mesh3d(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> eval_fem_mesh3d_box(
    double x0, double y0, double z0, double x1, double y1, double z1, std::size_t nx,
    std::size_t ny, std::size_t nz);

Result<Matrix<double>> eval_fem_stiffness_3d(const Matrix<double>& mesh_m);

Result<Matrix<double>> eval_fem_load_3d(const Matrix<double>& mesh_m, double f_const);

Matrix<double> pack_cfd_grid3d(const cfd::Grid3D& grid);

Result<cfd::Grid3D> cfd_grid3d_from_packed_matrix(const Matrix<double>& m, const char* fn);

Result<std::vector<std::vector<std::vector<double>>>> matrix_to_grid3d(const Matrix<double>& m,
                                                                       std::size_t nx,
                                                                       std::size_t ny,
                                                                       std::size_t nz,
                                                                       const char* fn);

Result<Matrix<double>> eval_cfd_grid3d(double x0, double x1, double y0, double y1, double z0,
                                      double z1, std::size_t nx, std::size_t ny, std::size_t nz);

Result<Matrix<double>> eval_cfd_square_pulse_3d(const Matrix<double>& grid_m, double xc, double yc,
                                                double zc, double width_x, double width_y,
                                                double width_z, double amplitude);

Result<Matrix<double>> eval_cfd_upwind_step_3d(const Matrix<double>& grid_m,
                                               const Matrix<double>& u_m, double vx, double vy,
                                               double vz, double dt,
                                               cfd::BoundaryCondition bc_x,
                                               cfd::BoundaryCondition bc_y,
                                               cfd::BoundaryCondition bc_z);

Result<double> eval_cfd_integrated_mass_3d(const Matrix<double>& grid_m,
                                           const Matrix<double>& u_m);

Result<Matrix<double>> eval_gria_gf2n_generate_field(int n);

Result<Matrix<double>> eval_quantum_eigenspectrum(const Matrix<double>& H_m);

Result<Matrix<double>> eval_quantum_ground_state(const Matrix<double>& H_m);

Result<Matrix<double>> eval_quantum_time_evolve_psi(const Matrix<double>& H_m,
                                                      const Matrix<double>& psi_m, double t);

Result<Matrix<double>> eval_quantum_anticommutator(const Matrix<double>& A_m,
                                                   const Matrix<double>& B_m);

Result<Matrix<double>> eval_quantum_schmidt_decomposition(const Matrix<double>& psi_m, int dim_a,
                                                          int dim_b);

Result<double> eval_izaac_exponential_mechanism(const Matrix<double>& scores_m, double epsilon,
                                                double sensitivity);

Result<Matrix<double>> eval_mpc_split(uint64_t secret, int n, int k);

Result<double> eval_mpc_reconstruct(const Matrix<double>& shares_m);

Result<Matrix<double>> eval_simulate_gbm_path(double s0, double mu, double sigma, double dt,
                                              size_t steps);

Result<Matrix<double>> eval_run_backtest(const Matrix<double>& prices_m,
                                         const Matrix<double>& positions_m,
                                         double initial_capital);

Result<Matrix<double>> eval_run_backtest_equity(const Matrix<double>& prices_m,
                                                const Matrix<double>& positions_m,
                                                double initial_capital);

Result<double> eval_run_backtest_sharpe(const Matrix<double>& prices_m,
                                        const Matrix<double>& positions_m,
                                        double initial_capital);

Result<double> eval_run_backtest_max_drawdown(const Matrix<double>& prices_m,
                                              const Matrix<double>& positions_m,
                                              double initial_capital);

Result<double> eval_run_backtest_total_return(const Matrix<double>& prices_m,
                                              const Matrix<double>& positions_m,
                                              double initial_capital);

Result<Matrix<double>> eval_izaac_vrf_keygen();

Result<Matrix<double>> eval_izaac_fuzz_mutate(const Matrix<double>& input_m, size_t max_edits);

Result<izaac::VRFKey> vrf_key_from_matrix(const Matrix<double>& m, const char* fn);

Result<std::array<uint8_t, 32>> key32_from_matrix(const Matrix<double>& m, const char* fn);

Result<std::vector<uint8_t>> byte_vector_from_matrix(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> pack_vrf_proof(const izaac::VRFProof& proof);

Result<izaac::VRFProof> vrf_proof_from_matrix(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> pack_izaac_ciphertext(const izaac::crypto::CipherText& ct);

Result<izaac::crypto::CipherText> izaac_ciphertext_from_matrix(const Matrix<double>& m,
                                                               const char* fn);

Result<Matrix<double>> eval_izaac_vrf_prove(const Matrix<double>& key_m,
                                            const Matrix<double>& msg_m);

Result<double> eval_izaac_vrf_verify(const Matrix<double>& pub_m, const Matrix<double>& msg_m,
                                     const Matrix<double>& proof_m);

Result<Matrix<double>> eval_izaac_encrypt(const Matrix<double>& plaintext_m,
                                          const Matrix<double>& key_m);

Result<Matrix<double>> eval_izaac_decrypt(const Matrix<double>& ciphertext_m,
                                          const Matrix<double>& key_m);

Result<Matrix<double>> eval_izaac_randn_matrix(size_t rows, size_t cols);

Result<double> eval_quantum_schmidt_number(const Matrix<double>& psi_m, int dim_a, int dim_b);

Result<Matrix<double>> eval_quantum_ket_tensor_product(const Matrix<double>& psi1_m,
                                                       const Matrix<double>& psi2_m);

Result<Matrix<double>> eval_quantum_outer(const Matrix<double>& ket_m,
                                          const Matrix<double>& bra_m);

Result<Matrix<double>> eval_cfd_run_advection_3d(const Matrix<double>& grid_m,
                                                 const Matrix<double>& u_m, double vx, double vy,
                                                 double vz, double t_end, double dt);

Result<Matrix<double>> eval_quantum_dagger(const Matrix<double>& op_m);

Result<Matrix<double>> eval_quantum_matmul_dm(const Matrix<double>& A_m,
                                              const Matrix<double>& B_m);

Result<Matrix<double>> eval_izaac_rand_matrix(size_t rows, size_t cols);

Result<Matrix<double>> eval_quantum_schmidt_bases(const Matrix<double>& psi_m, int dim_a,
                                                  int dim_b);

Result<Matrix<double>> eval_quantum_bell_states();

Result<Matrix<double>> eval_graph_floyd_warshall(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_poly_integ(const Matrix<double>& coeffs_m, double c);

Result<double> eval_stats_spearman(const Matrix<double>& x_m, const Matrix<double>& y_m);

Result<double> eval_stats_median(const Matrix<double>& x_m);

Result<double> eval_graph_is_connected(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_fft_dct2(const Matrix<double>& x_m);

Result<Matrix<double>> eval_poly_add(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_poly_lagrange(const Matrix<double>& xs_m, const Matrix<double>& ys_m);

Result<Matrix<double>> eval_poly_interp_newton(const Matrix<double>& xs_m,
                                               const Matrix<double>& ys_m);

Result<Matrix<double>> eval_poly_roots(const Matrix<double>& coeffs_m);

Result<Matrix<double>> eval_poly_fit(const Matrix<double>& xs_m, const Matrix<double>& ys_m,
                                     int degree);

Result<Matrix<double>> eval_poly_interp_hermite(const Matrix<double>& xs_m,
                                                const Matrix<double>& ys_m,
                                                const Matrix<double>& dys_m);

Result<Matrix<double>> eval_poly_gcd(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_poly_squarefree(const Matrix<double>& coeffs_m);

Result<Matrix<double>> eval_poly_factor(const Matrix<double>& coeffs_m);

Result<Matrix<double>> eval_poly_rational_roots(const Matrix<double>& coeffs_m, double tol);

Result<Matrix<double>> eval_poly_factor_rational(const Matrix<double>& coeffs_m, double tol);

Result<Matrix<double>> eval_poly_partial_fractions(const Matrix<double>& num_m,
                                                   const Matrix<double>& den_m);

Result<double> eval_poly_root_count(const Matrix<double>& coeffs_m, double a, double b);

Result<double> eval_poly_cheb_eval(const Matrix<double>& cheb_m, double x);

Result<Matrix<double>> eval_poly_cheb_expand(const Matrix<double>& coeffs_m, int n,
                                              double a = -1.0, double b = 1.0);

Result<Matrix<double>> eval_poly_monic(const Matrix<double>& coeffs_m);

Result<Matrix<double>> eval_poly_reverse(const Matrix<double>& coeffs_m);

Result<Matrix<double>> eval_poly_shift(const Matrix<double>& coeffs_m, double a);

Result<Matrix<double>> eval_poly_scale(const Matrix<double>& coeffs_m, double a);

Result<Matrix<double>> eval_poly_pow(const Matrix<double>& coeffs_m, int n);

Result<Matrix<double>> eval_poly_lcm(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_poly_div_quot(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_poly_mod(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_poly_eval_at(const Matrix<double>& coeffs_m,
                                         const Matrix<double>& xs_m);

Result<Matrix<double>> eval_poly_sylvester(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<Matrix<double>> eval_quantum_tensor_product(const Matrix<double>& A_m,
                                                   const Matrix<double>& B_m);

Result<double> eval_stats_kendall(const Matrix<double>& x_m, const Matrix<double>& y_m);

Result<double> eval_stats_partial_correlation(const Matrix<double>& x_m,
                                              const Matrix<double>& y_m,
                                              const Matrix<double>& z_m);

Result<double> eval_stats_weighted_mean(const Matrix<double>& x_m, const Matrix<double>& w_m);

Result<double> eval_stats_weighted_variance(const Matrix<double>& x_m, const Matrix<double>& w_m);

Result<double> eval_stats_weighted_correlation(const Matrix<double>& x_m,
                                               const Matrix<double>& y_m,
                                               const Matrix<double>& w_m);

Result<double> eval_stats_bootstrap_mean(const Matrix<double>& x_m, int n_boot, unsigned seed);

Result<double> eval_stats_trimmed_mean(const Matrix<double>& x_m, double frac);

Result<double> eval_stats_vif(const Matrix<double>& X_m, double j_d, const char* fn);

Result<Matrix<double>> eval_stats_arfit(const Matrix<double>& x_m, int p);

Result<Matrix<double>> eval_stats_multiple_regression(const Matrix<double>& X_m,
                                                      const Matrix<double>& y_m);

Result<Matrix<double>> eval_graph_mst_kruskal(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_signal_correlate(const Matrix<double>& a_m,
                                             const Matrix<double>& b_m);

Result<Matrix<double>> eval_signal_xcorr(const Matrix<double>& a_m, const Matrix<double>& b_m,
                                         int max_lag);

Result<Matrix<double>> eval_signal_xcov(const Matrix<double>& a_m, const Matrix<double>& b_m,
                                        int max_lag);

Result<Matrix<double>> eval_signal_autocorr(const Matrix<double>& x_m, int max_lag);

Result<double> eval_stats_stddev(const Matrix<double>& x_m);

Result<Matrix<double>> eval_fft_idct2(const Matrix<double>& x_m);

Result<Matrix<double>> eval_poly_mul(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<Matrix<double>> eval_graph_mst_prim(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_min_arborescence(const Matrix<double>& adj_m, int root);

Result<double> eval_stats_skewness(const Matrix<double>& x_m);

Result<Matrix<double>> eval_poly_sub(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<double> eval_stats_kurtosis(const Matrix<double>& x_m);

Result<double> eval_stats_var(const Matrix<double>& x_m);

Result<Matrix<double>> eval_poly_compose(const Matrix<double>& p_m, const Matrix<double>& q_m);

Result<Matrix<double>> eval_graph_bfs(const Matrix<double>& adj_m, int source);

Result<double> eval_graph_is_tree(const Matrix<double>& adj_m);

Result<double> eval_graph_is_planar(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_dfs(const Matrix<double>& adj_m, int source);

Result<Matrix<double>> eval_graph_astar(const Matrix<double>& adj_m, int source, int target,
                                        const Matrix<double>& h_m);

Result<Matrix<double>> shortest_path_dist_parent_matrix(const std::vector<double>& dist,
                                                        const std::vector<int>& parent);

Result<Matrix<double>> eval_graph_dijkstra(const Matrix<double>& adj_m, int source);

Result<Matrix<double>> eval_graph_bellman_ford(const Matrix<double>& adj_m, int source);

Result<double> eval_stats_percentile(const Matrix<double>& x_m, double p);

Result<Matrix<double>> eval_signal_lowpass(const Matrix<double>& x_m, double cutoff, double fs);

Result<Matrix<double>> eval_signal_butterworth(const Matrix<double>& x_m, double cutoff, double fs);

Result<Matrix<double>> eval_signal_highpass(const Matrix<double>& x_m, double cutoff, double fs);

Result<Matrix<double>> eval_signal_bandpass(const Matrix<double>& x_m, double low, double high,
                                            double fs);

Result<Matrix<double>> eval_signal_cheby2(int order, double rs_db, double cutoff, double fs,
                                          FilterType type);

Result<Matrix<double>> eval_signal_cheby1(int order, double rp_db, double cutoff, double fs,
                                          FilterType type);

Result<Matrix<double>> eval_signal_firwin(int n_taps, double cutoff, FirWindow window);

Result<Matrix<double>> eval_signal_firwin_highpass(int n_taps, double cutoff, FirWindow window);

Result<Matrix<double>> eval_signal_filtfilt(const Matrix<double>& b_m, const Matrix<double>& a_m,
                                            const Matrix<double>& x_m);

Result<Matrix<double>> eval_signal_filter(const Matrix<double>& b_m, const Matrix<double>& a_m,
                                          const Matrix<double>& x_m);

Result<Matrix<double>> eval_signal_sosfilt(const Matrix<double>& sos_m,
                                           const Matrix<double>& x_m);

Result<Matrix<double>> eval_signal_periodogram(const Matrix<double>& x_m, double fs);

Result<Matrix<double>> eval_signal_welch_psd(const Matrix<double>& x_m, double fs,
                                             size_t nperseg);

Result<Matrix<double>> eval_signal_coherence(const Matrix<double>& x_m, const Matrix<double>& y_m,
                                             double fs, size_t nperseg);

Result<Matrix<double>> eval_signal_spectrogram(const Matrix<double>& x_m, double fs);

Result<Matrix<double>> eval_signal_envelope(const Matrix<double>& x_m);

Result<Matrix<double>> eval_signal_hilbert(const Matrix<double>& x_m);

Result<Matrix<double>> complex_vec_to_re_im_matrix(const std::vector<std::complex<double>>& z,
                                                   const char* fn);

Result<Matrix<double>> eval_signal_czt(const Matrix<double>& x_m, int m, double w_re, double w_im,
                                       double a_re, double a_im);

Result<Matrix<double>> eval_signal_czt_zoom(const Matrix<double>& x_m, double f_start,
                                            double f_stop, int m, double fs);

Result<Matrix<double>> eval_signal_instantaneous_freq(const Matrix<double>& x_m, double fs);

Result<Matrix<double>> eval_signal_instantaneous_phase(const Matrix<double>& x_m);

Result<Matrix<double>> eval_signal_unwrap(const Matrix<double>& x_m);

Result<Matrix<double>> eval_graph_topological_sort(const Matrix<double>& adj_m);

Result<double> eval_stats_mode(const Matrix<double>& x_m);

Result<double> eval_stats_geometric_mean(const Matrix<double>& x_m);

Result<double> eval_stats_ttest(const Matrix<double>& x_m, double mu);

Result<double> eval_stats_harmonic_mean(const Matrix<double>& x_m);

Result<double> eval_stats_rms(const Matrix<double>& x_m);

Result<double> eval_stats_mad(const Matrix<double>& x_m);

Result<double> eval_stats_iqr(const Matrix<double>& x_m);

Result<double> eval_stats_ztest(const Matrix<double>& x_m, double mu, double sigma);

Result<double> eval_stats_ks_norm(const Matrix<double>& x_m, double mu, double sigma);

Result<Matrix<double>> eval_stats_acf(const Matrix<double>& x_m, int max_lag);

Result<Matrix<double>> eval_stats_linear_regression(const Matrix<double>& x_m,
                                                    const Matrix<double>& y_m);

Result<Matrix<double>> eval_stats_pacf(const Matrix<double>& x_m, int max_lag);

Result<Matrix<double>> eval_stats_kde(const Matrix<double>& samples_m,
                                      const Matrix<double>& grid_m, double h,
                                      const char* kernel = "gaussian");

Result<Matrix<double>> eval_stats_bootstrap_ci(const Matrix<double>& x_m);

Result<double> eval_stats_two_sample_ttest(const Matrix<double>& a_m, const Matrix<double>& b_m);

Result<double> eval_stats_chi2_gof(const Matrix<double>& obs_m, const Matrix<double>& exp_m);

Result<Matrix<double>> eval_fft_fft2(const Matrix<double>& spectrum_m);

Result<Matrix<double>> eval_fft_dft(const Matrix<double>& x_m);

Result<Matrix<double>> eval_fft_goertzel(const Matrix<double>& x_m, double f, double fs);

Result<Matrix<double>> eval_sph_harm(int l, int m, double theta, double phi);

Result<Matrix<double>> eval_graph_greedy_colour(const Matrix<double>& adj_m);

Matrix<double> graph_to_adjacency_matrix(const graph::Graph& G);

Result<Matrix<double>> eval_graph_k_core_decomposition(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_k_core_subgraph(const Matrix<double>& adj_m, int k);

Result<double> eval_graph_chromatic_number(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_graph_euler_circuit(const Matrix<double>& adj_m);

Result<Matrix<double>> eval_fft_dst2(const Matrix<double>& x_m);

Result<Matrix<double>> eval_fft_ifft2(const Matrix<double>& spectrum_m);

Result<Matrix<double>> eval_fft_idst2(const Matrix<double>& x_m);

Result<std::vector<std::vector<double>>> matrix_to_groups(const Matrix<double>& m,
                                                           const char* fn);

Result<Matrix<double>> eval_kruskal_wallis(const Matrix<double>& groups_m);

Result<Matrix<double>> eval_stats_shapiro_wilk(const Matrix<double>& x_m);

Result<Matrix<double>> eval_stats_mann_whitney_u(const Matrix<double>& a_m,
                                                  const Matrix<double>& b_m);

Result<Matrix<double>> eval_stats_one_way_anova(const Matrix<double>& groups_m);

Result<Matrix<double>> eval_stats_levene(const Matrix<double>& groups_m);

Result<Matrix<double>> eval_stats_bartlett(const Matrix<double>& groups_m);

Result<Matrix<double>> eval_stats_fligner(const Matrix<double>& groups_m);

Result<Matrix<double>> eval_stats_wilcoxon_signed_rank(const Matrix<double>& x_m,
                                                       const Matrix<double>& y_m);

Result<Matrix<double>> eval_stats_friedman(const Matrix<double>& data_m);

Result<Matrix<double>> eval_stats_ks_2sample(const Matrix<double>& a_m,
                                             const Matrix<double>& b_m);

Result<Matrix<double>> eval_stats_jarque_bera(const Matrix<double>& x_m);

Result<Matrix<double>> eval_stats_ljung_box(const Matrix<double>& x_m, int max_lag);

Result<Matrix<double>> eval_combo_derangements(int n);

Result<Matrix<double>> eval_combo_gray_code(int n);

Result<Matrix<double>> eval_combo_dyck_paths(int n);

Result<Matrix<double>> eval_combo_necklaces(int n, int k);

Result<Matrix<double>> eval_combo_bracelets(int n, int k);

Result<Matrix<double>> eval_combo_lyndon_words(int n, int k);

Result<Matrix<double>> eval_combo_de_bruijn_sequence(int k, int n);

Result<Matrix<double>> eval_combo_motzkin_paths(int n);

Result<Matrix<double>> eval_combo_set_partitions(int n);

Result<Matrix<double>> eval_combo_restricted_partitions(int n, int k);

Result<double> eval_cplx_line_integral_one();

Result<Matrix<double>> eval_quantum_density_matrix(const Matrix<double>& psi_m);

Result<std::vector<topo::PersistencePair>> matrix_to_persistence_diagram(const Matrix<double>& m,
                                                                         const char* fn);

Result<double> eval_topo_bottleneck_distance(const Matrix<double>& dgm1_m,
                                             const Matrix<double>& dgm2_m, int dim);

Result<double> eval_topo_wasserstein_distance(const Matrix<double>& dgm1_m,
                                              const Matrix<double>& dgm2_m, int dim);

diffgeo::MetricFn unit_sphere_metric_fn();

Result<double> eval_diffgeo_christoffel_sphere(double k_d, double i_d, double j_d, double u,
                                               double v);

diffgeo::CurveFn circular_helix_curve(double a, double b);

Result<double> eval_diffgeo_helix_torsion(double t, double a, double b);

Result<double> eval_diffgeo_ricci_scalar_sphere(double u, double v);

Result<double> eval_diffgeo_einstein_scalar_sphere(double u, double v);

Result<double> eval_cplx_joukowski_inv(double re, double im);

Result<double> eval_cplx_residue_inv(double pole_re, double pole_im);

Result<double> eval_cplx_cauchy_integral(double z0re, double z0im);

Result<double> eval_cplx_contour_integral_oneoverz_im();

Result<Matrix<double>> eval_quantum_time_evolution_matrix(const Matrix<double>& H_m, double t);

Result<Matrix<double>> eval_run_length_encode_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_run_length_decode_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_topo_betti_curve(const Matrix<double>& dist_m,
                                             const Matrix<double>& thresholds_m, int max_dim);

Matrix<double> simplicial_complex_to_matrix(const topo::SimplicialComplex& sc);

Result<topo::SimplicialComplex> matrix_to_simplicial_complex(const Matrix<double>& m,
                                                             const char* fn);

Result<Matrix<double>> eval_topo_cech_complex(const Matrix<double>& dist_m, double epsilon,
                                               int max_dim);

Result<Matrix<double>> eval_topo_vietoris_rips(const Matrix<double>& dist_m, double r,
                                              int max_dim);

Result<Matrix<double>> eval_topo_simplicial_betti(const Matrix<double>& sc_m);

Result<double> eval_topo_simplicial_euler(const Matrix<double>& sc_m);

Result<Matrix<double>> eval_topo_simplicial_counts(const Matrix<double>& sc_m);

Result<double> eval_topo_simplicial_dimension(const Matrix<double>& sc_m);

Result<std::vector<std::vector<double>>> matrix_to_topo_points2d(const Matrix<double>& m,
                                                                 const char* fn);

Result<std::vector<int>> matrix_to_index_column(const Matrix<double>& m, const char* fn);

Result<Matrix<double>> eval_topo_alpha_complex(const Matrix<double>& P_m, double alpha,
                                               int max_dim);

Result<Matrix<double>> eval_topo_select_landmarks(const Matrix<double>& P_m, int n_landmarks,
                                                  int seed_index);

Result<Matrix<double>> eval_topo_witness_complex(const Matrix<double>& P_m,
                                                 const Matrix<double>& landmarks_m,
                                                 double max_epsilon, int max_dim);

Result<Matrix<double>> eval_topo_persistence_landscape(const Matrix<double>& dgm_m, int n_layers,
                                                       int n_samples, double t_min, double t_max);

Result<Matrix<double>> eval_quantum_bell_state(int index);

Result<Matrix<double>> eval_bzip2_compress_vec(const Matrix<double>& m);

Result<Matrix<double>> eval_bzip2_decompress_vec(const Matrix<double>& c_m);

Result<Matrix<double>> eval_control_place(const Matrix<double>& A_m,
                                          const Matrix<double>& B_m,
                                          const Matrix<double>& poles_m);

diffgeo::SurfaceFn unit_sphere_surface();

Result<double> eval_diffgeo_sphere_gauss_bonnet(int n);

Result<double> eval_diffgeo_sphere_gauss_bonnet_residual(int n);

Result<Matrix<double>> eval_diffgeo_surface_normal_sphere(double u, double v);

Result<double> eval_diffgeo_gaussian_sphere();

Result<double> eval_diffgeo_gaussian_curvature_sphere(double u, double v);

Result<double> eval_diffgeo_mean_curvature_sphere(double u, double v);

Result<double> eval_diffgeo_mean_sphere();

Result<double> eval_topo_euler_tetrahedron();

Result<double> eval_diffgeo_principal_curvature_sphere();

Result<double> eval_topo_euler_sphere_surface();

bool is_nullary_scalar_callee(const std::string& callee);

Result<double> eval_nullary_scalar_call(const std::string& fn);

bool is_nullary_matrix_callee(const std::string& callee);

Result<Matrix<double>> eval_nullary_matrix_call(const std::string& fn);

bool is_unary_scalar_matrix_callee(const std::string& callee);

Result<Matrix<double>> eval_unary_scalar_matrix_call(const std::string& fn, double arg);

struct ScalarDualMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
};

struct ScalarTripleMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
    std::string arg_c;
};

struct MatrixDualMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
};

struct MatrixTripleMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
    std::string arg_c;
};

struct MatrixQuadMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg_a;
    std::string arg_b;
    std::string arg_c;
    std::string arg_d;
};

struct ScalarMatrixMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_scalar;
    std::string arg_matrix;
};

struct MatrixScalarMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_matrix;
    std::string arg_scalar;
};

struct TwoScalarMatrixMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_scalar_a;
    std::string arg_scalar_b;
    std::string arg_matrix;
};

struct AxiomExprMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string expr_arg;
    std::string inputs_arg;
};

struct AxiomExprFitnessCallAssign {
    std::string target;
    std::string callee;
    std::string expr_arg;
    std::string inputs_arg;
    std::string targets_arg;
};

bool try_parse_axiom_expr_matrix_call_assignment(const std::string& line,
                                                 AxiomExprMatrixCallAssign& assign);

bool try_parse_axiom_expr_fitness_call_assignment(const std::string& line,
                                                  AxiomExprFitnessCallAssign& assign);

bool is_scalar_matrix_mixed_call_callee(const std::string& callee);

bool is_two_scalar_matrix_mixed_call_callee(const std::string& callee);

bool is_matrix_scalar_mixed_call_callee(const std::string& callee);

bool is_matrix_dual_matrix_call_callee(const std::string& callee);

bool is_matrix_triple_matrix_call_callee(const std::string& callee);

bool is_matrix_quad_matrix_call_callee(const std::string& callee);

bool is_scalar_dual_matrix_call_callee(const std::string& callee);

bool try_parse_scalar_dual_matrix_call_assignment(const std::string& line,
                                                  ScalarDualMatrixCallAssign& assign);

bool is_scalar_triple_matrix_call_callee(const std::string& callee);

bool try_parse_scalar_triple_matrix_call_assignment(const std::string& line,
                                                    ScalarTripleMatrixCallAssign& assign);

bool try_parse_matrix_dual_matrix_call_assignment(const std::string& line,
                                                  MatrixDualMatrixCallAssign& assign);

bool try_parse_matrix_triple_matrix_call_assignment(const std::string& line,
                                                    MatrixTripleMatrixCallAssign& assign);

bool try_parse_matrix_quad_matrix_call_assignment(const std::string& line,
                                                  MatrixQuadMatrixCallAssign& assign);

bool try_parse_scalar_matrix_mixed_call_assignment(const std::string& line,
                                                   ScalarMatrixMixedCallAssign& assign);

bool try_parse_matrix_scalar_mixed_call_assignment(const std::string& line,
                                                   MatrixScalarMixedCallAssign& assign);

bool try_parse_two_scalar_matrix_mixed_call_assignment(const std::string& line,
                                                       TwoScalarMatrixMixedCallAssign& assign);

struct MatrixTwoScalarMixedCallAssign {
    std::string target;
    std::string callee;
    std::string arg_matrix;
    std::string arg_scalar_a;
    std::string arg_scalar_b;
};

bool is_matrix_two_scalar_mixed_call_callee(const std::string& callee);

bool try_parse_matrix_two_scalar_mixed_call_assignment(
    const std::string& line, MatrixTwoScalarMixedCallAssign& assign);

Result<double> eval_ml_metric(const std::string& callee, const ml::Vec& y_pred, const ml::Vec& y_true);

bool try_parse_bigint_unary_call(const std::string& line, std::string& name, std::string& fn,
                                 int& n);

bool try_parse_bigint_gcd_assignment(const std::string& line, std::string& name, std::string& a,
                                     std::string& b);

Result<double> eval_bigint_unary(const std::string& fn, int n);

Result<double> eval_bigint_gcd_strings(const std::string& a, const std::string& b);

Result<SymExpr> parse_sym_quoted_expr(const std::string& quoted_arg, const char* fn);

Result<std::string> eval_sym_diff_strings(const std::string& expr_arg, const std::string& var_arg);

Result<std::string> eval_sym_simplify_string(const std::string& expr_arg);

Result<std::string> eval_sym_integrate_strings(const std::string& expr_arg, const std::string& var_arg);

Result<std::string> eval_sym_eval_strings(const std::string& expr_arg, const std::string& binding_arg);

Result<std::string> eval_sym_expand_string(const std::string& expr_arg);

using SymTransform3 = SymExpr (*)(const SymExpr&, const std::string&, const std::string&);

Result<std::string> eval_sym_transform_strings(const std::string& expr_arg, const std::string& var_a_arg,
                                               const std::string& var_b_arg, const char* fn,
                                               SymTransform3 transform);

Result<std::string> eval_sym_collect_strings(const std::string& expr_arg, const std::string& var_arg);

Result<std::string> eval_sym_substitute_strings(const std::string& expr_arg, const std::string& var_arg,
                                                const std::string& repl_arg);

Result<std::string> eval_sym_limit_strings(const std::string& expr_arg, const std::string& var_arg,
                                           const std::string& point_arg);

Result<std::string> eval_sym_series_strings(const std::string& expr_arg, const std::string& var_arg,
                                            const std::string& point_arg, const std::string& order_arg);

Result<std::vector<std::string>> parse_sym_semicolon_identifiers(const std::string& arg,
                                                                 const char* fn);

Result<std::string> eval_sym_solve_linear_strings(const std::string& eqs_arg,
                                                  const std::string& vars_arg);

Result<std::string> format_ode_trajectory(const OdeResult& result);

OdeResult ode_trapezoidal_wrapped(OdeFunc f, double t0, double y0, double t_end, size_t steps);

OdeResult ode_rosenbrock23_wrapped(OdeFunc f, double t0, double y0, double t_end, size_t steps);

Result<Matrix<double>> eval_ode_fixed_step_matrix(const std::string& fn,
                                                  const std::string& formula_arg,
                                                  const std::string& t0_arg,
                                                  const std::string& y0_arg,
                                                  const std::string& t_end_arg,
                                                  const std::string& steps_arg,
                                                  OdeResult (*solver)(OdeFunc, double, double,
                                                                      double, size_t));

Result<std::string> eval_ode_fixed_step_call(const std::string& fn, const std::string& formula_arg,
                                             const std::string& t0_arg,
                                             const std::string& y0_arg,
                                             const std::string& t_end_arg,
                                             const std::string& steps_arg,
                                             OdeResult (*solver)(OdeFunc, double, double, double,
                                                                 size_t));

Result<std::string> eval_ode_rk45_call(const std::string& formula_arg, const std::string& t0_arg,
                                       const std::string& y0_arg, const std::string& t_end_arg,
                                       const std::string& rtol_arg, const std::string& atol_arg);

Result<std::string> eval_ode_adaptive_call(
    const char* fn, const std::string& formula_arg, const std::string& t0_arg,
    const std::string& y0_arg, const std::string& t_end_arg, const std::string& rtol_arg,
    const std::string& atol_arg,
    OdeResult (*solver)(OdeFunc, double, double, double, double, double));

Result<std::string> eval_ode_trapezoidal_call(const std::string& formula_arg,
                                              const std::string& t0_arg,
                                              const std::string& y0_arg,
                                              const std::string& t_end_arg,
                                              const std::string& steps_arg);

Result<std::string> eval_ode_rosenbrock23_call(const std::string& formula_arg,
                                               const std::string& t0_arg,
                                               const std::string& y0_arg,
                                               const std::string& t_end_arg,
                                               const std::string& steps_arg);

Result<std::string> eval_ode_exponential_euler_call(const std::string& formula_arg,
                                                  const std::string& lambda_arg,
                                                  const std::string& t0_arg,
                                                  const std::string& y0_arg,
                                                  const std::string& t_end_arg,
                                                  const std::string& steps_arg);

Result<std::vector<double>> parse_comma_separated_numbers(const std::string& row_text,
                                                          const char* fn);

Result<Matrix<double>> parse_bracket_matrix_literal(const std::string& text, const char* fn);

Result<std::vector<double>> parse_bracket_vector_literal(const std::string& text, const char* fn);

std::map<std::string, double> build_optim_env(const std::vector<double>& x);

Result<std::string> format_optim_result(const OptimResult& result);

Result<std::string> format_scalar_optim_result(double x_opt, double f_val);

struct NdOptimInputs {
    std::shared_ptr<SymExpr> expr;
    std::vector<double> x0;
    FuncND f;
};

Result<NdOptimInputs> parse_nd_optim_inputs(const std::string& formula_arg,
                                            const std::string& x0_arg, const char* fn);

GradND make_finite_diff_grad(FuncND f, int n);

Result<double> parse_optional_positive_number(const std::string& text, const char* fn,
                                              const char* label, double default_value);

Result<int> parse_optional_positive_int(const std::string& text, const char* fn, const char* label,
                                        int default_value);

Result<std::string> eval_bfgs_call(const std::string& formula_arg, const std::string& x0_arg,
                                   const std::string& tol_arg, const std::string& max_iter_arg);

Result<std::string> eval_nelder_mead_call(const std::string& formula_arg, const std::string& x0_arg,
                                          const std::string& tol_arg,
                                          const std::string& max_iter_arg);

Result<std::string> eval_lbfgs_call(const std::string& formula_arg, const std::string& x0_arg,
                                    const std::string& m_arg, const std::string& tol_arg,
                                    const std::string& max_iter_arg);

Result<std::string> eval_adam_call(const std::string& formula_arg, const std::string& x0_arg,
                                   const std::string& alpha_arg, const std::string& max_iter_arg);

Result<std::string> eval_conjugate_gradient_call(const std::string& formula_arg,
                                                 const std::string& x0_arg,
                                                 const std::string& tol_arg,
                                                 const std::string& max_iter_arg);

Result<std::string> eval_rmsprop_call(const std::string& formula_arg, const std::string& x0_arg,
                                      const std::string& alpha_arg,
                                      const std::string& max_iter_arg);

Result<std::string> eval_adadelta_call(const std::string& formula_arg, const std::string& x0_arg,
                                       const std::string& lr_arg, const std::string& max_iter_arg);

Result<std::string> eval_golden_section_call(const std::string& formula_arg,
                                             const std::string& a_arg, const std::string& b_arg,
                                             const std::string& tol_arg);

Result<std::string> eval_levenberg_marquardt_call(const std::string& formulas_arg,
                                                  const std::string& x0_arg,
                                                  const std::string& max_iter_arg,
                                                  const std::string& tol_arg);

Result<std::string> eval_cmaes_call(const std::string& formula_arg, const std::string& x0_arg,
                                    const std::string& sigma_arg,
                                    const std::string& max_iter_arg,
                                    const std::string& seed_arg);

axiom::Axiom make_axiom_repl_engine();

Result<axiom::Algorithm> make_axiom_algorithm(const std::string& expr_arg, const char* fn);

Result<Matrix<double>> eval_axiom_evaluate_call(const std::string& expr_arg,
                                                const Matrix<double>& inputs, const char* fn);

Result<double> eval_axiom_mse_fitness_call(const std::string& expr_arg,
                                           const Matrix<double>& inputs,
                                           const std::vector<double>& targets, const char* fn);

Result<double> eval_axiom_rmse_fitness_call(const std::string& expr_arg,
                                            const Matrix<double>& inputs,
                                            const std::vector<double>& targets, const char* fn);

Result<double> eval_axiom_gria_fitness_call(const std::string& expr_arg,
                                            const Matrix<double>& data, const char* fn);

Result<double> eval_axiom_evolve_call(const Matrix<double>& data, size_t population_size,
                                      size_t max_generations);

Result<std::vector<double>> resolve_axiom_targets(const std::string& text, const char* fn,
                                                    const std::function<Result<Matrix<double>>(const std::string&)>&
                                                        resolve_matrix_arg);

Result<std::vector<std::pair<double, double>>> parse_bounds_pairs_literal(const std::string& text,
                                                                          const char* fn);

Result<unsigned> parse_optional_seed(const std::string& seed_arg, const char* fn);

Func1D make_scalar_formula_func(SymExpr expr);

using BracketRootSolver = double (*)(Func1D, double, double, double, int);

Result<std::string> eval_bracket_root_call(const char* fn, BracketRootSolver solver,
                                           const std::string& formula_arg,
                                           const std::string& a_arg, const std::string& b_arg,
                                           const std::string& tol_arg,
                                           const std::string& max_iter_arg);

Result<std::string> eval_secant_call(const std::string& formula_arg, const std::string& x0_arg,
                                     const std::string& x1_arg, const std::string& tol_arg,
                                     const std::string& max_iter_arg);

Result<std::string> eval_halley_call(const std::string& f_arg, const std::string& df_arg,
                                     const std::string& d2f_arg, const std::string& x0_arg,
                                     const std::string& tol_arg, const std::string& max_iter_arg);

Result<std::string> eval_fixed_point_call(const std::string& formula_arg, const std::string& x0_arg,
                                          const std::string& tol_arg,
                                          const std::string& max_iter_arg);

Result<std::string> eval_simulated_annealing_call(const std::string& formula_arg,
                                                  const std::string& x0_arg,
                                                  const std::string& t0_arg,
                                                  const std::string& cooling_arg,
                                                  const std::string& max_iter_arg,
                                                  const std::string& seed_arg);

Result<std::string> eval_differential_evolution_call(
    const std::string& formula_arg, const std::string& bounds_arg, const std::string& pop_arg,
    const std::string& f_arg, const std::string& cr_arg, const std::string& max_iter_arg,
    const std::string& seed_arg);

Result<std::string> eval_particle_swarm_call(const std::string& formula_arg,
                                             const std::string& bounds_arg,
                                             const std::string& n_particles_arg,
                                             const std::string& max_iter_arg,
                                             const std::string& seed_arg);

Result<std::vector<SymExpr>> parse_sym_semicolon_formulas(const std::string& formula_arg,
                                                          const char* fn);

std::map<std::string, double> build_vec_ode_env(double t, const std::vector<double>& y);

std::map<std::string, double> build_vec_accel_env(double t, const std::vector<double>& q);

Result<std::string> format_ode_trajectory_vec(const OdeResultVec& result);

Result<std::string> format_ode_verlet_trajectory(const OdeVerletResult& result);

Result<std::string> format_ode_verlet_trajectory_vec(const OdeVerletResultVec& result);

Result<std::string> eval_ode_verlet_call(const std::string& formula_arg, const std::string& t0_arg,
                                         const std::string& q0_arg, const std::string& v0_arg,
                                         const std::string& t_end_arg, const std::string& steps_arg);

Result<std::string> eval_ode_vec_fixed_step_call(
    const std::string& fn, const std::string& formula_arg, const std::string& t0_arg,
    const std::string& y0_arg, const std::string& t_end_arg, const std::string& steps_arg,
    OdeResultVec (*solver)(OdeFuncVec, double, const std::vector<double>&, double, size_t));

Result<std::string> eval_ode_rk45_vec_call(const std::string& formula_arg, const std::string& t0_arg,
                                           const std::string& y0_arg, const std::string& t_end_arg,
                                           const std::string& rtol_arg, const std::string& atol_arg);

Result<std::string> eval_ode_rosenbrock23_vec_call(const std::string& formula_arg,
                                                   const std::string& t0_arg,
                                                   const std::string& y0_arg,
                                                   const std::string& t_end_arg,
                                                   const std::string& steps_arg);

Result<std::string> eval_ode_verlet_vec_call(const std::string& formula_arg,
                                             const std::string& t0_arg, const std::string& q0_arg,
                                             const std::string& v0_arg, const std::string& t_end_arg,
                                             const std::string& steps_arg);

std::map<std::string, double> build_dae_env(double t, const std::vector<double>& y,
                                            const std::vector<double>& z);

Result<std::string> format_dae_trajectory(const DaeResult& result);

Result<std::string> format_ode_bvp_trajectory(const OdeBvpResult& result);

Result<std::string> format_ode_event_trajectory(const OdeEventResult& result);

Result<std::string> eval_ode_dae_index1_call(const std::string& diff_formula_arg,
                                             const std::string& alg_formula_arg,
                                             const std::string& t0_arg,
                                             const std::string& y0_arg,
                                             const std::string& z0_arg,
                                             const std::string& t_end_arg,
                                             const std::string& steps_arg);

Result<std::string> eval_ode_bvp_shooting_call(const std::string& formula_arg,
                                               const std::string& t0_arg,
                                               const std::string& y_a_arg,
                                               const std::string& t_end_arg,
                                               const std::string& y_b_arg,
                                               const std::string& steps_arg);

Result<std::string> eval_ode_dde_fixed_step_call(const std::string& formula_arg,
                                                 const std::string& history_formula_arg,
                                                 const std::string& t0_arg,
                                                 const std::string& t_end_arg,
                                                 const std::string& tau_arg,
                                                 const std::string& steps_arg);

Result<std::string> eval_ode_event_detect_call(const std::string& formula_arg,
                                               const std::string& event_formula_arg,
                                               const std::string& t0_arg,
                                               const std::string& y0_arg,
                                               const std::string& t_end_arg,
                                               const std::string& steps_arg);

std::optional<Result<std::string>> try_eval_sym_command(const std::string& cmd);

std::optional<Result<std::string>> try_eval_crypto_command(const std::string& cmd);

std::optional<std::vector<std::string>> split_call_args(const std::string& cmd);

bool is_identifier(const std::string& text);

bool is_identifier_view(std::string_view text);

bool parse_scalar_operand_view(std::string_view text, ScalarOperand& out);

std::optional<std::pair<std::string_view, std::string_view>> parse_scalar_unary_call_view(
    std::string_view expr);

size_t split_scalar_call_args_view(std::string_view args_text, std::string_view* out, size_t cap);

bool try_parse_bigint_assignment(const std::string& line, std::string& name, std::string& decimal);

bool parse_scalar_operand(const std::string& text, ScalarOperand& out);

bool is_binary_minus(const std::string& expr, size_t index);

std::optional<std::pair<size_t, char>> find_top_level_op(const std::string& expr, const char* ops);

std::optional<std::pair<size_t, char>> find_scalar_binop(const std::string& rhs);

std::string strip_outer_parens(std::string expr);

bool contains_scalar_operator(const std::string& rhs);

std::optional<std::pair<std::string, std::string>> parse_scalar_unary_call(const std::string& expr);

std::optional<std::string> parse_nullary_scalar_call(const std::string& expr);

std::optional<std::string> parse_nullary_matrix_call(const std::string& expr);

std::optional<std::pair<std::string, std::string>> parse_unary_scalar_matrix_call(
    const std::string& expr);

std::vector<std::string> split_scalar_call_args(const std::string& args_text);

std::optional<std::pair<std::string, std::vector<std::string>>> parse_scalar_call(
    const std::string& expr);

bool is_scalar_expression_rhs(const std::string& rhs);

Result<double> resolve_scalar_operand(const SessionState& state, const ScalarOperand& operand);

Result<double> eval_scalar_call_cached(std::string_view fn_name, std::span<const double> args);

Result<double> eval_scalar_expr_impl(const SessionState& state, std::string_view expr_text);

Result<double> eval_scalar_expr(const SessionState& state, const std::string& expr_text);

void append_unique_var(std::vector<std::string>& vars, const std::string& name);

void collect_scalar_expr_variables(const std::string& expr_text, std::vector<std::string>& vars);

const char* format_compute_class(gria::ComputeClass cls);

Result<void> require_session_rng(const char* fn);

std::string format_nig_params(const cypha::NIGParams& params);

const char* session_object_kind_name(const SessionObject& object);

template <typename T>
const char* session_object_type_label() {
    if constexpr (std::is_same_v<T, izaac::bloom::BloomFilter>) {
        return "BloomFilter";
    } else if constexpr (std::is_same_v<T, izaac::ratelimit::TokenBucket>) {
        return "TokenBucket";
    } else if constexpr (std::is_same_v<T, cellai::CellMemory>) {
        return "CellMemory";
    } else if constexpr (std::is_same_v<T, cypha::DifModel>) {
        return "DifModel";
    } else if constexpr (std::is_same_v<T, izaac::consensus::Cluster>) {
        return "Cluster";
    } else if constexpr (std::is_same_v<T, tensorops::CPDecomposition>) {
        return "CPDecomposition";
    } else if constexpr (std::is_same_v<T, tensorops::TuckerDecomposition>) {
        return "TuckerDecomposition";
    } else if constexpr (std::is_same_v<T, tensorops::NMFDecomposition>) {
        return "NMFDecomposition";
    } else if constexpr (std::is_same_v<T, tensorops::TTDecomposition>) {
        return "TTDecomposition";
    } else {
        return "session object";
    }
}

const char* format_node_role(izaac::consensus::NodeRole role);

template <typename T>
Result<void> require_session_object_type(
    std::map<std::string, SessionObject>& registry,
    const std::string& handle,
    const char* fn,
    T*& out) {
    const auto it = registry.find(handle);
    if (it == registry.end()) {
        return std::unexpected(DomainError{fn, "session object not found: " + handle});
    }
    if (!std::holds_alternative<T>(it->second)) {
        return std::unexpected(DomainError{
            fn, std::string("session object '") + handle + "' is not a " +
                    session_object_type_label<T>()});
    }
    out = &std::get<T>(it->second);
    return {};
}

Result<std::vector<int>> parse_bracket_int_vector_literal(const std::string& text, const char* fn);

Result<void> parse_session_handle(const std::string& text, const char* fn, std::string& handle);

std::span<const uint8_t> string_item_bytes(const std::string& item);

} // namespace ms::interp::detail
