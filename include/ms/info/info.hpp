#pragma once
#include <span>
#include <vector>

namespace ms {
namespace info {

// --- Shannon entropy (bits or nats depending on base) ---
double entropy(std::span<const double> p, double base = 2.0);
double joint_entropy(std::span<const double> pxy, int rows, int cols,
                     double base = 2.0);
double conditional_entropy(std::span<const double> pxy, int rows, int cols,
                           double base = 2.0); // H(Y|X)
double mutual_info(std::span<const double> pxy, int rows, int cols,
                   double base = 2.0);          // I(X;Y)
double cross_entropy(std::span<const double> p, std::span<const double> q,
                     double base = 2.0);

// --- Divergences ---
double kl_divergence(std::span<const double> p, std::span<const double> q,
                     double base = 2.0);        // D_KL(P||Q)
double js_divergence(std::span<const double> p, std::span<const double> q,
                     double base = 2.0);        // Jensen-Shannon (symmetric)
double renyi_entropy(std::span<const double> p, double alpha,
                     double base = 2.0);        // H_alpha
double tsallis_entropy(std::span<const double> p, double q);
double hellinger_dist(std::span<const double> p, std::span<const double> q);
double tv_distance(std::span<const double> p, std::span<const double> q);

// --- Channel capacity (binary symmetric channel) ---
double channel_capacity_bsc(double p_error);    // 1 - H_b(p)
double channel_capacity_bec(double epsilon);    // erasure: 1 - epsilon
// Shannon-Hartley: C = B log2(1 + SNR)
double shannon_hartley(double bandwidth_hz, double snr_linear);

// --- Channel capacity (general discrete memoryless channel) ---
// Result of the Blahut-Arimoto capacity computation: the capacity itself
// (in bits) together with the input distribution p(x) that achieves it.
struct ChannelCapacityResult {
    double capacity = 0.0;                 // bits
    std::vector<double> input_distribution; // p(x), size = number of rows of W
};

/// Computes the capacity of a discrete memoryless channel via the
/// Blahut-Arimoto algorithm.
///
/// The channel is specified by a transition matrix W where
/// W[x][y] = P(Y=y | X=x), x = 0..n_inputs-1, y = 0..n_outputs-1. Each row
/// of W must be a valid probability distribution (non-negative, sums to 1).
///
/// The algorithm alternates:
///   1. q(x|y) = p(x) W(y|x) / sum_x' p(x') W(y|x')      (Bayes' rule)
///   2. p(x)   <- exp( sum_y W(y|x) log q(x|y) ), then normalise
/// and tracks the mutual information I(X;Y) under the current p(x) as the
/// capacity estimate, iterating until it changes by less than @p eps or
/// @p max_iter iterations have elapsed.
///
/// @param W        Channel transition matrix, W[x][y] = P(Y=y|X=x).
/// @param eps       Convergence tolerance on the capacity estimate (bits).
/// @param max_iter  Maximum number of Blahut-Arimoto iterations.
/// @return Capacity in bits (base-2), consistent with entropy()/mutual_info().
/// @note Defensive: returns 0.0 for an empty matrix, a matrix with
///       mismatched/empty row lengths, or rows that do not sum to ~1.
/// @accuracy Converges to the true capacity within @p eps for well-posed
///           channels; typically a few hundred iterations suffice for eps=1e-9.
double blahut_arimoto(const std::vector<std::vector<double>>& W,
                       double eps = 1e-9, int max_iter = 1000);

/// Same as blahut_arimoto(), but also returns the optimal input
/// distribution p(x) that achieves the reported capacity.
/// @param W        Channel transition matrix, W[x][y] = P(Y=y|X=x).
/// @param eps       Convergence tolerance on the capacity estimate (bits).
/// @param max_iter  Maximum number of Blahut-Arimoto iterations.
/// @return A ChannelCapacityResult{capacity, input_distribution}; on
///         degenerate/invalid input, capacity = 0.0 and input_distribution
///         is empty.
ChannelCapacityResult channel_capacity(const std::vector<std::vector<double>>& W,
                                        double eps = 1e-9, int max_iter = 1000);

// --- Rate distortion ---
double rate_distortion_gaussian(double variance, double distortion);

// --- Source coding bounds ---
double source_coding_rate(std::span<const double> p);  // min avg code length

// --- Compression related ---
// Lempel-Ziv complexity (normalised) of a binary sequence
double lz_complexity(std::span<const int> sequence);

// --- Redundancy & efficiency ---
double redundancy(std::span<const double> p);   // H_max - H(X)
double efficiency(std::span<const double> p);   // H(X) / H_max

// --- Differential entropy (Gaussian) ---
double differential_entropy_gaussian(double sigma);    // 0.5 ln(2πe σ²)
double differential_entropy_uniform(double a, double b);

// --- Entropy rate from empirical counts (sample entropy) ---
double sample_entropy(std::span<const double> x, int m, double r);

} // namespace info
} // namespace ms
