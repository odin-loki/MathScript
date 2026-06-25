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
