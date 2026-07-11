#pragma once

#include "ms/core/matrix.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ms::izaac {

struct VRFKey {
    std::array<uint8_t, 32> private_key{};
    std::array<uint8_t, 32> public_key{};
};

struct VRFProof {
    std::array<uint8_t, 80> proof{};
    std::array<uint8_t, 64> output{};
};

VRFKey keygen();
VRFProof prove(const VRFKey& key, std::span<const uint8_t> msg);
bool verify(
    const std::array<uint8_t, 32>& pub,
    std::span<const uint8_t> msg,
    const VRFProof& proof);

class CSPRNG {
public:
    explicit CSPRNG(const VRFProof& seed);
    explicit CSPRNG(std::array<uint8_t, 32> seed);

    void fill(std::span<uint8_t> buf);
    uint64_t next_u64();
    double next_f64();
    double next_normal();

private:
    std::array<uint8_t, 32> state_{};
    uint64_t stream_ = 0;
    bool has_spare_normal_ = false;
    double spare_normal_ = 0.0;
};

extern std::optional<CSPRNG> g_session_rng;

void seed_session(const VRFKey& key, std::span<const uint8_t> nonce);
void seed_session(uint64_t deterministic_seed);
void clear_session();
bool session_active();

Matrix<double> randn_matrix(size_t rows, size_t cols);
Matrix<double> rand_matrix(size_t rows, size_t cols);

namespace mc {

double estimate_pi(size_t samples, CSPRNG& rng);

} // namespace mc

namespace bloom {

class BloomFilter {
public:
    BloomFilter(size_t expected_items, double false_positive_rate, CSPRNG& rng);
    void insert(std::span<const uint8_t> item);
    bool might_contain(std::span<const uint8_t> item) const;
    size_t bit_count() const;
    size_t hash_count() const;

private:
    std::vector<uint8_t> bits_;
    size_t bit_count_ = 0;
    size_t hash_count_ = 0;
    uint64_t hash_seed1_ = 0;
    uint64_t hash_seed2_ = 0;

    size_t hash_index(std::span<const uint8_t> item, size_t i) const;
    void set_bit(size_t index);
    bool get_bit(size_t index) const;
};

} // namespace bloom

namespace ratelimit {

class TokenBucket {
public:
    TokenBucket(double capacity, double refill_rate_per_sec);
    bool try_consume(double tokens, double now_seconds);
    double available_tokens(double now_seconds) const;

private:
    double capacity_;
    double refill_rate_;
    double tokens_;
    double last_update_ = 0.0;

    void refill_to(double now_seconds);
};

} // namespace ratelimit

namespace diffpriv {

double laplace_mechanism(double true_value, double epsilon, double sensitivity, CSPRNG& rng);
double gaussian_mechanism(
    double true_value,
    double epsilon,
    double delta,
    double sensitivity,
    CSPRNG& rng);

} // namespace diffpriv

namespace backtest {

struct BacktestResult {
    std::vector<double> equity_curve;
    double total_return = 0.0;
    double max_drawdown = 0.0;
    double sharpe_ratio = 0.0;
};

std::vector<double> simulate_gbm_path(
    double s0,
    double mu,
    double sigma,
    double dt,
    size_t steps,
    CSPRNG& rng);
BacktestResult run_backtest(
    const std::vector<double>& prices,
    const std::vector<int>& positions,
    double initial_capital);

} // namespace backtest

} // namespace ms::izaac
