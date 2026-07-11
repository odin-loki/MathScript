#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/error_types.hpp"
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

namespace crypto {

/// Ciphertext produced by encrypt(): nonce (16 bytes) + XOR-encrypted payload, plus a tag.
/// This is a CSPRNG keystream XOR cipher with a keyed checksum for tamper detection.
/// Suitable for internal/demo use in this codebase — NOT a substitute for a vetted crypto library.
struct CipherText {
    std::vector<uint8_t> data;
    std::array<uint8_t, 32> tag{};
};

/// Encrypts plaintext with a CSPRNG keystream (key + fresh random nonce) and attaches a tag.
CipherText encrypt(std::span<const uint8_t> plaintext, std::array<uint8_t, 32> key);

/// Verifies the tag and decrypts; returns an error if authentication fails.
Result<std::vector<uint8_t>> decrypt(const CipherText& ct, std::array<uint8_t, 32> key);

} // namespace crypto

namespace mpc {

/// Prime field modulus for Shamir secret sharing: 2^61 - 1 (61-bit Mersenne prime).
/// Secrets and share values must be strictly less than this value.
constexpr uint64_t PRIME = 2305843009213693951ULL;

struct Share {
    int x = 0;
    uint64_t y = 0;
};

/// Splits secret into n shares with k-of-n threshold via Shamir's scheme over PRIME.
std::vector<Share> split_secret(uint64_t secret, int n, int k, CSPRNG& rng);

/// Reconstructs the secret from shares via Lagrange interpolation at x = 0.
/// Requires at least two shares; fewer than the original threshold yields a wrong value.
Result<uint64_t> reconstruct_secret(const std::vector<Share>& shares);

} // namespace mpc

} // namespace ms::izaac
