#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/core/rng.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace ms::izaac {

namespace {

CSPRNG* active_rng() {
    return g_session_rng ? &*g_session_rng : nullptr;
}

double uniform_from_rng() {
    auto* rng = active_rng();
    return rng ? rng->next_f64() : 0.0;
}

double normal_from_rng() {
    auto* rng = active_rng();
    return rng ? rng->next_normal() : 0.0;
}

std::array<uint8_t, 32> mix_seed(std::array<uint8_t, 32> seed) {
    for (size_t i = 0; i < seed.size(); ++i) {
        seed[i] ^= static_cast<uint8_t>(seed[(i + 7) % seed.size()] + i * 31);
    }
    return seed;
}

uint64_t fnv1a_64(std::span<const uint8_t> data, uint64_t seed) {
    constexpr uint64_t offset = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset ^ seed;
    for (uint8_t byte : data) {
        hash ^= static_cast<uint64_t>(byte);
        hash *= prime;
    }
    return hash;
}

} // namespace

std::optional<CSPRNG> g_session_rng;

VRFKey keygen() {
    VRFKey key;
    std::mt19937_64 gen{std::random_device{}()};
    for (auto& b : key.private_key) {
        b = static_cast<uint8_t>(gen() & 0xFF);
    }
    for (size_t i = 0; i < key.public_key.size(); ++i) {
        key.public_key[i] = static_cast<uint8_t>(key.private_key[i] ^ key.private_key[(i + 13) % 32]);
    }
    return key;
}

VRFProof prove(const VRFKey& key, std::span<const uint8_t> msg) {
    VRFProof proof;
    for (size_t i = 0; i < proof.output.size(); ++i) {
        proof.output[i] = static_cast<uint8_t>(key.private_key[i % 32] ^ (msg.empty() ? 0 : msg[i % msg.size()]));
    }
    for (size_t i = 0; i < proof.proof.size(); ++i) {
        proof.proof[i] = static_cast<uint8_t>(proof.output[i % 64] + key.public_key[i % 32]);
    }
    return proof;
}

bool verify(
    const std::array<uint8_t, 32>& pub,
    std::span<const uint8_t> msg,
    const VRFProof& proof) {
    const auto recomputed = prove(VRFKey{{}, pub}, msg);
    return std::memcmp(recomputed.output.data(), proof.output.data(), proof.output.size()) == 0;
}

CSPRNG::CSPRNG(const VRFProof& seed) {
    std::array<uint8_t, 32> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = seed.output[i];
    }
    state_ = mix_seed(bytes);
}

CSPRNG::CSPRNG(std::array<uint8_t, 32> seed) : state_(mix_seed(seed)) {}

void CSPRNG::fill(std::span<uint8_t> buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>(next_u64() & 0xFF);
    }
}

uint64_t CSPRNG::next_u64() {
    uint64_t x = 0;
    for (size_t i = 0; i < 8; ++i) {
        state_[i % 32] = static_cast<uint8_t>(state_[i % 32] ^ state_[(i + 11) % 32] ^ static_cast<uint8_t>(stream_));
        x = (x << 8) | state_[i % 32];
    }
    ++stream_;
    return x;
}

double CSPRNG::next_f64() {
    return static_cast<double>(next_u64() & 0xFFFFFFFFFFFFFULL) / static_cast<double>(0xFFFFFFFFFFFFFULL);
}

double CSPRNG::next_normal() {
    if (has_spare_normal_) {
        has_spare_normal_ = false;
        return spare_normal_;
    }
    double u1 = 0.0;
    double u2 = 0.0;
    while (u1 <= 1e-12) {
        u1 = next_f64();
        u2 = next_f64();
    }
    const double mag = std::sqrt(-2.0 * std::log(u1));
    spare_normal_ = mag * std::sin(2.0 * 3.14159265358979323846 * u2);
    has_spare_normal_ = true;
    return mag * std::cos(2.0 * 3.14159265358979323846 * u2);
}

void seed_session(const VRFKey& key, std::span<const uint8_t> nonce) {
    const auto proof = prove(key, nonce);
    g_session_rng = CSPRNG(proof);
    set_session_rng(uniform_from_rng, normal_from_rng);
}

void seed_session(uint64_t deterministic_seed) {
    std::array<uint8_t, 32> seed{};
    std::memcpy(seed.data(), &deterministic_seed, sizeof(deterministic_seed));
    for (size_t i = sizeof(deterministic_seed); i < seed.size(); ++i) {
        seed[i] = static_cast<uint8_t>((deterministic_seed >> (i % 8)) & 0xFF);
    }
    g_session_rng = CSPRNG(seed);
    set_session_rng(uniform_from_rng, normal_from_rng);
}

void clear_session() {
    g_session_rng.reset();
    clear_session_rng();
}

bool session_active() {
    return g_session_rng.has_value();
}

Matrix<double> randn_matrix(size_t rows, size_t cols) {
    Matrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out(i, j) = session_active() ? normal_from_rng() : 0.0;
        }
    }
    return out;
}

Matrix<double> rand_matrix(size_t rows, size_t cols) {
    Matrix<double> out(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out(i, j) = session_active() ? uniform_from_rng() : 0.0;
        }
    }
    return out;
}

namespace mc {

double estimate_pi(size_t samples, CSPRNG& rng) {
    size_t inside = 0;
    for (size_t i = 0; i < samples; ++i) {
        const double x = rng.next_f64();
        const double y = rng.next_f64();
        if (x * x + y * y <= 1.0) {
            ++inside;
        }
    }
    return 4.0 * static_cast<double>(inside) / static_cast<double>(samples);
}

} // namespace mc

namespace bloom {

BloomFilter::BloomFilter(size_t expected_items, double false_positive_rate, CSPRNG& rng) {
    const double ln2 = std::log(2.0);
    const double ln2_sq = ln2 * ln2;
    const double n = static_cast<double>(std::max(expected_items, size_t{1}));
    const double p = std::max(false_positive_rate, 1e-12);

    bit_count_ = static_cast<size_t>(std::ceil(-n * std::log(p) / ln2_sq));
    hash_count_ = static_cast<size_t>(std::round(static_cast<double>(bit_count_) / n * ln2));
    if (hash_count_ == 0) {
        hash_count_ = 1;
    }

    bits_.assign((bit_count_ + 7) / 8, 0);
    hash_seed1_ = rng.next_u64();
    hash_seed2_ = rng.next_u64() | 1ULL;
}

size_t BloomFilter::hash_index(std::span<const uint8_t> item, size_t i) const {
    const uint64_t h1 = fnv1a_64(item, hash_seed1_);
    const uint64_t h2 = fnv1a_64(item, hash_seed2_);
    const uint64_t combined = h1 + static_cast<uint64_t>(i) * h2;
    return static_cast<size_t>(combined % static_cast<uint64_t>(bit_count_));
}

void BloomFilter::set_bit(size_t index) {
    bits_[index / 8] |= static_cast<uint8_t>(1u << (index % 8));
}

bool BloomFilter::get_bit(size_t index) const {
    return (bits_[index / 8] & static_cast<uint8_t>(1u << (index % 8))) != 0;
}

void BloomFilter::insert(std::span<const uint8_t> item) {
    for (size_t i = 0; i < hash_count_; ++i) {
        set_bit(hash_index(item, i));
    }
}

bool BloomFilter::might_contain(std::span<const uint8_t> item) const {
    for (size_t i = 0; i < hash_count_; ++i) {
        if (!get_bit(hash_index(item, i))) {
            return false;
        }
    }
    return true;
}

size_t BloomFilter::bit_count() const {
    return bit_count_;
}

size_t BloomFilter::hash_count() const {
    return hash_count_;
}

} // namespace bloom

namespace ratelimit {

TokenBucket::TokenBucket(double capacity, double refill_rate_per_sec)
    : capacity_(capacity),
      refill_rate_(refill_rate_per_sec),
      tokens_(capacity) {}

void TokenBucket::refill_to(double now_seconds) {
    if (now_seconds > last_update_) {
        const double elapsed = now_seconds - last_update_;
        tokens_ = std::min(capacity_, tokens_ + elapsed * refill_rate_);
        last_update_ = now_seconds;
    }
}

bool TokenBucket::try_consume(double tokens, double now_seconds) {
    refill_to(now_seconds);
    if (tokens_ < tokens) {
        return false;
    }
    tokens_ -= tokens;
    return true;
}

double TokenBucket::available_tokens(double now_seconds) const {
    TokenBucket copy = *this;
    copy.refill_to(now_seconds);
    return copy.tokens_;
}

} // namespace ratelimit

namespace diffpriv {

double laplace_mechanism(double true_value, double epsilon, double sensitivity, CSPRNG& rng) {
    const double scale = sensitivity / epsilon;
    const uint64_t bits = rng.next_u64();
    double u = (static_cast<double>(static_cast<uint32_t>(bits & 0xFFFFFFFFu)) + 1.0) /
        static_cast<double>(0x100000000ULL);
    double noise = 0.0;
    if (u < 0.5) {
        noise = scale * std::log(2.0 * u);
    } else {
        noise = -scale * std::log(2.0 * (1.0 - u));
    }
    return true_value + noise;
}

double gaussian_mechanism(
    double true_value,
    double epsilon,
    double delta,
    double sensitivity,
    CSPRNG& rng) {
    const double sigma = std::sqrt(2.0 * std::log(1.25 / delta)) * sensitivity / epsilon;
    return true_value + rng.next_normal() * sigma;
}

} // namespace diffpriv

namespace backtest {

std::vector<double> simulate_gbm_path(
    double s0,
    double mu,
    double sigma,
    double dt,
    size_t steps,
    CSPRNG& rng) {
    std::vector<double> path;
    path.reserve(steps + 1);
    path.push_back(s0);

    const double drift = (mu - 0.5 * sigma * sigma) * dt;
    const double diffusion = sigma * std::sqrt(dt);

    double price = s0;
    for (size_t i = 0; i < steps; ++i) {
        const double z = rng.next_normal();
        price *= std::exp(drift + diffusion * z);
        path.push_back(price);
    }
    return path;
}

BacktestResult run_backtest(
    const std::vector<double>& prices,
    const std::vector<int>& positions,
    double initial_capital) {
    BacktestResult result;
    if (prices.empty() || positions.size() != prices.size()) {
        return result;
    }

    result.equity_curve.reserve(prices.size());
    result.equity_curve.push_back(initial_capital);

    std::vector<double> step_returns;
    if (prices.size() > 1) {
        step_returns.reserve(prices.size() - 1);
    }

    double equity = initial_capital;
    for (size_t i = 0; i + 1 < prices.size(); ++i) {
        const double price_return = (prices[i + 1] - prices[i]) / prices[i];
        const double portfolio_return = static_cast<double>(positions[i]) * price_return;
        const double prev_equity = equity;
        equity *= (1.0 + portfolio_return);
        result.equity_curve.push_back(equity);
        if (prev_equity > 0.0) {
            step_returns.push_back(equity / prev_equity - 1.0);
        }
    }

    if (!result.equity_curve.empty()) {
        result.total_return = result.equity_curve.back() / initial_capital - 1.0;
    }

    double peak = result.equity_curve.empty() ? 0.0 : result.equity_curve.front();
    for (double value : result.equity_curve) {
        peak = std::max(peak, value);
        if (peak > 0.0) {
            const double drawdown = (peak - value) / peak;
            result.max_drawdown = std::max(result.max_drawdown, drawdown);
        }
    }

    if (step_returns.size() >= 2) {
        double mean = 0.0;
        for (double r : step_returns) {
            mean += r;
        }
        mean /= static_cast<double>(step_returns.size());

        double variance = 0.0;
        for (double r : step_returns) {
            const double diff = r - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(step_returns.size());
        const double stddev = std::sqrt(variance);
        if (stddev > 1e-12) {
            result.sharpe_ratio = mean / stddev;
        }
    }

    return result;
}

} // namespace backtest

} // namespace ms::izaac
