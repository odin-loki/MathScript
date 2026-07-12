#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/core/rng.hpp"
#include "ms/error/error_types.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

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

uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// SplitMix64 (Vigna): used only to expand/diffuse a raw seed into well-distributed 64-bit
// words. Guarantees good avalanche behaviour even for small, low-entropy seeds (e.g. a
// deterministic small integer passed to seed_session), unlike a single-pass byte XOR.
uint64_t splitmix64_next(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Expands an arbitrary 32-byte seed into a well-mixed 256-bit state for the xoshiro256**
// generator used by next_u64(). Folds all 32 input bytes into a single 64-bit accumulator
// (order- and position-sensitive via rotation) and then runs that accumulator through
// SplitMix64 four times to produce the four state words, each forced non-zero (xoshiro256**
// is undefined for an all-zero state).
std::array<uint8_t, 32> mix_seed(std::array<uint8_t, 32> seed) {
    uint64_t acc = 0x243F6A8885A308D3ULL;
    for (size_t i = 0; i < seed.size(); ++i) {
        acc = rotl64(acc ^ (static_cast<uint64_t>(seed[i]) << ((i % 8) * 8)), 13) +
              static_cast<uint64_t>(i);
    }
    std::array<uint8_t, 32> out{};
    for (int w = 0; w < 4; ++w) {
        uint64_t word = splitmix64_next(acc);
        if (word == 0) {
            word = 1;
        }
        std::memcpy(out.data() + w * 8, &word, sizeof(word));
    }
    return out;
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

std::array<uint8_t, 32> xor_key_with_constant(
    std::array<uint8_t, 32> key,
    std::array<uint8_t, 32> constant) {
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] ^= constant[i];
    }
    return mix_seed(key);
}

std::array<uint8_t, 32> derive_cipher_seed(
    std::array<uint8_t, 32> key,
    std::span<const uint8_t> nonce) {
    std::array<uint8_t, 32> seed = key;
    for (size_t i = 0; i < seed.size(); ++i) {
        seed[i] ^= nonce[i % nonce.size()];
    }
    return mix_seed(seed);
}

std::array<uint8_t, 32> compute_tag(std::span<const uint8_t> data, std::array<uint8_t, 32> key) {
    constexpr std::array<uint8_t, 32> tag_constant = {
        0x54, 0x41, 0x47, 0x5F, 0x4B, 0x45, 0x59, 0x5F,
        0x44, 0x49, 0x53, 0x54, 0x49, 0x4E, 0x47, 0x55,
        0x49, 0x53, 0x48, 0x5F, 0x43, 0x4F, 0x4E, 0x53,
        0x54, 0x41, 0x4E, 0x54, 0x5F, 0x56, 0x31, 0x00};

    CSPRNG rng(xor_key_with_constant(key, tag_constant));
    std::array<uint8_t, 32> state{};

    for (size_t i = 0; i < data.size(); ++i) {
        const uint64_t mix = rng.next_u64() ^ static_cast<uint64_t>(data[i]) ^ static_cast<uint64_t>(i);
        state[i % 32] ^= static_cast<uint8_t>(mix & 0xFF);
        state[(i + 7) % 32] ^= static_cast<uint8_t>((mix >> 8) & 0xFF);
        state[(i + 13) % 32] ^= static_cast<uint8_t>((mix >> 16) & 0xFF);
        state[(i + 19) % 32] ^= static_cast<uint8_t>((mix >> 24) & 0xFF);

        const uint8_t carry = static_cast<uint8_t>(state[31] >> 7);
        for (int j = 31; j > 0; --j) {
            state[static_cast<size_t>(j)] =
                static_cast<uint8_t>((state[static_cast<size_t>(j)] << 1) |
                    (state[static_cast<size_t>(j - 1)] >> 7));
        }
        state[0] = static_cast<uint8_t>((state[0] << 1) | carry);
    }

    for (int round = 0; round < 4; ++round) {
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] ^= static_cast<uint8_t>(rng.next_u64() & 0xFF);
        }
    }

    return state;
}

uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>(static_cast<__uint128_t>(a) * b % mod);
#elif defined(_MSC_VER)
    unsigned __int64 hi = 0;
    const unsigned __int64 lo = _umul128(a, b, &hi);
    uint64_t rem = 0;
    for (int i = 63; i >= 0; --i) {
        rem = (rem << 1) | ((hi >> i) & 1);
        if (rem >= mod) {
            rem -= mod;
        }
    }
    for (int i = 63; i >= 0; --i) {
        rem = (rem << 1) | ((lo >> i) & 1);
        if (rem >= mod) {
            rem -= mod;
        }
    }
    return rem;
#else
    uint64_t res = 0;
    a %= mod;
    while (b > 0) {
        if (b & 1) {
            res += a;
            if (res >= mod) {
                res -= mod;
            }
        }
        a = (a << 1) % mod;
        b >>= 1;
    }
    return res;
#endif
}

uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, mod);
        }
        base = mod_mul(base, base, mod);
        exp >>= 1;
    }
    return result;
}

uint64_t mod_inverse(uint64_t value, uint64_t mod) {
    if (value == 0) {
        return 0;
    }
    return mod_pow(value, mod - 2, mod);
}

uint64_t eval_polynomial(
    const std::vector<uint64_t>& coeffs,
    int x,
    uint64_t mod) {
    uint64_t result = 0;
    uint64_t power = 1;
    const uint64_t x_mod = static_cast<uint64_t>(x) % mod;
    for (uint64_t coeff : coeffs) {
        result = (result + mod_mul(coeff % mod, power, mod)) % mod;
        power = mod_mul(power, x_mod, mod);
    }
    return result;
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
    size_t i = 0;
    while (i < buf.size()) {
        const uint64_t word = next_u64();
        for (size_t b = 0; b < sizeof(word) && i < buf.size(); ++b, ++i) {
            buf[i] = static_cast<uint8_t>((word >> (b * 8)) & 0xFF);
        }
    }
}

// xoshiro256** (Blackman & Vigna, public domain): a fast, high-quality, non-cryptographic
// PRNG operating on the full 256-bit state (state_ interpreted as four uint64_t words).
// The previous implementation only ever mutated 8 of the 32 state bytes per call and relied
// on a single weak XOR mixing pass at construction, which produced badly biased output (e.g.
// Monte Carlo pi estimates off by >20%) especially for small deterministic seeds. This
// generator has a long period (2^256 - 1) and passes standard statistical test suites; it is
// still explicitly NOT cryptographically secure (see the CSPRNG class doc — the name reflects
// its role as a seedable session generator, not a security guarantee).
uint64_t CSPRNG::next_u64() {
    std::array<uint64_t, 4> s{};
    std::memcpy(s.data(), state_.data(), sizeof(s));

    const uint64_t result = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    std::memcpy(state_.data(), s.data(), sizeof(s));
    ++stream_;
    return result;
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

namespace crypto {

namespace {

constexpr size_t kNonceSize = 16;

} // namespace

CipherText encrypt(std::span<const uint8_t> plaintext, std::array<uint8_t, 32> key) {
    CipherText ct;

    CSPRNG nonce_rng(mix_seed(key));
    std::array<uint8_t, kNonceSize> nonce{};
    nonce_rng.fill(std::span<uint8_t>(nonce));

    ct.data.resize(kNonceSize + plaintext.size());
    std::memcpy(ct.data.data(), nonce.data(), kNonceSize);

    CSPRNG keystream_rng(derive_cipher_seed(key, std::span<const uint8_t>(nonce)));
    for (size_t i = 0; i < plaintext.size(); ++i) {
        const uint8_t stream_byte = static_cast<uint8_t>(keystream_rng.next_u64() & 0xFF);
        ct.data[kNonceSize + i] = static_cast<uint8_t>(plaintext[i] ^ stream_byte);
    }

    ct.tag = compute_tag(std::span<const uint8_t>(ct.data), key);
    return ct;
}

Result<std::vector<uint8_t>> decrypt(const CipherText& ct, std::array<uint8_t, 32> key) {
    if (ct.data.size() < kNonceSize) {
        return std::unexpected(DomainError{"izaac::crypto::decrypt", "ciphertext too short"});
    }

    const std::array<uint8_t, 32> expected_tag = compute_tag(std::span<const uint8_t>(ct.data), key);
    if (std::memcmp(expected_tag.data(), ct.tag.data(), ct.tag.size()) != 0) {
        return std::unexpected(DomainError{"izaac::crypto::decrypt", "authentication tag mismatch"});
    }

    const std::span<const uint8_t> nonce(ct.data.data(), kNonceSize);
    const size_t ciphertext_len = ct.data.size() - kNonceSize;
    std::vector<uint8_t> plaintext(ciphertext_len);

    CSPRNG keystream_rng(derive_cipher_seed(key, nonce));
    for (size_t i = 0; i < ciphertext_len; ++i) {
        const uint8_t stream_byte = static_cast<uint8_t>(keystream_rng.next_u64() & 0xFF);
        plaintext[i] = static_cast<uint8_t>(ct.data[kNonceSize + i] ^ stream_byte);
    }

    return plaintext;
}

} // namespace crypto

namespace mpc {

std::vector<Share> split_secret(uint64_t secret, int n, int k, CSPRNG& rng) {
    std::vector<Share> shares;
    if (n < k || k < 2 || n < 2 || secret >= PRIME) {
        return shares;
    }

    std::vector<uint64_t> coeffs(static_cast<size_t>(k));
    coeffs[0] = secret % PRIME;
    for (int i = 1; i < k; ++i) {
        coeffs[static_cast<size_t>(i)] = rng.next_u64() % PRIME;
    }

    shares.reserve(static_cast<size_t>(n));
    for (int x = 1; x <= n; ++x) {
        shares.push_back(Share{x, eval_polynomial(coeffs, x, PRIME)});
    }
    return shares;
}

Result<uint64_t> reconstruct_secret(const std::vector<Share>& shares) {
    if (shares.size() < 2) {
        return std::unexpected(
            DomainError{"izaac::mpc::reconstruct_secret", "need at least two shares"});
    }

    uint64_t secret = 0;
    for (size_t i = 0; i < shares.size(); ++i) {
        uint64_t numerator = 1;
        uint64_t denominator = 1;

        const uint64_t xi = static_cast<uint64_t>(shares[i].x) % PRIME;
        for (size_t j = 0; j < shares.size(); ++j) {
            if (i == j) {
                continue;
            }
            const uint64_t xj = static_cast<uint64_t>(shares[j].x) % PRIME;
            numerator = mod_mul(numerator, (PRIME - xj) % PRIME, PRIME);
            const uint64_t diff = (xi + PRIME - xj) % PRIME;
            if (diff == 0) {
                return std::unexpected(
                    DomainError{"izaac::mpc::reconstruct_secret", "duplicate share index"});
            }
            denominator = mod_mul(denominator, diff, PRIME);
        }

        const uint64_t lagrange_basis =
            mod_mul(numerator, mod_inverse(denominator, PRIME), PRIME);
        secret = (secret + mod_mul(shares[i].y % PRIME, lagrange_basis, PRIME)) % PRIME;
    }

    return secret;
}

} // namespace mpc

namespace consensus {

Cluster::Cluster(int n, unsigned seed) : seed(seed), rng_(seed) {
    nodes.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        nodes.push_back(Node{i});
    }
}

int Cluster::run_election() {
    if (nodes.empty()) {
        return -1;
    }

    const int n = static_cast<int>(nodes.size());
    const int majority = n / 2 + 1;

    std::uniform_int_distribution<int> node_dist(0, n - 1);
    std::vector<int> candidates;
    candidates.push_back(node_dist(rng_));

    // ~20% of rounds with n >= 4 spawn a second candidate in the same term; followers pick
    // one eligible candidate via the seeded RNG, which can produce a split vote (no majority).
    const bool split_round = (n >= 4) && ((rng_() % 5) == 0);
    if (split_round) {
        int second = node_dist(rng_);
        while (second == candidates[0]) {
            second = node_dist(rng_);
        }
        candidates.push_back(second);
    }
    std::sort(candidates.begin(), candidates.end());

    uint64_t new_term = 0;
    for (const Node& node : nodes) {
        new_term = std::max(new_term, node.current_term);
    }
    ++new_term;

    for (Node& node : nodes) {
        node.role = NodeRole::Follower;
        if (new_term > node.current_term) {
            node.current_term = new_term;
            node.voted_for = -1;
        }
    }

    for (int cid : candidates) {
        nodes[static_cast<size_t>(cid)].role = NodeRole::Candidate;
    }

    std::vector<int> votes(static_cast<size_t>(n), 0);
    for (int cid : candidates) {
        Node& candidate = nodes[static_cast<size_t>(cid)];
        candidate.voted_for = cid;
        ++votes[static_cast<size_t>(cid)];
    }

    for (int i = 0; i < n; ++i) {
        Node& voter = nodes[static_cast<size_t>(i)];
        if (voter.voted_for != -1) {
            continue;
        }

        std::vector<int> eligible;
        eligible.reserve(candidates.size());
        for (int cid : candidates) {
            const Node& candidate = nodes[static_cast<size_t>(cid)];
            if (candidate.current_term >= voter.current_term) {
                eligible.push_back(cid);
            }
        }
        if (eligible.empty()) {
            continue;
        }

        const int choice = eligible.size() == 1
            ? eligible[0]
            : eligible[static_cast<size_t>(rng_() % eligible.size())];
        voter.voted_for = choice;
        ++votes[static_cast<size_t>(choice)];
    }

    int winner = -1;
    int winner_votes = 0;
    for (int cid : candidates) {
        const int count = votes[static_cast<size_t>(cid)];
        if (count >= majority && count > winner_votes) {
            winner = cid;
            winner_votes = count;
        }
    }

    if (winner >= 0) {
        for (Node& node : nodes) {
            node.role = (node.id == winner) ? NodeRole::Leader : NodeRole::Follower;
        }
        return winner;
    }

    for (Node& node : nodes) {
        node.role = NodeRole::Follower;
    }
    return -1;
}

bool Cluster::replicate(int leader_id, const std::string& command) {
    if (leader_id < 0 || leader_id >= static_cast<int>(nodes.size())) {
        return false;
    }

    Node& leader = nodes[static_cast<size_t>(leader_id)];
    if (leader.role != NodeRole::Leader) {
        return false;
    }

    const LogEntry entry{leader.current_term, command};
    leader.log.push_back(entry);
    const uint64_t entry_index = leader.log.size();

    const int n = static_cast<int>(nodes.size());
    const int majority = n / 2 + 1;
    int holding = 1;

    for (int i = 0; i < n && holding < majority; ++i) {
        if (i == leader_id) {
            continue;
        }
        nodes[static_cast<size_t>(i)].log.push_back(entry);
        ++holding;
    }

    if (holding < majority) {
        return false;
    }

    for (Node& node : nodes) {
        if (node.log.size() >= entry_index) {
            node.commit_index = entry_index;
        }
    }
    return true;
}

int Cluster::current_leader() const {
    int leader = -1;
    for (const Node& node : nodes) {
        if (node.role == NodeRole::Leader) {
            if (leader != -1) {
                return -1;
            }
            leader = node.id;
        }
    }
    return leader;
}

} // namespace consensus

namespace fuzz {

namespace {

constexpr std::size_t kMaxSpliceLength = 8;

// Draws a uniform value in [0, bound); returns 0 for bound == 0 rather than dividing by
// zero, matching this module's defensive no-crash convention.
uint64_t bounded(CSPRNG& rng, uint64_t bound) {
    if (bound == 0) {
        return 0;
    }
    return rng.next_u64() % bound;
}

uint8_t random_byte(CSPRNG& rng) {
    return static_cast<uint8_t>(bounded(rng, 256));
}

enum class EditKind : uint8_t { Flip = 0, Insert = 1, Delete = 2, Splice = 3 };

void apply_flip(std::vector<uint8_t>& buf, CSPRNG& rng) {
    const std::size_t pos = static_cast<std::size_t>(bounded(rng, buf.size()));
    buf[pos] = random_byte(rng);
}

void apply_insert(std::vector<uint8_t>& buf, CSPRNG& rng) {
    const std::size_t pos = static_cast<std::size_t>(bounded(rng, buf.size() + 1));
    buf.insert(buf.begin() + static_cast<std::ptrdiff_t>(pos), random_byte(rng));
}

void apply_delete(std::vector<uint8_t>& buf, CSPRNG& rng) {
    const std::size_t pos = static_cast<std::size_t>(bounded(rng, buf.size()));
    buf.erase(buf.begin() + static_cast<std::ptrdiff_t>(pos));
}

void apply_splice(std::vector<uint8_t>& buf, CSPRNG& rng) {
    const std::size_t start = static_cast<std::size_t>(bounded(rng, buf.size()));
    const std::size_t max_len = std::min(kMaxSpliceLength, buf.size() - start);
    const std::size_t len = 1 + static_cast<std::size_t>(bounded(rng, max_len));
    const std::vector<uint8_t> chunk(
        buf.begin() + static_cast<std::ptrdiff_t>(start),
        buf.begin() + static_cast<std::ptrdiff_t>(start + len));
    const std::size_t insert_pos = static_cast<std::size_t>(bounded(rng, buf.size() + 1));
    buf.insert(buf.begin() + static_cast<std::ptrdiff_t>(insert_pos), chunk.begin(), chunk.end());
}

} // namespace

std::vector<uint8_t> mutate(std::span<const uint8_t> input, CSPRNG& rng, std::size_t max_edits) {
    std::vector<uint8_t> buf(input.begin(), input.end());
    if (max_edits == 0) {
        return buf;
    }

    const std::size_t edit_count = 1 + static_cast<std::size_t>(bounded(rng, max_edits));
    for (std::size_t i = 0; i < edit_count; ++i) {
        if (buf.empty()) {
            // Only insertion is well-defined on an empty buffer; still consumes rng state
            // for the byte value so edit i's randomness usage stays uniform across edits.
            buf.push_back(random_byte(rng));
            continue;
        }

        switch (static_cast<EditKind>(bounded(rng, 4))) {
        case EditKind::Flip:
            apply_flip(buf, rng);
            break;
        case EditKind::Insert:
            apply_insert(buf, rng);
            break;
        case EditKind::Delete:
            apply_delete(buf, rng);
            break;
        case EditKind::Splice:
            apply_splice(buf, rng);
            break;
        }
    }
    return buf;
}

} // namespace fuzz

} // namespace ms::izaac
