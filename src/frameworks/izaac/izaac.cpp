#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/core/rng.hpp"

#include <cmath>
#include <cstring>
#include <random>

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

} // namespace ms::izaac
