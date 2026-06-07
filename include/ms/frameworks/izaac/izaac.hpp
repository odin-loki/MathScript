#pragma once

#include "ms/core/matrix.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

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

} // namespace ms::izaac
