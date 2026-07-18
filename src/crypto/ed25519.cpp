#include "ms/crypto/crypto.hpp"

extern "C" {
#include "ed25519/ed25519.h"
#include "ed25519/ge.h"
}

#include <cstring>

namespace ms {
namespace crypto {
namespace {

void ed25519_public_from_secret(std::array<uint8_t, ed25519_public_key_size>& public_key,
                                const std::array<uint8_t, ed25519_secret_key_size>& secret_key) {
    ge_p3 point{};
    ge_scalarmult_base(&point, secret_key.data());
    ge_p3_tobytes(public_key.data(), &point);
}

} // namespace

Ed25519Keypair ed25519_keypair(std::span<const uint8_t> seed32) {
    Ed25519Keypair out{};
    if (seed32.size() != ed25519_seed_size) {
        return out;
    }
    ed25519_create_keypair(out.public_key.data(), out.secret_key.data(), seed32.data());
    return out;
}

std::array<uint8_t, ed25519_signature_size> ed25519_sign(std::span<const uint8_t> secret_or_seed,
                                                         std::span<const uint8_t> message) {
    std::array<uint8_t, ed25519_signature_size> sig{};
    if (secret_or_seed.size() != ed25519_seed_size &&
        secret_or_seed.size() != ed25519_secret_key_size) {
        return sig;
    }

    Ed25519Keypair kp{};
    if (secret_or_seed.size() == ed25519_seed_size) {
        kp = ed25519_keypair(secret_or_seed);
    } else {
        std::memcpy(kp.secret_key.data(), secret_or_seed.data(), ed25519_secret_key_size);
        ed25519_public_from_secret(kp.public_key, kp.secret_key);
    }

    if (kp.public_key == std::array<uint8_t, ed25519_public_key_size>{}) {
        return sig;
    }

    ::ed25519_sign(sig.data(), message.data(), message.size(), kp.public_key.data(),
                   kp.secret_key.data());
    return sig;
}

bool ed25519_verify(std::span<const uint8_t> public_key, std::span<const uint8_t> message,
                    std::span<const uint8_t> signature) {
    if (public_key.size() != ed25519_public_key_size ||
        signature.size() != ed25519_signature_size) {
        return false;
    }
    return ::ed25519_verify(signature.data(), message.data(), message.size(),
                            public_key.data()) == 1;
}

} // namespace crypto
} // namespace ms
