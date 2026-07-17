#include "ms/crypto/crypto.hpp"

#include <array>
#include <cstring>

namespace ms {
namespace crypto {
namespace {

constexpr std::uint32_t kSha256Init[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::uint64_t kSha512Init[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
    0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
    0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull,
};

constexpr std::uint64_t kSha512K[80] = {
    0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full,
    0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull, 0x59f111f1b605d019ull,
    0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull,
    0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
    0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
    0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull,
    0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull, 0x2de92c6f592b0275ull,
    0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
    0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full,
    0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
    0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull,
    0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
    0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull,
    0x92722c851482353bull, 0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull,
    0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
    0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
    0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull,
    0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull,
    0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull,
    0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
    0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull,
    0xc67178f2e372532bull, 0xca273eceea26619cull, 0xd186b8c721c0c207ull,
    0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull,
    0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
    0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
    0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull,
    0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull,
};

inline std::uint32_t rotr32(std::uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}

inline std::uint64_t rotr64(std::uint64_t x, unsigned n) {
    return (x >> n) | (x << (64u - n));
}

inline void store_be32(std::uint8_t* out, std::uint32_t v) {
    out[0] = static_cast<std::uint8_t>((v >> 24) & 0xffu);
    out[1] = static_cast<std::uint8_t>((v >> 16) & 0xffu);
    out[2] = static_cast<std::uint8_t>((v >> 8) & 0xffu);
    out[3] = static_cast<std::uint8_t>(v & 0xffu);
}

inline void store_be64(std::uint8_t* out, std::uint64_t v) {
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<std::uint8_t>(v & 0xffull);
        v >>= 8;
    }
}

inline std::uint32_t load_be32(const std::uint8_t* in) {
    return (static_cast<std::uint32_t>(in[0]) << 24) |
           (static_cast<std::uint32_t>(in[1]) << 16) |
           (static_cast<std::uint32_t>(in[2]) << 8) |
           static_cast<std::uint32_t>(in[3]);
}

inline std::uint64_t load_be64(const std::uint8_t* in) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v = (v << 8) | static_cast<std::uint64_t>(in[i]);
    }
    return v;
}

void sha256_transform(std::uint32_t state[8], const std::uint8_t block[64]) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = load_be32(block + i * 4);
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^
                                 (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^
                                 (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    for (int i = 0; i < 64; ++i) {
        const std::uint32_t t1 = h + (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) +
                                 ((e & f) ^ (~e & g)) + kSha256K[i] + w[i];
        const std::uint32_t t2 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) +
                                 ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha512_transform(std::uint64_t state[8], const std::uint8_t block[128]) {
    std::uint64_t w[80];
    for (int i = 0; i < 16; ++i) {
        w[i] = load_be64(block + i * 8);
    }
    for (int i = 16; i < 80; ++i) {
        const std::uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^
                                 (w[i - 15] >> 7);
        const std::uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^
                                 (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint64_t a = state[0];
    std::uint64_t b = state[1];
    std::uint64_t c = state[2];
    std::uint64_t d = state[3];
    std::uint64_t e = state[4];
    std::uint64_t f = state[5];
    std::uint64_t g = state[6];
    std::uint64_t h = state[7];

    for (int i = 0; i < 80; ++i) {
        const std::uint64_t t1 = h + (rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41)) +
                                 ((e & f) ^ (~e & g)) + kSha512K[i] + w[i];
        const std::uint64_t t2 = (rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39)) +
                                 ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256_hash(std::span<const std::uint8_t> data, std::uint8_t digest[32]) {
    std::uint32_t state[8];
    std::memcpy(state, kSha256Init, sizeof(state));

    const std::uint64_t total_bits = static_cast<std::uint64_t>(data.size()) * 8u;
    std::size_t offset = 0;

    while (offset + 64 <= data.size()) {
        sha256_transform(state, data.data() + offset);
        offset += 64;
    }

    std::array<std::uint8_t, 64> block{};
    const std::size_t rem = data.size() - offset;
    if (rem > 0) {
        std::memcpy(block.data(), data.data() + offset, rem);
    }
    block[rem] = 0x80;

    if (rem + 1 + 8 > 64) {
        sha256_transform(state, block.data());
        block.fill(0);
    }

    store_be64(block.data() + 56, total_bits);
    sha256_transform(state, block.data());

    for (int i = 0; i < 8; ++i) {
        store_be32(digest + i * 4, state[i]);
    }
}

void sha512_hash(std::span<const std::uint8_t> data, std::uint8_t digest[64]) {
    std::uint64_t state[8];
    std::memcpy(state, kSha512Init, sizeof(state));

    const std::uint64_t total_bits = static_cast<std::uint64_t>(data.size()) * 8u;
    std::size_t offset = 0;

    while (offset + 128 <= data.size()) {
        sha512_transform(state, data.data() + offset);
        offset += 128;
    }

    std::array<std::uint8_t, 128> block{};
    const std::size_t rem = data.size() - offset;
    if (rem > 0) {
        std::memcpy(block.data(), data.data() + offset, rem);
    }
    block[rem] = 0x80;

    if (rem + 1 + 16 > 128) {
        sha512_transform(state, block.data());
        block.fill(0);
    }

    store_be64(block.data() + 120, total_bits);
    sha512_transform(state, block.data());

    for (int i = 0; i < 8; ++i) {
        store_be64(digest + i * 8, state[i]);
    }
}

char hex_digit(int v) {
    return static_cast<char>(v < 10 ? ('0' + v) : ('a' + v - 10));
}

} // namespace

std::vector<uint8_t> sha256(std::span<const uint8_t> data) {
    std::vector<uint8_t> out(sha256_digest_size);
    sha256_hash(data, out.data());
    return out;
}

std::vector<uint8_t> sha512(std::span<const uint8_t> data) {
    std::vector<uint8_t> out(sha512_digest_size);
    sha512_hash(data, out.data());
    return out;
}

std::vector<uint8_t> hmac_sha256(std::span<const uint8_t> key,
                                   std::span<const uint8_t> data) {
    constexpr std::size_t block_size = 64;
    std::array<std::uint8_t, block_size> k{};
    if (key.size() > block_size) {
        auto kh = sha256(key);
        std::memcpy(k.data(), kh.data(), kh.size());
    } else {
        std::memcpy(k.data(), key.data(), key.size());
    }

    std::array<std::uint8_t, block_size> ipad{};
    std::array<std::uint8_t, block_size> opad{};
    for (std::size_t i = 0; i < block_size; ++i) {
        ipad[i] = k[i] ^ 0x36u;
        opad[i] = k[i] ^ 0x5cu;
    }

    std::vector<std::uint8_t> inner;
    inner.reserve(block_size + data.size());
    inner.insert(inner.end(), ipad.begin(), ipad.end());
    inner.insert(inner.end(), data.begin(), data.end());
    auto inner_hash = sha256(inner);

    std::vector<std::uint8_t> outer;
    outer.reserve(block_size + inner_hash.size());
    outer.insert(outer.end(), opad.begin(), opad.end());
    outer.insert(outer.end(), inner_hash.begin(), inner_hash.end());
    return sha256(outer);
}

std::vector<uint8_t> sha256(std::string_view data) {
    return sha256(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

std::vector<uint8_t> sha512(std::string_view data) {
    return sha512(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

std::vector<uint8_t> hmac_sha256(std::string_view key, std::string_view data) {
    return hmac_sha256(
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(key.data()),
                                 key.size()),
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()),
                                 data.size()));
}

std::string to_hex(std::span<const uint8_t> bytes) {
    std::string out;
    out.resize(bytes.size() * 2);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        out[i * 2] = hex_digit((bytes[i] >> 4) & 0x0f);
        out[i * 2 + 1] = hex_digit(bytes[i] & 0x0f);
    }
    return out;
}

std::string sha256_hex(std::span<const uint8_t> data) {
    return to_hex(sha256(data));
}

std::string sha512_hex(std::span<const uint8_t> data) {
    return to_hex(sha512(data));
}

std::string hmac_sha256_hex(std::span<const uint8_t> key,
                            std::span<const uint8_t> data) {
    return to_hex(hmac_sha256(key, data));
}

std::string sha256_hex(std::string_view data) {
    return sha256_hex(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

std::string sha512_hex(std::string_view data) {
    return sha512_hex(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

std::string hmac_sha256_hex(std::string_view key, std::string_view data) {
    return hmac_sha256_hex(
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(key.data()),
                                 key.size()),
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()),
                                 data.size()));
}

} // namespace crypto
} // namespace ms
