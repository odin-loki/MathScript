#include "ms/crypto/crypto.hpp"

#include <array>
#include <cstring>

#if defined(_MSC_VER)
#include <stdlib.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__has_builtin)
#if __has_builtin(__builtin_bswap32)
#define MS_HAVE_BSWAP32 1
#endif
#if __has_builtin(__builtin_bswap64)
#define MS_HAVE_BSWAP64 1
#endif
#endif
#ifndef MS_HAVE_BSWAP32
#define MS_HAVE_BSWAP32 1
#endif
#ifndef MS_HAVE_BSWAP64
#define MS_HAVE_BSWAP64 1
#endif
#endif

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

inline std::uint32_t bswap32(std::uint32_t v) {
#if defined(_MSC_VER)
    return _byteswap_ulong(v);
#elif defined(MS_HAVE_BSWAP32)
    return __builtin_bswap32(v);
#else
    return ((v & 0x000000ffu) << 24) | ((v & 0x0000ff00u) << 8) |
           ((v & 0x00ff0000u) >> 8) | ((v & 0xff000000u) >> 24);
#endif
}

inline std::uint64_t bswap64(std::uint64_t v) {
#if defined(_MSC_VER)
    return _byteswap_uint64(v);
#elif defined(MS_HAVE_BSWAP64)
    return __builtin_bswap64(v);
#else
    return ((v & 0x00000000000000ffull) << 56) | ((v & 0x000000000000ff00ull) << 40) |
           ((v & 0x0000000000ff0000ull) << 24) | ((v & 0x00000000ff000000ull) << 8) |
           ((v & 0x000000ff00000000ull) >> 8) | ((v & 0x0000ff0000000000ull) >> 24) |
           ((v & 0x00ff000000000000ull) >> 40) | ((v & 0xff00000000000000ull) >> 56);
#endif
}

inline void store_be32(std::uint8_t* out, std::uint32_t v) {
    const std::uint32_t be = bswap32(v);
    std::memcpy(out, &be, sizeof(be));
}

inline void store_be64(std::uint8_t* out, std::uint64_t v) {
    const std::uint64_t be = bswap64(v);
    std::memcpy(out, &be, sizeof(be));
}

inline std::uint32_t load_be32(const std::uint8_t* in) {
    std::uint32_t v;
    std::memcpy(&v, in, sizeof(v));
    return bswap32(v);
}

inline std::uint64_t load_be64(const std::uint8_t* in) {
    std::uint64_t v;
    std::memcpy(&v, in, sizeof(v));
    return bswap64(v);
}

#define SHA256_SIGMA0(x) (rotr32((x), 7) ^ rotr32((x), 18) ^ ((x) >> 3))
#define SHA256_SIGMA1(x) (rotr32((x), 17) ^ rotr32((x), 19) ^ ((x) >> 10))
#define SHA256_S0(x) (rotr32((x), 2) ^ rotr32((x), 13) ^ rotr32((x), 22))
#define SHA256_S1(x) (rotr32((x), 6) ^ rotr32((x), 11) ^ rotr32((x), 25))
#define SHA256_CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define SHA256_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define SHA256_ROUND(i, a, b, c, d, e, f, g, h) \
    do { \
        const std::uint32_t t1 = \
            (h) + SHA256_S1(e) + SHA256_CH(e, f, g) + kSha256K[i] + w[i]; \
        const std::uint32_t t2 = SHA256_S0(a) + SHA256_MAJ(a, b, c); \
        (h) = (g); \
        (g) = (f); \
        (f) = (e); \
        (e) = (d) + t1; \
        (d) = (c); \
        (c) = (b); \
        (b) = (a); \
        (a) = t1 + t2; \
    } while (0)

#define SHA512_SIGMA0(x) (rotr64((x), 1) ^ rotr64((x), 8) ^ ((x) >> 7))
#define SHA512_SIGMA1(x) (rotr64((x), 19) ^ rotr64((x), 61) ^ ((x) >> 6))
#define SHA512_S0(x) (rotr64((x), 28) ^ rotr64((x), 34) ^ rotr64((x), 39))
#define SHA512_S1(x) (rotr64((x), 14) ^ rotr64((x), 18) ^ rotr64((x), 41))
#define SHA512_CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define SHA512_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

#define SHA512_ROUND(i, a, b, c, d, e, f, g, h) \
    do { \
        const std::uint64_t t1 = \
            (h) + SHA512_S1(e) + SHA512_CH(e, f, g) + kSha512K[i] + w[i]; \
        const std::uint64_t t2 = SHA512_S0(a) + SHA512_MAJ(a, b, c); \
        (h) = (g); \
        (g) = (f); \
        (f) = (e); \
        (e) = (d) + t1; \
        (d) = (c); \
        (c) = (b); \
        (b) = (a); \
        (a) = t1 + t2; \
    } while (0)

inline void sha256_expand(std::uint32_t w[64], const std::uint8_t block[64]) {
    w[0] = load_be32(block + 0);
    w[1] = load_be32(block + 4);
    w[2] = load_be32(block + 8);
    w[3] = load_be32(block + 12);
    w[4] = load_be32(block + 16);
    w[5] = load_be32(block + 20);
    w[6] = load_be32(block + 24);
    w[7] = load_be32(block + 28);
    w[8] = load_be32(block + 32);
    w[9] = load_be32(block + 36);
    w[10] = load_be32(block + 40);
    w[11] = load_be32(block + 44);
    w[12] = load_be32(block + 48);
    w[13] = load_be32(block + 52);
    w[14] = load_be32(block + 56);
    w[15] = load_be32(block + 60);

    for (int i = 16; i < 64; i += 4) {
        w[i] = w[i - 16] + SHA256_SIGMA0(w[i - 15]) + w[i - 7] + SHA256_SIGMA1(w[i - 2]);
        w[i + 1] =
            w[i - 15] + SHA256_SIGMA0(w[i - 14]) + w[i - 6] + SHA256_SIGMA1(w[i - 1]);
        w[i + 2] =
            w[i - 14] + SHA256_SIGMA0(w[i - 13]) + w[i - 5] + SHA256_SIGMA1(w[i]);
        w[i + 3] =
            w[i - 13] + SHA256_SIGMA0(w[i - 12]) + w[i - 4] + SHA256_SIGMA1(w[i + 1]);
    }
}

inline void sha256_rounds(std::uint32_t w[64],
                          std::uint32_t& a,
                          std::uint32_t& b,
                          std::uint32_t& c,
                          std::uint32_t& d,
                          std::uint32_t& e,
                          std::uint32_t& f,
                          std::uint32_t& g,
                          std::uint32_t& h) {
    SHA256_ROUND(0, a, b, c, d, e, f, g, h);
    SHA256_ROUND(1, a, b, c, d, e, f, g, h);
    SHA256_ROUND(2, a, b, c, d, e, f, g, h);
    SHA256_ROUND(3, a, b, c, d, e, f, g, h);
    SHA256_ROUND(4, a, b, c, d, e, f, g, h);
    SHA256_ROUND(5, a, b, c, d, e, f, g, h);
    SHA256_ROUND(6, a, b, c, d, e, f, g, h);
    SHA256_ROUND(7, a, b, c, d, e, f, g, h);
    SHA256_ROUND(8, a, b, c, d, e, f, g, h);
    SHA256_ROUND(9, a, b, c, d, e, f, g, h);
    SHA256_ROUND(10, a, b, c, d, e, f, g, h);
    SHA256_ROUND(11, a, b, c, d, e, f, g, h);
    SHA256_ROUND(12, a, b, c, d, e, f, g, h);
    SHA256_ROUND(13, a, b, c, d, e, f, g, h);
    SHA256_ROUND(14, a, b, c, d, e, f, g, h);
    SHA256_ROUND(15, a, b, c, d, e, f, g, h);
    SHA256_ROUND(16, a, b, c, d, e, f, g, h);
    SHA256_ROUND(17, a, b, c, d, e, f, g, h);
    SHA256_ROUND(18, a, b, c, d, e, f, g, h);
    SHA256_ROUND(19, a, b, c, d, e, f, g, h);
    SHA256_ROUND(20, a, b, c, d, e, f, g, h);
    SHA256_ROUND(21, a, b, c, d, e, f, g, h);
    SHA256_ROUND(22, a, b, c, d, e, f, g, h);
    SHA256_ROUND(23, a, b, c, d, e, f, g, h);
    SHA256_ROUND(24, a, b, c, d, e, f, g, h);
    SHA256_ROUND(25, a, b, c, d, e, f, g, h);
    SHA256_ROUND(26, a, b, c, d, e, f, g, h);
    SHA256_ROUND(27, a, b, c, d, e, f, g, h);
    SHA256_ROUND(28, a, b, c, d, e, f, g, h);
    SHA256_ROUND(29, a, b, c, d, e, f, g, h);
    SHA256_ROUND(30, a, b, c, d, e, f, g, h);
    SHA256_ROUND(31, a, b, c, d, e, f, g, h);
    SHA256_ROUND(32, a, b, c, d, e, f, g, h);
    SHA256_ROUND(33, a, b, c, d, e, f, g, h);
    SHA256_ROUND(34, a, b, c, d, e, f, g, h);
    SHA256_ROUND(35, a, b, c, d, e, f, g, h);
    SHA256_ROUND(36, a, b, c, d, e, f, g, h);
    SHA256_ROUND(37, a, b, c, d, e, f, g, h);
    SHA256_ROUND(38, a, b, c, d, e, f, g, h);
    SHA256_ROUND(39, a, b, c, d, e, f, g, h);
    SHA256_ROUND(40, a, b, c, d, e, f, g, h);
    SHA256_ROUND(41, a, b, c, d, e, f, g, h);
    SHA256_ROUND(42, a, b, c, d, e, f, g, h);
    SHA256_ROUND(43, a, b, c, d, e, f, g, h);
    SHA256_ROUND(44, a, b, c, d, e, f, g, h);
    SHA256_ROUND(45, a, b, c, d, e, f, g, h);
    SHA256_ROUND(46, a, b, c, d, e, f, g, h);
    SHA256_ROUND(47, a, b, c, d, e, f, g, h);
    SHA256_ROUND(48, a, b, c, d, e, f, g, h);
    SHA256_ROUND(49, a, b, c, d, e, f, g, h);
    SHA256_ROUND(50, a, b, c, d, e, f, g, h);
    SHA256_ROUND(51, a, b, c, d, e, f, g, h);
    SHA256_ROUND(52, a, b, c, d, e, f, g, h);
    SHA256_ROUND(53, a, b, c, d, e, f, g, h);
    SHA256_ROUND(54, a, b, c, d, e, f, g, h);
    SHA256_ROUND(55, a, b, c, d, e, f, g, h);
    SHA256_ROUND(56, a, b, c, d, e, f, g, h);
    SHA256_ROUND(57, a, b, c, d, e, f, g, h);
    SHA256_ROUND(58, a, b, c, d, e, f, g, h);
    SHA256_ROUND(59, a, b, c, d, e, f, g, h);
    SHA256_ROUND(60, a, b, c, d, e, f, g, h);
    SHA256_ROUND(61, a, b, c, d, e, f, g, h);
    SHA256_ROUND(62, a, b, c, d, e, f, g, h);
    SHA256_ROUND(63, a, b, c, d, e, f, g, h);
}

void sha256_transform(std::uint32_t state[8], const std::uint8_t block[64]) {
    std::uint32_t w[64];
    sha256_expand(w, block);

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    sha256_rounds(w, a, b, c, d, e, f, g, h);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

inline void sha512_expand(std::uint64_t w[80], const std::uint8_t block[128]) {
    w[0] = load_be64(block + 0);
    w[1] = load_be64(block + 8);
    w[2] = load_be64(block + 16);
    w[3] = load_be64(block + 24);
    w[4] = load_be64(block + 32);
    w[5] = load_be64(block + 40);
    w[6] = load_be64(block + 48);
    w[7] = load_be64(block + 56);
    w[8] = load_be64(block + 64);
    w[9] = load_be64(block + 72);
    w[10] = load_be64(block + 80);
    w[11] = load_be64(block + 88);
    w[12] = load_be64(block + 96);
    w[13] = load_be64(block + 104);
    w[14] = load_be64(block + 112);
    w[15] = load_be64(block + 120);

    for (int i = 16; i < 80; i += 4) {
        w[i] = w[i - 16] + SHA512_SIGMA0(w[i - 15]) + w[i - 7] + SHA512_SIGMA1(w[i - 2]);
        w[i + 1] =
            w[i - 15] + SHA512_SIGMA0(w[i - 14]) + w[i - 6] + SHA512_SIGMA1(w[i - 1]);
        w[i + 2] =
            w[i - 14] + SHA512_SIGMA0(w[i - 13]) + w[i - 5] + SHA512_SIGMA1(w[i]);
        w[i + 3] =
            w[i - 13] + SHA512_SIGMA0(w[i - 12]) + w[i - 4] + SHA512_SIGMA1(w[i + 1]);
    }
}

inline void sha512_rounds(std::uint64_t w[80],
                           std::uint64_t& a,
                           std::uint64_t& b,
                           std::uint64_t& c,
                           std::uint64_t& d,
                           std::uint64_t& e,
                           std::uint64_t& f,
                           std::uint64_t& g,
                           std::uint64_t& h) {
    SHA512_ROUND(0, a, b, c, d, e, f, g, h);
    SHA512_ROUND(1, a, b, c, d, e, f, g, h);
    SHA512_ROUND(2, a, b, c, d, e, f, g, h);
    SHA512_ROUND(3, a, b, c, d, e, f, g, h);
    SHA512_ROUND(4, a, b, c, d, e, f, g, h);
    SHA512_ROUND(5, a, b, c, d, e, f, g, h);
    SHA512_ROUND(6, a, b, c, d, e, f, g, h);
    SHA512_ROUND(7, a, b, c, d, e, f, g, h);
    SHA512_ROUND(8, a, b, c, d, e, f, g, h);
    SHA512_ROUND(9, a, b, c, d, e, f, g, h);
    SHA512_ROUND(10, a, b, c, d, e, f, g, h);
    SHA512_ROUND(11, a, b, c, d, e, f, g, h);
    SHA512_ROUND(12, a, b, c, d, e, f, g, h);
    SHA512_ROUND(13, a, b, c, d, e, f, g, h);
    SHA512_ROUND(14, a, b, c, d, e, f, g, h);
    SHA512_ROUND(15, a, b, c, d, e, f, g, h);
    SHA512_ROUND(16, a, b, c, d, e, f, g, h);
    SHA512_ROUND(17, a, b, c, d, e, f, g, h);
    SHA512_ROUND(18, a, b, c, d, e, f, g, h);
    SHA512_ROUND(19, a, b, c, d, e, f, g, h);
    SHA512_ROUND(20, a, b, c, d, e, f, g, h);
    SHA512_ROUND(21, a, b, c, d, e, f, g, h);
    SHA512_ROUND(22, a, b, c, d, e, f, g, h);
    SHA512_ROUND(23, a, b, c, d, e, f, g, h);
    SHA512_ROUND(24, a, b, c, d, e, f, g, h);
    SHA512_ROUND(25, a, b, c, d, e, f, g, h);
    SHA512_ROUND(26, a, b, c, d, e, f, g, h);
    SHA512_ROUND(27, a, b, c, d, e, f, g, h);
    SHA512_ROUND(28, a, b, c, d, e, f, g, h);
    SHA512_ROUND(29, a, b, c, d, e, f, g, h);
    SHA512_ROUND(30, a, b, c, d, e, f, g, h);
    SHA512_ROUND(31, a, b, c, d, e, f, g, h);
    SHA512_ROUND(32, a, b, c, d, e, f, g, h);
    SHA512_ROUND(33, a, b, c, d, e, f, g, h);
    SHA512_ROUND(34, a, b, c, d, e, f, g, h);
    SHA512_ROUND(35, a, b, c, d, e, f, g, h);
    SHA512_ROUND(36, a, b, c, d, e, f, g, h);
    SHA512_ROUND(37, a, b, c, d, e, f, g, h);
    SHA512_ROUND(38, a, b, c, d, e, f, g, h);
    SHA512_ROUND(39, a, b, c, d, e, f, g, h);
    SHA512_ROUND(40, a, b, c, d, e, f, g, h);
    SHA512_ROUND(41, a, b, c, d, e, f, g, h);
    SHA512_ROUND(42, a, b, c, d, e, f, g, h);
    SHA512_ROUND(43, a, b, c, d, e, f, g, h);
    SHA512_ROUND(44, a, b, c, d, e, f, g, h);
    SHA512_ROUND(45, a, b, c, d, e, f, g, h);
    SHA512_ROUND(46, a, b, c, d, e, f, g, h);
    SHA512_ROUND(47, a, b, c, d, e, f, g, h);
    SHA512_ROUND(48, a, b, c, d, e, f, g, h);
    SHA512_ROUND(49, a, b, c, d, e, f, g, h);
    SHA512_ROUND(50, a, b, c, d, e, f, g, h);
    SHA512_ROUND(51, a, b, c, d, e, f, g, h);
    SHA512_ROUND(52, a, b, c, d, e, f, g, h);
    SHA512_ROUND(53, a, b, c, d, e, f, g, h);
    SHA512_ROUND(54, a, b, c, d, e, f, g, h);
    SHA512_ROUND(55, a, b, c, d, e, f, g, h);
    SHA512_ROUND(56, a, b, c, d, e, f, g, h);
    SHA512_ROUND(57, a, b, c, d, e, f, g, h);
    SHA512_ROUND(58, a, b, c, d, e, f, g, h);
    SHA512_ROUND(59, a, b, c, d, e, f, g, h);
    SHA512_ROUND(60, a, b, c, d, e, f, g, h);
    SHA512_ROUND(61, a, b, c, d, e, f, g, h);
    SHA512_ROUND(62, a, b, c, d, e, f, g, h);
    SHA512_ROUND(63, a, b, c, d, e, f, g, h);
    SHA512_ROUND(64, a, b, c, d, e, f, g, h);
    SHA512_ROUND(65, a, b, c, d, e, f, g, h);
    SHA512_ROUND(66, a, b, c, d, e, f, g, h);
    SHA512_ROUND(67, a, b, c, d, e, f, g, h);
    SHA512_ROUND(68, a, b, c, d, e, f, g, h);
    SHA512_ROUND(69, a, b, c, d, e, f, g, h);
    SHA512_ROUND(70, a, b, c, d, e, f, g, h);
    SHA512_ROUND(71, a, b, c, d, e, f, g, h);
    SHA512_ROUND(72, a, b, c, d, e, f, g, h);
    SHA512_ROUND(73, a, b, c, d, e, f, g, h);
    SHA512_ROUND(74, a, b, c, d, e, f, g, h);
    SHA512_ROUND(75, a, b, c, d, e, f, g, h);
    SHA512_ROUND(76, a, b, c, d, e, f, g, h);
    SHA512_ROUND(77, a, b, c, d, e, f, g, h);
    SHA512_ROUND(78, a, b, c, d, e, f, g, h);
    SHA512_ROUND(79, a, b, c, d, e, f, g, h);
}

void sha512_transform(std::uint64_t state[8], const std::uint8_t block[128]) {
    std::uint64_t w[80];
    sha512_expand(w, block);

    std::uint64_t a = state[0];
    std::uint64_t b = state[1];
    std::uint64_t c = state[2];
    std::uint64_t d = state[3];
    std::uint64_t e = state[4];
    std::uint64_t f = state[5];
    std::uint64_t g = state[6];
    std::uint64_t h = state[7];

    sha512_rounds(w, a, b, c, d, e, f, g, h);

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
    const std::uint8_t* ptr = data.data();
    const std::size_t full_blocks = data.size() / 64;
    for (std::size_t i = 0; i < full_blocks; ++i) {
        sha256_transform(state, ptr);
        ptr += 64;
    }

    std::array<std::uint8_t, 64> block{};
    const std::size_t rem = data.size() - full_blocks * 64;
    if (rem > 0) {
        std::memcpy(block.data(), ptr, rem);
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
    const std::uint8_t* ptr = data.data();
    const std::size_t full_blocks = data.size() / 128;
    for (std::size_t i = 0; i < full_blocks; ++i) {
        sha512_transform(state, ptr);
        ptr += 128;
    }

    std::array<std::uint8_t, 128> block{};
    const std::size_t rem = data.size() - full_blocks * 128;
    if (rem > 0) {
        std::memcpy(block.data(), ptr, rem);
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

constexpr std::uint8_t kAesSbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
};

constexpr std::uint8_t kAesInvSbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
};

constexpr std::uint8_t kAesRcon[11] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
};

inline std::uint8_t aes_xtime(std::uint8_t x) {
    return static_cast<std::uint8_t>(((x << 1) & 0xff) ^ (((x >> 7) & 1) * 0x1b));
}

inline std::uint8_t aes_mul(std::uint8_t x, std::uint8_t y) {
    return static_cast<std::uint8_t>(
        ((y & 1) * x) ^ ((y >> 1 & 1) * aes_xtime(x)) ^ ((y >> 2 & 1) * aes_xtime(aes_xtime(x))) ^
        ((y >> 3 & 1) * aes_xtime(aes_xtime(aes_xtime(x)))) ^
        ((y >> 4 & 1) * aes_xtime(aes_xtime(aes_xtime(aes_xtime(x))))));
}

inline void aes_sub_word(std::uint8_t word[4]) {
    for (int i = 0; i < 4; ++i) {
        word[i] = kAesSbox[word[i]];
    }
}

inline void aes_rot_word(std::uint8_t word[4]) {
    const std::uint8_t tmp = word[0];
    word[0] = word[1];
    word[1] = word[2];
    word[2] = word[3];
    word[3] = tmp;
}

void aes_key_expand(std::span<const std::uint8_t> key,
                    int rounds,
                    std::array<std::uint8_t, 240>& round_keys) {
    const int nk = static_cast<int>(key.size() / 4);
    const int nr = rounds;
    const int expanded_words = 4 * (nr + 1);

    std::memcpy(round_keys.data(), key.data(), key.size());

    std::uint8_t temp[4];
    int i = nk;
    while (i < expanded_words) {
        std::memcpy(temp, round_keys.data() + (i - 1) * 4, 4);
        if (i % nk == 0) {
            aes_rot_word(temp);
            aes_sub_word(temp);
            temp[0] ^= kAesRcon[i / nk];
        } else if (nk > 6 && i % nk == 4) {
            aes_sub_word(temp);
        }
        for (int j = 0; j < 4; ++j) {
            round_keys[i * 4 + j] =
                static_cast<std::uint8_t>(round_keys[(i - nk) * 4 + j] ^ temp[j]);
        }
        ++i;
    }
}

inline void aes_add_round_key(std::uint8_t state[16], const std::uint8_t* round_key, int round) {
    const int offset = round * 16;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            state[row * 4 + col] ^=
                round_key[offset + row * 4 + col];
        }
    }
}

inline void aes_sub_bytes(std::uint8_t state[16]) {
    for (int i = 0; i < 16; ++i) {
        state[i] = kAesSbox[state[i]];
    }
}

inline void aes_inv_sub_bytes(std::uint8_t state[16]) {
    for (int i = 0; i < 16; ++i) {
        state[i] = kAesInvSbox[state[i]];
    }
}

inline void aes_shift_rows(std::uint8_t state[16]) {
    std::uint8_t tmp;
    tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    tmp = state[2];
    state[2] = state[10];
    state[10] = tmp;
    tmp = state[6];
    state[6] = state[14];
    state[14] = tmp;

    tmp = state[3];
    state[3] = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = tmp;
}

inline void aes_inv_shift_rows(std::uint8_t state[16]) {
    std::uint8_t tmp;
    tmp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = tmp;

    tmp = state[2];
    state[2] = state[10];
    state[10] = tmp;
    tmp = state[6];
    state[6] = state[14];
    state[14] = tmp;

    tmp = state[3];
    state[3] = state[7];
    state[7] = state[11];
    state[11] = state[15];
    state[15] = tmp;
}

inline void aes_mix_columns(std::uint8_t state[16]) {
    for (int row = 0; row < 4; ++row) {
        const int i = row * 4;
        const std::uint8_t t = state[i];
        const std::uint8_t tmp =
            static_cast<std::uint8_t>(state[i] ^ state[i + 1] ^ state[i + 2] ^ state[i + 3]);
        std::uint8_t tm =
            static_cast<std::uint8_t>(state[i] ^ state[i + 1]);
        tm = aes_xtime(tm);
        state[i] ^= static_cast<std::uint8_t>(tm ^ tmp);
        tm = static_cast<std::uint8_t>(state[i + 1] ^ state[i + 2]);
        tm = aes_xtime(tm);
        state[i + 1] ^= static_cast<std::uint8_t>(tm ^ tmp);
        tm = static_cast<std::uint8_t>(state[i + 2] ^ state[i + 3]);
        tm = aes_xtime(tm);
        state[i + 2] ^= static_cast<std::uint8_t>(tm ^ tmp);
        tm = static_cast<std::uint8_t>(state[i + 3] ^ t);
        tm = aes_xtime(tm);
        state[i + 3] ^= static_cast<std::uint8_t>(tm ^ tmp);
    }
}

inline void aes_inv_mix_columns(std::uint8_t state[16]) {
    for (int row = 0; row < 4; ++row) {
        const int i = row * 4;
        const std::uint8_t a = state[i];
        const std::uint8_t b = state[i + 1];
        const std::uint8_t c = state[i + 2];
        const std::uint8_t d = state[i + 3];
        state[i] = static_cast<std::uint8_t>(aes_mul(a, 0x0e) ^ aes_mul(b, 0x0b) ^ aes_mul(c, 0x0d) ^
                                               aes_mul(d, 0x09));
        state[i + 1] = static_cast<std::uint8_t>(aes_mul(a, 0x09) ^ aes_mul(b, 0x0e) ^ aes_mul(c, 0x0b) ^
                                                   aes_mul(d, 0x0d));
        state[i + 2] = static_cast<std::uint8_t>(aes_mul(a, 0x0d) ^ aes_mul(b, 0x09) ^ aes_mul(c, 0x0e) ^
                                                   aes_mul(d, 0x0b));
        state[i + 3] = static_cast<std::uint8_t>(aes_mul(a, 0x0b) ^ aes_mul(b, 0x0d) ^ aes_mul(c, 0x09) ^
                                                   aes_mul(d, 0x0e));
    }
}

void aes_encrypt_block_impl(std::span<const std::uint8_t> key,
                            const std::uint8_t in[16],
                            std::uint8_t out[16]) {
    const int rounds = (key.size() == 16) ? 10 : 14;
    std::array<std::uint8_t, 240> round_keys{};
    aes_key_expand(key, rounds, round_keys);

    std::uint8_t state[16];
    std::memcpy(state, in, 16);
    aes_add_round_key(state, round_keys.data(), 0);

    for (int round = 1; round < rounds; ++round) {
        aes_sub_bytes(state);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, round_keys.data(), round);
    }

    aes_sub_bytes(state);
    aes_shift_rows(state);
    aes_add_round_key(state, round_keys.data(), rounds);
    std::memcpy(out, state, 16);
}

void aes_decrypt_block_impl(std::span<const std::uint8_t> key,
                            const std::uint8_t in[16],
                            std::uint8_t out[16]) {
    const int rounds = (key.size() == 16) ? 10 : 14;
    std::array<std::uint8_t, 240> round_keys{};
    aes_key_expand(key, rounds, round_keys);

    std::uint8_t state[16];
    std::memcpy(state, in, 16);
    aes_add_round_key(state, round_keys.data(), rounds);

    for (int round = rounds - 1; round >= 0; --round) {
        aes_inv_shift_rows(state);
        aes_inv_sub_bytes(state);
        aes_add_round_key(state, round_keys.data(), round);
        if (round == 0) {
            break;
        }
        aes_inv_mix_columns(state);
    }

    std::memcpy(out, state, 16);
}

std::vector<std::uint8_t> aes_encrypt_block(std::span<const std::uint8_t> key,
                                            std::span<const std::uint8_t> block) {
    if (block.size() != 16 || (key.size() != 16 && key.size() != 32)) {
        return {};
    }
    std::vector<std::uint8_t> out(16);
    aes_encrypt_block_impl(key, block.data(), out.data());
    return out;
}

std::vector<std::uint8_t> aes_cbc_encrypt(std::span<const std::uint8_t> key,
                                          std::span<const std::uint8_t> iv,
                                          std::span<const std::uint8_t> plaintext) {
    if (iv.size() != 16 || (key.size() != 16 && key.size() != 32)) {
        return {};
    }

    const std::size_t pad_len = 16 - (plaintext.size() % 16);
    std::vector<std::uint8_t> padded(plaintext.size() + pad_len);
    if (!plaintext.empty()) {
        std::memcpy(padded.data(), plaintext.data(), plaintext.size());
    }
    std::memset(padded.data() + plaintext.size(), static_cast<int>(pad_len), pad_len);

    std::vector<std::uint8_t> out(padded.size());
    std::array<std::uint8_t, 16> chain{};
    std::memcpy(chain.data(), iv.data(), 16);

    for (std::size_t offset = 0; offset < padded.size(); offset += 16) {
        std::array<std::uint8_t, 16> block{};
        for (int i = 0; i < 16; ++i) {
            block[static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>(padded[offset + static_cast<std::size_t>(i)] ^ chain[i]);
        }
        aes_encrypt_block_impl(key, block.data(), out.data() + offset);
        std::memcpy(chain.data(), out.data() + offset, 16);
    }
    return out;
}

std::vector<std::uint8_t> aes_cbc_decrypt(std::span<const std::uint8_t> key,
                                          std::span<const std::uint8_t> iv,
                                          std::span<const std::uint8_t> ciphertext) {
    if (iv.size() != 16 || (key.size() != 16 && key.size() != 32) ||
        ciphertext.empty() || ciphertext.size() % 16 != 0) {
        return {};
    }

    std::vector<std::uint8_t> out(ciphertext.size());
    std::array<std::uint8_t, 16> chain{};
    std::memcpy(chain.data(), iv.data(), 16);

    for (std::size_t offset = 0; offset < ciphertext.size(); offset += 16) {
        std::array<std::uint8_t, 16> plain{};
        aes_decrypt_block_impl(key, ciphertext.data() + offset, plain.data());
        for (int i = 0; i < 16; ++i) {
            out[offset + static_cast<std::size_t>(i)] =
                static_cast<std::uint8_t>(plain[static_cast<std::size_t>(i)] ^ chain[i]);
        }
        std::memcpy(chain.data(), ciphertext.data() + offset, 16);
    }

    const std::uint8_t pad_len = out.back();
    if (pad_len == 0 || pad_len > 16) {
        return {};
    }
    for (std::size_t i = out.size() - pad_len; i < out.size(); ++i) {
        if (out[i] != pad_len) {
            return {};
        }
    }
    out.resize(out.size() - pad_len);
    return out;
}

#undef SHA256_SIGMA0
#undef SHA256_SIGMA1
#undef SHA256_S0
#undef SHA256_S1
#undef SHA256_CH
#undef SHA256_MAJ
#undef SHA256_ROUND
#undef SHA512_SIGMA0
#undef SHA512_SIGMA1
#undef SHA512_S0
#undef SHA512_S1
#undef SHA512_CH
#undef SHA512_MAJ
#undef SHA512_ROUND

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

std::vector<uint8_t> aes128_encrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block) {
    if (key.size() != aes128_key_size) {
        return {};
    }
    return aes_encrypt_block(key, block);
}

std::vector<uint8_t> aes256_encrypt_block(std::span<const uint8_t> key,
                                          std::span<const uint8_t> block) {
    if (key.size() != aes256_key_size) {
        return {};
    }
    return aes_encrypt_block(key, block);
}

std::vector<uint8_t> aes128_cbc_encrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> plaintext) {
    if (key.size() != aes128_key_size) {
        return {};
    }
    return aes_cbc_encrypt(key, iv, plaintext);
}

std::vector<uint8_t> aes128_cbc_decrypt(std::span<const uint8_t> key,
                                        std::span<const uint8_t> iv,
                                        std::span<const uint8_t> ciphertext) {
    if (key.size() != aes128_key_size) {
        return {};
    }
    return aes_cbc_decrypt(key, iv, ciphertext);
}

} // namespace crypto
} // namespace ms
