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

} // namespace crypto
} // namespace ms
