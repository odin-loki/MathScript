#include "ms/numthy/numthy.hpp"
#include "ms/error/error_types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ms {
namespace numthy {

// --- Miller-Rabin primality test (deterministic for n < 3,317,044,064,679,887,385,961,981) ---
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
#if defined(__SIZEOF_INT128__)
    return static_cast<__uint128_t>(a) * b % m;
#elif defined(_MSC_VER)
    unsigned __int64 hi = 0;
    const unsigned __int64 lo = _umul128(a, b, &hi);
    uint64_t rem = 0;
    for (int i = 63; i >= 0; --i) {
        rem = (rem << 1) | ((hi >> i) & 1);
        if (rem >= m) rem -= m;
    }
    for (int i = 63; i >= 0; --i) {
        rem = (rem << 1) | ((lo >> i) & 1);
        if (rem >= m) rem -= m;
    }
    return rem;
#else
    uint64_t res = 0;
    a %= m;
    while (b > 0) {
        if (b & 1) {
            res += a;
            if (res >= m) res -= m;
        }
        a = (a << 1) % m;
        b >>= 1;
    }
    return res;
#endif
}

static uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = mulmod(result, base, mod);
        base = mulmod(base, base, mod);
        exp >>= 1;
    }
    return result;
}

static bool miller_rabin_witness(uint64_t n, uint64_t a, uint64_t d, int r) {
    uint64_t x = powmod(a, d, n);
    if (x == 1 || x == n - 1) return false;
    for (int i = 0; i < r - 1; ++i) {
        x = mulmod(x, x, n);
        if (x == n - 1) return false;
    }
    return true; // composite
}

bool isprime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    uint64_t d = n - 1;
    int r = 0;
    while (d % 2 == 0) { d /= 2; ++r; }

    // Deterministic witnesses for all n < 3.3e24
    for (uint64_t a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
        if (a >= n) continue;
        if (miller_rabin_witness(n, a, d, r)) return false;
    }
    return true;
}

uint64_t nextprime(uint64_t n) {
    if (n < 2) return 2;
    uint64_t c = (n % 2 == 0) ? n + 1 : n + 2;
    while (!isprime(c)) c += 2;
    return c;
}

uint64_t prevprime(uint64_t n) {
    if (n <= 2) return 0;
    if (n == 3) return 2;
    uint64_t c = (n % 2 == 0) ? n - 1 : n - 2;
    while (c > 2 && !isprime(c)) c -= 2;
    return isprime(c) ? c : 0;
}

std::vector<uint64_t> primes(uint64_t lo, uint64_t hi) {
    if (hi < lo || hi < 2) return {};
    if (lo < 2) lo = 2;
    // Segmented sieve
    std::vector<bool> sieve(hi - lo + 1, true);
    if (lo == 1) sieve[0] = false;
    uint64_t sqrthi = static_cast<uint64_t>(std::sqrt(static_cast<double>(hi))) + 1;
    for (uint64_t p = 2; p <= sqrthi; ++p) {
        if (!isprime(p)) continue;
        uint64_t start = ((lo + p - 1) / p) * p;
        if (start == p) start += p;
        for (uint64_t j = start; j <= hi; j += p)
            if (j >= lo) sieve[j - lo] = false;
    }
    std::vector<uint64_t> result;
    for (uint64_t i = lo; i <= hi; ++i)
        if (sieve[i - lo]) result.push_back(i);
    return result;
}

uint64_t prime_pi(uint64_t n) {
    auto ps = primes(2, n);
    return static_cast<uint64_t>(ps.size());
}

uint64_t prime_nth(uint64_t n) {
    if (n == 0) return 0;
    uint64_t c = 0, p = 1;
    while (c < n) { p = nextprime(p); ++c; }
    return p;
}

// --- Pollard rho factorisation ---
static uint64_t pollard_rho(uint64_t n) {
    if (n % 2 == 0) return 2;
    uint64_t x = 2, y = 2, c = 1, d = 1;
    while (d == 1) {
        x = (mulmod(x, x, n) + c) % n;
        y = (mulmod(y, y, n) + c) % n;
        y = (mulmod(y, y, n) + c) % n;
        d = std::gcd(x > y ? x - y : y - x, n);
    }
    return (d != n) ? d : 0; // 0 = retry needed
}

static void factor_recursive(uint64_t n, std::vector<uint64_t>& factors) {
    if (n <= 1) return;
    if (isprime(n)) { factors.push_back(n); return; }
    // Trial division up to small primes
    for (uint64_t p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL}) {
        if (n % p == 0) {
            while (n % p == 0) { factors.push_back(p); n /= p; }
            if (n > 1) factor_recursive(n, factors);
            return;
        }
    }
    uint64_t d = 0;
    uint64_t c = 1;
    while (d == 0 || d == n) {
        d = pollard_rho(n);
        ++c;
        if (c > 20) {
            // fallback trial
            for (uint64_t i = 17; i * i <= n; i += 2) {
                if (n % i == 0) { d = i; break; }
            }
            if (d == 0 || d == n) { factors.push_back(n); return; }
            break;
        }
    }
    factor_recursive(d, factors);
    factor_recursive(n / d, factors);
}

std::vector<uint64_t> factor(uint64_t n) {
    std::vector<uint64_t> factors;
    factor_recursive(n, factors);
    std::sort(factors.begin(), factors.end());
    return factors;
}

std::vector<std::pair<uint64_t,int>> factor_exp(uint64_t n) {
    auto f = factor(n);
    std::vector<std::pair<uint64_t,int>> result;
    for (size_t i = 0; i < f.size(); ) {
        size_t j = i;
        while (j < f.size() && f[j] == f[i]) ++j;
        result.push_back({f[i], static_cast<int>(j - i)});
        i = j;
    }
    return result;
}

std::vector<uint64_t> divisors(uint64_t n) {
    std::vector<uint64_t> divs;
    for (uint64_t i = 1; i * i <= n; ++i) {
        if (n % i == 0) {
            divs.push_back(i);
            if (i != n / i) divs.push_back(n / i);
        }
    }
    std::sort(divs.begin(), divs.end());
    return divs;
}

uint64_t num_divisors(uint64_t n) {
    auto fe = factor_exp(n);
    uint64_t tau = 1;
    for (auto& [p, e] : fe) tau *= (e + 1);
    return tau;
}

uint64_t sum_divisors(uint64_t n) {
    auto d = divisors(n);
    uint64_t s = 0;
    for (auto x : d) s += x;
    return s;
}

uint64_t euler_phi(uint64_t n) {
    auto fe = factor_exp(n);
    uint64_t result = n;
    for (auto& [p, e] : fe) {
        result -= result / p;
    }
    return result;
}

static bool pow_u64(uint64_t base, uint32_t exp, uint64_t& out) {
#if defined(__SIZEOF_INT128__)
    __uint128_t result = 1;
    __uint128_t b = base;
    uint32_t e = exp;
    while (e > 0) {
        if (e & 1) {
            result *= b;
            if (result > UINT64_MAX) return false;
        }
        if (e > 1) {
            b *= b;
            if (b > UINT64_MAX) return false;
        }
        e >>= 1;
    }
    out = static_cast<uint64_t>(result);
    return true;
#else
    uint64_t result = 1;
    uint64_t b = base;
    uint32_t e = exp;
    while (e > 0) {
        if (e & 1) {
            if (b != 0 && result > UINT64_MAX / b) return false;
            result *= b;
        }
        if (e > 1) {
            if (b > 0 && b > UINT64_MAX / b) return false;
            b *= b;
        }
        e >>= 1;
    }
    out = result;
    return true;
#endif
}

uint64_t jordan_totient(uint32_t k, uint64_t n) {
    if (n == 0 || k == 0) return 0;
    if (n == 1) return 1;
    auto fe = factor_exp(n);
#if defined(__SIZEOF_INT128__)
    __uint128_t result = 1;
    for (auto& [p, e] : fe) {
        uint64_t pk_e;
        if (!pow_u64(p, k * static_cast<uint32_t>(e), pk_e)) return 0;
        uint64_t pk_e1;
        if (k * static_cast<uint32_t>(e) <= 1) {
            pk_e1 = 1;
        } else if (!pow_u64(p, k * static_cast<uint32_t>(e) - 1, pk_e1)) {
            return 0;
        }
        if (pk_e < pk_e1) return 0;
        __uint128_t term = static_cast<__uint128_t>(pk_e) - pk_e1;
        result *= term;
        if (result > UINT64_MAX) return 0;
    }
    return static_cast<uint64_t>(result);
#else
    uint64_t result = 1;
    for (auto& [p, e] : fe) {
        uint64_t pk_e;
        if (!pow_u64(p, k * static_cast<uint32_t>(e), pk_e)) return 0;
        uint64_t pk_e1;
        if (k * static_cast<uint32_t>(e) <= 1) {
            pk_e1 = 1;
        } else if (!pow_u64(p, k * static_cast<uint32_t>(e) - 1, pk_e1)) {
            return 0;
        }
        if (pk_e < pk_e1) return 0;
        uint64_t term = pk_e - pk_e1;
        if (result > UINT64_MAX / term) return 0;
        result *= term;
    }
    return result;
#endif
}

int64_t mobius(uint64_t n) {
    auto fe = factor_exp(n);
    for (auto& [p, e] : fe)
        if (e > 1) return 0;
    return (fe.size() % 2 == 0) ? 1 : -1;
}

int liouville(uint64_t n) {
    auto f = factor(n);
    return (f.size() % 2 == 0) ? 1 : -1;
}

double von_mangoldt(uint64_t n) {
    if (n <= 1) return 0.0;
    auto fe = factor_exp(n);
    if (fe.size() != 1) return 0.0;
    return std::log(static_cast<double>(fe[0].first));
}

uint64_t gcd(uint64_t a, uint64_t b) {
    return std::gcd(a, b);
}

uint64_t lcm(uint64_t a, uint64_t b) {
    return std::lcm(a, b);
}

std::tuple<int64_t,int64_t,int64_t> extended_gcd(int64_t a, int64_t b) {
    int64_t old_r = a, r = b;
    int64_t old_s = 1, s = 0;
    int64_t old_t = 0, t = 1;
    while (r != 0) {
        int64_t q = old_r / r;
        int64_t tmp = r; r = old_r - q * r; old_r = tmp;
        tmp = s; s = old_s - q * s; old_s = tmp;
        tmp = t; t = old_t - q * t; old_t = tmp;
    }
    return {old_r, old_s, old_t};
}

Result<uint64_t> mod_inv(uint64_t a, uint64_t m) {
    auto [g, x, y] = extended_gcd(static_cast<int64_t>(a),
                                   static_cast<int64_t>(m));
    if (g != 1)
        return std::unexpected(Error{DomainError{"mod_inv", "gcd(a,m) != 1"}});
    return static_cast<uint64_t>((x % static_cast<int64_t>(m) + m) % m);
}

uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    return powmod(base, exp, mod);
}

Result<uint64_t> crt(const std::vector<uint64_t>& r,
                     const std::vector<uint64_t>& m) {
    if (r.size() != m.size() || r.empty())
        return std::unexpected(Error{DomainError{"crt", "bad input sizes"}});
    uint64_t x = r[0], M = m[0];
    for (size_t i = 1; i < r.size(); ++i) {
        auto inv = mod_inv(M % m[i], m[i]);
        if (!inv) return std::unexpected(inv.error());
        uint64_t t = (r[i] + m[i] - x % m[i]) % m[i] * inv.value() % m[i];
        x = x + M * t;
        M *= m[i];
        x %= M;
    }
    return x;
}

int legendre_symbol(int64_t a, uint64_t p) {
    if (p < 2 || !isprime(p)) return 0;
    int64_t mod = static_cast<int64_t>(p);
    a = ((a % mod) + mod) % mod;
    if (a == 0) return 0;
    uint64_t ls = powmod(static_cast<uint64_t>(a), (p - 1) / 2, p);
    return (ls == p - 1) ? -1 : static_cast<int>(ls);
}

std::vector<uint64_t> quadratic_residues(uint64_t p) {
    if (p <= 2 || !isprime(p)) return {};
    std::set<uint64_t> residues;
    for (uint64_t a = 1; a < p; ++a)
        residues.insert(mulmod(a, a, p));
    return {residues.begin(), residues.end()};
}

int jacobi_symbol(int64_t a, uint64_t n) {
    if (n == 0) {
        return 0;
    }
    if (n % 2 == 0) {
        return 0;
    }
    int result = 1;
    int64_t aa = ((a % static_cast<int64_t>(n)) + static_cast<int64_t>(n)) % static_cast<int64_t>(n);
    uint64_t nn = n;
    while (aa != 0) {
        while (aa % 2 == 0) {
            aa /= 2;
            const uint64_t r8 = nn % 8;
            if (r8 == 3 || r8 == 5) {
                result = -result;
            }
        }
        if (aa % 4 == 3 && nn % 4 == 3) {
            result = -result;
        }
        const int64_t t = aa;
        aa = static_cast<int64_t>(nn % static_cast<uint64_t>(aa));
        nn = static_cast<uint64_t>(t);
    }
    return nn == 1 ? result : 0;
}

int kronecker_symbol(int64_t a, int64_t n) {
    if (n == 0) return (a == 1 || a == -1) ? 1 : 0;
    if (n == 1) return 1;
    if (n == -1) return (a < 0) ? -1 : 1;
    if (n < 0) return kronecker_symbol(a, -n) * ((a < 0) ? -1 : 1);
    return jacobi_symbol(a, static_cast<uint64_t>(n));
}

Result<uint64_t> discrete_log(uint64_t g, uint64_t h, uint64_t p) {
    // Baby-step giant-step
    uint64_t m = static_cast<uint64_t>(std::ceil(std::sqrt(static_cast<double>(p)))) + 1;
    std::unordered_map<uint64_t,uint64_t> table;
    uint64_t gj = 1;
    for (uint64_t j = 0; j < m; ++j) {
        table[gj] = j;
        gj = mulmod(gj, g, p);
    }
    uint64_t gm_inv = powmod(powmod(g, m, p), p - 2, p); // g^{-m} mod p
    uint64_t cur = h;
    for (uint64_t i = 0; i <= m; ++i) {
        if (table.count(cur)) {
            uint64_t x = i * m + table[cur];
            if (powmod(g, x, p) == h) return x;
        }
        cur = mulmod(cur, gm_inv, p);
    }
    return std::unexpected(Error{DomainError{"discrete_log", "no solution"}});
}

bool is_primitive_root(uint64_t g, uint64_t p) {
    if (!isprime(p)) return false;
    uint64_t phi = p - 1;
    auto fe = factor_exp(phi);
    for (auto& [q, e] : fe)
        if (powmod(g, phi / q, p) == 1) return false;
    return true;
}

uint64_t primitive_root(uint64_t p) {
    for (uint64_t g = 2; g < p; ++g)
        if (is_primitive_root(g, p)) return g;
    return 0;
}

Result<uint64_t> tonelli_shanks(uint64_t n, uint64_t p) {
    if (legendre_symbol(static_cast<int64_t>(n), p) != 1)
        return std::unexpected(Error{DomainError{"tonelli_shanks", "n is not a QR mod p"}});
    if (n == 0) return 0ULL;
    if (p == 2) return n % 2;
    if (p % 4 == 3) {
        return powmod(n, (p + 1) / 4, p);
    }
    uint64_t Q = p - 1, S = 0;
    while (Q % 2 == 0) { Q /= 2; ++S; }
    uint64_t z = 2;
    while (legendre_symbol(static_cast<int64_t>(z), p) != -1) ++z;
    uint64_t M = S, c = powmod(z, Q, p), t = powmod(n, Q, p),
             R = powmod(n, (Q + 1) / 2, p);
    while (true) {
        if (t == 1) return R;
        uint64_t i = 1, tmp = mulmod(t, t, p);
        while (tmp != 1) { tmp = mulmod(tmp, tmp, p); ++i; }
        uint64_t b = c;
        for (uint64_t j = 0; j < M - i - 1; ++j) b = mulmod(b, b, p);
        M = i; c = mulmod(b, b, p); t = mulmod(t, c, p);
        R = mulmod(R, b, p);
    }
}

std::vector<int64_t> continued_fraction(double x, int max_terms) {
    std::vector<int64_t> cf;
    for (int i = 0; i < max_terms; ++i) {
        int64_t a = static_cast<int64_t>(std::floor(x));
        cf.push_back(a);
        double frac = x - a;
        if (std::abs(frac) < 1e-10) break;
        x = 1.0 / frac;
    }
    return cf;
}

std::vector<std::pair<int64_t,int64_t>> convergents(
    const std::vector<int64_t>& cf) {
    std::vector<std::pair<int64_t,int64_t>> conv;
    int64_t h_prev = 1, h_curr = cf[0];
    int64_t k_prev = 0, k_curr = 1;
    conv.push_back({h_curr, k_curr});
    for (size_t i = 1; i < cf.size(); ++i) {
        int64_t h_next = cf[i] * h_curr + h_prev;
        int64_t k_next = cf[i] * k_curr + k_prev;
        conv.push_back({h_next, k_next});
        h_prev = h_curr; h_curr = h_next;
        k_prev = k_curr; k_curr = k_next;
    }
    return conv;
}

static uint64_t isqrt_u64(uint64_t n) {
    uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(n)));
    while ((r + 1) <= n / (r + 1)) ++r;
    while (r > 0 && r > n / r) --r;
    return r;
}

struct u128 {
    uint64_t hi;
    uint64_t lo;
};

static u128 u128_mul64(uint64_t a, uint64_t b) {
#if defined(_MSC_VER)
    u128 r;
    r.lo = _umul128(a, b, &r.hi);
    return r;
#elif defined(__SIZEOF_INT128__)
    const __uint128_t v = static_cast<__uint128_t>(a) * b;
    return {static_cast<uint64_t>(v >> 64), static_cast<uint64_t>(v)};
#else
    u128 r{};
    r.lo = a * b;
    return r;
#endif
}

static u128 u128_mul_u128_u64(u128 a, uint64_t b) {
#if defined(_MSC_VER)
    unsigned __int64 mid;
    const unsigned __int64 lo = _umul128(a.lo, b, &mid);
    u128 r;
    r.lo = lo;
    r.hi = a.hi * b + mid;
    return r;
#elif defined(__SIZEOF_INT128__)
    const __uint128_t v = (static_cast<__uint128_t>(a.hi) << 64 | a.lo) * b;
    return {static_cast<uint64_t>(v >> 64), static_cast<uint64_t>(v)};
#else
    return u128_mul64(a.lo, b);
#endif
}

static bool u128_sub_u128(u128 a, u128 b, u128& out) {
    const bool borrow = a.lo < b.lo;
    out.lo = a.lo - b.lo;
    out.hi = a.hi - b.hi - (borrow ? 1u : 0u);
    return a.hi >= b.hi + (borrow ? 1u : 0u);
}

static bool pell_norm_is_one(int64_t p, int64_t q, uint64_t D) {
    if (q <= 0) return false;
    const uint64_t ap = static_cast<uint64_t>(std::llabs(p));
    const uint64_t aq = static_cast<uint64_t>(q);
    const u128 p2 = u128_mul64(ap, ap);
    const u128 q2 = u128_mul64(aq, aq);
    const u128 dq2 = u128_mul_u128_u64(q2, D);
    u128 diff{};
    if (!u128_sub_u128(p2, dq2, diff)) return false;
    return diff.hi == 0 && diff.lo == 1;
}

static std::vector<int64_t> sqrt_cf_period(uint64_t D) {
    const uint64_t a0 = isqrt_u64(D);
    std::vector<int64_t> cf;
    cf.push_back(static_cast<int64_t>(a0));

    uint64_t m = 0, d = 1, a = a0;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> seen;

    while (true) {
        m = d * a - m;
        d = (D - m * m) / d;
        a = (a0 + m) / d;
        const uint64_t key = (m << 32) ^ d;
        if (seen.count(key)) break;
        seen[key] = {m, d};
        cf.push_back(static_cast<int64_t>(a));
    }
    return cf;
}

std::vector<std::pair<uint64_t,uint64_t>> farey(uint32_t n) {
    if (n == 0) return {};
    std::vector<std::pair<uint64_t,uint64_t>> fracs;
    for (uint64_t q = 1; q <= n; ++q) {
        for (uint64_t p = 0; p <= q; ++p) {
            if (gcd(p, q) == 1)
                fracs.push_back({p, q});
        }
    }
    std::sort(fracs.begin(), fracs.end(),
        [](const auto& lhs, const auto& rhs) {
#if defined(__SIZEOF_INT128__)
            __uint128_t left = static_cast<__uint128_t>(lhs.first) * rhs.second;
            __uint128_t right = static_cast<__uint128_t>(rhs.first) * lhs.second;
            return left < right;
#else
            return lhs.first * rhs.second < rhs.first * lhs.second;
#endif
        });
    return fracs;
}

std::vector<std::pair<uint64_t,uint64_t>> stern_brocot(uint64_t n) {
    std::vector<std::pair<uint64_t,uint64_t>> result;
    if (n == 0) return result;
    std::queue<std::pair<uint64_t,uint64_t>> frontier;
    frontier.push({1, 1});
    while (!frontier.empty() && result.size() < n) {
        auto [p, q] = frontier.front();
        frontier.pop();
        result.push_back({p, q});
        if (result.size() >= n) break;
        frontier.push({p, p + q});
        frontier.push({p + q, q});
    }
    return result;
}

Result<std::pair<uint64_t,uint64_t>> pell_solve(uint64_t D) {
    if (D == 0)
        return std::unexpected(Error{DomainError{"pell_solve", "D == 0"}});
    const uint64_t s = isqrt_u64(D);
    if (s * s == D)
        return std::unexpected(Error{DomainError{"pell_solve", "D is a perfect square"}});

    auto period_cf = sqrt_cf_period(D);
    if (period_cf.size() < 2)
        return std::unexpected(Error{DomainError{"pell_solve", "invalid CF"}});

    std::vector<int64_t> cf = period_cf;
    cf.insert(cf.end(), period_cf.begin() + 1, period_cf.end());

    auto conv = convergents(cf);
    for (size_t i = 1; i < conv.size(); ++i) {
        const int64_t p = conv[i].first;
        const int64_t q = conv[i].second;
        if (pell_norm_is_one(p, q, D))
            return std::make_pair(static_cast<uint64_t>(std::llabs(p)),
                                  static_cast<uint64_t>(q));
    }
    return std::unexpected(Error{DomainError{"pell_solve", "no solution found"}});
}

// b*b > bound, computed without risking uint64_t overflow for large b.
static bool sq_exceeds(uint64_t b, uint64_t bound) {
    const u128 b2 = u128_mul64(b, b);
    return b2.hi != 0 || b2.lo > bound;
}

Result<std::pair<uint64_t,uint64_t>> cornacchia(uint64_t d, uint64_t p) {
    if (!isprime(p))
        return std::unexpected(Error{DomainError{"cornacchia", "p is not prime"}});
    if (d == 0)
        return std::unexpected(Error{DomainError{"cornacchia", "d == 0"}});
    if (d >= p)
        return std::unexpected(Error{DomainError{"cornacchia", "d >= p"}});
    if (p == 2)
        return std::make_pair(1ULL, 1ULL); // 0 < d < 2 forces d == 1; 1^2 + 1*1^2 = 2

    // Solve t^2 ≡ -d ≡ (p - d) (mod p).
    auto troot = tonelli_shanks(p - d, p);
    if (!troot)
        return std::unexpected(Error{DomainError{"cornacchia", "-d is not a quadratic residue mod p; no solution"}});

    uint64_t t = troot.value();
    if (t > p - t) t = p - t;

    // Partial Euclidean algorithm on (p, t) until the remainder b satisfies b^2 <= p.
    uint64_t a = p, b = t;
    while (sq_exceeds(b, p)) {
        uint64_t r = a % b;
        a = b;
        b = r;
    }

    const u128 b2 = u128_mul64(b, b);
    if (b2.hi != 0 || b2.lo > p)
        return std::unexpected(Error{DomainError{"cornacchia", "no solution"}});
    const uint64_t rem = p - b2.lo;
    if (rem % d != 0)
        return std::unexpected(Error{DomainError{"cornacchia", "no solution"}});
    const uint64_t c2 = rem / d;
    const uint64_t c = isqrt_u64(c2);
    if (c * c != c2)
        return std::unexpected(Error{DomainError{"cornacchia", "no solution"}});

    return std::make_pair(b, c);
}

bool is_carmichael(uint64_t n) {
    if (n < 3) return false;
    if (isprime(n)) return false;
    auto fe = factor_exp(n);
    if (fe.size() < 3) return false; // Carmichael numbers have >= 3 distinct prime factors
    for (auto& [p, e] : fe) {
        if (e != 1) return false; // not squarefree
        if ((n - 1) % (p - 1) != 0) return false; // Korselt's criterion
    }
    return true;
}

namespace {
struct Mat2 { int64_t a, b, c, d; }; // [[a,b],[c,d]]

Mat2 mat2_mul(const Mat2& x, const Mat2& y) {
    return {x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d,
            x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
}

// M^k for M = [[P,-Q],[1,0]] via binary exponentiation, analogous to mod_pow's approach.
Mat2 mat2_pow(int64_t P, int64_t Q, int64_t k) {
    Mat2 result{1, 0, 0, 1}; // identity
    Mat2 base{P, -Q, 1, 0};
    while (k > 0) {
        if (k & 1) result = mat2_mul(result, base);
        base = mat2_mul(base, base);
        k >>= 1;
    }
    return result;
}
} // namespace

std::pair<int64_t, int64_t> lucas_sequence(int64_t k, int64_t P, int64_t Q) {
    if (k < 0) k = 0; // negative indices need division; clamp defensively (documented in header)
    // [U_{k+1}; U_k] = M^k * [1; 0], M = [[P,-Q],[1,0]]
    Mat2 mk = mat2_pow(P, Q, k);
    int64_t u_next = mk.a; // U_{k+1}
    int64_t u_k = mk.c;    // U_k
    int64_t v_k = 2 * u_next - P * u_k; // V_k = 2*U_{k+1} - P*U_k
    return {u_k, v_k};
}

uint64_t partition(uint32_t n) {
    // Dynamic programming via Euler's pentagonal theorem
    std::vector<uint64_t> p(n + 1, 0);
    p[0] = 1;
    for (uint32_t i = 1; i <= n; ++i) {
        // pentagonal numbers: k*(3k-1)/2 for k = 1,-1,2,-2,...
        int64_t k = 1;
        while (true) {
            int64_t pent1 = k * (3 * k - 1) / 2;
            int64_t pent2 = k * (3 * k + 1) / 2;
            if (pent1 > static_cast<int64_t>(i) && pent2 > static_cast<int64_t>(i)) break;
            if (pent1 <= static_cast<int64_t>(i))
                p[i] += ((k % 2 == 1) ? 1 : -1) * p[i - pent1];
            if (pent2 <= static_cast<int64_t>(i))
                p[i] += ((k % 2 == 1) ? 1 : -1) * p[i - pent2];
            ++k;
        }
    }
    return p[n];
}

} // namespace numthy
} // namespace ms
