#include "ms/bignum/bignum.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>

namespace ms {
namespace bignum {

namespace {

int digit_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    return -1;
}

char digit_char(int d) {
    if (d < 10) return static_cast<char>('0' + d);
    return static_cast<char>('a' + d - 10);
}

} // namespace

// ========================== BigInt ==========================

BigInt::BigInt(long long v) {
    negative = v < 0;
    unsigned long long uv = static_cast<unsigned long long>(v);
    if (negative) {
        uv = 0ull - uv;
    }
    digits.clear();
    if (uv == 0) { digits.push_back(0); return; }
    while (uv > 0) { digits.push_back(uv % BASE); uv /= BASE; }
}

// Defensive decimal constructor: unparsable input yields zero, per the header. It used
// std::stoul on each 9-digit chunk, which THROWS std::invalid_argument on a non-numeric
// chunk -- and this library is built with -fno-exceptions, so BigInt("x") called
// std::terminate and aborted the process instead of constructing zero. The digits are
// accumulated by hand now, and any non-digit character abandons the whole parse.
BigInt::BigInt(const std::string& s) {
    std::string str = s;
    negative = false;
    if (!str.empty() && str[0] == '-') { negative = true; str = str.substr(1); }
    if (!str.empty() && str[0] == '+') str = str.substr(1);
    digits.clear();
    for (char c : str) {
        if (c < '0' || c > '9') { str.clear(); break; }
    }
    digits.reserve((str.size() + 8) / 9);
    // Parse from the right in chunks of 9 decimal digits (BASE == 1e9).
    int i = (int)str.size();
    while (i > 0) {
        int start = std::max(0, i-9);
        uint32_t chunk = 0;
        for (int k = start; k < i; ++k) {
            chunk = chunk * 10u + static_cast<uint32_t>(str[static_cast<size_t>(k)] - '0');
        }
        digits.push_back(chunk);
        i = start;
    }
    if (digits.empty()) digits.push_back(0);
    trim();
    if (is_zero()) negative = false;
}

Result<BigInt> BigInt::parse(const std::string& s, int base) {
    if (base < 2 || base > 36)
        return std::unexpected(DomainError{"BigInt::parse", "base must be in [2, 36]"});
    std::string str = s;
    bool is_neg = false;
    if (!str.empty() && str[0] == '-') { is_neg = true; str = str.substr(1); }
    if (!str.empty() && str[0] == '+') str = str.substr(1);
    BigInt result(0LL);
    // "" and a lone sign are not numerals. The defensive BigInt(string) constructor
    // turns malformed input into zero on purpose; parse() is the checked entry point
    // and has to say so, as APFloat::parse already does for an empty significand.
    if (str.empty())
        return std::unexpected(DomainError{"BigInt::parse", "empty significand"});

    BigInt bbase(static_cast<long long>(base));
    for (char c : str) {
        int d = digit_value(c);
        if (d < 0 || d >= base)
            return std::unexpected(DomainError{"BigInt::parse", "invalid digit for base"});
        result = result * bbase + BigInt(static_cast<long long>(d));
    }
    result.negative = is_neg && !result.is_zero();
    return result;
}

Result<BigInt> BigInt::parse(const std::string& s) {
    return parse(s, 10);
}

BigInt::BigInt(const std::string& s, int base) : digits(1, 0), negative(false) {
    if (auto r = parse(s, base))
        *this = std::move(*r);
}

void BigInt::trim() {
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
}

bool BigInt::is_zero() const { return digits.size()==1 && digits[0]==0; }
bool BigInt::is_one()  const { return digits.size()==1 && digits[0]==1 && !negative; }

std::string BigInt::to_string() const {
    if (is_zero()) return "0";
    std::string s;
    s.reserve((negative ? 1 : 0) + digits.size() * 9);
    if (negative) s += '-';
    s += std::to_string(digits.back());
    for (int i=(int)digits.size()-2; i>=0; --i) {
        std::string chunk = std::to_string(digits[i]);
        s += std::string(9 - chunk.size(), '0') + chunk;
    }
    return s;
}

std::string BigInt::to_string(int base) const {
    if (base < 2 || base > 36)
        return {};
    if (base == 10) return to_string();
    if (is_zero()) return "0";

    BigInt n = negative ? -*this : *this;
    BigInt bbase(static_cast<long long>(base));
    std::string digits_out;
    while (!n.is_zero()) {
        BigInt rem = n % bbase;
        digits_out.push_back(digit_char(static_cast<int>(rem.to_ll())));
        n = n / bbase;
    }
    std::reverse(digits_out.begin(), digits_out.end());
    if (negative) digits_out.insert(digits_out.begin(), '-');
    return digits_out;
}

long long BigInt::to_ll() const {
    long long v = 0;
    long long base = 1;
    const size_t n = digits.size() < 3 ? digits.size() : 3;
    for (size_t i = 0; i < n; ++i) {
        v += static_cast<long long>(digits[i]) * base;
        if (i + 1 < n) {
            if (base > (std::numeric_limits<long long>::max)() / BASE) {
                break;
            }
            base *= BASE;
        }
    }
    return negative ? -v : v;
}

double BigInt::to_double() const {
    double v = 0, b = 1;
    for (auto d : digits) { v += d*b; b *= BASE; }
    return negative ? -v : v;
}

// |a| vs |b|
int BigInt::cmp_abs(const BigInt& o) const {
    if (digits.size() != o.digits.size()) return digits.size() < o.digits.size() ? -1 : 1;
    for (int i=(int)digits.size()-1; i>=0; --i)
        if (digits[i] != o.digits[i]) return digits[i] < o.digits[i] ? -1 : 1;
    return 0;
}

bool BigInt::operator==(const BigInt& o) const {
    return negative==o.negative && digits==o.digits;
}
bool BigInt::operator<(const BigInt& o) const {
    if (negative != o.negative) return negative;
    int c = cmp_abs(o);
    return negative ? c > 0 : c < 0;
}

BigInt BigInt::add_abs(const BigInt& a, const BigInt& b) {
    BigInt r;
    r.digits.clear();
    r.digits.reserve(std::max(a.digits.size(), b.digits.size()) + 1);
    uint64_t carry = 0;
    for (size_t i=0; i<std::max(a.digits.size(),b.digits.size())||carry; ++i) {
        uint64_t sum = carry;
        if (i<a.digits.size()) sum+=a.digits[i];
        if (i<b.digits.size()) sum+=b.digits[i];
        r.digits.push_back(sum%BASE); carry=sum/BASE;
    }
    return r;
}

BigInt BigInt::sub_abs(const BigInt& a, const BigInt& b) {
    // |a| >= |b|
    BigInt r; r.digits=a.digits;
    int borrow=0;
    for (size_t i=0; i<b.digits.size()||borrow; ++i) {
        int diff=(int)r.digits[i] - (i<b.digits.size()?(int)b.digits[i]:0) - borrow;
        if (diff<0){diff+=BASE;borrow=1;} else borrow=0;
        r.digits[i]=(uint32_t)diff;
    }
    r.trim(); return r;
}

BigInt BigInt::operator+(const BigInt& o) const {
    if (negative == o.negative) {
        auto r = add_abs(*this, o); r.negative = negative; r.trim();
        if (r.is_zero()) r.negative=false;
        return r;
    }
    // Different signs
    int c = cmp_abs(o);
    if (c == 0) return BigInt(0LL);
    BigInt r = c > 0 ? sub_abs(*this, o) : sub_abs(o, *this);
    r.negative = c > 0 ? negative : o.negative;
    r.trim(); if (r.is_zero()) r.negative=false;
    return r;
}

BigInt BigInt::operator-(const BigInt& o) const {
    BigInt neg_o = o; neg_o.negative = !o.negative;
    if (neg_o.is_zero()) neg_o.negative = false;
    return *this + neg_o;
}

BigInt BigInt::operator*(const BigInt& o) const {
    const std::size_t n = digits.size();
    const std::size_t m = o.digits.size();
    BigInt r; r.digits.assign(n+m, 0);
    for (std::size_t i=0;i<n;++i) {
        uint64_t carry=0;
        for (std::size_t j=0;j<m||carry;++j) {
            uint64_t cur=(uint64_t)r.digits[i+j]+carry;
            if (j<m) cur+=(uint64_t)digits[i]*o.digits[j];
            r.digits[i+j]=cur%BASE; carry=cur/BASE;
        }
    }
    r.negative = negative != o.negative;
    r.trim(); if (r.is_zero()) r.negative=false;
    return r;
}

// Long division: computes quotient and remainder of a/b in a single pass,
// shared by operator/, operator%, and bigint_divmod so neither of the latter
// two need to re-run the (expensive) long-division loop.
std::pair<BigInt, BigInt> BigInt::divmod(const BigInt& a, const BigInt& b) {
    if (b.is_zero()) return {BigInt(0LL), a};
    int c = a.cmp_abs(b);
    if (c < 0) return {BigInt(0LL), a};
    if (c == 0) {
        BigInt q(1LL); q.negative = a.negative != b.negative;
        return {q, BigInt(0LL)};
    }
    // Simple long division in base BASE
    BigInt divisor=b; divisor.negative=false;
    BigInt remainder; remainder.digits.clear();
    remainder.digits.reserve(a.digits.size() + 1);
    BigInt quotient; quotient.digits.resize(a.digits.size(),0);
    BigInt t;
    t.negative = false;
    for (int i=(int)a.digits.size()-1;i>=0;--i) {
        remainder.digits.insert(remainder.digits.begin(), a.digits[i]);
        remainder.trim();
        // Binary search for q
        uint32_t lo=0,hi=BASE-1;
        while (lo<hi) {
            uint32_t mid=(lo+hi+1)/2;
            t.digits.assign(1, mid);
            if ((t*divisor).cmp_abs(remainder)<=0) lo=mid; else hi=mid-1;
        }
        quotient.digits[i]=lo;
        t.digits.assign(1, lo);
        remainder=remainder-t*divisor;
        if (remainder.is_zero()) remainder.digits={0};
    }
    quotient.negative = a.negative != b.negative; quotient.trim();
    if (quotient.is_zero()) quotient.negative=false;
    remainder.negative = a.negative; remainder.trim();
    if (remainder.is_zero()) remainder.negative=false;
    return {quotient, remainder};
}

BigInt BigInt::operator/(const BigInt& o) const {
    return divmod(*this, o).first;
}

BigInt BigInt::operator%(const BigInt& o) const {
    return divmod(*this, o).second;
}

BigInt BigInt::shift10(int n) const {
    if (is_zero()) return *this;
    std::string s=to_string();
    if (negative) s=s.substr(1);
    s+=std::string(n,'0');
    BigInt r(s); r.negative=negative;
    return r;
}

// ========================== Number Theory ==========================

std::pair<BigInt, BigInt> bigint_divmod(const BigInt& a, const BigInt& b) {
    return BigInt::divmod(a, b);
}

BigInt bigint_gcd(BigInt a, BigInt b) {
    a.negative=false; b.negative=false;
    while (!b.is_zero()) { BigInt t=std::move(a)%b; a=std::move(b); b=std::move(t); }
    return a;
}

std::tuple<BigInt, BigInt, BigInt> bigint_extended_gcd(BigInt a, BigInt b) {
    BigInt old_r = a, r = b;
    BigInt old_s(1LL), s(0LL);
    BigInt old_t(0LL), t(1LL);
    while (!r.is_zero()) {
        BigInt q = old_r / r;
        BigInt tmp = r;
        r = old_r - q * r;
        old_r = tmp;
        tmp = s;
        s = old_s - q * s;
        old_s = tmp;
        tmp = t;
        t = old_t - q * t;
        old_t = tmp;
    }
    BigInt g = old_r;
    BigInt x = old_s;
    BigInt y = old_t;
    if (g.negative) {
        g = -g;
        x = -x;
        y = -y;
    }
    return {g, x, y};
}

int bigint_bit_length(const BigInt& a) {
    if (a.is_zero()) return 0;
    BigInt n = a.negative ? -a : a;
    int bits = 0;
    BigInt power(1LL);
    while (power <= n) {
        ++bits;
        power = power * BigInt(2LL);
    }
    return bits;
}

bool bigint_is_even(const BigInt& a) {
    return (a % BigInt(2LL)).is_zero();
}

bool bigint_is_odd(const BigInt& a) {
    return !bigint_is_even(a);
}

BigInt bigint_lcm(const BigInt& a, const BigInt& b) {
    return a/bigint_gcd(a,b)*b;
}

BigInt bigint_pow(const BigInt& base, long long exp) {
    if (exp==0) return BigInt(1LL);
    if (exp==1) return base;
    BigInt half=bigint_pow(base,exp/2);
    BigInt r=half*half;
    if (exp%2) r=r*base;
    return r;
}

BigInt bigint_pow_mod(BigInt base, BigInt exp, const BigInt& mod) {
    BigInt result(1LL);
    base=base%mod;
    while (!exp.is_zero()) {
        if ((exp%BigInt(2LL)).digits[0]%2==1) result=result*base%mod;
        base=base*base%mod; exp=exp/BigInt(2LL);
    }
    return result;
}

BigInt bigint_mod_inv(const BigInt& a, const BigInt& m) {
    if (m <= BigInt(1LL)) return BigInt(0LL);
    BigInt mod = m.negative ? -m : m;
    auto [g, x, y] = bigint_extended_gcd(a, mod);
    (void)y;
    if (!g.is_one()) return BigInt(0LL);
    BigInt inv = x % mod;
    if (inv.negative) inv = inv + mod;
    return inv;
}

BigInt bigint_factorial(int n) {
    BigInt r(1LL);
    for (int i=2;i<=n;++i) r=r*BigInt((long long)i);
    return r;
}

BigInt bigint_fibonacci(int n) {
    if (n<=0) return BigInt(0LL);
    if (n==1) return BigInt(1LL);
    BigInt a(0LL), b(1LL);
    for (int i=2;i<=n;++i) { BigInt c=std::move(a)+b; a=std::move(b); b=std::move(c); }
    return b;
}

bool bigint_is_prime(const BigInt& n, int rounds) {
    if (n<=BigInt(1LL)) return false;
    if (n<=BigInt(3LL)) return true;
    if ((n%BigInt(2LL)).is_zero()) return false;
    // Miller-Rabin
    // Write n-1 = 2^r * d
    BigInt nm1=n-BigInt(1LL);
    BigInt d=nm1; int r=0;
    while ((d%BigInt(2LL)).is_zero()){d=d/BigInt(2LL);++r;}
    std::mt19937_64 rng(42);
    long long nll=nm1.to_ll();
    if (nll<=2) nll=3;
    for (int i=0;i<rounds;++i) {
        long long ull = 2 + (long long)(rng()%(std::abs(nll)-2));
        BigInt a((long long)ull);
        if (a>=n) a=BigInt(2LL);
        BigInt x=bigint_pow_mod(a,d,n);
        if (x.is_one()||x==nm1) continue;
        bool composite=true;
        for (int j=0;j<r-1;++j) {
            x=x*x%n;
            if (x==nm1){composite=false;break;}
        }
        if (composite) return false;
    }
    return true;
}

BigInt bigint_next_prime(const BigInt& n) {
    if (n <= BigInt(1LL)) return BigInt(2LL);
    BigInt candidate = n.negative ? BigInt(2LL) : n;
    if (candidate == BigInt(2LL)) return BigInt(2LL);
    if (bigint_is_even(candidate)) candidate = candidate + BigInt(1LL);
    while (!bigint_is_prime(candidate)) candidate = candidate + BigInt(2LL);
    return candidate;
}

BigInt bigint_isqrt(const BigInt& n) {
    if (n.negative) return BigInt(0LL);
    if (n.is_zero() || n.is_one()) return n;
    // Seed Newton's method with a guess >= the true root: n has bit_length bl bits,
    // i.e. n < 2^bl, so sqrt(n) < 2^ceil(bl/2). Starting above the root guarantees
    // the iteration decreases monotonically until it settles on floor(sqrt(n)).
    int bl = bigint_bit_length(n);
    BigInt x = bigint_pow(BigInt(2LL), (bl + 1) / 2);
    BigInt y = (x + n / x) / BigInt(2LL);
    while (y < x) {
        x = y;
        y = (x + n / x) / BigInt(2LL);
    }
    return x;
}

// ========================== Rational ==========================

Rational::Rational(long long n, long long d) : num(n), den(d) {
    if (d < 0) { num = -num; den = -den; }
    reduce();
}
Rational::Rational(BigInt n, BigInt d) : num(std::move(n)), den(std::move(d)) {
    if (den.negative) { num.negative=!num.negative; den.negative=false; }
    reduce();
}

void Rational::reduce() {
    bool neg = num.negative;
    num.negative = false;
    BigInt g = bigint_gcd(num, den);
    if (!g.is_zero()&&!g.is_one()) { num=num/g; den=den/g; }
    num.negative = neg && !num.is_zero();
}

Rational::Rational(const std::string& s) {
    auto pos = s.find('/');
    if (pos != std::string::npos) {
        num = BigInt(s.substr(0, pos));
        den = BigInt(s.substr(pos+1));
    } else {
        // Decimal string
        auto dot = s.find('.');
        if (dot == std::string::npos) { num = BigInt(s); den = BigInt(1LL); }
        else {
            std::string int_part = s.substr(0, dot);
            std::string frac_part = s.substr(dot+1);
            bool neg = !int_part.empty() && int_part[0]=='-';
            if (neg) int_part=int_part.substr(1);
            BigInt n_int(int_part.empty()?"0":int_part);
            BigInt n_frac(frac_part.empty()?"0":frac_part);
            BigInt d_pow = bigint_pow(BigInt(10LL), static_cast<long long>(frac_part.size()));
            num = n_int*d_pow + n_frac;
            den = d_pow;
            if (neg) num.negative=true;
        }
    }
    if (den.negative){num.negative=!num.negative;den.negative=false;}
    reduce();
}

Rational Rational::operator+(const Rational& o) const {
    return Rational(num*o.den + o.num*den, den*o.den);
}
Rational Rational::operator-(const Rational& o) const {
    return Rational(num*o.den - o.num*den, den*o.den);
}
Rational Rational::operator*(const Rational& o) const {
    return Rational(num*o.num, den*o.den);
}
Rational Rational::operator/(const Rational& o) const {
    return Rational(num*o.den, den*o.num);
}

bool Rational::operator==(const Rational& o) const {
    return num*o.den == o.num*den;
}
bool Rational::operator<(const Rational& o) const {
    BigInt lhs = num*o.den, rhs = o.num*den;
    return lhs < rhs;
}

std::string Rational::to_string() const {
    if (den.is_one()) return num.to_string();
    return num.to_string() + "/" + den.to_string();
}
double Rational::to_double() const { return num.to_double() / den.to_double(); }

BigInt Rational::floor() const {
    BigInt q = num / den;
    BigInt r = num % den;
    if (num.negative && !r.is_zero()) q = q - BigInt(1LL);
    return q;
}

BigInt Rational::ceil() const {
    BigInt q = num / den;
    BigInt r = num % den;
    if (!num.negative && !r.is_zero()) q = q + BigInt(1LL);
    return q;
}

BigInt Rational::round() const {
    BigInt r = num % den;
    BigInt abs_r = r.negative ? -r : r;
    BigInt twice_abs_r = abs_r * BigInt(2LL);
    if (twice_abs_r < den) return num.negative ? ceil() : floor();
    if (twice_abs_r > den) return num.negative ? floor() : ceil();
    if (!num.negative) return ceil();
    BigInt fl = floor();
    if ((fl % BigInt(2LL)).is_zero()) return fl;
    return ceil();
}

// ========================== APFloat ==========================
//
// value == mantissa * 10^exponent, carried to `prec` significant decimal digits.
// The exponent is decimal because BigInt's limbs are base 10^9: scaling by a power
// of ten is a limb slice, rounding to p significant digits is a digit drop, and
// printing needs no base conversion. See the class comment in bignum.hpp.

namespace {

// ---- BigInt-level helpers ------------------------------------------------

// BigInt::trim() is private, so anything that builds a BigInt by assigning .digits
// directly has to strip the leading zero limbs itself.
void ap_trim(BigInt& a) {
    while (a.digits.size() > 1 && a.digits.back() == 0u) a.digits.pop_back();
    if (a.digits.size() == 1 && a.digits[0] == 0u) a.negative = false;
}

// 10^n built limb-wise in O(n/9); bigint_pow(BigInt(10), n) would instead do
// O(log n) full multiplications of n-digit numbers.
BigInt ap_pow10(int n) {
    if (n <= 0) return BigInt(1LL);
    const std::size_t limbs = static_cast<std::size_t>(n) / 9u;
    const int rest = n % 9;
    uint32_t top = 1u;
    for (int i = 0; i < rest; ++i) top *= 10u;
    BigInt r;
    r.digits.assign(limbs + 1u, 0u);
    r.digits[limbs] = top;
    r.negative = false;
    return r;
}

// Truncating division by a 32-bit divisor in O(limbs), keeping a's sign.
// Precondition: d != 0. rem < d <= 2^32-1, so rem*10^9 + limb < 2^64.
std::pair<BigInt, uint32_t> ap_divmod_small(const BigInt& a, uint32_t d) {
    BigInt q;
    q.digits.assign(a.digits.size(), 0u);
    q.negative = a.negative;
    uint64_t rem = 0;
    for (std::size_t i = a.digits.size(); i-- > 0;) {
        const uint64_t cur = rem * static_cast<uint64_t>(BigInt::BASE)
                           + static_cast<uint64_t>(a.digits[i]);
        q.digits[i] = static_cast<uint32_t>(cur / d);
        rem = cur % d;
    }
    ap_trim(q);
    return {q, static_cast<uint32_t>(rem)};
}

BigInt ap_div_small(const BigInt& a, uint32_t d) { return ap_divmod_small(a, d).first; }

// Decimal digits of |a|; zero counts as one digit. Requires no leading zero limbs.
int ap_digit_count(const BigInt& a) {
    const uint32_t top = a.digits.back();
    int w = 1;
    for (uint32_t t = top; t >= 10u; t /= 10u) ++w;
    return static_cast<int>(a.digits.size() - 1u) * 9 + w;
}

// a * 10^n in O(n/9 + limbs). Replaces BigInt::shift10, which round-trips through
// std::to_string and the string constructor and is undefined for n < 0.
BigInt ap_shift10(const BigInt& a, int n) {
    if (n <= 0 || a.is_zero()) return a;
    const std::size_t limbs = static_cast<std::size_t>(n) / 9u;
    const int rest = n % 9;
    BigInt r = a;
    if (rest > 0) {
        uint32_t f = 1u;
        for (int i = 0; i < rest; ++i) f *= 10u;
        r = r * BigInt(static_cast<long long>(f));
    }
    if (limbs > 0) r.digits.insert(r.digits.begin(), limbs, 0u);
    return r;
}

// round(|m| / 10^k) with ties to even, sign preserved, in O(limbs).
// 10^9 is even, so the parity of the whole number is the parity of limb 0, which
// is why the half-even test only has to look at one limb.
BigInt ap_round_div_pow10(const BigInt& m, int k) {
    if (k <= 0 || m.is_zero()) return m;
    BigInt a = m;
    a.negative = false;
    bool sticky = false;
    const int kk = k - 1;  // drop k-1 digits, keep one to inspect
    if (kk > 0) {
        const std::size_t limbs = static_cast<std::size_t>(kk) / 9u;
        const int rest = kk % 9;
        if (limbs >= a.digits.size()) {
            sticky = true;
            a = BigInt(0LL);
        } else if (limbs > 0) {
            for (std::size_t i = 0; i < limbs; ++i)
                if (a.digits[i] != 0u) sticky = true;
            a.digits.erase(a.digits.begin(),
                           a.digits.begin() + static_cast<std::ptrdiff_t>(limbs));
        }
        if (rest > 0) {
            uint32_t d = 1u;
            for (int i = 0; i < rest; ++i) d *= 10u;
            const std::pair<BigInt, uint32_t> pr = ap_divmod_small(a, d);
            a = pr.first;
            if (pr.second != 0u) sticky = true;
        }
    }
    const std::pair<BigInt, uint32_t> last = ap_divmod_small(a, 10u);
    BigInt q = last.first;
    const uint32_t d1 = last.second;
    bool up = d1 > 5u;
    if (d1 == 5u) up = sticky || ((q.digits[0] % 2u) == 1u);
    if (up) q = q + BigInt(1LL);
    q.negative = m.negative && !q.is_zero();
    return q;
}

// |m| / 10^k truncated toward zero, sign preserved; `lost` reports whether any
// nonzero digit was dropped.
BigInt ap_trunc_div_pow10(const BigInt& m, int k, bool& lost) {
    lost = false;
    if (k <= 0 || m.is_zero()) return m;
    BigInt a = m;
    const bool neg = a.negative;
    a.negative = false;
    const std::size_t limbs = static_cast<std::size_t>(k) / 9u;
    const int rest = k % 9;
    if (limbs >= a.digits.size()) {
        lost = true;
        return BigInt(0LL);
    }
    if (limbs > 0) {
        for (std::size_t i = 0; i < limbs; ++i)
            if (a.digits[i] != 0u) lost = true;
        a.digits.erase(a.digits.begin(),
                       a.digits.begin() + static_cast<std::ptrdiff_t>(limbs));
    }
    if (rest > 0) {
        uint32_t d = 1u;
        for (int i = 0; i < rest; ++i) d *= 10u;
        const std::pair<BigInt, uint32_t> pr = ap_divmod_small(a, d);
        a = pr.first;
        if (pr.second != 0u) lost = true;
    }
    a.negative = neg && !a.is_zero();
    return a;
}

// Drop trailing decimal zeros from the significand, moving them into the exponent.
void ap_strip_zeros(BigInt& m, int& e) {
    if (m.is_zero()) { m = BigInt(0LL); e = 0; return; }
    while (m.digits.size() > 1 && m.digits[0] == 0u) {
        m.digits.erase(m.digits.begin());
        e += 9;
    }
    for (int i = 0; i < 9; ++i) {
        const std::pair<BigInt, uint32_t> pr = ap_divmod_small(m, 10u);
        if (pr.second != 0u) break;
        if (pr.first.is_zero()) break;
        m = pr.first;
        e += 1;
    }
}

// ---- precision policy ----------------------------------------------------

int ap_clamp_precision(int p) {
    if (p <= 0) return APFloat::DEFAULT_PRECISION;
    if (p > APFloat::MAX_PRECISION) return APFloat::MAX_PRECISION;
    return p;
}

// Guard digits: 10 plus the number of decimal digits of P. Every routine here
// performs O(W) rounded operations whose errors are damped by the next division,
// so the accumulated relative error stays below 10^(2+log10(W)-W); requiring that
// under 10^(-P-1) needs W >= P + 3 + log10(W), which this beats by six digits at
// every precision in [1, MAX_PRECISION].
int ap_guard_digits(int P) {
    int d = 1;
    for (int t = P; t >= 10; t /= 10) ++d;
    return 10 + d;
}

// Number of argument halvings before the exp Taylor series. With |r| ~ 1.15/2^m the
// term count N satisfies N*(log10 N + 0.301m) ~ W, and N + m multiplications are
// minimised near m = 1.5*sqrt(W). Only the run time depends on this, never the value.
int ap_halving_count(int W) {
    const double s = 1.5 * std::sqrt(static_cast<double>(W));
    int m = static_cast<int>(s);
    if (m < 4) m = 4;
    if (m > 400) m = 400;
    return m;
}

// Hard defensive cap on any series loop; the reductions above never come close.
long long ap_series_cap(int W) { return 20LL * static_cast<long long>(W) + 100LL; }

// ---- APFloat-level helpers ----------------------------------------------

// Build mantissa*10^e at precision P, range-checking the exponent in long long
// before it is narrowed to int (the unchecked sum of two exponents would reach 2e9).
APFloat ap_make(BigInt m, long long e, int P) {
    if (m.is_zero()) return APFloat::zero(P);
    const long long lim = static_cast<long long>(APFloat::EXP10_LIMIT)
                        + static_cast<long long>(APFloat::MAX_PRECISION) + 64LL;
    if (e >  lim) return APFloat::saturated(m.negative ? -1 : 1, P);
    if (e < -lim) return APFloat::zero(P);
    return APFloat(std::move(m), static_cast<int>(e), P);
}

// x / n for a small integer n. The mantissa MUST be extended to W+2 digits first:
// ap_div_small truncates at the mantissa's current length, so dividing an exact
// short value (1, or 0.001) would silently collapse the working precision to that
// length -- BigInt(1)/2 is 0. Truncating at W+2 digits costs under 1 ulp at W.
APFloat ap_div_small_ap(const APFloat& x, uint32_t n, int W) {
    if (x.mantissa.is_zero() || n == 0u) return x;
    int need = W + 2 - ap_digit_count(x.mantissa);
    if (need < 0) need = 0;
    const BigInt mm = ap_shift10(x.mantissa, need);
    return ap_make(ap_div_small(mm, n), static_cast<long long>(x.exponent) - need, W);
}

// x * n, exact before the single rounding.
APFloat ap_mul_small_ap(const APFloat& x, long long n, int W) {
    if (x.mantissa.is_zero() || n == 0) return APFloat::zero(W);
    return ap_make(x.mantissa * BigInt(n), static_cast<long long>(x.exponent), W);
}

// x / 2^m, EXACT: x/2^m == (mantissa * 5^m) * 10^(exponent-m). The mantissa grows
// by exactly m digits before rounding, so no precision is lost the way a plain
// division by a small integer would lose it.
APFloat ap_scale_pow2_down(const APFloat& x, int m, int W) {
    if (x.mantissa.is_zero() || m <= 0) return x.with_precision(W);
    return ap_make(x.mantissa * bigint_pow(BigInt(5LL), m),
                   static_cast<long long>(x.exponent) - m, W);
}

// A series term is negligible once it sits more than W+2 digits below the sum.
bool ap_negligible(const APFloat& sum, const APFloat& term, int W) {
    if (term.is_zero()) return true;
    if (sum.is_zero()) return false;
    return sum.dec_exp() - term.dec_exp() > W + 2;
}

// Correctly rounded (ma*10^ea) / (mb*10^eb) at precision P.
// The exact quotient generally does not terminate, so computing P+2 digits and
// rounding would mis-round the tie case. Writing the true quotient as Q + R/|b|
// with Q carrying t >= 1 digits more than P, a genuine tie needs both
// Q mod 10^t == 10^t/2 AND R == 0; when R != 0, replacing Q by 10Q+1 makes the
// dropped part 10*(Q mod 10^t)+1 against a threshold of 10*(10^t/2), which exceeds
// it exactly when Q mod 10^t >= 10^t/2 -- precisely the true comparison.
APFloat ap_div_exact(const BigInt& ma, long long ea,
                     const BigInt& mb, long long eb, int P) {
    if (ma.is_zero() || mb.is_zero()) return APFloat::zero(P);
    const int na = ap_digit_count(ma);
    const int nb = ap_digit_count(mb);
    long long kk = static_cast<long long>(P) + 2
                 - (static_cast<long long>(na) - static_cast<long long>(nb));
    if (kk < 0) kk = 0;
    const int k = static_cast<int>(kk);
    const std::pair<BigInt, BigInt> qr = bigint_divmod(ap_shift10(ma, k), mb);
    BigInt q = qr.first;
    long long e = ea - static_cast<long long>(k) - eb;
    if (!qr.second.is_zero()) {  // sticky digit
        q = ap_shift10(q, 1) + BigInt(q.negative ? -1LL : 1LL);
        e -= 1;
    }
    return ap_make(std::move(q), e, P);
}

// ---- scaled-integer series kernels --------------------------------------
// These run entirely on BigInt with O(n) small divisions: no APFloat arithmetic and
// no big division, which is what keeps ap_pi(1000) around a millisecond. Each
// iteration truncates twice and the error is damped by the next /q^2, so the total
// truncation error is under 2N ulp at 10^-W -- about six digits, well inside the
// eleven-plus guard digits.

// arctan(1/q) * 10^W, truncated. Requires q >= 2 and q*q < 2^32.
BigInt ap_atan_inv_int(uint32_t q, int W) {
    BigInt term = ap_div_small(ap_pow10(W), q);
    BigInt total = term;
    const uint32_t q2 = q * q;
    long long k = 1;
    for (;;) {
        term = ap_div_small(term, q2);
        if (term.is_zero()) break;
        const BigInt t = ap_div_small(term, static_cast<uint32_t>(2 * k + 1));
        if (t.is_zero()) break;
        if ((k % 2) == 1) total = total - t;
        else              total = total + t;
        ++k;
    }
    return total;
}

// artanh(1/q) * 10^W, truncated. Requires q >= 2 and q*q < 2^32.
BigInt ap_atanh_inv_int(uint32_t q, int W) {
    BigInt term = ap_div_small(ap_pow10(W), q);
    BigInt total = term;
    const uint32_t q2 = q * q;
    long long k = 1;
    for (;;) {
        term = ap_div_small(term, q2);
        if (term.is_zero()) break;
        const BigInt t = ap_div_small(term, static_cast<uint32_t>(2 * k + 1));
        if (t.is_zero()) break;
        total = total + t;
        ++k;
    }
    return total;
}

// e * 10^W, truncated; stops at the smallest N with N! > 10^W.
BigInt ap_e_int(int W) {
    BigInt term = ap_pow10(W);
    BigInt total = term;
    for (uint32_t k = 1; k < 4000000u; ++k) {
        term = ap_div_small(term, k);
        if (term.is_zero()) break;
        total = total + term;
    }
    return total;
}

// floor(n^(1/3)) for n >= 0, by Newton's method on BigInt in the style of
// bigint_isqrt: x <- (2x + n/x^2)/3 from a seed above the root, then an exact
// +-1 correction so the result is the true floor for every n.
BigInt ap_icbrt(const BigInt& n) {
    if (n.negative) return BigInt(0LL);
    if (n.is_zero() || n.is_one()) return n;
    const int bl = bigint_bit_length(n);
    BigInt x = bigint_pow(BigInt(2LL), (bl + 2) / 3);  // >= the true root
    BigInt y = (x * BigInt(2LL) + n / (x * x)) / BigInt(3LL);
    while (y < x) {
        x = y;
        y = (x * BigInt(2LL) + n / (x * x)) / BigInt(3LL);
    }
    while ((x * x * x) > n) x = x - BigInt(1LL);
    for (;;) {
        const BigInt t = x + BigInt(1LL);
        if ((t * t * t) <= n) x = t;
        else break;
    }
    return x;
}

// Shared sin/cos core: one range reduction feeds both series, so ap_tan and the
// complex routines do not pay for it twice. Results come back at precision P.
void ap_sincos_core(const APFloat& x, int P0, APFloat& sin_out, APFloat& cos_out);

} // namespace

// ---- construction --------------------------------------------------------

APFloat::APFloat() : mantissa(0LL), exponent(0), prec(DEFAULT_PRECISION) {}

APFloat::APFloat(int precision)
    : mantissa(0LL), exponent(0), prec(ap_clamp_precision(precision)) {}

APFloat::APFloat(long long v, int precision)
    : mantissa(v), exponent(0), prec(ap_clamp_precision(precision)) {
    normalize();
}

APFloat::APFloat(BigInt m, int exp10, int precision)
    : mantissa(std::move(m)), exponent(exp10), prec(ap_clamp_precision(precision)) {
    normalize();
}

APFloat::APFloat(double v, int precision)
    : mantissa(0LL), exponent(0), prec(ap_clamp_precision(precision)) {
    if (!std::isfinite(v) || v == 0.0) return;
    int exp2 = 0;
    const double frac = std::frexp(v, &exp2);          // frac in [0.5, 1)
    const long long m = static_cast<long long>(std::ldexp(frac, 53));
    exp2 -= 53;
    if (exp2 >= 0) {
        mantissa = BigInt(m) * bigint_pow(BigInt(2LL), exp2);
        exponent = 0;
    } else {
        // m / 2^|e| == m * 5^|e| / 10^|e|: exact and finite in decimal.
        mantissa = BigInt(m) * bigint_pow(BigInt(5LL), -exp2);
        exponent = exp2;
    }
    normalize();
}

Result<APFloat> APFloat::parse(const std::string& s, int precision) {
    const int P = ap_clamp_precision(precision);
    const std::size_t n = s.size();
    std::size_t i = 0;
    bool neg = false;
    if (i < n && (s[i] == '+' || s[i] == '-')) { neg = (s[i] == '-'); ++i; }
    std::string ip;
    std::string fp;
    while (i < n && s[i] >= '0' && s[i] <= '9') { ip.push_back(s[i]); ++i; }
    if (i < n && s[i] == '.') {
        ++i;
        while (i < n && s[i] >= '0' && s[i] <= '9') { fp.push_back(s[i]); ++i; }
    }
    if (ip.empty() && fp.empty())
        return std::unexpected(DomainError{"APFloat::parse",
                                           "no digits in the significand"});
    long long exp_field = 0;
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        bool eneg = false;
        if (i < n && (s[i] == '+' || s[i] == '-')) { eneg = (s[i] == '-'); ++i; }
        if (i >= n || s[i] < '0' || s[i] > '9')
            return std::unexpected(DomainError{"APFloat::parse",
                                               "exponent marker without digits"});
        while (i < n && s[i] >= '0' && s[i] <= '9') {
            if (exp_field <= static_cast<long long>(APFloat::EXP10_LIMIT))
                exp_field = exp_field * 10 + static_cast<long long>(s[i] - '0');
            ++i;
        }
        if (eneg) exp_field = -exp_field;
    }
    if (i != n)
        return std::unexpected(DomainError{"APFloat::parse",
                                           "unexpected trailing characters"});
    if (exp_field >  static_cast<long long>(APFloat::EXP10_LIMIT) ||
        exp_field < -static_cast<long long>(APFloat::EXP10_LIMIT))
        return std::unexpected(DomainError{"APFloat::parse", "exponent out of range"});

    // digits_str is validated all-digits; BigInt(const std::string&) now abandons the
    // parse and yields zero on a non-digit rather than aborting, but feeding it validated
    // input keeps this path's meaning explicit.
    const std::string digits_str = ip + fp;
    BigInt m(digits_str);
    m.negative = neg && !m.is_zero();
    const long long e = exp_field - static_cast<long long>(fp.size());
    return ap_make(std::move(m), e, P);
}

APFloat::APFloat(const std::string& s, int precision)
    : mantissa(0LL), exponent(0), prec(ap_clamp_precision(precision)) {
    if (Result<APFloat> r = parse(s, precision)) *this = std::move(*r);
}

APFloat APFloat::from_rational(const Rational& r, int precision) {
    return ap_div_exact(r.num, 0, r.den, 0, ap_clamp_precision(precision));
}

APFloat APFloat::zero(int precision) { return APFloat(ap_clamp_precision(precision)); }
APFloat APFloat::one(int precision)  { return APFloat(1LL, precision); }

APFloat APFloat::saturated(int sign_, int precision) {
    const int P = ap_clamp_precision(precision);
    BigInt m = ap_pow10(P) - BigInt(1LL);
    m.negative = sign_ < 0;
    return APFloat(std::move(m), APFloat::EXP10_LIMIT - P, P);
}

// ---- normalisation -------------------------------------------------------

void APFloat::normalize() {
    prec = ap_clamp_precision(prec);
    if (mantissa.is_zero()) { mantissa = BigInt(0LL); exponent = 0; return; }

    const int sgn = mantissa.negative ? -1 : 1;
    const int limit = APFloat::EXP10_LIMIT + APFloat::MAX_PRECISION + 64;
    if (exponent >  limit) { *this = APFloat::saturated(sgn, prec); return; }
    if (exponent < -limit) { mantissa = BigInt(0LL); exponent = 0; return; }

    ap_strip_zeros(mantissa, exponent);
    if (mantissa.is_zero()) { exponent = 0; return; }

    const int n = ap_digit_count(mantissa);
    if (n > prec) {
        const int k = n - prec;
        mantissa = ap_round_div_pow10(mantissa, k);
        exponent += k;
        // Rounding can carry (9995 -> 1000 at three digits) and add a digit; the
        // second strip removes it again, so one pass is enough.
        ap_strip_zeros(mantissa, exponent);
        if (mantissa.is_zero()) { exponent = 0; return; }
    }

    const long long de = static_cast<long long>(exponent) + ap_digit_count(mantissa);
    if (de >  static_cast<long long>(APFloat::EXP10_LIMIT)) {
        *this = APFloat::saturated(sgn, prec);
        return;
    }
    if (de < -static_cast<long long>(APFloat::EXP10_LIMIT)) {
        mantissa = BigInt(0LL);
        exponent = 0;
    }
}

void APFloat::set_precision(int p) {
    prec = ap_clamp_precision(p);
    normalize();
}

APFloat APFloat::with_precision(int p) const {
    APFloat r = *this;
    r.set_precision(p);
    return r;
}

// ---- predicates ----------------------------------------------------------

bool APFloat::is_zero() const { return mantissa.is_zero(); }
bool APFloat::is_negative() const { return mantissa.negative && !mantissa.is_zero(); }
bool APFloat::is_integer() const { return mantissa.is_zero() || exponent >= 0; }
bool APFloat::is_saturated() const {
    return !is_zero() && dec_exp() >= APFloat::EXP10_LIMIT;
}
int APFloat::sign() const {
    if (mantissa.is_zero()) return 0;
    return mantissa.negative ? -1 : 1;
}
int APFloat::dec_exp() const {
    if (mantissa.is_zero()) return 0;
    return exponent + ap_digit_count(mantissa);
}

APFloat APFloat::abs() const {
    APFloat r = *this;
    r.mantissa.negative = false;
    return r;
}
APFloat APFloat::neg() const {
    APFloat r = *this;
    if (!r.mantissa.is_zero()) r.mantissa.negative = !r.mantissa.negative;
    return r;
}
APFloat APFloat::operator-() const { return neg(); }

// ---- arithmetic ----------------------------------------------------------

APFloat APFloat::operator+(const APFloat& o) const {
    const int P = std::max(prec, o.prec);
    if (mantissa.is_zero())   return o.with_precision(P);
    if (o.mantissa.is_zero()) return with_precision(P);

    // Which operand dominates is decided by dec_exp, i.e. by magnitude -- comparing
    // the stored exponents instead would call 5.2e-11 negligible next to an exact 1,
    // whose mantissa is a single digit.
    const bool this_hi = dec_exp() >= o.dec_exp();
    const APFloat& hi = this_hi ? *this : o;
    const APFloat& lo = this_hi ? o : *this;
    if (static_cast<long long>(hi.dec_exp()) - lo.dec_exp() > static_cast<long long>(P) + 8) {
        // `lo` sits more than P+8 digits below hi's leading digit, so it can only
        // matter in the exact-tie case. Keeping one signed unit that far down breaks
        // that tie the way the exact sum would, without aligning a mantissa across a
        // billion digits (1e300 + 1e-300 must not allocate 600 digits).
        const int drop = P + 8;
        BigInt m = ap_shift10(hi.mantissa, drop);
        m = m + BigInt(lo.mantissa.negative ? -1LL : 1LL);
        return ap_make(std::move(m),
                       static_cast<long long>(hi.exponent) - drop, P);
    }
    // Otherwise the alignment is exact, and bounded: the two dec_exps differ by at
    // most P+8, so the shift stays under 2P+8 digits.
    const int emin = std::min(exponent, o.exponent);
    const BigInt ma = ap_shift10(mantissa, exponent - emin);
    const BigInt mb = ap_shift10(o.mantissa, o.exponent - emin);
    return ap_make(ma + mb, static_cast<long long>(emin), P);
}

APFloat APFloat::operator-(const APFloat& o) const { return *this + o.neg(); }

APFloat APFloat::operator*(const APFloat& o) const {
    const int P = std::max(prec, o.prec);
    if (mantissa.is_zero() || o.mantissa.is_zero()) return APFloat::zero(P);
    // The BigInt product is exact, so there is exactly one rounding.
    return ap_make(mantissa * o.mantissa,
                   static_cast<long long>(exponent) + static_cast<long long>(o.exponent),
                   P);
}

APFloat APFloat::operator/(const APFloat& o) const {
    return ap_div_exact(mantissa, static_cast<long long>(exponent),
                        o.mantissa, static_cast<long long>(o.exponent),
                        std::max(prec, o.prec));
}

APFloat& APFloat::operator+=(const APFloat& o) { *this = *this + o; return *this; }
APFloat& APFloat::operator-=(const APFloat& o) { *this = *this - o; return *this; }
APFloat& APFloat::operator*=(const APFloat& o) { *this = *this * o; return *this; }
APFloat& APFloat::operator/=(const APFloat& o) { *this = *this / o; return *this; }

int APFloat::cmp(const APFloat& o) const {
    const int sa = sign();
    const int sb = o.sign();
    if (sa != sb) return sa < sb ? -1 : 1;
    if (sa == 0) return 0;
    const int da = dec_exp();
    const int db = o.dec_exp();
    if (da != db) return (da < db) ? -sa : sa;
    // Same sign and same decimal exponent, so the alignment shift below is bounded
    // by the two digit counts and never by the exponents themselves.
    const int emin = std::min(exponent, o.exponent);
    const BigInt ma = ap_shift10(mantissa, exponent - emin);
    const BigInt mb = ap_shift10(o.mantissa, o.exponent - emin);
    if (ma < mb) return -1;
    if (mb < ma) return 1;
    return 0;
}

// ---- integer extraction --------------------------------------------------

BigInt APFloat::trunc() const {
    if (mantissa.is_zero()) return BigInt(0LL);
    if (exponent >= 0) {
        if (dec_exp() > APFloat::MATERIALIZE_LIMIT) return BigInt(0LL);
        return ap_shift10(mantissa, exponent);
    }
    bool lost = false;
    return ap_trunc_div_pow10(mantissa, -exponent, lost);
}

BigInt APFloat::floor() const {
    if (mantissa.is_zero()) return BigInt(0LL);
    if (exponent >= 0) return trunc();
    bool lost = false;
    BigInt q = ap_trunc_div_pow10(mantissa, -exponent, lost);
    if (mantissa.negative && lost) q = q - BigInt(1LL);
    return q;
}

BigInt APFloat::ceil() const {
    if (mantissa.is_zero()) return BigInt(0LL);
    if (exponent >= 0) return trunc();
    bool lost = false;
    BigInt q = ap_trunc_div_pow10(mantissa, -exponent, lost);
    if (!mantissa.negative && lost) q = q + BigInt(1LL);
    return q;
}

BigInt APFloat::round() const {
    if (mantissa.is_zero()) return BigInt(0LL);
    if (exponent >= 0) return trunc();
    return ap_round_div_pow10(mantissa, -exponent);
}

// ---- conversion / formatting --------------------------------------------

double APFloat::to_double() const {
    if (is_zero()) return 0.0;
    const APFloat t = with_precision(18);
    std::string s = t.mantissa.to_string();
    const bool neg = t.mantissa.negative;
    if (neg) s.erase(s.begin());
    std::string buf;
    if (neg) buf += '-';
    buf += s.substr(0, 1);
    if (s.size() > 1) { buf += '.'; buf += s.substr(1); }
    buf += 'e';
    buf += std::to_string(t.dec_exp() - 1);
    // std::stod would throw on an out-of-range literal; strtod reports it through
    // errno and returns +-HUGE_VAL (or +-0.0 on underflow), which is the documented
    // behaviour of this function.
    return std::strtod(buf.c_str(), nullptr);
}

long long APFloat::to_ll() const { return trunc().to_ll(); }

Rational APFloat::to_rational() const {
    if (mantissa.is_zero()) return Rational(0LL, 1LL);
    if (exponent >= 0) {
        if (exponent > APFloat::MATERIALIZE_LIMIT) return Rational(0LL, 1LL);
        return Rational(ap_shift10(mantissa, exponent), BigInt(1LL));
    }
    if (-exponent > APFloat::MATERIALIZE_LIMIT) return Rational(0LL, 1LL);
    return Rational(mantissa, ap_pow10(-exponent));
}

std::string APFloat::to_string() const { return to_string(prec); }

std::string APFloat::to_string(int digits) const {
    if (digits <= 0) digits = prec;
    if (digits > APFloat::MAX_PRECISION) digits = APFloat::MAX_PRECISION;
    if (is_zero()) return "0";
    const APFloat t = with_precision(digits);
    std::string s = t.mantissa.to_string();
    const bool neg = t.mantissa.negative;
    if (neg) s.erase(s.begin());
    const int D = t.dec_exp();
    // Pad on the right so trailing zeros survive: to_string(10) of 12345 has to
    // render "12345.00000", not "12345".
    if (static_cast<int>(s.size()) < digits)
        s.append(static_cast<std::size_t>(digits - static_cast<int>(s.size())), '0');
    std::string out;
    if (D >= -4 && D <= digits) {
        if (D > 0) {
            out = s.substr(0, static_cast<std::size_t>(D));
            if (D < digits) {
                out += '.';
                out += s.substr(static_cast<std::size_t>(D));
            }
        } else {
            out = "0.";
            out.append(static_cast<std::size_t>(-D), '0');
            out += s;
        }
    } else {
        out = s.substr(0, 1);
        if (digits > 1) { out += '.'; out += s.substr(1); }
        out += 'e';
        const long long ex = static_cast<long long>(D) - 1;
        out += (ex < 0) ? '-' : '+';
        out += std::to_string(ex < 0 ? -ex : ex);
    }
    return neg ? ("-" + out) : out;
}

std::string APFloat::to_string_fixed(int decimals) const {
    if (decimals < 0) decimals = 0;
    if (decimals > APFloat::MAX_PRECISION) decimals = APFloat::MAX_PRECISION;
    const long long k = -static_cast<long long>(exponent)
                      - static_cast<long long>(decimals);
    BigInt m = mantissa;
    if (k > 0) {
        m = ap_round_div_pow10(m, static_cast<int>(k));
    } else if (k < 0) {
        if (-k > static_cast<long long>(APFloat::MATERIALIZE_LIMIT))
            return "0";  // would need a million-digit rendering; see MATERIALIZE_LIMIT
        m = ap_shift10(m, static_cast<int>(-k));
    }
    std::string s = m.to_string();
    const bool neg = m.negative && !m.is_zero();
    if (neg) s.erase(s.begin());
    while (static_cast<int>(s.size()) <= decimals) s.insert(s.begin(), '0');
    std::string out;
    if (neg) out += '-';
    const std::size_t cut = s.size() - static_cast<std::size_t>(decimals);
    out += s.substr(0, cut);
    if (decimals > 0) { out += '.'; out += s.substr(cut); }
    return out;
}

// ---- constants -----------------------------------------------------------

APFloat ap_pi(int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    // Machin: pi = 16*arctan(1/5) - 4*arctan(1/239). The two series need about
    // 0.715*W and 0.210*W terms respectively.
    const BigInt scaled = ap_atan_inv_int(5u, W) * BigInt(16LL)
                        - ap_atan_inv_int(239u, W) * BigInt(4LL);
    return APFloat(scaled, -W, P);
}

APFloat ap_e(int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    return APFloat(ap_e_int(W), -W, P);
}

APFloat ap_ln2(int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    // ln 2 = 2*artanh(1/3); about 1.05*W terms.
    return APFloat(ap_atanh_inv_int(3u, W) * BigInt(2LL), -W, P);
}

APFloat ap_ln10(int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    // ln 10 = 3*ln2 + ln(10/8) = 6*artanh(1/3) + 2*artanh(1/9).
    const BigInt scaled = ap_atanh_inv_int(3u, W) * BigInt(6LL)
                        + ap_atanh_inv_int(9u, W) * BigInt(2LL);
    return APFloat(scaled, -W, P);
}

// ---- algebraic -----------------------------------------------------------

APFloat ap_abs(const APFloat& x) { return x.abs(); }
APFloat ap_neg(const APFloat& x) { return x.neg(); }

APFloat ap_sqrt(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero() || x.is_negative()) return APFloat::zero(P);
    const int W = P + ap_guard_digits(P);
    // value = m*10^e, so sqrt(m*10^shift) * 10^((e-shift)/2) = sqrt(m*10^e) as long
    // as e-shift is even. C++'s % can return -1 here, so test against 0, not 1.
    int shift = 2 * W + 8;
    if (((x.exponent - shift) % 2) != 0) ++shift;
    const BigInt r = bigint_isqrt(ap_shift10(x.mantissa, shift));
    // The radicand is at least 10^(2W+8), so the root has at least W+5 digits and
    // the floor costs under one ulp there: four spare digits on top of the guard.
    return ap_make(r, (static_cast<long long>(x.exponent) - shift) / 2, P);
}

Result<APFloat> ap_sqrt_checked(const APFloat& x, int prec) {
    if (x.is_negative())
        return std::unexpected(DomainError{"ap_sqrt", "square root of a negative value"});
    return ap_sqrt(x, prec);
}

APFloat ap_cbrt(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    const bool neg = x.is_negative();
    const int W = P + ap_guard_digits(P);
    int shift = 3 * W + 9;
    while (((x.exponent - shift) % 3) != 0) ++shift;  // at most two extra
    BigInt m = x.mantissa;
    m.negative = false;
    BigInt r = ap_icbrt(ap_shift10(m, shift));
    r.negative = neg && !r.is_zero();
    return ap_make(std::move(r), (static_cast<long long>(x.exponent) - shift) / 3, P);
}

APFloat ap_hypot(const APFloat& a, const APFloat& b, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    const APFloat aa = (a * a).with_precision(W);
    const APFloat bb = (b * b).with_precision(W);
    return ap_sqrt((aa + bb).with_precision(W), P);
}

// ---- exp -----------------------------------------------------------------

APFloat ap_exp(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::one(P);
    const int c = std::max(0, x.dec_exp());
    // |x| >= 10^10 gives |x/ln10| > EXP10_LIMIT, so the result cannot land in range.
    if (c >= 11) return x.is_negative() ? APFloat::zero(P) : APFloat::saturated(1, P);
    // Representing x to W relative digits gives absolute accuracy 10^(c-W), which is
    // exactly the relative accuracy the reduced argument passes to the result.
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + c);

    // (1) decimal reduction x = k*ln10 + r with |r| <= ln10/2.
    const APFloat L10 = ap_ln10(W);
    const BigInt kb = (x / L10).with_precision(W).round();
    const long long k = kb.to_ll();
    if (k >  static_cast<long long>(APFloat::EXP10_LIMIT)) return APFloat::saturated(1, P);
    if (k < -static_cast<long long>(APFloat::EXP10_LIMIT)) return APFloat::zero(P);
    const APFloat r = (x - APFloat(kb, 0, W) * L10).with_precision(W);

    // (2) halve the argument so the Taylor series converges in ~1.5*sqrt(W) terms.
    const int m = ap_halving_count(W);
    const APFloat rr = ap_scale_pow2_down(r, m, W);

    // (3) exp(rr) = sum rr^n / n!
    APFloat sum = APFloat::one(W);
    APFloat term = APFloat::one(W);
    const long long cap = ap_series_cap(W);
    for (long long n = 1; n <= cap; ++n) {
        term = (term * rr).with_precision(W);
        term = ap_div_small_ap(term, static_cast<uint32_t>(n), W);
        if (term.is_zero()) break;
        const bool done = ap_negligible(sum, term, W);
        sum = (sum + term).with_precision(W);
        if (done) break;
    }

    // (4) undo the halving, then (5) undo the decimal reduction -- which with a
    // decimal exponent is a single addition, no 10^k has to be materialised.
    for (int i = 0; i < m; ++i) sum = (sum * sum).with_precision(W);
    return ap_make(sum.mantissa, static_cast<long long>(sum.exponent) + k, P);
}

Result<APFloat> ap_exp_checked(const APFloat& x, int prec) {
    const APFloat r = ap_exp(x, prec);
    // Underflow to zero is the correct limit and is not reported as an error.
    if (r.is_saturated()) return std::unexpected(OverflowError{"ap_exp"});
    return r;
}

// ---- log -----------------------------------------------------------------

APFloat ap_log(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero() || x.is_negative()) return APFloat::zero(P);
    const int W = P + ap_guard_digits(P);
    const int nx = ap_digit_count(x.mantissa);
    // EP is chosen so that steps (1) and (2) below cannot round at all.
    const int EP = ap_clamp_precision(std::max(W, nx + 8));
    const int E = x.dec_exp() - 1;

    // (1) exact decimal split x = f * 10^E with f in [1, 10): the same digits with
    // the point moved, since x.exponent - E is exactly 1 - nx.
    APFloat f(x.mantissa, 1 - nx, EP);

    // (2) exact binary trim f -> [0.75, 1.5); at most three halvings, each of which
    // multiplies the mantissa by 5 and so stays inside EP digits.
    int j = 0;
    const APFloat c15(BigInt(15LL), -1, EP);
    while (f.cmp(c15) >= 0) {
        f = ap_scale_pow2_down(f, 1, EP);
        ++j;
    }

    // (3) t = f - 1 is therefore EXACT: for x = 1 + eps the series still gets all of
    // eps, which is why log(1 + 1e-10) keeps every requested digit.
    const APFloat t = (f - APFloat::one(EP)).with_precision(EP);
    APFloat res(0LL, W);
    if (!t.is_zero()) {
        const APFloat d = (f + APFloat::one(EP)).with_precision(EP);
        const APFloat z = (t / d).with_precision(W);   // |z| <= 0.2
        const APFloat z2 = (z * z).with_precision(W);
        APFloat sum = z;
        APFloat term = z;
        const long long cap = ap_series_cap(W);
        for (long long k = 1; k <= cap; ++k) {
            term = (term * z2).with_precision(W);
            const APFloat tt = ap_div_small_ap(term, static_cast<uint32_t>(2 * k + 1), W);
            if (ap_negligible(sum, tt, W)) break;
            sum = (sum + tt).with_precision(W);
        }
        res = ap_mul_small_ap(sum, 2LL, W);            // log f = 2*artanh(z)
    }
    if (j != 0) res = (res + ap_mul_small_ap(ap_ln2(W), j, W)).with_precision(W);
    if (E != 0) res = (res + ap_mul_small_ap(ap_ln10(W), E, W)).with_precision(W);
    res.set_precision(P);
    return res;
}

Result<APFloat> ap_log_checked(const APFloat& x, int prec) {
    if (x.is_zero())
        return std::unexpected(DomainError{"ap_log", "logarithm of zero"});
    if (x.is_negative())
        return std::unexpected(DomainError{"ap_log", "logarithm of a negative value"});
    return ap_log(x, prec);
}

APFloat ap_log10(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero() || x.is_negative()) return APFloat::zero(P);
    const int W = P + ap_guard_digits(P);
    APFloat r = (ap_log(x, W) / ap_ln10(W)).with_precision(W);
    r.set_precision(P);
    return r;
}

// ---- trigonometric -------------------------------------------------------

namespace {

void ap_sincos_core(const APFloat& x, int P0, APFloat& sin_out, APFloat& cos_out) {
    const int P = ap_clamp_precision(P0);
    if (x.is_zero()) {
        sin_out = APFloat::zero(P);
        cos_out = APFloat::one(P);
        return;
    }
    if (x.dec_exp() > APFloat::TRIG_ARG_LIMIT) {
        // Beyond this the reduction would need a pi with more digits than the
        // working precision can justify; degrade defensively (see ap_sin_checked).
        sin_out = APFloat::zero(P);
        cos_out = APFloat::zero(P);
        return;
    }
    const int c = std::max(0, x.dec_exp());

    // (1) reduce modulo 2*pi, then (2) by quadrant to |s| <= pi/4. W > c holds by
    // construction, so the quotient q is exactly representable.
    // The reduced argument is only accurate to 10^(c-W) in ABSOLUTE terms, so an x
    // that sits very close to a multiple of pi/2 (ap_sin(ap_pi(120)) is the extreme
    // case) leaves a reduced argument whose leading digits have all cancelled. When
    // that happens, redo the reduction with as many extra digits of pi as the
    // cancellation ate; since x is a finite decimal and pi is irrational the true
    // residual is nonzero, so this settles after one retry.
    int extra = 0;
    int W = 0;
    long long n = 0;
    APFloat s;
    for (int attempt = 0; attempt < 4; ++attempt) {
        W = ap_clamp_precision(P + ap_guard_digits(P) + c + extra);
        const APFloat PI = ap_pi(W);
        const APFloat TWOPI = ap_mul_small_ap(PI, 2LL, W);
        const APFloat HALFPI = ap_scale_pow2_down(PI, 1, W);
        const BigInt qb = (x / TWOPI).with_precision(W).round();
        const APFloat r = (x - APFloat(qb, 0, W) * TWOPI).with_precision(W);
        const BigInt nb = (r / HALFPI).with_precision(W).round();
        n = nb.to_ll();
        s = (r - APFloat(nb, 0, W) * HALFPI).with_precision(W);
        const int need = s.is_zero() ? (W + 8) : std::max(0, -s.dec_exp());
        if (need <= extra + 4 || W >= APFloat::MAX_PRECISION) break;
        extra = need + 8;
    }

    const APFloat s2 = (s * s).with_precision(W);
    const long long cap = ap_series_cap(W);

    // sin s = s - s^3/3! + ...; the factorial denominator is applied as two small
    // divisions so each divisor stays inside uint32_t for any reachable k.
    APFloat sn = s;
    APFloat st = s;
    for (long long k = 1; k <= cap; ++k) {
        st = (st * s2).with_precision(W);
        st = ap_div_small_ap(st, static_cast<uint32_t>(2 * k), W);
        st = ap_div_small_ap(st, static_cast<uint32_t>(2 * k + 1), W);
        st = st.neg();
        if (ap_negligible(sn, st, W)) break;
        sn = (sn + st).with_precision(W);
    }

    // cos s = 1 - s^2/2! + ...
    APFloat cs = APFloat::one(W);
    APFloat ct = APFloat::one(W);
    for (long long k = 1; k <= cap; ++k) {
        ct = (ct * s2).with_precision(W);
        ct = ap_div_small_ap(ct, static_cast<uint32_t>(2 * k - 1), W);
        ct = ap_div_small_ap(ct, static_cast<uint32_t>(2 * k), W);
        ct = ct.neg();
        if (ap_negligible(cs, ct, W)) break;
        cs = (cs + ct).with_precision(W);
    }

    long long q4 = n % 4;
    if (q4 < 0) q4 += 4;
    switch (q4) {
        case 0:  sin_out = sn;       cos_out = cs;       break;
        case 1:  sin_out = cs;       cos_out = sn.neg(); break;
        case 2:  sin_out = sn.neg(); cos_out = cs.neg(); break;
        default: sin_out = cs.neg(); cos_out = sn;       break;
    }
    sin_out.set_precision(P);
    cos_out.set_precision(P);
}

} // namespace

APFloat ap_sin(const APFloat& x, int prec) {
    APFloat s;
    APFloat c;
    ap_sincos_core(x, ap_clamp_precision(prec), s, c);
    return s;
}

APFloat ap_cos(const APFloat& x, int prec) {
    APFloat s;
    APFloat c;
    ap_sincos_core(x, ap_clamp_precision(prec), s, c);
    return c;
}

APFloat ap_tan(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + std::max(0, x.dec_exp()));
    APFloat s;
    APFloat c;
    ap_sincos_core(x, W, s, c);
    if (c.is_zero()) return APFloat::zero(P);
    APFloat r = (s / c).with_precision(W);
    r.set_precision(P);
    return r;
}

Result<APFloat> ap_sin_checked(const APFloat& x, int prec) {
    if (x.dec_exp() > APFloat::TRIG_ARG_LIMIT)
        return std::unexpected(DomainError{
            "ap_sin", "argument magnitude exceeds the range-reduction limit (10^10000)"});
    return ap_sin(x, prec);
}

Result<APFloat> ap_cos_checked(const APFloat& x, int prec) {
    if (x.dec_exp() > APFloat::TRIG_ARG_LIMIT)
        return std::unexpected(DomainError{
            "ap_cos", "argument magnitude exceeds the range-reduction limit (10^10000)"});
    return ap_cos(x, prec);
}

Result<APFloat> ap_tan_checked(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.dec_exp() > APFloat::TRIG_ARG_LIMIT)
        return std::unexpected(DomainError{
            "ap_tan", "argument magnitude exceeds the range-reduction limit (10^10000)"});
    if (x.is_zero()) return APFloat::zero(P);
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + std::max(0, x.dec_exp()));
    APFloat s;
    APFloat c;
    ap_sincos_core(x, W, s, c);
    if (c.is_zero())
        return std::unexpected(DomainError{"ap_tan", "tangent at an odd multiple of pi/2"});
    APFloat r = (s / c).with_precision(W);
    r.set_precision(P);
    return r;
}

// ---- inverse trigonometric ----------------------------------------------

APFloat ap_atan(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    const int W = P + ap_guard_digits(P);
    const bool xneg = x.is_negative();
    APFloat a = x.abs().with_precision(W);

    bool recip = false;
    if (a.cmp(APFloat::one(W)) > 0) {          // atan a = pi/2 - atan(1/a)
        a = (APFloat::one(W) / a).with_precision(W);
        recip = true;
    }

    // Rational reduction atan(a) = atan(1/n) + atan((n*a - 1)/(n + a)) with
    // n = round(1/|a|). The atan(1/n) pieces come from the same scaled-integer
    // series that drives ap_pi, which is far cheaper than the square roots the
    // half-angle identity would need. Four levels take |a| <= 1 below 1e-7.
    APFloat acc = APFloat::zero(W);
    for (int level = 0; level < 10; ++level) {
        if (a.is_zero()) break;
        if (a.dec_exp() < -3) break;                   // |a| < 1e-4: the series is fast
        const long long s = a.is_negative() ? -1LL : 1LL;
        const BigInt nb = (APFloat::one(W) / a.abs()).with_precision(W).round();
        long long n = nb.to_ll();
        if (n < 1) n = 1;
        if (n > 10000) break;                          // defensive; unreachable above
        const APFloat piece = (n == 1)
            ? ap_scale_pow2_down(ap_pi(W), 2, W)       // atan(1) = pi/4; the n == 1
                                                       // series would be Leibniz's
            : APFloat(ap_atan_inv_int(static_cast<uint32_t>(n), W), -W, W);
        acc = ((s > 0) ? (acc + piece) : (acc - piece)).with_precision(W);
        const APFloat num = (ap_mul_small_ap(a, n, W) - APFloat(s, W)).with_precision(W);
        const APFloat den = (APFloat(n, W) + ((s > 0) ? a : a.neg())).with_precision(W);
        a = (num / den).with_precision(W);
    }

    const APFloat a2 = (a * a).with_precision(W);
    APFloat sum = a;
    APFloat term = a;
    const long long cap = ap_series_cap(W);
    for (long long k = 1; k <= cap; ++k) {
        term = (term * a2).with_precision(W);
        APFloat t = ap_div_small_ap(term, static_cast<uint32_t>(2 * k + 1), W);
        if ((k % 2) == 1) t = t.neg();
        if (ap_negligible(sum, t, W)) break;
        sum = (sum + t).with_precision(W);
    }
    APFloat res = (acc + sum).with_precision(W);
    if (recip) res = (ap_scale_pow2_down(ap_pi(W), 1, W) - res).with_precision(W);
    if (xneg) res = res.neg();
    res.set_precision(P);
    return res;
}

APFloat ap_asin(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    if (x.is_zero()) return APFloat::zero(P);
    if (x.abs().cmp(APFloat::one(W)) > 0) return APFloat::zero(P);
    // 1 - x^2 is formed at a precision that makes it exact, so the usual
    // cancellation as |x| -> 1 simply does not happen.
    const int EP = ap_clamp_precision(std::max(W, 2 * ap_digit_count(x.mantissa) + 8));
    const APFloat xx = x.with_precision(EP);
    const APFloat u = (APFloat::one(EP) - (xx * xx).with_precision(EP)).with_precision(EP);
    if (u.is_zero()) {                                  // |x| == 1
        APFloat h = ap_scale_pow2_down(ap_pi(W), 1, W);
        if (x.is_negative()) h = h.neg();
        h.set_precision(P);
        return h;
    }
    const APFloat den = (APFloat::one(W) + ap_sqrt(u, W)).with_precision(W);
    APFloat r = ap_mul_small_ap(ap_atan((xx / den).with_precision(W), W), 2LL, W);
    r.set_precision(P);
    return r;
}

Result<APFloat> ap_asin_checked(const APFloat& x, int prec) {
    const int W = ap_clamp_precision(prec) + 2;
    if (x.abs().cmp(APFloat::one(W)) > 0)
        return std::unexpected(DomainError{"ap_asin", "argument outside [-1, 1]"});
    return ap_asin(x, prec);
}

APFloat ap_acos(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    const APFloat one_ = APFloat::one(W);
    const int c = x.cmp(one_);
    if (x.abs().cmp(one_) > 0) return APFloat::zero(P);
    if (c == 0) return APFloat::zero(P);                // acos(1) == 0
    if (x.cmp(one_.neg()) == 0) {                       // acos(-1) == pi
        APFloat p = ap_pi(W);
        p.set_precision(P);
        return p;
    }
    // acos(x) = 2*atan(sqrt((1-x)/(1+x))). Both 1-x and 1+x are formed exactly, and
    // unlike pi/2 - asin(x) this form does not cancel as x -> 1.
    const int EP = ap_clamp_precision(std::max(W, ap_digit_count(x.mantissa) + 8));
    const APFloat xx = x.with_precision(EP);
    const APFloat num = (APFloat::one(EP) - xx).with_precision(EP);
    const APFloat den = (APFloat::one(EP) + xx).with_precision(EP);
    const APFloat q = (num / den).with_precision(W);
    APFloat r = ap_mul_small_ap(ap_atan(ap_sqrt(q, W), W), 2LL, W);
    r.set_precision(P);
    return r;
}

Result<APFloat> ap_acos_checked(const APFloat& x, int prec) {
    const int W = ap_clamp_precision(prec) + 2;
    if (x.abs().cmp(APFloat::one(W)) > 0)
        return std::unexpected(DomainError{"ap_acos", "argument outside [-1, 1]"});
    return ap_acos(x, prec);
}

APFloat ap_atan2(const APFloat& y, const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = P + ap_guard_digits(P);
    const int sx = x.sign();
    const int sy = y.sign();
    if (sx == 0 && sy == 0) return APFloat::zero(P);    // no value; degrade to zero
    APFloat r(0LL, W);
    if (sx > 0) {
        r = ap_atan((y / x).with_precision(W), W);
    } else if (sx < 0) {
        const APFloat base = ap_atan((y / x).with_precision(W), W);
        const APFloat pi = ap_pi(W);
        r = (sy >= 0) ? (base + pi).with_precision(W) : (base - pi).with_precision(W);
    } else {
        r = ap_scale_pow2_down(ap_pi(W), 1, W);
        if (sy < 0) r = r.neg();
    }
    r.set_precision(P);
    return r;
}

// ---- powers --------------------------------------------------------------

APFloat ap_pow_int(const APFloat& x, long long n, int prec) {
    const int P = ap_clamp_precision(prec);
    if (n == 0) return APFloat::one(P);                 // 0^0 == 1, as bigint_pow has it
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    if (n < 0) {
        if (x.is_zero()) return APFloat::zero(P);
        long long m = n;
        if (m == (std::numeric_limits<long long>::min)())
            m = (std::numeric_limits<long long>::max)();  // clamp; the result saturates
        else
            m = -m;
        const APFloat p = ap_pow_int(x, m, W);
        if (p.is_zero()) return APFloat::zero(P);
        APFloat r = (APFloat::one(W) / p).with_precision(W);
        r.set_precision(P);
        return r;
    }
    APFloat base = x.with_precision(W);
    APFloat acc = APFloat::one(W);
    long long e = n;
    while (e > 0) {
        if ((e & 1LL) != 0) acc = (acc * base).with_precision(W);
        e >>= 1;
        if (e > 0) base = (base * base).with_precision(W);
    }
    acc.set_precision(P);
    return acc;
}

APFloat ap_pow(const APFloat& x, const APFloat& y, int prec) {
    const int P = ap_clamp_precision(prec);
    if (y.is_zero()) return APFloat::one(P);
    if (x.is_zero()) return APFloat::zero(P);
    // An integer exponent is both faster and exact, and it is the only way a
    // negative base has a real value.
    if (y.is_integer() && y.dec_exp() <= 18) return ap_pow_int(x, y.to_ll(), P);
    if (x.is_negative()) return APFloat::zero(P);
    const int W = P + ap_guard_digits(P);
    const APFloat t0 = (y * ap_log(x, W + 8)).with_precision(W + 8);
    // exp's relative error is the absolute error of its argument, so a large
    // y*log(x) needs that many more digits; one recomputation is enough.
    const int ex = std::max(0, t0.dec_exp());
    const APFloat t = (ex == 0)
        ? t0
        : (y * ap_log(x, ap_clamp_precision(W + 8 + ex)))
              .with_precision(ap_clamp_precision(W + 8 + ex));
    return ap_exp(t, P);
}

Result<APFloat> ap_pow_checked(const APFloat& x, const APFloat& y, int prec) {
    const int P = ap_clamp_precision(prec);
    if (y.is_zero()) return APFloat::one(P);
    if (x.is_zero()) {
        if (y.is_negative())
            return std::unexpected(DomainError{"ap_pow",
                                               "zero raised to a non-positive power"});
        return APFloat::zero(P);
    }
    if (x.is_negative() && !(y.is_integer() && y.dec_exp() <= 18))
        return std::unexpected(DomainError{"ap_pow",
                                           "negative base with a non-integer exponent"});
    const APFloat r = ap_pow(x, y, P);
    if (r.is_saturated()) return std::unexpected(OverflowError{"ap_pow"});
    return r;
}

// ---- hyperbolic ----------------------------------------------------------

APFloat ap_sinh(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + std::max(0, x.dec_exp()));
    APFloat r(0LL, W);
    if (x.dec_exp() <= 0) {
        // |x| < 1: (e^x - e^-x)/2 would cancel away -dec_exp(x) digits, so sum the
        // Taylor series instead.
        const APFloat xx = x.with_precision(W);
        const APFloat x2 = (xx * xx).with_precision(W);
        APFloat sum = xx;
        APFloat term = xx;
        const long long cap = ap_series_cap(W);
        for (long long k = 1; k <= cap; ++k) {
            term = (term * x2).with_precision(W);
            term = ap_div_small_ap(term, static_cast<uint32_t>(2 * k), W);
            term = ap_div_small_ap(term, static_cast<uint32_t>(2 * k + 1), W);
            if (ap_negligible(sum, term, W)) break;
            sum = (sum + term).with_precision(W);
        }
        r = sum;
    } else {
        const APFloat e = ap_exp(x, W);
        if (e.is_zero()) return APFloat::zero(P);
        const APFloat inv = (APFloat::one(W) / e).with_precision(W);
        r = ap_scale_pow2_down((e - inv).with_precision(W), 1, W);
    }
    r.set_precision(P);
    return r;
}

APFloat ap_cosh(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::one(P);
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + std::max(0, x.dec_exp()));
    const APFloat e = ap_exp(x.abs(), W);
    if (e.is_saturated()) return APFloat::saturated(1, P);
    const APFloat inv = (APFloat::one(W) / e).with_precision(W);
    APFloat r = ap_scale_pow2_down((e + inv).with_precision(W), 1, W);
    r.set_precision(P);
    return r;
}

APFloat ap_tanh(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + std::max(0, x.dec_exp()));
    const APFloat c = ap_cosh(x, W);
    if (c.is_zero() || c.is_saturated()) {
        APFloat one_ = APFloat::one(P);
        return x.is_negative() ? one_.neg() : one_;
    }
    const APFloat s = ap_sinh(x, W);
    APFloat r = (s / c).with_precision(W);
    r.set_precision(P);
    return r;
}

APFloat ap_asinh(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    // For small |x| the argument of the logarithm is 1 + |x| + O(x^2), so |x| has to
    // survive next to the 1: carry -dec_exp(x) extra digits.
    const int c = std::min(APFloat::MAX_PRECISION, std::max(0, -x.dec_exp()));
    const int W = ap_clamp_precision(P + ap_guard_digits(P) + c);
    const APFloat a = x.abs().with_precision(W);
    const APFloat u = ((a * a).with_precision(W) + APFloat::one(W)).with_precision(W);
    APFloat r = ap_log((a + ap_sqrt(u, W)).with_precision(W), W);
    r.set_precision(P);
    return x.is_negative() ? r.neg() : r;
}

APFloat ap_acosh(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W0 = P + ap_guard_digits(P);
    if (x.cmp(APFloat::one(W0)) < 0) return APFloat::zero(P);
    const int EP0 = ap_clamp_precision(std::max(W0, ap_digit_count(x.mantissa) + 8));
    const APFloat d = (x.with_precision(EP0) - APFloat::one(EP0)).with_precision(EP0);
    if (d.is_zero()) return APFloat::zero(P);           // acosh(1) == 0
    // x + sqrt(x^2-1) approaches 1 like sqrt(2*(x-1)) as x -> 1, so allow for that
    // many digits of cancellation inside the logarithm.
    const int c = std::min(APFloat::MAX_PRECISION, std::max(0, -d.dec_exp()));
    const int W = ap_clamp_precision(W0 + c);
    const int EP = ap_clamp_precision(std::max(W, 2 * ap_digit_count(x.mantissa) + 8));
    const APFloat xx = x.with_precision(EP);
    const APFloat u = ((xx * xx).with_precision(EP) - APFloat::one(EP)).with_precision(EP);
    APFloat r = ap_log((xx + ap_sqrt(u, W)).with_precision(W), W);
    r.set_precision(P);
    return r;
}

Result<APFloat> ap_acosh_checked(const APFloat& x, int prec) {
    const int W = ap_clamp_precision(prec) + 2;
    if (x.cmp(APFloat::one(W)) < 0)
        return std::unexpected(DomainError{"ap_acosh", "argument below 1"});
    return ap_acosh(x, prec);
}

APFloat ap_atanh(const APFloat& x, int prec) {
    const int P = ap_clamp_precision(prec);
    if (x.is_zero()) return APFloat::zero(P);
    const int W0 = P + ap_guard_digits(P);
    if (x.abs().cmp(APFloat::one(W0)) >= 0) return APFloat::zero(P);
    const int c = std::min(APFloat::MAX_PRECISION, std::max(0, -x.dec_exp()));
    const int W = ap_clamp_precision(W0 + c);
    const int EP = ap_clamp_precision(std::max(W, ap_digit_count(x.mantissa) + 8));
    const APFloat xx = x.with_precision(EP);
    const APFloat num = (APFloat::one(EP) + xx).with_precision(EP);
    const APFloat den = (APFloat::one(EP) - xx).with_precision(EP);
    APFloat r = ap_scale_pow2_down(ap_log((num / den).with_precision(W), W), 1, W);
    r.set_precision(P);
    return r;
}

Result<APFloat> ap_atanh_checked(const APFloat& x, int prec) {
    const int W = ap_clamp_precision(prec) + 2;
    if (x.abs().cmp(APFloat::one(W)) >= 0)
        return std::unexpected(DomainError{"ap_atanh", "argument outside (-1, 1)"});
    return ap_atanh(x, prec);
}

// ========================== APComplex ==========================

APComplex::APComplex() : re(), im() {}

APComplex::APComplex(int precision)
    : re(APFloat::zero(precision)), im(APFloat::zero(precision)) {}

APComplex::APComplex(APFloat r)
    : re(std::move(r)), im(APFloat::zero(re.precision())) {}

APComplex::APComplex(APFloat r, APFloat i) : re(std::move(r)), im(std::move(i)) {
    const int p = std::max(re.precision(), im.precision());
    re.set_precision(p);
    im.set_precision(p);
}

APComplex::APComplex(long long r, long long i, int precision)
    : re(r, precision), im(i, precision) {}

APComplex APComplex::from_doubles(double r, double i, int precision) {
    return APComplex(APFloat(r, precision), APFloat(i, precision));
}

APComplex APComplex::zero(int precision) { return APComplex(APFloat::zero(precision)); }
APComplex APComplex::one(int precision)  { return APComplex(APFloat::one(precision)); }
APComplex APComplex::i_unit(int precision) {
    return APComplex(APFloat::zero(precision), APFloat::one(precision));
}

void APComplex::set_precision(int p) {
    re.set_precision(p);
    im.set_precision(p);
}

APComplex APComplex::with_precision(int p) const {
    APComplex r = *this;
    r.set_precision(p);
    return r;
}

bool APComplex::is_zero() const { return re.is_zero() && im.is_zero(); }
bool APComplex::is_real() const { return im.is_zero(); }

APComplex APComplex::operator+(const APComplex& o) const {
    return APComplex(re + o.re, im + o.im);
}
APComplex APComplex::operator-(const APComplex& o) const {
    return APComplex(re - o.re, im - o.im);
}
APComplex APComplex::operator*(const APComplex& o) const {
    return APComplex((re * o.re - im * o.im), (re * o.im + im * o.re));
}
APComplex APComplex::operator/(const APComplex& o) const {
    const int P = std::max(precision(), o.precision());
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    const APFloat q = ((o.re * o.re).with_precision(W)
                     + (o.im * o.im).with_precision(W)).with_precision(W);
    if (q.is_zero()) return APComplex::zero(P);
    const APFloat nr = ((re * o.re).with_precision(W)
                      + (im * o.im).with_precision(W)).with_precision(W);
    const APFloat ni = ((im * o.re).with_precision(W)
                      - (re * o.im).with_precision(W)).with_precision(W);
    APComplex r((nr / q).with_precision(W), (ni / q).with_precision(W));
    r.set_precision(P);
    return r;
}
APComplex APComplex::operator-() const { return APComplex(re.neg(), im.neg()); }

bool APComplex::operator==(const APComplex& o) const {
    return re.cmp(o.re) == 0 && im.cmp(o.im) == 0;
}

APComplex APComplex::conj() const { return APComplex(re, im.neg()); }

std::string APComplex::to_string(int digits) const {
    const std::string rs = re.to_string(digits);
    const bool neg = im.is_negative();
    const std::string is = im.abs().to_string(digits);
    return rs + (neg ? " - " : " + ") + is + "i";
}

APFloat ap_cabs(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    if (z.im.is_zero()) return z.re.abs().with_precision(P);
    if (z.re.is_zero()) return z.im.abs().with_precision(P);
    const APFloat s = ((z.re * z.re).with_precision(W)
                     + (z.im * z.im).with_precision(W)).with_precision(W);
    return ap_sqrt(s, P);
}

APFloat ap_carg(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    if (z.is_zero()) return APFloat::zero(P);
    return ap_atan2(z.im, z.re, P);
}

APComplex ap_cconj(const APComplex& z) { return z.conj(); }

APComplex ap_cexp(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    const APFloat e = ap_exp(z.re, W);
    APFloat s;
    APFloat c;
    ap_sincos_core(z.im, W, s, c);
    APComplex r((e * c).with_precision(W), (e * s).with_precision(W));
    r.set_precision(P);
    return r;
}

APComplex ap_clog(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    if (z.is_zero()) return APComplex::zero(P);
    // log|z| == log(re^2 + im^2)/2: the sum of squares is exact, and this avoids a
    // square root entirely.
    const APFloat s = ((z.re * z.re).with_precision(W)
                     + (z.im * z.im).with_precision(W)).with_precision(W);
    const APFloat lr = ap_scale_pow2_down(ap_log(s, W), 1, W);
    const APFloat li = ap_atan2(z.im, z.re, W);
    APComplex r(lr, li);
    r.set_precision(P);
    return r;
}

Result<APComplex> ap_clog_checked(const APComplex& z, int prec) {
    if (z.is_zero())
        return std::unexpected(DomainError{"ap_clog", "logarithm of zero"});
    return ap_clog(z, prec);
}

APComplex ap_csqrt(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    const APFloat r = ap_cabs(z, W);
    if (r.is_zero()) return APComplex::zero(P);
    // Principal branch in the numerically stable form: one of the two components is
    // built from (|z| + |re|)/2, which never cancels, and the other from a division
    // by it. sgn(0) is taken as +1, so sqrt(-4) is +2i.
    const APFloat w = ap_sqrt(ap_scale_pow2_down((r + z.re.abs()).with_precision(W), 1, W), W);
    const APFloat w2 = ap_mul_small_ap(w, 2LL, W);
    APComplex out(APFloat::zero(W), APFloat::zero(W));
    if (!z.re.is_negative()) {
        out = APComplex(w, (z.im / w2).with_precision(W));
    } else {
        const APFloat a = (z.im.abs() / w2).with_precision(W);
        out = APComplex(a, z.im.is_negative() ? w.neg() : w);
    }
    out.set_precision(P);
    return out;
}

APComplex ap_csin(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    APFloat s;
    APFloat c;
    ap_sincos_core(z.re, W, s, c);
    const APFloat ch = ap_cosh(z.im, W);
    const APFloat sh = ap_sinh(z.im, W);
    APComplex r((s * ch).with_precision(W), (c * sh).with_precision(W));
    r.set_precision(P);
    return r;
}

APComplex ap_ccos(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    APFloat s;
    APFloat c;
    ap_sincos_core(z.re, W, s, c);
    const APFloat ch = ap_cosh(z.im, W);
    const APFloat sh = ap_sinh(z.im, W);
    APComplex r((c * ch).with_precision(W), (s * sh).with_precision(W).neg());
    r.set_precision(P);
    return r;
}

APComplex ap_ctan(const APComplex& z, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    const APComplex c = ap_ccos(z, W);
    if (c.is_zero()) return APComplex::zero(P);
    APComplex r = ap_csin(z, W) / c;
    r.set_precision(P);
    return r;
}

APComplex ap_cpow_int(const APComplex& z, long long n, int prec) {
    const int P = ap_clamp_precision(prec);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    if (n == 0) return APComplex::one(P);
    if (n < 0) {
        if (z.is_zero()) return APComplex::zero(P);
        long long m = n;
        if (m == (std::numeric_limits<long long>::min)())
            m = (std::numeric_limits<long long>::max)();
        else
            m = -m;
        const APComplex p = ap_cpow_int(z, m, W);
        if (p.is_zero()) return APComplex::zero(P);
        APComplex r = APComplex::one(W) / p;
        r.set_precision(P);
        return r;
    }
    APComplex base = z.with_precision(W);
    APComplex acc = APComplex::one(W);
    long long e = n;
    while (e > 0) {
        if ((e & 1LL) != 0) acc = acc * base;
        e >>= 1;
        if (e > 0) base = base * base;
    }
    acc.set_precision(P);
    return acc;
}

APComplex ap_cpow(const APComplex& z, const APComplex& w, int prec) {
    const int P = ap_clamp_precision(prec);
    if (w.is_zero()) return APComplex::one(P);
    if (z.is_zero()) return APComplex::zero(P);
    if (w.is_real() && w.re.is_integer() && w.re.dec_exp() <= 18)
        return ap_cpow_int(z, w.re.to_ll(), P);
    const int W = ap_clamp_precision(P + ap_guard_digits(P));
    const APComplex t0 = (w * ap_clog(z, W + 8)).with_precision(W + 8);
    const int ex = std::max(std::max(0, t0.re.dec_exp()), std::max(0, t0.im.dec_exp()));
    const APComplex t = (ex == 0)
        ? t0
        : (w * ap_clog(z, ap_clamp_precision(W + 8 + ex)))
              .with_precision(ap_clamp_precision(W + 8 + ex));
    return ap_cexp(t, P);
}

Result<APComplex> ap_cpow_checked(const APComplex& z, const APComplex& w, int prec) {
    const int P = ap_clamp_precision(prec);
    if (w.is_zero()) return APComplex::one(P);
    if (z.is_zero()) {
        if (w.is_real() && !w.re.is_negative()) return APComplex::zero(P);
        return std::unexpected(DomainError{"ap_cpow",
                                           "zero raised to a non-positive power"});
    }
    return ap_cpow(z, w, P);
}

} // namespace bignum
} // namespace ms
