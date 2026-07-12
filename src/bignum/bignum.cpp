#include "ms/bignum/bignum.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
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
    unsigned long long uv = negative ? -(unsigned long long)v : (unsigned long long)v;
    digits.clear();
    if (uv == 0) { digits.push_back(0); return; }
    while (uv > 0) { digits.push_back(uv % BASE); uv /= BASE; }
}

BigInt::BigInt(const std::string& s) {
    std::string str = s;
    negative = false;
    if (!str.empty() && str[0] == '-') { negative = true; str = str.substr(1); }
    if (!str.empty() && str[0] == '+') str = str.substr(1);
    digits.clear();
    // Parse from right in chunks of 9
    int i = (int)str.size();
    while (i > 0) {
        int start = std::max(0, i-9);
        std::string chunk = str.substr(start, i-start);
        digits.push_back((uint32_t)std::stoul(chunk));
        i = start;
    }
    if (digits.empty()) digits.push_back(0);
    trim();
    if (is_zero()) negative = false;
}

BigInt::BigInt(const std::string& s, int base) {
    if (base < 2 || base > 36)
        throw std::invalid_argument("BigInt base must be in [2, 36]");
    std::string str = s;
    bool is_neg = false;
    if (!str.empty() && str[0] == '-') { is_neg = true; str = str.substr(1); }
    if (!str.empty() && str[0] == '+') str = str.substr(1);
    if (str.empty()) { digits = {0}; negative = false; return; }

    BigInt result(0LL);
    BigInt bbase(static_cast<long long>(base));
    for (char c : str) {
        int d = digit_value(c);
        if (d < 0 || d >= base)
            throw std::invalid_argument("invalid digit for BigInt base");
        result = result * bbase + BigInt(static_cast<long long>(d));
    }
    *this = std::move(result);
    negative = is_neg && !is_zero();
}

void BigInt::trim() {
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
}

bool BigInt::is_zero() const { return digits.size()==1 && digits[0]==0; }
bool BigInt::is_one()  const { return digits.size()==1 && digits[0]==1 && !negative; }

std::string BigInt::to_string() const {
    if (is_zero()) return "0";
    std::string s = negative ? "-" : "";
    s += std::to_string(digits.back());
    for (int i=(int)digits.size()-2; i>=0; --i) {
        std::string chunk = std::to_string(digits[i]);
        s += std::string(9 - chunk.size(), '0') + chunk;
    }
    return s;
}

std::string BigInt::to_string(int base) const {
    if (base < 2 || base > 36)
        throw std::invalid_argument("BigInt base must be in [2, 36]");
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
    long long v = 0, base = 1;
    for (size_t i=0; i<digits.size() && i<3; ++i) { v += (long long)digits[i]*base; base*=BASE; }
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
    BigInt r; r.digits.clear();
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
    int n=digits.size(), m=o.digits.size();
    BigInt r; r.digits.assign(n+m, 0);
    for (int i=0;i<n;++i) {
        uint64_t carry=0;
        for (int j=0;j<m||carry;++j) {
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
    BigInt quotient; quotient.digits.resize(a.digits.size(),0);
    for (int i=(int)a.digits.size()-1;i>=0;--i) {
        remainder.digits.insert(remainder.digits.begin(), a.digits[i]);
        remainder.trim();
        // Binary search for q
        uint32_t lo=0,hi=BASE-1;
        while (lo<hi) {
            uint32_t mid=(lo+hi+1)/2;
            BigInt t(0LL); t.digits={mid};
            if ((t*divisor).cmp_abs(remainder)<=0) lo=mid; else hi=mid-1;
        }
        quotient.digits[i]=lo;
        BigInt t(0LL); t.digits={lo};
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
    while (!b.is_zero()) { BigInt t=a%b; a=b; b=t; }
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
    for (int i=2;i<=n;++i) { BigInt c=a+b; a=b; b=c; }
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
            int flen = frac_part.size();
            // Remove leading '-' for computation
            bool neg = !int_part.empty() && int_part[0]=='-';
            if (neg) int_part=int_part.substr(1);
            BigInt n_int(int_part.empty()?"0":int_part);
            BigInt n_frac(frac_part.empty()?"0":frac_part);
            BigInt d_pow = bigint_pow(BigInt(10LL), flen);
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

} // namespace bignum
} // namespace ms
