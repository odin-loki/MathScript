#include "ms/domain/domain.hpp"

namespace ms {

size_t factorial(size_t n) {
    size_t result = 1;
    for (size_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

size_t nchoosek(size_t n, size_t k) {
    if (k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }
    size_t result = 1;
    for (size_t i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}

size_t gcd(size_t a, size_t b) {
    while (b != 0) {
        const size_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

size_t lcm(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return a / gcd(a, b) * b;
}

ExtGcdResult extended_gcd(long long a, long long b) {
    if (a == 0 && b == 0) {
        return ExtGcdResult{0, 0, 0};
    }

    long long old_r = a, r = b;
    long long old_s = 1, s = 0;
    long long old_t = 0, t = 1;

    while (r != 0) {
        const long long quotient = old_r / r;

        const long long new_r = old_r - quotient * r;
        old_r = r;
        r = new_r;

        const long long new_s = old_s - quotient * s;
        old_s = s;
        s = new_s;

        const long long new_t = old_t - quotient * t;
        old_t = t;
        t = new_t;
    }

    // By convention gcd is non-negative; flip signs if the algorithm
    // produced a negative gcd (happens when both a and b are negative).
    if (old_r < 0) {
        old_r = -old_r;
        old_s = -old_s;
        old_t = -old_t;
    }

    return ExtGcdResult{old_r, old_s, old_t};
}

size_t graph_num_edges(const Graph& g) {
    size_t count = 0;
    for (const auto& row : g.adj) {
        count += row.size();
    }
    return count / 2;
}

} // namespace ms
