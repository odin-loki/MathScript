#pragma once

#include <cstddef>
#include <vector>

namespace ms {

size_t factorial(size_t n);
size_t nchoosek(size_t n, size_t k);
size_t gcd(size_t a, size_t b);

// Least common multiple of a and b, computed as a / gcd(a, b) * b (dividing
// before multiplying to avoid unnecessary overflow). By convention,
// lcm(0, n) == lcm(n, 0) == 0.
size_t lcm(size_t a, size_t b);

// Result of extended_gcd(): the gcd of the inputs plus Bezout coefficients
// x, y satisfying a * x + b * y == gcd.
struct ExtGcdResult {
    long long gcd = 0;
    long long x = 0;
    long long y = 0;
};

// Extended Euclidean algorithm: computes gcd(a, b) together with Bezout
// coefficients x, y such that a * x + b * y == gcd(a, b). Uses signed
// integers (unlike the rest of this module) because Bezout coefficients
// can be negative. By convention, extended_gcd(0, 0) returns {0, 0, 0}.
ExtGcdResult extended_gcd(long long a, long long b);

struct Graph {
    size_t n = 0;
    std::vector<std::vector<size_t>> adj;
};

size_t graph_num_edges(const Graph& g);

} // namespace ms
