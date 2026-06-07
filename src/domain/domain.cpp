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

size_t graph_num_edges(const Graph& g) {
    size_t count = 0;
    for (const auto& row : g.adj) {
        count += row.size();
    }
    return count / 2;
}

} // namespace ms
