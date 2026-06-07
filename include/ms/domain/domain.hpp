#pragma once

#include <cstddef>
#include <vector>

namespace ms {

size_t factorial(size_t n);
size_t nchoosek(size_t n, size_t k);
size_t gcd(size_t a, size_t b);

struct Graph {
    size_t n = 0;
    std::vector<std::vector<size_t>> adj;
};

size_t graph_num_edges(const Graph& g);

} // namespace ms
