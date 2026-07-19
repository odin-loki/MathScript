#include "ms/combo/combo.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace ms {
namespace combo {

uint64_t factorial(uint32_t n) {
    if (n > 20) return UINT64_MAX; // overflow sentinel
    uint64_t r = 1;
    for (uint32_t i = 2; i <= n; ++i) r *= i;
    return r;
}

uint64_t double_factorial(uint32_t n) {
    if (n == 0 || n == 1) return 1;
    uint64_t r = 1;
    for (uint32_t i = n; i >= 2; i -= 2) r *= i;
    return r;
}

uint64_t subfactorial(uint32_t n) {
    if (n == 0) return 1;
    if (n == 1) return 0;
    // D(n) = (n-1) * (D(n-1) + D(n-2))
    uint64_t a = 1, b = 0;
    for (uint32_t i = 2; i <= n; ++i) {
        uint64_t c = (i - 1) * (a + b);
        a = b; b = c;
    }
    return b;
}

uint64_t binomial(uint32_t n, uint32_t k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    uint64_t r = 1;
    for (uint32_t i = 0; i < k; ++i) {
        r = r * (n - i) / (i + 1);
    }
    return r;
}

uint64_t multinomial(uint32_t n, const std::vector<uint32_t>& ks) {
    uint64_t sum = 0;
    for (auto k : ks) sum += k;
    if (sum != n) return 0;
    uint64_t r = factorial(n);
    for (auto k : ks) r /= factorial(k);
    return r;
}

uint64_t permutations(uint32_t n, uint32_t k) {
    if (k > n) return 0;
    uint64_t r = 1;
    for (uint32_t i = n; i > n - k; --i) r *= i;
    return r;
}

uint64_t combinations(uint32_t n, uint32_t k) {
    return binomial(n, k);
}

uint64_t combinations_with_rep(uint32_t n, uint32_t k) {
    return binomial(n + k - 1, k);
}

bool next_perm(std::vector<int>& v) {
    return std::next_permutation(v.begin(), v.end());
}

bool prev_perm(std::vector<int>& v) {
    return std::prev_permutation(v.begin(), v.end());
}

bool next_comb(std::vector<int>& v, int n) {
    int k = static_cast<int>(v.size());
    int i = k - 1;
    while (i >= 0 && v[i] == n - k + i) --i;
    if (i < 0) return false;
    ++v[i];
    for (int j = i + 1; j < k; ++j) v[j] = v[i] + j - i;
    return true;
}

bool prev_comb(std::vector<int>& v, int n) {
    int k = static_cast<int>(v.size());
    int i = k - 1;
    while (i >= 0) {
        const int min_val = (i == 0) ? 0 : v[i - 1] + 1;
        if (v[i] > min_val) {
            break;
        }
        --i;
    }
    if (i < 0) {
        return false;
    }
    --v[i];
    for (int j = i + 1; j < k; ++j) {
        v[j] = n - k + j;
    }
    return true;
}

uint64_t rank_permutation(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    uint64_t rank = 0;
    std::vector<bool> used(n, false);
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        for (int j = 0; j < v[i]; ++j) if (!used[j]) ++cnt;
        rank += static_cast<uint64_t>(cnt) * factorial(n - 1 - i);
        used[v[i]] = true;
    }
    return rank;
}

std::vector<int> unrank_permutation(int n, uint64_t rank) {
    std::vector<int> v;
    v.reserve(static_cast<size_t>(n));
    std::vector<int> avail(n);
    std::iota(avail.begin(), avail.end(), 0);
    for (int i = n; i > 0; --i) {
        uint64_t f = factorial(i - 1);
        int idx = static_cast<int>(rank / f);
        v.push_back(avail[idx]);
        avail.erase(avail.begin() + idx);
        rank %= f;
    }
    return v;
}

uint64_t rank_combination(const std::vector<int>& v, int n) {
    int k = static_cast<int>(v.size());
    uint64_t rank = 0;
    for (int i = 0; i < k; ++i) {
        int vi = (i == 0) ? 0 : v[i - 1] + 1;
        for (int j = vi; j < v[i]; ++j)
            rank += binomial(n - 1 - j, k - 1 - i);
    }
    return rank;
}

std::vector<int> unrank_combination(int n, int k, uint64_t rank) {
    std::vector<int> v;
    int start = 0;
    for (int i = k; i > 0; --i) {
        for (int j = start; j < n; ++j) {
            uint64_t c = binomial(n - 1 - j, i - 1);
            if (rank < c) { v.push_back(j); start = j + 1; break; }
            rank -= c;
        }
    }
    return v;
}

std::vector<std::vector<int>> all_permutations(int n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::vector<std::vector<int>> result;
    if (n <= 20)
        result.reserve(static_cast<size_t>(factorial(static_cast<uint32_t>(n))));
    do { result.push_back(v); } while (std::next_permutation(v.begin(), v.end()));
    return result;
}

std::vector<std::vector<int>> all_subsets(int n) {
    std::vector<std::vector<int>> result;
    if (n >= 0 && n < 31)
        result.reserve(static_cast<size_t>(1) << n);
    for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> subset;
        for (int i = 0; i < n; ++i)
            if (mask & (1 << i)) subset.push_back(i);
        result.push_back(std::move(subset));
    }
    return result;
}

static void compositions_helper(int n, int max_parts, int min_val,
                                  std::vector<int>& current,
                                  std::vector<std::vector<int>>& result) {
    if (n == 0) { result.push_back(current); return; }
    if (max_parts == 0) return;
    for (int i = 1; i <= n; ++i) {
        current.push_back(i);
        compositions_helper(n - i, max_parts - 1, i, current, result);
        current.pop_back();
    }
}

std::vector<std::vector<int>> all_compositions(int n, int max_parts) {
    if (max_parts < 0) max_parts = n;
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    compositions_helper(n, max_parts, 1, current, result);
    return result;
}

static void partitions_helper(int n, int max_part, std::vector<int>& current,
                                std::vector<std::vector<int>>& result) {
    if (n == 0) { result.push_back(current); return; }
    for (int i = std::min(n, max_part); i >= 1; --i) {
        current.push_back(i);
        partitions_helper(n - i, i, current, result);
        current.pop_back();
    }
}

std::vector<std::vector<int>> all_partitions(int n) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    partitions_helper(n, n, current, result);
    return result;
}

static void restricted_partitions_helper(int n, int max_part, int parts_left,
                                           std::vector<int>& current,
                                           std::vector<std::vector<int>>& result) {
    if (parts_left == 0) {
        if (n == 0) result.push_back(current);
        return;
    }
    if (n < parts_left) return;
    if (parts_left == 1) {
        if (n <= max_part) {
            current.push_back(n);
            result.push_back(current);
            current.pop_back();
        }
        return;
    }
    int upper = std::min(max_part, n - (parts_left - 1));
    for (int i = upper; i >= 1; --i) {
        current.push_back(i);
        restricted_partitions_helper(n - i, i, parts_left - 1, current, result);
        current.pop_back();
    }
}

std::vector<std::vector<int>> restricted_partitions(int n, int k) {
    if (n < 0) return {};
    if (n == 0) return k == 0 ? std::vector<std::vector<int>>{std::vector<int>{}} : std::vector<std::vector<int>>{};
    if (k <= 0) return {};
    if (k > n) return {};
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    restricted_partitions_helper(n, n, k, current, result);
    return result;
}

static void derangements_helper(int pos, std::vector<int>& perm,
                                  std::vector<bool>& used, int n,
                                  std::vector<std::vector<int>>& result) {
    if (pos == n) { result.push_back(perm); return; }
    for (int i = 0; i < n; ++i) {
        if (!used[i] && i != pos) {
            used[i] = true;
            perm[pos] = i;
            derangements_helper(pos + 1, perm, used, n, result);
            used[i] = false;
        }
    }
}

std::vector<std::vector<int>> derangements(int n) {
    std::vector<std::vector<int>> result;
    std::vector<int> perm(n, -1);
    std::vector<bool> used(n, false);
    derangements_helper(0, perm, used, n, result);
    return result;
}

uint64_t catalan_num(uint32_t n) {
    return binomial(2 * n, n) / (n + 1);
}

uint64_t stirling1(uint32_t n, uint32_t k) {
    if (n == 0 && k == 0) return 1;
    if (n == 0 || k == 0) return 0;
    if (k > n) return 0;
    // |s(n,k)| = (n-1)*|s(n-1,k)| + |s(n-1,k-1)|
    std::vector<std::vector<uint64_t>> s(n + 1, std::vector<uint64_t>(k + 1, 0));
    s[0][0] = 1;
    for (uint32_t i = 1; i <= n; ++i)
        for (uint32_t j = 1; j <= std::min(i, k); ++j)
            s[i][j] = (i - 1) * s[i-1][j] + s[i-1][j-1];
    return s[n][k];
}

uint64_t stirling2(uint32_t n, uint32_t k) {
    if (n == 0 && k == 0) return 1;
    if (n == 0 || k == 0) return 0;
    if (k > n) return 0;
    std::vector<std::vector<uint64_t>> s(n + 1, std::vector<uint64_t>(k + 1, 0));
    s[0][0] = 1;
    for (uint32_t i = 1; i <= n; ++i)
        for (uint32_t j = 1; j <= std::min(i, k); ++j)
            s[i][j] = j * s[i-1][j] + s[i-1][j-1];
    return s[n][k];
}

uint64_t eulerian_number(uint32_t n, uint32_t k) {
    if (n == 0) return k == 0 ? 1 : 0;
    if (k >= n) return 0;
    if (k == 0) return 1;
    // A(n,k) = (k+1)*A(n-1,k) + (n-k)*A(n-1,k-1)
    std::vector<std::vector<uint64_t>> a(n + 1, std::vector<uint64_t>(k + 1, 0));
    a[0][0] = 1;
    for (uint32_t i = 1; i <= n; ++i) {
        a[i][0] = 1;
        for (uint32_t j = 1; j <= std::min(i - 1, k); ++j)
            a[i][j] = (j + 1) * a[i - 1][j] + (i - j) * a[i - 1][j - 1];
    }
    return a[n][k];
}

uint64_t bell_num(uint32_t n) {
    // Bell triangle method
    if (n == 0) return 1;
    std::vector<uint64_t> row = {1};
    for (uint32_t i = 1; i <= n; ++i) {
        std::vector<uint64_t> next(i + 1);
        next[0] = row.back();
        for (uint32_t j = 1; j <= i; ++j)
            next[j] = next[j-1] + row[j-1];
        row = next;
    }
    return row[0];
}

uint64_t motzkin_num(uint32_t n) {
    // M(n) = M(n-1) + sum_{k=0}^{n-2} M(k)*M(n-2-k)
    if (n == 0 || n == 1) return 1;
    std::vector<uint64_t> m(n + 1, 0);
    m[0] = 1; m[1] = 1;
    for (uint32_t i = 2; i <= n; ++i) {
        m[i] = m[i-1];
        for (uint32_t k = 0; k + 2 <= i; ++k)
            m[i] += m[k] * m[i - 2 - k];
    }
    return m[n];
}

static void set_partitions_extend(
    const std::vector<std::vector<std::vector<int>>>& prev, int elem,
    std::vector<std::vector<std::vector<int>>>& result) {
    for (const auto& part : prev) {
        auto with_new_block = part;
        with_new_block.push_back({elem});
        result.push_back(std::move(with_new_block));
        for (size_t i = 0; i < part.size(); ++i) {
            auto extended = part;
            extended[i].push_back(elem);
            result.push_back(std::move(extended));
        }
    }
}

std::vector<std::vector<std::vector<int>>> set_partitions(int n) {
    if (n == 0) return {{}};
    auto prev = set_partitions(n - 1);
    std::vector<std::vector<std::vector<int>>> result;
    set_partitions_extend(prev, n - 1, result);
    return result;
}

uint64_t involutions(uint32_t n) {
    if (n == 0 || n == 1) return 1;
    uint64_t prev2 = 1, prev1 = 1;
    for (uint32_t i = 2; i <= n; ++i) {
        uint64_t cur = prev1 + static_cast<uint64_t>(i - 1) * prev2;
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

static void dyck_paths_helper(int n, int open, int close, std::string& current,
                              std::vector<std::string>& result) {
    if (static_cast<int>(current.size()) == 2 * n) {
        result.push_back(current);
        return;
    }
    if (open < n) {
        current.push_back('(');
        dyck_paths_helper(n, open + 1, close, current, result);
        current.pop_back();
    }
    if (close < open) {
        current.push_back(')');
        dyck_paths_helper(n, open, close + 1, current, result);
        current.pop_back();
    }
}

std::vector<std::string> dyck_paths(int n) {
    if (n < 0) return {};
    std::vector<std::string> result;
    std::string current;
    dyck_paths_helper(n, 0, 0, current, result);
    return result;
}

static void motzkin_paths_helper(int n, int pos, int height, std::string& current,
                                 std::vector<std::string>& result) {
    if (pos == n) {
        if (height == 0) result.push_back(current);
        return;
    }
    current.push_back('U');
    motzkin_paths_helper(n, pos + 1, height + 1, current, result);
    current.pop_back();

    current.push_back('F');
    motzkin_paths_helper(n, pos + 1, height, current, result);
    current.pop_back();

    if (height > 0) {
        current.push_back('D');
        motzkin_paths_helper(n, pos + 1, height - 1, current, result);
        current.pop_back();
    }
}

std::vector<std::string> motzkin_paths(int n) {
    if (n < 0) return {};
    std::vector<std::string> result;
    std::string current;
    motzkin_paths_helper(n, 0, 0, current, result);
    return result;
}

static bool is_necklace_representative(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    for (int shift = 1; shift < n; ++shift) {
        for (int i = 0; i < n; ++i) {
            int rotated = v[(i + shift) % n];
            if (rotated < v[i]) return false;
            if (rotated > v[i]) break;
        }
    }
    return true;
}

static void necklaces_odometer(std::vector<int>& v, int k) {
    int n = static_cast<int>(v.size());
    int i = n - 1;
    while (i >= 0 && v[i] == k - 1) {
        v[i] = 0;
        --i;
    }
    if (i >= 0) ++v[i];
}

std::vector<std::vector<int>> necklaces(int n, int k) {
    if (n < 0 || k <= 0) return {};
    if (n == 0) return {{}};
    std::vector<int> v(n, 0);
    std::vector<std::vector<int>> result;
    while (true) {
        if (is_necklace_representative(v))
            result.push_back(v);
        necklaces_odometer(v, k);
        if (std::all_of(v.begin(), v.end(), [](int x) { return x == 0; }))
            break;
    }
    return result;
}

static bool is_bracelet_representative(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    // v must be <= every one of its own rotations (necklace condition).
    for (int shift = 1; shift < n; ++shift) {
        for (int i = 0; i < n; ++i) {
            int rotated = v[(i + shift) % n];
            if (rotated < v[i]) return false;
            if (rotated > v[i]) break;
        }
    }
    // v must also be <= every rotation of its reversal (reflection condition).
    std::vector<int> rev(v.rbegin(), v.rend());
    for (int shift = 0; shift < n; ++shift) {
        for (int i = 0; i < n; ++i) {
            int reflected = rev[(i + shift) % n];
            if (reflected < v[i]) return false;
            if (reflected > v[i]) break;
        }
    }
    return true;
}

std::vector<std::vector<int>> bracelets(int n, int k) {
    if (n < 0 || k <= 0) return {};
    if (n == 0) return {{}};
    std::vector<int> v(n, 0);
    std::vector<std::vector<int>> result;
    while (true) {
        if (is_bracelet_representative(v))
            result.push_back(v);
        necklaces_odometer(v, k);
        if (std::all_of(v.begin(), v.end(), [](int x) { return x == 0; }))
            break;
    }
    return result;
}

static bool is_aperiodic(const std::vector<int>& v) {
    int n = static_cast<int>(v.size());
    if (n <= 1) return true;
    for (int period = 1; period < n; ++period) {
        if (n % period != 0) continue;
        bool periodic = true;
        for (int i = period; i < n; ++i) {
            if (v[i] != v[i - period]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return false;
    }
    return true;
}

std::vector<std::vector<int>> lyndon_words(int n, int k) {
    if (n < 0 || k <= 0) return {};
    auto reps = necklaces(n, k);
    std::vector<std::vector<int>> result;
    result.reserve(reps.size());
    for (const auto& w : reps) {
        if (is_aperiodic(w))
            result.push_back(w);
    }
    return result;
}

std::vector<std::vector<int>> gray_code(int n) {
    if (n < 0) return {};
    if (n == 0) return {{}};
    int total = 1 << n;
    std::vector<std::vector<int>> result;
    result.reserve(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) {
        int g = i ^ (i >> 1);
        std::vector<int> code(n);
        for (int bit = 0; bit < n; ++bit)
            code[bit] = (g >> (n - 1 - bit)) & 1; // MSB first
        result.push_back(std::move(code));
    }
    return result;
}

// FKM (Fredricksen-Kessler-Maiorana) algorithm: builds B(k,n) as the
// concatenation, in lexicographic order, of all Lyndon words over
// {0,...,k-1} whose length divides n. `a` holds the current candidate
// prefix (1-indexed; a[0] is unused), `t` is the next position to fill,
// and `p` tracks the length of the current longest proper "period" seen
// so far (used to detect when a[1..t-1] is itself a Lyndon word).
static void de_bruijn_helper(int t, int p, int k, int n,
                              std::vector<int>& a, std::vector<int>& seq) {
    if (t > n) {
        if (n % p == 0) {
            for (int i = 1; i <= p; ++i) seq.push_back(a[i]);
        }
        return;
    }
    a[t] = a[t - p];
    de_bruijn_helper(t + 1, p, k, n, a, seq);
    for (int j = a[t - p] + 1; j < k; ++j) {
        a[t] = j;
        de_bruijn_helper(t + 1, t, k, n, a, seq);
    }
}

std::vector<int> de_bruijn_sequence(int k, int n) {
    if (k <= 0 || n <= 0) return {};
    std::vector<int> a(static_cast<std::size_t>(n) + 1, 0);
    std::size_t expected_len = 1;
    for (int i = 0; i < n; ++i) expected_len *= static_cast<std::size_t>(k);
    std::vector<int> seq;
    seq.reserve(expected_len);
    de_bruijn_helper(1, 1, k, n, a, seq);
    return seq;
}

} // namespace combo
} // namespace ms
