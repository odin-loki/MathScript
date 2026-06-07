#include "ms/linalg/linalg.hpp"
#include "ms/core/rng.hpp"
#include "detail.hpp"
#include <cmath>
#include <random>

namespace ms {

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> rand(size_t m, size_t n, unsigned seed) {
    Matrix<S, StorageOrder::ColMajor, Alloc> R(m, n);
    if (session_rng_active()) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                R(i, j) = static_cast<S>(session_uniform());
            }
        }
        return R;
    }
    std::mt19937 gen(seed);
    std::uniform_real_distribution<S> dist(S(0), S(1));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            R(i, j) = dist(gen);
        }
    }
    return R;
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> randn(size_t m, size_t n, unsigned seed) {
    Matrix<S, StorageOrder::ColMajor, Alloc> R(m, n);
    if (session_rng_active()) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                R(i, j) = static_cast<S>(session_normal());
            }
        }
        return R;
    }
    std::mt19937 gen(seed);
    std::normal_distribution<S> dist(S(0), S(1));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            R(i, j) = dist(gen);
        }
    }
    return R;
}

template<typename S, template<typename> class Alloc>
Matrix<S, StorageOrder::ColMajor, Alloc> diag(const std::vector<S>& v) {
    const size_t n = v.size();
    Matrix<S, StorageOrder::ColMajor, Alloc> D(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        D(i, i) = v[i];
    }
    return D;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
std::vector<S> diag(const Matrix<S, OA, Alloc>& A) {
    const size_t n = std::min(A.rows(), A.cols());
    std::vector<S> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = A(i, i);
    }
    return v;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> tril(const Matrix<S, OA, Alloc>& A, int k) {
    Matrix<S, OA, Alloc> L(A.rows(), A.cols(), S(0));
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            if (static_cast<int>(j) <= static_cast<int>(i) + k) {
                L(i, j) = A(i, j);
            }
        }
    }
    return L;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> triu(const Matrix<S, OA, Alloc>& A, int k) {
    Matrix<S, OA, Alloc> U(A.rows(), A.cols(), S(0));
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            if (static_cast<int>(j) >= static_cast<int>(i) + k) {
                U(i, j) = A(i, j);
            }
        }
    }
    return U;
}

template Matrix<double> rand<double>(size_t, size_t, unsigned);
template Matrix<float> rand<float>(size_t, size_t, unsigned);
template Matrix<double> randn<double>(size_t, size_t, unsigned);
template Matrix<float> randn<float>(size_t, size_t, unsigned);
template Matrix<double> diag<double>(const std::vector<double>&);
template Matrix<float> diag<float>(const std::vector<float>&);
template std::vector<double> diag<double>(const Matrix<double>&);
template std::vector<float> diag<float>(const Matrix<float>&);
template Matrix<double> tril<double>(const Matrix<double>&, int);
template Matrix<double> triu<double>(const Matrix<double>&, int);

} // namespace ms
