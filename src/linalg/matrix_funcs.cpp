#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <cmath>

namespace ms {

namespace {

using namespace linalg_detail;

Matrix<double> apply_spectral_func(
    const Matrix<double>& values,
    const Matrix<double>& V,
    double (*fn)(double)) {
    const size_t n = V.rows();
    Matrix<double> D = zeros<double>(n, n);
    for (size_t i = 0; i < n; ++i) {
        D(i, i) = fn(values(i, 0));
    }
    auto Vt = transpose_copy(V);
    return multiply(V, multiply(D, Vt));
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> logm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    auto eig_result = is_symmetric(A) ? eig_sym(A) : eig(A);
    if (!eig_result) {
        return std::unexpected(eig_result.error());
    }

    return apply_spectral_func(eig_result->values, eig_result->vectors, [](double x) {
        return std::log(x);
    });
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sqrtm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    auto eig_result = is_symmetric(A) ? eig_sym(A) : eig(A);
    if (!eig_result) {
        return std::unexpected(eig_result.error());
    }

    return apply_spectral_func(eig_result->values, eig_result->vectors, [](double x) {
        return x >= 0.0 ? std::sqrt(x) : 0.0;
    });
}

template auto logm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto sqrtm<double>(const Matrix<double>&) -> Result<Matrix<double>>;

} // namespace ms
