#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <cmath>
#include <functional>

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

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sinm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    auto er = is_symmetric(A) ? eig_sym(A) : eig(A);
    if (!er) return std::unexpected(er.error());
    return apply_spectral_func(er->values, er->vectors,
                                [](double x) { return std::sin(x); });
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cosm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    auto er = is_symmetric(A) ? eig_sym(A) : eig(A);
    if (!er) return std::unexpected(er.error());
    return apply_spectral_func(er->values, er->vectors,
                                [](double x) { return std::cos(x); });
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> funm(const Matrix<S, OA, Alloc>& A,
                                   std::function<S(S)> func) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    auto er = is_symmetric(A) ? eig_sym(A) : eig(A);
    if (!er) return std::unexpected(er.error());
    const size_t n = A.rows();
    Matrix<double> D = zeros<double>(n, n);
    for (size_t i = 0; i < n; ++i)
        D(i, i) = static_cast<double>(func(
            static_cast<S>(er->values(i, 0))));
    auto& V = er->vectors;
    auto Vt = transpose_copy(V);
    return Matrix<S, OA, Alloc>(multiply(V, multiply(D, Vt)));
}

template auto sinm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto cosm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto funm<double>(const Matrix<double>&,
                            std::function<double(double)>) -> Result<Matrix<double>>;

} // namespace ms
