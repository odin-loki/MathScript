#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace ms {

namespace {

using namespace linalg_detail;

// f(A) = V * diag(f(lambda)) * V^T. Exact only when V is orthogonal, i.e. when
// A is symmetric; every caller guards this behind is_symmetric().
Matrix<double> spectral_symmetric(
    const Matrix<double>& values,
    const Matrix<double>& V,
    const std::function<double(double)>& fn) {
    const size_t n = V.rows();
    Matrix<double> D = zeros<double>(n, n);
    for (size_t i = 0; i < n; ++i) {
        D(i, i) = fn(values(i, 0));
    }
    auto Vt = transpose_copy(V);
    return multiply(V, multiply(D, Vt));
}

// True when the real Schur factor holds a 2x2 block, i.e. a complex-conjugate
// eigenvalue pair that a real-arithmetic Parlett recurrence cannot serve.
bool has_complex_block(const Matrix<double>& T) {
    for (size_t i = 1; i < T.rows(); ++i) {
        if (T(i, i - 1) != 0.0) {
            return true;
        }
    }
    return false;
}

// Schur-Parlett: f(A) = Q * F * Q^T where F = f(T) is built from the diagonal
// outwards by the Parlett recurrence
//   F(i,j) = ( T(i,j)*(F(j,j)-F(i,i))
//              + sum_{k=i+1}^{j-1} (T(i,k)F(k,j) - F(i,k)T(k,j)) )
//            / (T(j,j) - T(i,i)),
// which is the (i,j) entry of the commutation relation F*T == T*F.
// The recurrence is only usable when the diagonal entries are separated; a
// clustered pair is reported rather than divided through.
Result<Matrix<double>> schur_parlett(
    const Matrix<double>& A,
    const std::function<double(double)>& fn,
    const char* fname) {
    auto sr = linalg_detail::real_schur(A);
    if (!sr) {
        return std::unexpected(sr.error());
    }
    const Matrix<double>& T = sr->T;
    const Matrix<double>& Q = sr->Q;
    const size_t n = T.rows();

    if (has_complex_block(T)) {
        return std::unexpected(DomainError{
            fname,
            "matrix has complex-conjugate eigenvalues; a real Schur-Parlett "
            "evaluation of f(A) would need complex arithmetic"});
    }

    const double tnorm = std::max(frobenius_norm(T), 1.0);
    const double sep_tol = 1e-8 * tnorm;

    Matrix<double> F = zeros<double>(n, n);
    double fscale = 1.0;
    for (size_t i = 0; i < n; ++i) {
        F(i, i) = fn(T(i, i));
        if (!std::isfinite(F(i, i))) {
            return std::unexpected(DomainError{
                fname, "f is not finite at an eigenvalue of the matrix"});
        }
        fscale = std::max(fscale, std::abs(F(i, i)));
    }

    for (size_t d = 1; d < n; ++d) {
        for (size_t i = 0; i + d < n; ++i) {
            const size_t j = i + d;
            double num = T(i, j) * (F(j, j) - F(i, i));
            for (size_t k = i + 1; k < j; ++k) {
                num += T(i, k) * F(k, j) - F(i, k) * T(k, j);
            }
            const double den = T(j, j) - T(i, i);
            if (std::abs(den) <= sep_tol) {
                if (std::abs(num) <= 1e-12 * fscale) {
                    F(i, j) = 0.0;  // block is decoupled; the answer is zero
                    continue;
                }
                return std::unexpected(DomainError{
                    fname,
                    "repeated or clustered eigenvalues: the Parlett recurrence "
                    "is not applicable and a blocked Schur-Parlett evaluation "
                    "is not implemented"});
            }
            F(i, j) = num / den;
            fscale = std::max(fscale, std::abs(F(i, j)));
        }
    }

    auto Qt = transpose_copy(Q);
    return multiply(Q, multiply(F, Qt));
}

// Schur square root (Bjorck-Hammarling): R upper triangular with R*R == T,
//   R(i,i) = sqrt(T(i,i)),
//   R(i,j) = (T(i,j) - sum_{k=i+1}^{j-1} R(i,k)R(k,j)) / (R(i,i)+R(j,j)).
// Unlike the Parlett recurrence this stays well defined for repeated
// eigenvalues; it only fails when two diagonal square roots cancel, which for
// nonnegative eigenvalues means both are zero.
Result<Matrix<double>> schur_sqrt(const Matrix<double>& A) {
    auto sr = linalg_detail::real_schur(A);
    if (!sr) {
        return std::unexpected(sr.error());
    }
    const Matrix<double>& T = sr->T;
    const Matrix<double>& Q = sr->Q;
    const size_t n = T.rows();

    if (has_complex_block(T)) {
        return std::unexpected(DomainError{
            "sqrtm",
            "matrix has complex-conjugate eigenvalues; its real square root "
            "is not computed by this implementation"});
    }

    const double tnorm = std::max(frobenius_norm(T), 1.0);
    const double neg_tol = static_cast<double>(n) * tnorm
                         * std::numeric_limits<double>::epsilon() * 16.0;

    Matrix<double> R = zeros<double>(n, n);
    for (size_t i = 0; i < n; ++i) {
        const double t = T(i, i);
        if (t < -neg_tol) {
            return std::unexpected(DomainError{
                "sqrtm",
                "matrix has a negative eigenvalue; its square root is not real"});
        }
        R(i, i) = std::sqrt(t > 0.0 ? t : 0.0);
    }
    for (size_t d = 1; d < n; ++d) {
        for (size_t i = 0; i + d < n; ++i) {
            const size_t j = i + d;
            double num = T(i, j);
            for (size_t k = i + 1; k < j; ++k) {
                num -= R(i, k) * R(k, j);
            }
            const double den = R(i, i) + R(j, j);
            if (std::abs(den) <= std::numeric_limits<double>::epsilon() * tnorm) {
                if (std::abs(num) <= std::numeric_limits<double>::epsilon() * tnorm) {
                    R(i, j) = 0.0;
                    continue;
                }
                return std::unexpected(DomainError{
                    "sqrtm",
                    "matrix is singular and defective at a zero eigenvalue; it "
                    "has no square root"});
            }
            R(i, j) = num / den;
        }
    }

    auto Qt = transpose_copy(Q);
    return multiply(Q, multiply(R, Qt));
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> logm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    if (is_symmetric(A)) {
        auto er = eig_sym(A);
        if (!er) {
            return std::unexpected(er.error());
        }
        for (size_t i = 0; i < er->values.rows(); ++i) {
            if (er->values(i, 0) <= 0.0) {
                return std::unexpected(DomainError{
                    "logm",
                    "symmetric matrix is not positive definite; its logarithm "
                    "is not real"});
            }
        }
        return spectral_symmetric(er->values, er->vectors,
                                  [](double x) { return std::log(x); });
    }

    return schur_parlett(to_col_major(A),
                         [](double x) { return std::log(x); }, "logm");
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sqrtm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    if (is_symmetric(A)) {
        auto er = eig_sym(A);
        if (!er) {
            return std::unexpected(er.error());
        }
        double scale = 1.0;
        for (size_t i = 0; i < er->values.rows(); ++i) {
            scale = std::max(scale, std::abs(er->values(i, 0)));
        }
        const double neg_tol = scale * static_cast<double>(A.rows())
                             * std::numeric_limits<double>::epsilon() * 16.0;
        for (size_t i = 0; i < er->values.rows(); ++i) {
            if (er->values(i, 0) < -neg_tol) {
                return std::unexpected(DomainError{
                    "sqrtm",
                    "symmetric matrix has a negative eigenvalue; its square "
                    "root is not real"});
            }
        }
        return spectral_symmetric(er->values, er->vectors, [](double x) {
            return x > 0.0 ? std::sqrt(x) : 0.0;
        });
    }

    return schur_sqrt(to_col_major(A));
}

template auto logm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto sqrtm<double>(const Matrix<double>&) -> Result<Matrix<double>>;

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> sinm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (is_symmetric(A)) {
        auto er = eig_sym(A);
        if (!er) return std::unexpected(er.error());
        return spectral_symmetric(er->values, er->vectors,
                                  [](double x) { return std::sin(x); });
    }
    return schur_parlett(to_col_major(A),
                         [](double x) { return std::sin(x); }, "sinm");
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cosm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (is_symmetric(A)) {
        auto er = eig_sym(A);
        if (!er) return std::unexpected(er.error());
        return spectral_symmetric(er->values, er->vectors,
                                  [](double x) { return std::cos(x); });
    }
    return schur_parlett(to_col_major(A),
                         [](double x) { return std::cos(x); }, "cosm");
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> funm(const Matrix<S, OA, Alloc>& A,
                                   std::function<S(S)> func) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (!func) {
        return std::unexpected(DomainError{"funm", "empty function object"});
    }
    const std::function<double(double)> fn = [&func](double x) {
        return static_cast<double>(func(static_cast<S>(x)));
    };
    if (is_symmetric(A)) {
        auto er = eig_sym(A);
        if (!er) return std::unexpected(er.error());
        return Matrix<S, OA, Alloc>(
            spectral_symmetric(er->values, er->vectors, fn));
    }
    auto res = schur_parlett(to_col_major(A), fn, "funm");
    if (!res) {
        return std::unexpected(res.error());
    }
    return Matrix<S, OA, Alloc>(*res);
}

template auto sinm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto cosm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto funm<double>(const Matrix<double>&,
                            std::function<double(double)>) -> Result<Matrix<double>>;

} // namespace ms
