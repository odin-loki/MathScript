#include "ms/core/operations.hpp"
#include "ms/cpu/lapack.hpp"
#include "ms/cuda/solver.hpp"
#include "ms/runtime/dispatch.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace ms {

namespace {

template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> multiply(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& B) {
    Matrix<S, OA, Alloc> C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>>
solve(const Matrix<S, OA, Alloc>& A, const Matrix<S, OA, Alloc>& b) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }

    if constexpr (std::is_same_v<S, double>) {
        const auto decision = decide(A.rows(), ExecPolicy::AUTO);
        if (decision.backend == Backend::CUDA && b.cols() == 1) {
            ColMatrix<double> Ac(A.rows(), A.cols());
            ColMatrix<double> Bc(b.rows(), b.cols());
            for (size_t i = 0; i < A.rows(); ++i) {
                for (size_t j = 0; j < A.cols(); ++j) {
                    Ac(i, j) = A(i, j);
                }
            }
            for (size_t i = 0; i < b.rows(); ++i) {
                for (size_t j = 0; j < b.cols(); ++j) {
                    Bc(i, j) = b(i, j);
                }
            }
            if (auto gpu = cuda::solve(Ac, Bc, decision.cuda_device)) {
                Matrix<S, OA, Alloc> x(gpu->rows(), gpu->cols());
                for (size_t i = 0; i < x.rows(); ++i) {
                    for (size_t j = 0; j < x.cols(); ++j) {
                        x(i, j) = (*gpu)(i, j);
                    }
                }
                return x;
            }
        }
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        const int n = static_cast<int>(A.rows());
        const int nrhs = static_cast<int>(b.cols());
        Matrix<S, OA, Alloc> factor = A;
        std::vector<int> ipiv(static_cast<std::size_t>(n));
        Matrix<S, OA, Alloc> x = b;
        const int info = cpu::lapack::dgesv(n, nrhs, factor.data(), n, ipiv.data(), x.data(), n);
        if (info != 0) {
            return std::unexpected(SingularMatrix{});
        }
        return x;
    }

    auto lu_result = lu(A);
    if (!lu_result) {
        return std::unexpected(lu_result.error());
    }

    const auto& [L, U, P] = *lu_result;
    const size_t n = A.rows();
    const size_t nrhs = b.cols();
    Matrix<S, OA, Alloc> Pb = multiply(P, b);
    Matrix<S, OA, Alloc> y(n, nrhs, S(0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < nrhs; ++c) {
            S sum = Pb(i, c);
            for (size_t k = 0; k < i; ++k) {
                sum -= L(i, k) * y(k, c);
            }
            y(i, c) = sum;
        }
    }

    Matrix<S, OA, Alloc> x(n, nrhs, S(0));
    for (size_t i = n; i-- > 0;) {
        for (size_t c = 0; c < nrhs; ++c) {
            S sum = y(i, c);
            for (size_t k = i + 1; k < n; ++k) {
                sum -= U(i, k) * x(k, c);
            }
            if (std::abs(U(i, i)) < std::numeric_limits<S>::epsilon()) {
                return std::unexpected(SingularMatrix{});
            }
            x(i, c) = sum / U(i, i);
        }
    }

    return x;
}

template auto solve<double>(const Matrix<double>&, const Matrix<double>&)
    -> Result<Matrix<double>>;
template auto solve<float>(const Matrix<float>&, const Matrix<float>&)
    -> Result<Matrix<float>>;

} // namespace ms
