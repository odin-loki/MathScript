#include "ms/core/operations.hpp"
#include "ms/cpu/blas.hpp"
#include "ms/cuda/blas.hpp"
#include "ms/runtime/dispatch.hpp"
#include <algorithm>
#include <utility>

namespace ms {

namespace {

template<typename S, StorageOrder OA, template<typename> class AllocA>
ColMatrix<S> copy_to_col(const Matrix<S, OA, AllocA>& M) {
    ColMatrix<S> out(M.rows(), M.cols());
    for (size_t i = 0; i < M.rows(); ++i) {
        for (size_t j = 0; j < M.cols(); ++j) {
            out(i, j) = M(i, j);
        }
    }
    return out;
}

template<typename S, StorageOrder OA, StorageOrder OB,
         template<typename> class AllocA, template<typename> class AllocB>
std::pair<const ColMatrix<S>&, const ColMatrix<S>&> col_major_refs(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B,
    ColMatrix<S>& A_copy,
    ColMatrix<S>& B_copy) {
    if constexpr (OA == StorageOrder::ColMajor) {
        if constexpr (OB == StorageOrder::ColMajor) {
            return {A, B};
        } else {
            B_copy = copy_to_col(B);
            return {A, B_copy};
        }
    } else {
        A_copy = copy_to_col(A);
        if constexpr (OB == StorageOrder::ColMajor) {
            return {A_copy, B};
        }
        B_copy = copy_to_col(B);
        return {A_copy, B_copy};
    }
}

template<typename S, StorageOrder OC, template<typename> class AllocC>
Matrix<S, OC, AllocC> from_col_major(ColMatrix<S>&& src) {
    if constexpr (OC == StorageOrder::ColMajor) {
        return Matrix<S, OC, AllocC>(std::move(src));
    }
    Matrix<S, OC, AllocC> out(src.rows(), src.cols());
    for (size_t i = 0; i < src.rows(); ++i) {
        for (size_t j = 0; j < src.cols(); ++j) {
            out(i, j) = src(i, j);
        }
    }
    return out;
}

} // namespace

template<typename S, StorageOrder OA, StorageOrder OB, StorageOrder OC,
         template<typename> class AllocA, template<typename> class AllocB, template<typename> class AllocC>
Result<Matrix<S, OC, AllocC>> matmul(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B,
    int policy) {
    if (A.cols() != B.rows()) {
        return std::unexpected(DimensionMismatch{A.cols(), B.rows()});
    }

    if constexpr (std::is_same_v<S, double>) {
        const size_t n = (std::max)({A.rows(), A.cols(), B.cols()});
        const auto decision = decide(n, static_cast<ExecPolicy>(policy));
        if (decision.backend == Backend::CUDA) {
            ColMatrix<double> A_copy;
            ColMatrix<double> B_copy;
            const auto [A_col, B_col] = col_major_refs(A, B, A_copy, B_copy);
            if (auto gpu = cuda::matmul(A_col, B_col, decision.cuda_device)) {
                return from_col_major<S, OC, AllocC>(std::move(*gpu));
            }
        }
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor &&
                  OB == StorageOrder::ColMajor && OC == StorageOrder::ColMajor) {
        Matrix<S, OC, AllocC> C(A.rows(), B.cols());
        const int m = static_cast<int>(A.rows());
        const int k = static_cast<int>(A.cols());
        const int n = static_cast<int>(B.cols());
        cpu::blas::dgemm(
            'N',
            'N',
            m,
            n,
            k,
            1.0,
            A.data(),
            m,
            B.data(),
            k,
            0.0,
            C.data(),
            m);
        return C;
    }

    Matrix<S, OC, AllocC> C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            for (size_t j = 0; j < B.cols(); ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

template Result<Matrix<double>> matmul(
    const Matrix<double>&, const Matrix<double>&, int);
template Result<Matrix<float>> matmul(
    const Matrix<float>&, const Matrix<float>&, int);
template Result<Matrix<double>> matmul(
    const Matrix<double, StorageOrder::RowMajor>&,
    const Matrix<double, StorageOrder::RowMajor>&,
    int);

} // namespace ms
