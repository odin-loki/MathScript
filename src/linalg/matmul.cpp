#include "ms/core/operations.hpp"
#include "ms/cpu/blas.hpp"
#include "ms/cuda/blas.hpp"
#include "ms/runtime/dispatch.hpp"
#include <algorithm>

namespace ms {

namespace {

template<typename S, StorageOrder OA, StorageOrder OB, template<typename> class AllocA, template<typename> class AllocB>
void to_col_matrix(
    const Matrix<S, OA, AllocA>& A,
    const Matrix<S, OB, AllocB>& B,
    ColMatrix<S>& A_out,
    ColMatrix<S>& B_out) {
    A_out = ColMatrix<S>(A.rows(), A.cols());
    B_out = ColMatrix<S>(B.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            A_out(i, j) = A(i, j);
        }
    }
    for (size_t i = 0; i < B.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            B_out(i, j) = B(i, j);
        }
    }
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
            ColMatrix<double> Ac;
            ColMatrix<double> Bc;
            to_col_matrix(A, B, Ac, Bc);
            if (auto gpu = cuda::matmul(Ac, Bc, decision.cuda_device)) {
                Matrix<S, OC, AllocC> C(gpu->rows(), gpu->cols());
                for (size_t i = 0; i < C.rows(); ++i) {
                    for (size_t j = 0; j < C.cols(); ++j) {
                        C(i, j) = (*gpu)(i, j);
                    }
                }
                return C;
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

} // namespace ms
