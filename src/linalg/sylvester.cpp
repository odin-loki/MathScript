#include "ms/linalg/linalg.hpp"

namespace ms {

// Solve A*X + X*B = C via Kronecker-sum vectorization: build
// K = I_m (x) A + B^T (x) I_n  (size nm x nm) and solve K*vec(X) = vec(C),
// where vec(X) is the column-major flatten of X (index j*n+i for X(i,j)).
// See the doc comment in include/ms/linalg/linalg.hpp for why this approach
// (rather than a Schur/Bartels-Stewart reduction) was chosen.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> solve_sylvester(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& B,
    const Matrix<S, OA, Alloc>& C) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (B.rows() != B.cols()) {
        return std::unexpected(DimensionMismatch{B.rows(), B.cols()});
    }

    const size_t n = A.rows();
    const size_t m = B.rows();

    if (C.rows() != n || C.cols() != m) {
        return std::unexpected(DimensionMismatch{C.rows(), C.cols(), n, m});
    }

    const size_t nm = n * m;
    Matrix<S, OA, Alloc> K(nm, nm, S(0));

    for (size_t j = 0; j < m; ++j) {
        for (size_t i = 0; i < n; ++i) {
            const size_t row = j * n + i;
            // (I_m ⊗ A) block: contributes A(i,k) to column j*n+k.
            for (size_t k = 0; k < n; ++k) {
                K(row, j * n + k) += A(i, k);
            }
            // (B^T ⊗ I_n) block: contributes B(l,j) to column l*n+i.
            for (size_t l = 0; l < m; ++l) {
                K(row, l * n + i) += B(l, j);
            }
        }
    }

    Matrix<S, OA, Alloc> vecC(nm, 1, S(0));
    for (size_t j = 0; j < m; ++j) {
        for (size_t i = 0; i < n; ++i) {
            vecC(j * n + i, 0) = C(i, j);
        }
    }

    auto sol = solve(K, vecC);
    if (!sol) {
        return std::unexpected(sol.error());
    }

    Matrix<S, OA, Alloc> X(n, m, S(0));
    for (size_t j = 0; j < m; ++j) {
        for (size_t i = 0; i < n; ++i) {
            X(i, j) = (*sol)(j * n + i, 0);
        }
    }
    return X;
}

template auto solve_sylvester<double>(
    const Matrix<double>&, const Matrix<double>&, const Matrix<double>&)
    -> Result<Matrix<double>>;

} // namespace ms
