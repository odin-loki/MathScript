#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>

namespace ms {

namespace {

using namespace linalg_detail;

void apply_householder_left(Matrix<double>& A, Matrix<double>& Q, size_t k) {
    const size_t n = A.rows();
    const size_t m = A.cols();
    if (k + 1 >= n) {
        return;
    }

    double scale = 0.0;
    for (size_t i = k + 1; i < n; ++i) {
        scale += A(i, k) * A(i, k);
    }
    scale = std::sqrt(scale);
    if (scale < 1e-30) {
        return;
    }

    const double sign = A(k + 1, k) >= 0 ? 1.0 : -1.0;
    const double u1 = A(k + 1, k) + sign * scale;
    std::vector<double> v(n - k - 1);
    v[0] = u1;
    for (size_t i = k + 2; i < n; ++i) {
        v[i - k - 1] = A(i, k);
    }

    double beta = 0.0;
    for (double x : v) {
        beta += x * x;
    }
    beta = 2.0 / beta;

    for (size_t j = k; j < m; ++j) {
        double dot = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            dot += v[i] * A(k + 1 + i, j);
        }
        dot *= beta;
        for (size_t i = 0; i < v.size(); ++i) {
            A(k + 1 + i, j) -= dot * v[i];
        }
    }

    for (size_t j = 0; j < n; ++j) {
        double dot = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            dot += Q(j, k + 1 + i) * v[i];
        }
        dot *= beta;
        for (size_t i = 0; i < v.size(); ++i) {
            Q(j, k + 1 + i) -= dot * v[i];
        }
    }

    A(k + 1, k) = -sign * scale;
    for (size_t i = k + 2; i < n; ++i) {
        A(i, k) = 0.0;
    }
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<LdlResult> ldl(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (!is_symmetric(A)) {
        return std::unexpected(DomainError{"ldl", "matrix not symmetric"});
    }

    const size_t n = A.rows();
    Matrix<double> L = eye<double>(n);
    Matrix<double> D(n, 1);
    Matrix<double> P = eye<double>(n);
    Matrix<double> work = copy(A);

    for (size_t j = 0; j < n; ++j) {
        double dj = work(j, j);
        for (size_t k = 0; k < j; ++k) {
            dj -= L(j, k) * L(j, k) * D(k, 0);
        }
        D(j, 0) = dj;
        if (std::abs(dj) < 1e-14) {
            return std::unexpected(SingularMatrix{});
        }
        for (size_t i = j + 1; i < n; ++i) {
            double lij = work(i, j);
            for (size_t k = 0; k < j; ++k) {
                lij -= L(i, k) * L(j, k) * D(k, 0);
            }
            L(i, j) = lij / D(j, 0);
        }
    }

    return LdlResult{L, D, P};
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> hess(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    const size_t n = A.rows();
    Matrix<double> H = copy(A);
    Matrix<double> Q = eye<double>(n);

    for (size_t k = 0; k + 2 < n; ++k) {
        apply_householder_left(H, Q, k);
    }

    return H;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<BidiagResult> bidiag(const Matrix<S, OA, Alloc>& A) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    Matrix<double> U = eye<double>(m);
    Matrix<double> B = copy(A);
    Matrix<double> V = eye<double>(n);

    const size_t lim = std::min(m, n);
    for (size_t k = 0; k < lim; ++k) {
        apply_householder_left(B, U, k);
        if (k + 1 < n) {
            double scale = 0.0;
            for (size_t j = k + 1; j < n; ++j) {
                scale += B(k, j) * B(k, j);
            }
            scale = std::sqrt(scale);
            if (scale > 1e-30) {
                const double sign = B(k, k + 1) >= 0 ? 1.0 : -1.0;
                const double u1 = B(k, k + 1) + sign * scale;
                std::vector<double> v(n - k - 1);
                v[0] = u1;
                for (size_t j = k + 2; j < n; ++j) {
                    v[j - k - 1] = B(k, j);
                }
                double beta = 0.0;
                for (double x : v) {
                    beta += x * x;
                }
                beta = 2.0 / beta;
                for (size_t i = k; i < m; ++i) {
                    double dot = 0.0;
                    for (size_t j = 0; j < v.size(); ++j) {
                        dot += B(i, k + 1 + j) * v[j];
                    }
                    dot *= beta;
                    for (size_t j = 0; j < v.size(); ++j) {
                        B(i, k + 1 + j) -= dot * v[j];
                    }
                }
                for (size_t i = 0; i < n; ++i) {
                    double dot = 0.0;
                    for (size_t j = 0; j < v.size(); ++j) {
                        dot += V(i, k + 1 + j) * v[j];
                    }
                    dot *= beta;
                    for (size_t j = 0; j < v.size(); ++j) {
                        V(i, k + 1 + j) -= dot * v[j];
                    }
                }
                B(k, k + 1) = -sign * scale;
                for (size_t j = k + 2; j < n; ++j) {
                    B(k, j) = 0.0;
                }
            }
        }
    }

    return BidiagResult{U, B, V};
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<SchurResult> schur(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    auto H_result = hess(A);
    if (!H_result) {
        return std::unexpected(H_result.error());
    }

    Matrix<double> T = *H_result;
    Matrix<double> Q = eye<double>(A.rows());

    for (int iter = 0; iter < 200; ++iter) {
        double sub = 0.0;
        for (size_t i = 1; i < T.rows(); ++i) {
            sub = std::max(sub, std::abs(T(i, i - 1)));
        }
        if (sub < 1e-10) {
            break;
        }
        auto qr_result = qr(T);
        if (!qr_result) {
            break;
        }
        auto [Qi, Ri] = *qr_result;
        T = multiply(Ri, Qi);
        Q = multiply(Q, Qi);
    }

    return SchurResult{T, Q};
}

template auto ldl<double>(const Matrix<double>&) -> Result<LdlResult>;
template auto hess<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto bidiag<double>(const Matrix<double>&) -> Result<BidiagResult>;
template auto schur<double>(const Matrix<double>&) -> Result<SchurResult>;

} // namespace ms
