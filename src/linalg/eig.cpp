#include "ms/linalg/linalg.hpp"
#include "ms/cpu/lapack.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <vector>

namespace ms {

namespace {

using namespace linalg_detail;

Matrix<double> jacobi_eigen_symmetric(Matrix<double> A, Matrix<double>& V, double tol = 1e-12) {
    const size_t n = A.rows();
    V = eye<double>(n);

    for (int sweep = 0; sweep < 100; ++sweep) {
        double max_off = 0.0;
        size_t p = 0;
        size_t q = 1;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(A(i, j)) > max_off) {
                    max_off = std::abs(A(i, j));
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < tol) {
            break;
        }

        const double app = A(p, p);
        const double aqq = A(q, q);
        const double apq = A(p, q);
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        for (size_t k = 0; k < n; ++k) {
            const double akp = A(k, p);
            const double akq = A(k, q);
            A(k, p) = c * akp - s * akq;
            A(p, k) = A(k, p);
            A(k, q) = s * akp + c * akq;
            A(q, k) = A(k, q);
        }
        A(p, p) = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A(q, q) = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A(p, q) = A(q, p) = 0.0;

        for (size_t k = 0; k < n; ++k) {
            const double vkp = V(k, p);
            const double vkq = V(k, q);
            V(k, p) = c * vkp - s * vkq;
            V(k, q) = s * vkp + c * vkq;
        }
    }

    Matrix<double> values(n, 1);
    for (size_t i = 0; i < n; ++i) {
        values(i, 0) = A(i, i);
    }
    return values;
}

void sort_eig_descending(Matrix<double>& values, Matrix<double>& V) {
    const size_t n = values.rows();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return values(a, 0) > values(b, 0);
    });

    Matrix<double> sorted_values(n, 1);
    Matrix<double> sorted_V(n, n);
    for (size_t j = 0; j < n; ++j) {
        sorted_values(j, 0) = values(order[j], 0);
        for (size_t i = 0; i < n; ++i) {
            sorted_V(i, j) = V(i, order[j]);
        }
    }
    values = std::move(sorted_values);
    V = std::move(sorted_V);
}

Matrix<double> qr_eigenvalues_general(Matrix<double> A, Matrix<double>& V, int max_iter = 300) {
    const size_t n = A.rows();
    V = eye<double>(n);
    Matrix<double> Ak = copy(A);

    for (int iter = 0; iter < max_iter; ++iter) {
        double sub = 0.0;
        for (size_t i = 1; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) {
                sub = std::max(sub, std::abs(Ak(i, j)));
            }
        }
        if (sub < 1e-10) {
            break;
        }

        auto qr_result = qr(Ak);
        if (!qr_result) {
            break;
        }
        auto [Q, R] = *qr_result;
        Ak = multiply(R, Q);
        V = multiply(V, Q);
    }

    Matrix<double> values(n, 1);
    for (size_t i = 0; i < n; ++i) {
        values(i, 0) = Ak(i, i);
    }
    return values;
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig_sym(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }
    if (!is_symmetric(A)) {
        return std::unexpected(DomainError{"eig_sym", "matrix not symmetric"});
    }

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        const int n = static_cast<int>(A.rows());
        Matrix<double> factors = copy(A);
        std::vector<double> w(static_cast<std::size_t>(n));
        if (cpu::lapack::dsyev('V', n, factors.data(), n, w.data()) == 0) {
            Matrix<double> values(static_cast<std::size_t>(n), 1);
            Matrix<double> vectors(static_cast<std::size_t>(n), static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j) {
                values(static_cast<std::size_t>(j), 0) = w[static_cast<std::size_t>(j)];
                for (int i = 0; i < n; ++i) {
                    vectors(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                        factors(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                }
            }
            sort_eig_descending(values, vectors);
            return EigResult{values, vectors};
        }
    }

    Matrix<double> Ad = to_col_major(A);
    Matrix<double> V;
    Matrix<double> values = jacobi_eigen_symmetric(Ad, V);
    sort_eig_descending(values, V);
    return EigResult{values, V};
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    if (is_symmetric(A)) {
        return eig_sym(A);
    }

    Matrix<double> Ad = to_col_major(A);
    Matrix<double> V;
    Matrix<double> values = qr_eigenvalues_general(Ad, V);
    sort_eig_descending(values, V);
    return EigResult{values, V};
}

template auto eig_sym<double>(const Matrix<double>&) -> Result<EigResult>;
template auto eig<double>(const Matrix<double>&) -> Result<EigResult>;
template auto eig_sym<double, StorageOrder::RowMajor>(const Matrix<double, StorageOrder::RowMajor>&)
    -> Result<EigResult>;
template auto eig<double, StorageOrder::RowMajor>(const Matrix<double, StorageOrder::RowMajor>&) -> Result<EigResult>;

} // namespace ms
