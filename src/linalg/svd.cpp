#include "ms/linalg/linalg.hpp"
#include "ms/cpu/lapack.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ms {

namespace {

using namespace linalg_detail;

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<SvdResult> svd(const Matrix<S, OA, Alloc>& A) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    if (m == 0 || n == 0) {
        return std::unexpected(DimensionMismatch{m, n});
    }

    const size_t r = std::min(m, n);

    if constexpr (std::is_same_v<S, double> && OA == StorageOrder::ColMajor) {
        const int im = static_cast<int>(m);
        const int in = static_cast<int>(n);
        const int k = (std::min)(im, in);
        Matrix<double> work = copy(A);
        Matrix<double> U(m, r);
        Matrix<double> VT(static_cast<std::size_t>(k), n);
        std::vector<double> sigma(static_cast<std::size_t>(k));

        if (cpu::lapack::dgesvd(im, in, work.data(), im, sigma.data(), U.data(), im, VT.data(), in) == 0) {
            Matrix<double> sing_vals(r, 1, 0.0);
            Matrix<double> V(n, r);
            for (int j = 0; j < k; ++j) {
                sing_vals(static_cast<std::size_t>(j), 0) = sigma[static_cast<std::size_t>(j)];
                for (int i = 0; i < in; ++i) {
                    if (im >= in) {
                        V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                            VT(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                    } else {
                        V(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                            VT(static_cast<std::size_t>(j), static_cast<std::size_t>(i));
                    }
                }
            }
            return SvdResult{U, sing_vals, V};
        }
    }

    Matrix<double> gram;
    if (m >= n) {
        gram = to_col_major(multiply(transpose_copy(A), A));
    } else {
        gram = to_col_major(multiply(A, transpose_copy(A)));
    }
    const size_t gram_size = gram.rows();

    auto eig_result = eig_sym(gram);
    if (!eig_result) {
        return std::unexpected(eig_result.error());
    }

    std::vector<size_t> order(gram_size);
    for (size_t i = 0; i < gram_size; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return eig_result->values(a, 0) > eig_result->values(b, 0);
    });

    Matrix<double> sing_vals(r, 1, 0.0);
    for (size_t j = 0; j < r; ++j) {
        sing_vals(j, 0) = std::sqrt((std::max)(0.0, eig_result->values(order[j], 0)));
    }

    if (m >= n) {
        Matrix<double> right_vectors(n, r);
        for (size_t j = 0; j < r; ++j) {
            for (size_t i = 0; i < n; ++i) {
                right_vectors(i, j) = eig_result->vectors(i, order[j]);
            }
        }
        Matrix<double> left_vectors(m, r);
        for (size_t j = 0; j < r; ++j) {
            if (sing_vals(j, 0) < 1e-14) {
                continue;
            }
            for (size_t i = 0; i < m; ++i) {
                double sum = 0.0;
                for (size_t t = 0; t < n; ++t) {
                    sum += A(i, t) * right_vectors(t, j);
                }
                left_vectors(i, j) = sum / sing_vals(j, 0);
            }
        }
        return SvdResult{left_vectors, sing_vals, right_vectors};
    }

    Matrix<double> left_vectors(m, r);
    for (size_t j = 0; j < r; ++j) {
        for (size_t i = 0; i < m; ++i) {
            left_vectors(i, j) = eig_result->vectors(i, order[j]);
        }
    }
    Matrix<double> right_vectors(n, r);
    for (size_t j = 0; j < r; ++j) {
        if (sing_vals(j, 0) < 1e-14) {
            continue;
        }
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t t = 0; t < m; ++t) {
                sum += A(t, i) * left_vectors(t, j);
            }
            right_vectors(i, j) = sum / sing_vals(j, 0);
        }
    }
    return SvdResult{left_vectors, sing_vals, right_vectors};
}

template auto svd<double>(const Matrix<double>&) -> Result<SvdResult>;
template auto svd<double, StorageOrder::RowMajor>(const RowMatrix<double>&) -> Result<SvdResult>;

} // namespace ms
