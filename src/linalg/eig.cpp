#include "ms/linalg/linalg.hpp"
#include "ms/cpu/lapack.hpp"
#include "detail.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace ms {

namespace {

using namespace linalg_detail;

using cdouble = std::complex<double>;

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

// One diagonal block of a real Schur form: either a 1x1 (real eigenvalue) or a
// 2x2 carrying a complex-conjugate pair.
struct SchurBlock {
    size_t start;
    size_t size;
    double re;
    double im;  // > 0 for the first member of a conjugate pair, 0 for 1x1
};

std::vector<SchurBlock> schur_blocks(const Matrix<double>& T) {
    const size_t n = T.rows();
    std::vector<SchurBlock> blocks;
    for (size_t j = 0; j < n;) {
        if (j + 1 < n && T(j + 1, j) != 0.0) {
            const double a = T(j, j);
            const double b = T(j, j + 1);
            const double c = T(j + 1, j);
            const double d = T(j + 1, j + 1);
            const double half = 0.5 * (a - d);
            const double disc = half * half + b * c;
            const double im = std::sqrt(disc < 0.0 ? -disc : 0.0);
            blocks.push_back(SchurBlock{j, 2, 0.5 * (a + d), im});
            j += 2;
        } else {
            blocks.push_back(SchurBlock{j, 1, T(j, j), 0.0});
            j += 1;
        }
    }
    return blocks;
}

// Right eigenvector of the quasi-upper-triangular T for the eigenvalue of
// block `bi`, by back substitution (the real-arithmetic core of LAPACK's
// dtrevc, carried out in complex arithmetic so conjugate pairs are handled).
// Writes the (complex) result into y.
void trevc_column(const Matrix<double>& T,
                  const std::vector<SchurBlock>& blocks,
                  size_t bi,
                  double smin,
                  std::vector<cdouble>& y) {
    const size_t n = T.rows();
    y.assign(n, cdouble(0.0, 0.0));

    const SchurBlock& blk = blocks[bi];
    const cdouble lambda(blk.re, blk.im);
    const size_t p = blk.start;

    if (blk.size == 1) {
        y[p] = cdouble(1.0, 0.0);
    } else {
        // Eigenvector of the 2x2 block for lambda is [lambda - d, c].
        y[p] = lambda - cdouble(T(p + 1, p + 1), 0.0);
        y[p + 1] = cdouble(T(p + 1, p), 0.0);
    }

    for (size_t back = bi; back-- > 0;) {
        const size_t i = blocks[back].start;
        const size_t sz = blocks[back].size;
        if (sz == 1) {
            cdouble s(0.0, 0.0);
            for (size_t k = i + 1; k < n; ++k) {
                s += cdouble(T(i, k), 0.0) * y[k];
            }
            cdouble denom = cdouble(T(i, i), 0.0) - lambda;
            if (std::abs(denom) < smin) {
                denom = cdouble(smin, 0.0);
            }
            y[i] = -s / denom;
        } else {
            cdouble s0(0.0, 0.0);
            cdouble s1(0.0, 0.0);
            for (size_t k = i + 2; k < n; ++k) {
                s0 += cdouble(T(i, k), 0.0) * y[k];
                s1 += cdouble(T(i + 1, k), 0.0) * y[k];
            }
            const cdouble m00 = cdouble(T(i, i), 0.0) - lambda;
            const cdouble m01(T(i, i + 1), 0.0);
            const cdouble m10(T(i + 1, i), 0.0);
            const cdouble m11 = cdouble(T(i + 1, i + 1), 0.0) - lambda;
            cdouble det = m00 * m11 - m01 * m10;
            if (std::abs(det) < smin) {
                det = cdouble(smin, 0.0);
            }
            y[i] = (m01 * s1 - m11 * s0) / det;
            y[i + 1] = (m10 * s0 - m00 * s1) / det;
        }
    }
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
            return EigResult{values, vectors,
                             Matrix<double>(static_cast<std::size_t>(n), 1, 0.0)};
        }
    }

    Matrix<double> Ad = to_col_major(A);
    Matrix<double> V;
    Matrix<double> values = jacobi_eigen_symmetric(Ad, V);
    sort_eig_descending(values, V);
    return EigResult{values, V, Matrix<double>(A.rows(), 1, 0.0)};
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<EigResult> eig(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    if (is_symmetric(A)) {
        return eig_sym(A);
    }

    const size_t n = A.rows();
    Matrix<double> Ad = to_col_major(A);

    auto sr = linalg_detail::real_schur(Ad);
    if (!sr) {
        return std::unexpected(sr.error());
    }
    const Matrix<double>& T = sr->T;
    const Matrix<double>& Q = sr->Q;

    std::vector<SchurBlock> blocks = schur_blocks(T);

    // Order the blocks by descending real part, keeping each conjugate pair
    // together so the packed eigenvector layout stays valid.
    std::vector<size_t> order(blocks.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return blocks[a].re > blocks[b].re;
    });

    const double anorm = frobenius_norm(T);
    const double eps = std::numeric_limits<double>::epsilon();

    Matrix<double> values(n, 1, 0.0);
    Matrix<double> values_imag(n, 1, 0.0);
    Matrix<double> vectors(n, n, 0.0);
    std::vector<cdouble> y;

    size_t col = 0;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const SchurBlock& blk = blocks[order[oi]];
        const double lam_mag = std::hypot(blk.re, blk.im);
        double smin = eps * std::max(anorm, lam_mag);
        if (smin < std::numeric_limits<double>::min()) {
            smin = std::numeric_limits<double>::min();
        }
        trevc_column(T, blocks, order[oi], smin, y);

        // Back-transform to the original basis: v = Q*y.
        std::vector<double> vre(n, 0.0);
        std::vector<double> vim(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            double sre = 0.0;
            double sim = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sre += Q(i, k) * y[k].real();
                sim += Q(i, k) * y[k].imag();
            }
            vre[i] = sre;
            vim[i] = sim;
        }
        double nrm = 0.0;
        for (size_t i = 0; i < n; ++i) {
            nrm += vre[i] * vre[i] + vim[i] * vim[i];
        }
        nrm = std::sqrt(nrm);
        if (nrm > 0.0) {
            for (size_t i = 0; i < n; ++i) {
                vre[i] /= nrm;
                vim[i] /= nrm;
            }
        }

        if (blk.size == 1) {
            // Fix the sign so the dominant component is positive; this makes
            // the output deterministic without changing the eigenvector.
            size_t imax = 0;
            for (size_t i = 1; i < n; ++i) {
                if (std::abs(vre[i]) > std::abs(vre[imax])) {
                    imax = i;
                }
            }
            const double sgn = (vre[imax] < 0.0) ? -1.0 : 1.0;
            values(col, 0) = blk.re;
            values_imag(col, 0) = 0.0;
            for (size_t i = 0; i < n; ++i) {
                vectors(i, col) = sgn * vre[i];
            }
            ++col;
        } else {
            // LAPACK packing: v_col = vectors(:,col) + i*vectors(:,col+1) and
            // v_{col+1} is its conjugate.
            values(col, 0) = blk.re;
            values_imag(col, 0) = blk.im;
            values(col + 1, 0) = blk.re;
            values_imag(col + 1, 0) = -blk.im;
            for (size_t i = 0; i < n; ++i) {
                vectors(i, col) = vre[i];
                vectors(i, col + 1) = vim[i];
            }
            col += 2;
        }
    }

    return EigResult{values, vectors, values_imag};
}

template auto eig_sym<double>(const Matrix<double>&) -> Result<EigResult>;
template auto eig<double>(const Matrix<double>&) -> Result<EigResult>;
template auto eig_sym<double, StorageOrder::RowMajor>(const Matrix<double, StorageOrder::RowMajor>&)
    -> Result<EigResult>;
template auto eig<double, StorageOrder::RowMajor>(const Matrix<double, StorageOrder::RowMajor>&) -> Result<EigResult>;

} // namespace ms
