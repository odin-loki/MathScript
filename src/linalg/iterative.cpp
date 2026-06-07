#include "ms/linalg/linalg.hpp"
#include "detail.hpp"
#include <cmath>

namespace ms {

namespace {

using namespace linalg_detail;

Matrix<double> matvec(const Matrix<double>& A, const Matrix<double>& x) {
    return multiply(A, x);
}

double dotvec(const Matrix<double>& a, const Matrix<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.rows(); ++i) {
        sum += a(i, 0) * b(i, 0);
    }
    return sum;
}

Matrix<double> axpy(double alpha, const Matrix<double>& x, const Matrix<double>& y) {
    Matrix<double> r(x.rows(), x.cols());
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            r(i, j) = alpha * x(i, j) + y(i, j);
        }
    }
    return r;
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> cg(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    if (A.rows() != A.cols() || A.rows() != b.rows()) {
        return std::unexpected(DimensionMismatch{A.rows(), b.rows()});
    }
    if (!is_symmetric(A)) {
        return std::unexpected(DomainError{"cg", "matrix not symmetric"});
    }

    Matrix<double> x(b.rows(), b.cols(), 0.0);
    Matrix<double> r = copy(b);
    Matrix<double> p = copy(r);
    double rsold = dotvec(r, r);

    for (size_t k = 0; k < max_iter; ++k) {
        Matrix<double> Ap = matvec(A, p);
        const double alpha = rsold / dotvec(p, Ap);
        x = axpy(alpha, p, x);
        r = axpy(-alpha, Ap, r);
        const double rsnew = dotvec(r, r);
        if (std::sqrt(rsnew) < tol) {
            break;
        }
        p = axpy(rsnew / rsold, p, r);
        rsold = rsnew;
    }

    return x;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> bicgstab(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t max_iter,
    S tol) {
    Matrix<double> x(b.rows(), b.cols(), 0.0);
    Matrix<double> r = copy(b);
    Matrix<double> r0 = copy(r);
    Matrix<double> p = copy(r);
    Matrix<double> v(b.rows(), b.cols(), 0.0);

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (size_t k = 0; k < max_iter; ++k) {
        const double rho1 = dotvec(r0, r);
        if (std::abs(rho1) < 1e-30) {
            break;
        }
        if (k == 0) {
            p = copy(r);
        } else {
            const double beta = (rho1 / rho) * (alpha / omega);
            Matrix<double> ph = axpy(-omega, v, p);
            p = axpy(beta, ph, r);
        }
        v = matvec(A, p);
        alpha = rho1 / dotvec(r0, v);
        Matrix<double> s = axpy(-alpha, v, r);
        if (std::sqrt(dotvec(s, s)) < tol) {
            x = axpy(alpha, p, x);
            break;
        }
        Matrix<double> t = matvec(A, s);
        omega = dotvec(t, s) / dotvec(t, t);
        x = axpy(alpha, p, axpy(omega, s, x));
        r = axpy(-omega, t, s);
        if (std::sqrt(dotvec(r, r)) < tol) {
            break;
        }
        rho = rho1;
    }

    return x;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> gmres(
    const Matrix<S, OA, Alloc>& A,
    const Matrix<S, OA, Alloc>& b,
    size_t restart,
    size_t max_iter,
    S tol) {
    Matrix<double> x(b.rows(), b.cols(), 0.0);
    Matrix<double> r = copy(b);

    for (size_t outer = 0; outer < max_iter; outer += restart) {
        const double beta = std::sqrt(dotvec(r, r));
        if (beta < tol) {
            break;
        }

        std::vector<Matrix<double>> V(restart + 1);
        V[0] = Matrix<double>(b.rows(), 1);
        for (size_t i = 0; i < b.rows(); ++i) {
            V[0](i, 0) = r(i, 0) / beta;
        }

        std::vector<double> g(restart + 1, 0.0);
        g[0] = beta;
        std::vector<std::vector<double>> H(
            restart + 1, std::vector<double>(restart, 0.0));

        size_t j = 0;
        for (; j < restart && outer + j < max_iter; ++j) {
            Matrix<double> w = matvec(A, V[j]);
            for (size_t i = 0; i <= j; ++i) {
                H[i][j] = dotvec(V[i], w);
                w = axpy(-H[i][j], V[i], w);
            }
            H[j + 1][j] = std::sqrt(dotvec(w, w));
            if (H[j + 1][j] < 1e-14) {
                break;
            }
            V[j + 1] = Matrix<double>(b.rows(), 1);
            for (size_t i = 0; i < b.rows(); ++i) {
                V[j + 1](i, 0) = w(i, 0) / H[j + 1][j];
            }

            for (size_t i = 0; i < j + 1; ++i) {
                const double c = H[i][j];
                const double s = H[i + 1][j];
                const double h0 = c * H[i][j] + s * H[i + 1][j];
                const double h1 = -s * H[i][j] + c * H[i + 1][j];
                H[i][j] = h0;
                H[i + 1][j] = h1;
                const double g0 = g[i];
                const double g1 = g[i + 1];
                g[i] = c * g0 + s * g1;
                g[i + 1] = -s * g0 + c * g1;
            }

            if (std::abs(g[j + 1]) < tol) {
                ++j;
                break;
            }
        }

        Matrix<double> y(j, 1, 0.0);
        for (int i = static_cast<int>(j) - 1; i >= 0; --i) {
            double sum = g[i];
            for (size_t k = i + 1; k < j; ++k) {
                sum -= H[i][k] * y(k, 0);
            }
            y(i, 0) = sum / H[i][i];
        }

        for (size_t i = 0; i < j; ++i) {
            x = axpy(y(i, 0), V[i], x);
        }

        r = axpy(-1.0, matvec(A, x), b);
        if (std::sqrt(dotvec(r, r)) < tol) {
            break;
        }
    }

    return x;
}

template auto cg<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto bicgstab<double>(const Matrix<double>&, const Matrix<double>&, size_t, double)
    -> Result<Matrix<double>>;
template auto gmres<double>(const Matrix<double>&, const Matrix<double>&, size_t, size_t, double)
    -> Result<Matrix<double>>;

} // namespace ms
