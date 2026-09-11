// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/core/operations.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ms {

namespace {

// Dense C = A*B for square matrices of the same shape.
template<typename S, StorageOrder OA, template<typename> class Alloc>
Matrix<S, OA, Alloc> gemm_square(const Matrix<S, OA, Alloc>& A,
                                 const Matrix<S, OA, Alloc>& B) {
    const size_t n = A.rows();
    Matrix<S, OA, Alloc> C(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            const S aik = A(i, k);
            if (aik == S(0)) {
                continue;
            }
            for (size_t j = 0; j < n; ++j) {
                C(i, j) += aik * B(k, j);
            }
        }
    }
    return C;
}

// Solve A*X == B for X by Gaussian elimination with partial pivoting.
// Returns false if A is numerically singular.
template<typename S, StorageOrder OA, template<typename> class Alloc>
bool solve_left(const Matrix<S, OA, Alloc>& A,
                const Matrix<S, OA, Alloc>& B,
                Matrix<S, OA, Alloc>& X) {
    const size_t n = A.rows();
    Matrix<S, OA, Alloc> M = A;
    Matrix<S, OA, Alloc> R = B;

    for (size_t k = 0; k < n; ++k) {
        size_t piv = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(M(i, k)) > std::abs(M(piv, k))) {
                piv = i;
            }
        }
        if (!(std::abs(M(piv, k)) > S(0))) {
            return false;  // exactly singular (or NaN) column
        }
        if (piv != k) {
            for (size_t j = 0; j < n; ++j) {
                const S t1 = M(k, j);
                M(k, j) = M(piv, j);
                M(piv, j) = t1;
                const S t2 = R(k, j);
                R(k, j) = R(piv, j);
                R(piv, j) = t2;
            }
        }
        for (size_t i = k + 1; i < n; ++i) {
            const S f = M(i, k) / M(k, k);
            if (f == S(0)) {
                continue;
            }
            M(i, k) = S(0);
            for (size_t j = k + 1; j < n; ++j) {
                M(i, j) -= f * M(k, j);
            }
            for (size_t j = 0; j < n; ++j) {
                R(i, j) -= f * R(k, j);
            }
        }
    }

    X = Matrix<S, OA, Alloc>(n, n, S(0));
    for (size_t col = 0; col < n; ++col) {
        for (size_t ii = n; ii-- > 0;) {
            S sum = R(ii, col);
            for (size_t j = ii + 1; j < n; ++j) {
                sum -= M(ii, j) * X(j, col);
            }
            X(ii, col) = sum / M(ii, ii);
            if (!std::isfinite(X(ii, col))) {
                return false;
            }
        }
    }
    return true;
}

template<typename S, StorageOrder OA, template<typename> class Alloc>
double norm1(const Matrix<S, OA, Alloc>& A) {
    double best = 0.0;
    for (size_t j = 0; j < A.cols(); ++j) {
        double s = 0.0;
        for (size_t i = 0; i < A.rows(); ++i) {
            s += std::abs(static_cast<double>(A(i, j)));
        }
        best = std::max(best, s);
    }
    return best;
}

} // namespace

template<typename S, StorageOrder OA, template<typename> class Alloc>
Result<Matrix<S, OA, Alloc>> expm(const Matrix<S, OA, Alloc>& A) {
    if (A.rows() != A.cols()) {
        return std::unexpected(DimensionMismatch{A.rows(), A.cols()});
    }

    using Mat = Matrix<S, OA, Alloc>;
    const size_t n = A.rows();
    if (n == 0) {
        return Mat(0, 0, S(0));
    }

    const Mat I = eye<S, Alloc>(n);
    Mat X = A;

    // Scaling-and-squaring driver (Higham 2005, Alg. 2.3 / 10.20): pick the
    // smallest Pade order whose backward-error bound theta_m covers ||A||_1,
    // otherwise scale by 2^-s down to theta_13 and square s times afterwards.
    const double a1 = norm1(A);
    if (!std::isfinite(a1)) {
        return std::unexpected(DomainError{"expm", "matrix has non-finite entries"});
    }

    static constexpr double kTheta3 = 1.495585217958292e-2;
    static constexpr double kTheta5 = 2.539398330063230e-1;
    static constexpr double kTheta7 = 9.504178996162932e-1;
    static constexpr double kTheta9 = 2.097847961257068e0;
    static constexpr double kTheta13 = 5.371920351148152e0;

    // The order-13 coefficients reach 6.5e16, which a 24-bit significand
    // cannot hold, so a low-precision scalar type stops the ladder at m = 7
    // (coefficients <= 1.7e7) and squares a couple more times instead.
    const bool low_precision =
        (static_cast<double>(std::numeric_limits<S>::epsilon()) > 1e-10);
    const int m_max = low_precision ? 7 : 13;
    const double theta_max = low_precision ? kTheta7 : kTheta13;

    int m = m_max;
    int s = 0;
    if (a1 <= kTheta3) {
        m = 3;
    } else if (a1 <= kTheta5) {
        m = 5;
    } else if (a1 <= kTheta7) {
        m = 7;
    } else if (!low_precision && a1 <= kTheta9) {
        m = 9;
    } else {
        m = m_max;
        if (a1 > theta_max) {
            const double ratio = a1 / theta_max;
            s = static_cast<int>(std::ceil(std::log2(ratio)));
            if (s < 0) {
                s = 0;
            }
            const S scale = static_cast<S>(std::ldexp(1.0, -s));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    X(i, j) = A(i, j) * scale;
                }
            }
        }
    }

    // Padé coefficients b_k of the [m/m] approximant to exp.
    static const std::vector<double> kB3{120.0, 60.0, 12.0, 1.0};
    static const std::vector<double> kB5{30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0};
    static const std::vector<double> kB7{
        17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0};
    static const std::vector<double> kB9{
        17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
        2162160.0, 110880.0, 3960.0, 90.0, 1.0};
    static const std::vector<double> kB13{
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0};

    const std::vector<double>& b =
        (m == 3) ? kB3 : (m == 5) ? kB5 : (m == 7) ? kB7 : (m == 9) ? kB9 : kB13;

    // U = X * (odd part), V = even part, both from powers of X2 = X*X.
    const Mat X2 = gemm_square(X, X);
    Mat U(n, n, S(0));
    Mat V(n, n, S(0));

    if (m == 13) {
        const Mat X4 = gemm_square(X2, X2);
        const Mat X6 = gemm_square(X4, X2);

        // W = X6*(b13 X6 + b11 X4 + b9 X2) + b7 X6 + b5 X4 + b3 X2 + b1 I
        Mat inner(n, n, S(0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inner(i, j) = static_cast<S>(b[13]) * X6(i, j)
                            + static_cast<S>(b[11]) * X4(i, j)
                            + static_cast<S>(b[9]) * X2(i, j);
            }
        }
        Mat W = gemm_square(X6, inner);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                W(i, j) += static_cast<S>(b[7]) * X6(i, j)
                         + static_cast<S>(b[5]) * X4(i, j)
                         + static_cast<S>(b[3]) * X2(i, j)
                         + static_cast<S>(b[1]) * I(i, j);
            }
        }
        U = gemm_square(X, W);

        Mat inner2(n, n, S(0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inner2(i, j) = static_cast<S>(b[12]) * X6(i, j)
                             + static_cast<S>(b[10]) * X4(i, j)
                             + static_cast<S>(b[8]) * X2(i, j);
            }
        }
        V = gemm_square(X6, inner2);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                V(i, j) += static_cast<S>(b[6]) * X6(i, j)
                         + static_cast<S>(b[4]) * X4(i, j)
                         + static_cast<S>(b[2]) * X2(i, j)
                         + static_cast<S>(b[0]) * I(i, j);
            }
        }
    } else {
        // Horner on the powers of X2 for the low-order approximants.
        std::vector<Mat> pow2;
        pow2.reserve(static_cast<size_t>(m / 2) + 1);
        pow2.push_back(I);
        pow2.push_back(X2);
        for (int k = 2; k <= m / 2; ++k) {
            pow2.push_back(gemm_square(pow2.back(), X2));
        }
        Mat odd(n, n, S(0));
        for (int k = 0; k <= m / 2; ++k) {
            const S c = static_cast<S>(b[static_cast<size_t>(2 * k + 1)]);
            const Mat& P = pow2[static_cast<size_t>(k)];
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    odd(i, j) += c * P(i, j);
                }
            }
        }
        U = gemm_square(X, odd);
        for (int k = 0; k <= m / 2; ++k) {
            const S c = static_cast<S>(b[static_cast<size_t>(2 * k)]);
            const Mat& P = pow2[static_cast<size_t>(k)];
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    V(i, j) += c * P(i, j);
                }
            }
        }
    }

    // r_m(X) = (-U + V)^-1 (U + V)  ->  solve (V - U) * F = (V + U).
    Mat num(n, n, S(0));
    Mat den(n, n, S(0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            num(i, j) = V(i, j) + U(i, j);
            den(i, j) = V(i, j) - U(i, j);
        }
    }

    Mat F(n, n, S(0));
    if (!solve_left(den, num, F)) {
        return std::unexpected(SingularMatrix{});
    }

    for (int k = 0; k < s; ++k) {
        F = gemm_square(F, F);
    }
    return F;
}

template auto expm<double>(const Matrix<double>&) -> Result<Matrix<double>>;
template auto expm<float>(const Matrix<float>&) -> Result<Matrix<float>>;

} // namespace ms
