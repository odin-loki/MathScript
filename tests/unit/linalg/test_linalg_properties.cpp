// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.5: the invariants the numerical core has and nobody was checking.
//
// The plan's argument for this is that a fixed case proves a function returns the
// value someone wrote down once. An invariant proves it returns the right value for
// inputs nobody wrote down at all -- and the inputs nobody wrote down are where the
// defects have been. Every finding of the two audits was on a path a fixed test had
// walked past: a binomial that was right for the twelve values in the test and wrong
// at C(67,33), a Jordan totient that agreed with its own test and with nothing else.
//
// These are properties rather than cases, so each is checked over many generated
// matrices. The generator is seeded from a constant, so a failure is reproducible:
// the seed and the size are printed with the assertion, and rerunning the binary
// produces the same matrix. A property test that cannot be rerun on its own failure
// is a flake report, not a test.
//
// The matrices are deliberately well-conditioned. A random matrix is nearly singular
// often enough that an unconditioned generator would spend its failures on
// conditioning rather than on correctness, and "the residual was large because the
// matrix was bad" is not a finding. Conditioning is a separate question with its own
// tests; this file asks whether the algebra holds.

#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ms/error/error_types.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

namespace {

/// Every property runs over this many independently generated inputs. Large enough
/// that a defect confined to one sign pattern or one pivot order shows up; small
/// enough that the whole file stays inside the per-test budget the suite runs in.
constexpr int kTrials = 40;

/// A fixed root, so the corpus is the same corpus on every machine and on every run.
/// Each property derives its own stream from it, so adding a property does not move
/// the inputs of the ones already there -- otherwise a new test would silently change
/// what the old ones cover.
constexpr std::uint64_t kSeedRoot = 0x5DEECE66Dull;

std::mt19937_64 stream_for(int property, int trial) {
    return std::mt19937_64{kSeedRoot + static_cast<std::uint64_t>(property) * 1000003ull +
                           static_cast<std::uint64_t>(trial)};
}

/// Entries in [-1, 1] with `n` added along the diagonal. Diagonal dominance by that
/// margin bounds the condition number well away from where double precision runs out,
/// which is what lets these assertions use a tolerance rather than a shrug.
DMatrix well_conditioned(std::mt19937_64& rng, std::size_t n) {
    std::uniform_real_distribution<double> entry(-1.0, 1.0);
    DMatrix A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = entry(rng);
        }
        A(i, i) += static_cast<double>(n);
    }
    return A;
}

/// A^T A + n I: symmetric by construction and positive definite by the shift, which is
/// what `chol` requires and what a random symmetric matrix is not.
DMatrix symmetric_positive_definite(std::mt19937_64& rng, std::size_t n) {
    const DMatrix A = well_conditioned(rng, n);
    DMatrix S(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += A(k, i) * A(k, j);
            }
            S(i, j) = sum;
        }
        S(i, i) += static_cast<double>(n);
    }
    return S;
}

DMatrix random_vector(std::mt19937_64& rng, std::size_t n) {
    std::uniform_real_distribution<double> entry(-1.0, 1.0);
    DMatrix v(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        v(i, 0) = entry(rng);
    }
    return v;
}

/// The largest absolute entry of A - B, which is the quantity every assertion here is
/// really about. Reporting it rather than asserting entry by entry keeps a failure
/// message short enough to read.
template <typename MA, typename MB>
double max_abs_difference(const MA& a, const MB& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            worst = std::max(worst, std::abs(a(i, j) - b(i, j)));
        }
    }
    return worst;
}

template <typename M>
double max_abs_difference_from_identity(const M& a) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            worst = std::max(worst, std::abs(a(i, j) - (i == j ? 1.0 : 0.0)));
        }
    }
    return worst;
}

// --- The properties ------------------------------------------------------------------

// P*A = L*U. The decomposition is only useful because of this identity, and it is the
// one thing a fixed 3x3 case cannot establish for a pivot order it never produces.
TEST(LinalgProperties, LuFactorsThePermutedMatrix) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(1, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const DMatrix A = well_conditioned(rng, n);
        const auto factored = lu(A);
        ASSERT_TRUE(factored.has_value()) << "n=" << n << " trial=" << trial;
        const auto& [L, U, P] = *factored;
        const auto PA = matmul(P, A);
        const auto LU = matmul(L, U);
        ASSERT_TRUE(PA.has_value());
        ASSERT_TRUE(LU.has_value());
        EXPECT_LT(max_abs_difference(*PA, *LU), 1e-11)
            << "P*A != L*U at n=" << n << " trial=" << trial;
    }
}

// Q is orthogonal and Q*R reconstructs A. Losing orthogonality is the classic
// Gram-Schmidt failure, and it does not show up in a reconstruction test alone.
TEST(LinalgProperties, QrIsOrthogonalAndReconstructs) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(2, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const DMatrix A = well_conditioned(rng, n);
        const auto factored = qr(A);
        ASSERT_TRUE(factored.has_value()) << "n=" << n << " trial=" << trial;
        const auto& [Q, R] = *factored;
        const DMatrix QtQ = transpose(Q) * Q;
        EXPECT_LT(max_abs_difference_from_identity(QtQ), 1e-11)
            << "Q^T Q != I at n=" << n << " trial=" << trial;
        const DMatrix QR = Q * R;
        EXPECT_LT(max_abs_difference(QR, A), 1e-11)
            << "Q*R != A at n=" << n << " trial=" << trial;
        // R is upper triangular, which is the other half of what "QR" claims.
        for (std::size_t i = 1; i < R.rows(); ++i) {
            for (std::size_t j = 0; j < i && j < R.cols(); ++j) {
                EXPECT_NEAR(R(i, j), 0.0, 1e-11) << "R is not upper triangular at (" << i
                                                 << ", " << j << ") trial=" << trial;
            }
        }
    }
}

// L*L^T = A for a symmetric positive definite A.
TEST(LinalgProperties, CholeskyReconstructs) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(3, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 5);
        const DMatrix S = symmetric_positive_definite(rng, n);
        const auto L = chol(S);
        ASSERT_TRUE(L.has_value()) << "n=" << n << " trial=" << trial;
        const DMatrix reconstructed = *L * transpose(*L);
        // The entries of S grow like n^2, so the tolerance is relative to the
        // largest of them rather than absolute -- an absolute bound here would be
        // a test of the generator's scale, not of the factorisation.
        double scale = 1.0;
        for (std::size_t i = 0; i < n; ++i) {
            scale = std::max(scale, std::abs(S(i, i)));
        }
        EXPECT_LT(max_abs_difference(reconstructed, S), 1e-11 * scale)
            << "L*L^T != A at n=" << n << " trial=" << trial;
    }
}

// A*x = b for the x that solve returns. This is the definition of the function, and it
// is checkable without knowing anything about how it computes.
TEST(LinalgProperties, SolveSatisfiesItsOwnSystem) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(4, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const DMatrix A = well_conditioned(rng, n);
        const DMatrix b = random_vector(rng, n);
        const auto x = solve(A, b);
        ASSERT_TRUE(x.has_value()) << "n=" << n << " trial=" << trial;
        const auto residual = matmul(A, *x);
        ASSERT_TRUE(residual.has_value());
        EXPECT_LT(max_abs_difference(*residual, b), 1e-11)
            << "A*x != b at n=" << n << " trial=" << trial;
    }
}

// det(A*B) = det(A)*det(B). Nothing in the implementation uses this identity, so it
// cannot agree with it by construction -- which is the property that makes it worth
// asserting rather than a restatement of the code.
TEST(LinalgProperties, DeterminantIsMultiplicative) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(5, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 5);
        const DMatrix A = well_conditioned(rng, n);
        const DMatrix B = well_conditioned(rng, n);
        const auto AB = matmul(A, B);
        ASSERT_TRUE(AB.has_value());
        const auto det_a = det(A);
        const auto det_b = det(B);
        const auto det_ab = det(*AB);
        ASSERT_TRUE(det_a.has_value());
        ASSERT_TRUE(det_b.has_value());
        ASSERT_TRUE(det_ab.has_value());
        const double expected = *det_a * *det_b;
        // Determinants of a diagonally dominant n x n matrix run to about n^n, so the
        // comparison has to be relative. An absolute epsilon would pass vacuously at
        // n = 2 and fail on arithmetic alone at n = 6.
        EXPECT_LT(std::abs(*det_ab - expected), 1e-9 * std::max(1.0, std::abs(expected)))
            << "det(A*B)=" << *det_ab << " det(A)*det(B)=" << expected << " at n=" << n
            << " trial=" << trial;
    }
}

// trace(A*B) = trace(B*A), which holds even though A*B and B*A are different matrices.
TEST(LinalgProperties, TraceCommutesUnderProduct) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(6, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const DMatrix A = well_conditioned(rng, n);
        const DMatrix B = well_conditioned(rng, n);
        const auto AB = matmul(A, B);
        const auto BA = matmul(B, A);
        ASSERT_TRUE(AB.has_value());
        ASSERT_TRUE(BA.has_value());
        const auto trace_ab = trace(*AB);
        const auto trace_ba = trace(*BA);
        ASSERT_TRUE(trace_ab.has_value());
        ASSERT_TRUE(trace_ba.has_value());
        EXPECT_LT(std::abs(*trace_ab - *trace_ba),
                  1e-9 * std::max(1.0, std::abs(*trace_ab)))
            << "n=" << n << " trial=" << trial;
    }
}

// A * pinv(A) * A = A. This is the first Moore-Penrose condition, and it is the one
// that holds whether or not A has full rank -- so it is the right thing to check on a
// generated input, where rank is not something the test controls.
TEST(LinalgProperties, PseudoinverseSatisfiesMoorePenrose) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(7, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 5);
        const DMatrix A = well_conditioned(rng, n);
        const auto P = pinv(A);
        ASSERT_TRUE(P.has_value()) << "n=" << n << " trial=" << trial;
        const auto AP = matmul(A, *P);
        ASSERT_TRUE(AP.has_value());
        const auto APA = matmul(*AP, A);
        ASSERT_TRUE(APA.has_value());
        EXPECT_LT(max_abs_difference(*APA, A), 1e-9)
            << "A*pinv(A)*A != A at n=" << n << " trial=" << trial;
    }
}

// U * diag(S) * V^T = A, and U has orthonormal columns.
TEST(LinalgProperties, SvdReconstructsAndIsOrthogonal) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(8, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 5);
        const DMatrix A = well_conditioned(rng, n);
        const auto factored = svd(A);
        ASSERT_TRUE(factored.has_value()) << "n=" << n << " trial=" << trial;
        DMatrix Sigma = zeros<double>(factored->S.rows(), factored->S.rows());
        for (std::size_t i = 0; i < factored->S.rows(); ++i) {
            Sigma(i, i) = factored->S(i, 0);
            // Singular values are non-negative and ordered. A negative one means a
            // sign was moved into the wrong factor, which reconstruction alone hides.
            EXPECT_GE(factored->S(i, 0), -1e-12) << "negative singular value, trial=" << trial;
            if (i > 0) {
                EXPECT_LE(factored->S(i, 0), factored->S(i - 1, 0) + 1e-12)
                    << "singular values out of order at i=" << i << " trial=" << trial;
            }
        }
        const DMatrix reconstructed = factored->U * Sigma * transpose(factored->V);
        EXPECT_LT(max_abs_difference(reconstructed, A), 1e-9)
            << "U*S*V^T != A at n=" << n << " trial=" << trial;
        const DMatrix UtU = transpose(factored->U) * factored->U;
        EXPECT_LT(max_abs_difference_from_identity(UtU), 1e-9)
            << "U^T U != I at n=" << n << " trial=" << trial;
    }
}

// expm(A) * expm(-A) = I. A and -A commute, so the identity holds exactly in exact
// arithmetic for every A -- there is no eigenvalue condition to arrange for.
TEST(LinalgProperties, MatrixExponentialInvertsItsNegation) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(9, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 4);
        // Scaled down: expm of a diagonally dominant matrix with n on the diagonal is
        // e^n, and the product of two such matrices loses digits to cancellation
        // rather than to any defect. The identity is about the algorithm, so the input
        // is kept where the algorithm's accuracy is what is being measured.
        DMatrix A = well_conditioned(rng, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                A(i, j) *= 0.25;
            }
        }
        DMatrix minus_a = A;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                minus_a(i, j) = -A(i, j);
            }
        }
        const auto exp_a = expm(A);
        const auto exp_minus_a = expm(minus_a);
        ASSERT_TRUE(exp_a.has_value()) << "n=" << n << " trial=" << trial;
        ASSERT_TRUE(exp_minus_a.has_value()) << "n=" << n << " trial=" << trial;
        const auto product = matmul(*exp_a, *exp_minus_a);
        ASSERT_TRUE(product.has_value());
        EXPECT_LT(max_abs_difference_from_identity(*product), 1e-9)
            << "expm(A)*expm(-A) != I at n=" << n << " trial=" << trial;
    }
}

// transpose is an involution. Cheap, and it is the assertion that catches a storage
// order confused with a shape.
TEST(LinalgProperties, TransposeMovesEveryEntry) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(10, trial);
        std::uniform_real_distribution<double> entry(-1.0, 1.0);
        const std::size_t rows = 1 + static_cast<std::size_t>(trial % 5);
        const std::size_t cols = 1 + static_cast<std::size_t>((trial / 5) % 5);
        DMatrix A(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                A(i, j) = entry(rng);
            }
        }
        // Stated as the defining property rather than as `transpose(transpose(A))`.
        // `transpose` always returns a RowMajor matrix whatever it was given, and the
        // library instantiates it only for a ColMajor input -- a double transpose does
        // not link. It is also the weaker statement: `T(j,i) == A(i,j)` for every entry
        // is what makes the round trip hold, and it says which entry moved when it
        // fails.
        const auto T = transpose(A);
        ASSERT_EQ(T.rows(), cols);
        ASSERT_EQ(T.cols(), rows);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                EXPECT_EQ(T(j, i), A(i, j))
                    << "transpose moved (" << i << ", " << j << ") wrongly at " << rows
                    << "x" << cols;
            }
        }
    }
}

// ifft(fft(x)) = x. fft zero-pads to a power of two, so the round trip is only the
// identity on the padded length -- which is why the assertion reads the first n
// entries and checks the padding is zero rather than ignoring it.
TEST(LinalgProperties, InverseFftUndoesFft) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(11, trial);
        std::uniform_real_distribution<double> entry(-1.0, 1.0);
        const std::size_t n = 1 + static_cast<std::size_t>(trial % 16);
        std::vector<double> x(n);
        for (double& value : x) {
            value = entry(rng);
        }
        const auto spectrum = fft(x);
        ASSERT_TRUE(spectrum.has_value()) << "n=" << n << " trial=" << trial;
        const auto recovered = ifft(*spectrum);
        ASSERT_TRUE(recovered.has_value()) << "n=" << n << " trial=" << trial;
        ASSERT_GE(recovered->size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR((*recovered)[i], x[i], 1e-11)
                << "sample " << i << " of " << n << " trial=" << trial;
        }
        for (std::size_t i = n; i < recovered->size(); ++i) {
            EXPECT_NEAR((*recovered)[i], 0.0, 1e-11)
                << "padding sample " << i << " is not zero, trial=" << trial;
        }
    }
}

// idft(dft(x)) = x at every length, which is the property dft has and fft does not.
TEST(LinalgProperties, InverseDftUndoesDftAtEveryLength) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(12, trial);
        std::uniform_real_distribution<double> entry(-1.0, 1.0);
        const std::size_t n = 1 + static_cast<std::size_t>(trial % 13);
        std::vector<double> x(n);
        for (double& value : x) {
            value = entry(rng);
        }
        const auto spectrum = dft(x);
        ASSERT_TRUE(spectrum.has_value()) << "n=" << n << " trial=" << trial;
        ASSERT_EQ(spectrum->size(), n) << "dft padded, which is what fft does and dft does not";
        const auto recovered = idft(*spectrum);
        ASSERT_TRUE(recovered.has_value()) << "n=" << n << " trial=" << trial;
        ASSERT_EQ(recovered->size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR((*recovered)[i].real(), x[i], 1e-11)
                << "sample " << i << " of " << n << " trial=" << trial;
            EXPECT_NEAR((*recovered)[i].imag(), 0.0, 1e-11)
                << "a real input came back with an imaginary part at sample " << i;
        }
    }
}

// The entrywise 2-norm obeys the triangle inequality. It is the assertion that a norm
// is a norm, and it costs nothing.
TEST(LinalgProperties, NormObeysTheTriangleInequality) {
    for (int trial = 0; trial < kTrials; ++trial) {
        auto rng = stream_for(13, trial);
        const std::size_t n = 2 + static_cast<std::size_t>(trial % 6);
        const DMatrix A = well_conditioned(rng, n);
        const DMatrix B = well_conditioned(rng, n);
        DMatrix sum(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                sum(i, j) = A(i, j) + B(i, j);
            }
        }
        const auto norm_a = norm(A);
        const auto norm_b = norm(B);
        const auto norm_sum = norm(sum);
        ASSERT_TRUE(norm_a.has_value());
        ASSERT_TRUE(norm_b.has_value());
        ASSERT_TRUE(norm_sum.has_value());
        EXPECT_LE(*norm_sum, *norm_a + *norm_b + 1e-9)
            << "||A+B|| > ||A|| + ||B|| at n=" << n << " trial=" << trial;
        // And it is absolutely homogeneous, which the triangle inequality alone does
        // not imply.
        DMatrix scaled(n, n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                scaled(i, j) = -3.0 * A(i, j);
            }
        }
        const auto norm_scaled = norm(scaled);
        ASSERT_TRUE(norm_scaled.has_value());
        EXPECT_LT(std::abs(*norm_scaled - 3.0 * *norm_a), 1e-9 * std::max(1.0, *norm_a))
            << "|| -3A || != 3 ||A|| at n=" << n << " trial=" << trial;
    }
}

} // namespace
