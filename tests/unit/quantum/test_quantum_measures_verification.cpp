// Independent verification tests for the quantum information measures.
//
// Every expected value here was derived by hand or from an identity that does
// not go through the implementation: the qubit Bloch-vector formula for the
// trace distance, the qubit closed form F^2 = Tr(rho sigma) + 2 sqrt(det rho
// det sigma) for the fidelity, multiplicativity of the fidelity over tensor
// products, the X-state closed form for the concurrence, and a spectrum planted
// in a rotated basis for the entropy.  They complement
// test_quantum_information_measures.cpp rather than repeating it.
//
// The QuantumSchmidtResolution cases cover a separate defect: the Schmidt
// coefficients are square roots of Gram-matrix eigenvalues, so an exactly
// singular Gram matrix -- which every product state and every shape with
// dim_a > dim_b produces -- surfaced its round-off eigenvalue as a spurious
// coefficient of order sqrt(eps) ~ 7e-9, above schmidt_rank's 1e-10 default
// tolerance.  schmidt_rank therefore reported 2 for generic product states and
// up to dim_a for a dim_a > dim_b state whose rank cannot exceed dim_b.
#define _USE_MATH_DEFINES
#include "ms/quantum/quantum.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace ms::quantum;

namespace {

// A deterministic non-degenerate complex amplitude generator; no RNG, so the
// cases are byte-for-byte reproducible on every platform.
C amp(int i, double a, double b) {
    return C(std::cos(a * i + 0.37), std::sin(b * i - 0.21));
}

Ket deterministic_ket(int n, double a, double b) {
    Ket psi(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) psi[static_cast<std::size_t>(i)] = amp(i, a, b);
    return ket_normalise(psi);
}

// rho = 0.5 I + 0.5 (r . sigma) for Bloch vector r.
DensityMatrix from_bloch(double x, double y, double z) {
    return {{C(0.5 * (1.0 + z), 0.0), C(0.5 * x, -0.5 * y)},
            {C(0.5 * x, 0.5 * y), C(0.5 * (1.0 - z), 0.0)}};
}

}  // namespace

// ---------------------------------------------------------------------------
// Trace distance against the Bloch-vector formula T = |r_rho - r_sigma| / 2
// ---------------------------------------------------------------------------

// rho has Bloch vector (0.6, 0.2, 0), sigma has (0, -0.2, 0.6); the difference
// is (0.6, 0.4, -0.6) of length sqrt(0.88), so T = sqrt(0.88)/2 exactly.
// Half the Frobenius norm -- the pre-fix body -- gives 0.331662479036 here.
TEST(QuantumMeasuresIndependent, TraceDistanceMatchesBlochVectorDistance) {
    const DensityMatrix rho = from_bloch(0.6, 0.2, 0.0);
    const DensityMatrix sigma = from_bloch(0.0, -0.2, 0.6);
    // Guard the construction itself: both must be legitimate states.
    EXPECT_NEAR((rho[0][0] + rho[1][1]).real(), 1.0, 1e-15);
    EXPECT_NEAR((sigma[0][0] + sigma[1][1]).real(), 1.0, 1e-15);

    const double exact = 0.5 * std::sqrt(0.88);
    EXPECT_NEAR(exact, 0.469041575982343, 1e-14);
    EXPECT_NEAR(trace_distance(rho, sigma), exact, 1e-13);
    EXPECT_NEAR(trace_distance(sigma, rho), exact, 1e-13);
}

// ---------------------------------------------------------------------------
// Fidelity: qubit closed form, and multiplicativity over tensor products
// ---------------------------------------------------------------------------

// For the same pair, Tr(rho sigma) = 0.48 and det rho = det sigma = 0.15, so
// F^2 = 0.48 + 2 * 0.15 = 0.78 and F = sqrt(0.78) exactly.
TEST(QuantumMeasuresIndependent, QubitFidelityMatchesDeterminantClosedForm) {
    const DensityMatrix rho = from_bloch(0.6, 0.2, 0.0);
    const DensityMatrix sigma = from_bloch(0.0, -0.2, 0.6);

    const auto product = matmul_dm(rho, sigma);
    const double tr_rs = (product[0][0] + product[1][1]).real();
    const double det_rho = (rho[0][0] * rho[1][1] - rho[0][1] * rho[1][0]).real();
    const double det_sigma = (sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0]).real();
    EXPECT_NEAR(tr_rs, 0.48, 1e-15);
    EXPECT_NEAR(det_rho, 0.15, 1e-15);
    EXPECT_NEAR(det_sigma, 0.15, 1e-15);

    const double exact = std::sqrt(0.78);
    EXPECT_NEAR(exact, 0.883176086632785, 1e-14);
    EXPECT_NEAR(fidelity(rho, sigma), exact, 1e-13);  // pre-fix: 0.692820323028
}

// F is multiplicative on product states: F(a(x)b, c(x)d) = F(a,c) F(b,d).
// With the pair above this pins a genuine 4x4 fidelity to 0.78 exactly, using
// only the qubit determinant closed form as input.  Pre-fix: 0.48.
TEST(QuantumMeasuresIndependent, FidelityIsMultiplicativeOverTensorProducts) {
    const DensityMatrix rho = from_bloch(0.6, 0.2, 0.0);
    const DensityMatrix sigma = from_bloch(0.0, -0.2, 0.6);
    const DensityMatrix big_rho = tensor_product(rho, rho);
    const DensityMatrix big_sigma = tensor_product(sigma, sigma);

    EXPECT_NEAR(fidelity(big_rho, big_sigma), 0.78, 1e-12);
    EXPECT_NEAR(fidelity(big_rho, big_sigma),
                fidelity(rho, sigma) * fidelity(rho, sigma), 1e-12);
    // Mixed-pair version, so the two factors are not equal.
    const DensityMatrix mixed_a = tensor_product(rho, sigma);
    const DensityMatrix mixed_b = tensor_product(sigma, rho);
    EXPECT_NEAR(fidelity(mixed_a, mixed_b), 0.78, 1e-12);
}

// ---------------------------------------------------------------------------
// Concurrence of a MIXED X state whose coherence is purely imaginary
// ---------------------------------------------------------------------------

// rho00 = rho33 = 0.3, rho03 = 0.25i, rho11 = 0.25, rho22 = 0.15.
// C = 2 max(0, |rho03| - sqrt(rho11 rho22)) = 2 (0.25 - sqrt(0.0375)).
// The pre-fix expression used Re(rho03) and rho11 + rho22, giving exactly 0 --
// the imaginary coherence made it report a maximally-X-correlated state as
// separable.  A pure-state test cannot see the rho11 + rho22 half of that bug.
TEST(QuantumMeasuresIndependent, MixedXStateWithImaginaryCoherence) {
    DensityMatrix rho(4, std::vector<C>(4, C(0.0)));
    rho[0][0] = C(0.3, 0.0);
    rho[3][3] = C(0.3, 0.0);
    rho[0][3] = C(0.0, 0.25);
    rho[3][0] = C(0.0, -0.25);
    rho[1][1] = C(0.25, 0.0);
    rho[2][2] = C(0.15, 0.0);

    C trace = 0.0;
    for (int i = 0; i < 4; ++i) trace += rho[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    ASSERT_NEAR(trace.real(), 1.0, 1e-15);

    const double exact = 2.0 * (0.25 - std::sqrt(0.25 * 0.15));
    EXPECT_NEAR(exact, 0.112701665379258, 1e-14);
    EXPECT_NEAR(concurrence(rho), exact, 1e-10);
}

// ---------------------------------------------------------------------------
// Entropy of a planted spectrum in a rotated basis, dimension 12
// ---------------------------------------------------------------------------

// Dimension 12 is above the pre-fix n <= 8 exact path and is not a power of two,
// so nothing about the state is diagonal or degenerate.  The spectrum is
// planted as (0.5, 0.3, 0.2) on an orthonormal complex triple, so
// S = -0.5 ln 0.5 - 0.3 ln 0.3 - 0.2 ln 0.2 and Tr(rho^2) = 0.38 exactly.
TEST(QuantumMeasuresIndependent, EntropyOfPlantedRankThreeSpectrumAtDimTwelve) {
    const int n = 12;
    std::vector<Ket> basis = {deterministic_ket(n, 0.4, 0.13),
                              deterministic_ket(n, 0.9, 0.07),
                              deterministic_ket(n, 1.7, 0.53)};
    for (std::size_t k = 1; k < basis.size(); ++k) {
        for (std::size_t j = 0; j < k; ++j) {
            const C ov = inner(basis[j], basis[k]);
            for (int i = 0; i < n; ++i)
                basis[k][static_cast<std::size_t>(i)] -= ov * basis[j][static_cast<std::size_t>(i)];
        }
        basis[k] = ket_normalise(basis[k]);
    }
    for (std::size_t k = 1; k < basis.size(); ++k)
        for (std::size_t j = 0; j < k; ++j)
            ASSERT_NEAR(std::abs(inner(basis[j], basis[k])), 0.0, 1e-12) << j << "," << k;

    const double p[3] = {0.5, 0.3, 0.2};
    DensityMatrix rho(static_cast<std::size_t>(n), std::vector<C>(static_cast<std::size_t>(n), C(0.0)));
    for (std::size_t k = 0; k < basis.size(); ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                rho[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
                    p[k] * basis[k][static_cast<std::size_t>(i)] *
                    std::conj(basis[k][static_cast<std::size_t>(j)]);

    const double exact = -0.5 * std::log(0.5) - 0.3 * std::log(0.3) - 0.2 * std::log(0.2);
    EXPECT_NEAR(exact, 1.029653014064574, 1e-14);
    EXPECT_NEAR(von_neumann_entropy(rho), exact, 1e-12);
    EXPECT_NEAR(purity(rho), 0.38, 1e-12);
    // The Jacobi spectrum itself must reproduce the planted eigenvalues.
    const auto evals = eigenspectrum(rho);
    ASSERT_EQ(evals.size(), static_cast<std::size_t>(n));
    EXPECT_NEAR(evals[static_cast<std::size_t>(n) - 1], 0.5, 1e-12);
    EXPECT_NEAR(evals[static_cast<std::size_t>(n) - 2], 0.3, 1e-12);
    EXPECT_NEAR(evals[static_cast<std::size_t>(n) - 3], 0.2, 1e-12);
    for (int i = 0; i < n - 3; ++i)
        EXPECT_NEAR(evals[static_cast<std::size_t>(i)], 0.0, 1e-12) << "null eigenvalue " << i;
}

// The eigenvector path has to converge at n = 16 as well, not just the
// eigenvalues: ground_state must satisfy H v = lambda_min v componentwise.
TEST(QuantumMeasuresIndependent, GroundStateIsAnEigenvectorAtDimensionSixteen) {
    const int n = 16;
    DensityMatrix H(static_cast<std::size_t>(n), std::vector<C>(static_cast<std::size_t>(n), C(0.0)));
    for (int i = 0; i < n; ++i) {
        H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] =
            C(2.0 * std::sin(0.9 * i + 0.2), 0.0);
        for (int j = i + 1; j < n; ++j) {
            const C v(0.5 * std::cos(0.23 * i + 0.41 * j), 0.5 * std::sin(0.31 * i - 0.19 * j));
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
            H[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = std::conj(v);
        }
    }
    const auto evals = eigenspectrum(H);
    const auto v = ground_state(H);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(n));

    const auto Hv = op_apply(H, v);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(std::abs(Hv[static_cast<std::size_t>(i)] -
                             evals.front() * v[static_cast<std::size_t>(i)]),
                    0.0, 1e-10) << "component " << i;
    EXPECT_NEAR(inner(v, Hv).real(), evals.front(), 1e-10);
}

// ---------------------------------------------------------------------------
// Schmidt coefficient resolution / schmidt_rank
// ---------------------------------------------------------------------------

// A generic complex product state has Schmidt rank 1 in EVERY shape.  The
// existing ProductStateRankOne test uses |00>, whose Gram matrix is exactly
// diagonal so the pivot is skipped and the second eigenvalue is a hard zero;
// with a generic product state the Gram matrix is only singular to round-off
// and the second coefficient came out at ~7e-9, above the 1e-10 default tol.
TEST(QuantumSchmidtResolution, GenericProductStatesHaveRankOneInEveryShape) {
    for (int da = 2; da <= 4; ++da) {
        for (int db = 2; db <= 4; ++db) {
            const Ket a = deterministic_ket(da, 0.7 + 0.1 * da, 0.31);
            const Ket b = deterministic_ket(db, 1.3, 0.19 + 0.07 * db);
            const Ket psi = ket_normalise(tensor_product_states(a, b));

            EXPECT_EQ(schmidt_rank(psi, da, db), 1) << da << "(x)" << db;
            EXPECT_EQ(schmidt_number(psi, da, db), 1) << da << "(x)" << db;

            const auto decomp = schmidt_decomposition(psi, da, db);
            ASSERT_EQ(decomp.coefficients.size(), static_cast<std::size_t>(da));
            EXPECT_NEAR(decomp.coefficients[0], 1.0, 1e-12) << da << "(x)" << db;
            for (std::size_t k = 1; k < decomp.coefficients.size(); ++k)
                EXPECT_EQ(decomp.coefficients[k], 0.0) << da << "(x)" << db << " k=" << k;
        }
    }
}

// A dim_a x dim_b coefficient matrix has at most min(dim_a, dim_b) singular
// values, so schmidt_rank can never exceed that -- a 3 (x) 2 state cannot have
// rank 3.  Pre-fix, the third coefficient came back as 6.1e-9 and was counted.
TEST(QuantumSchmidtResolution, RankIsCappedByTheSmallerSubsystem) {
    struct Shape { int da; int db; };
    const std::vector<Shape> shapes = {{3, 2}, {4, 2}, {4, 3}, {2, 3}, {2, 4}, {3, 4}};
    for (const auto& sh : shapes) {
        const Ket psi = deterministic_ket(sh.da * sh.db, 0.53, 0.29);
        const int expected = std::min(sh.da, sh.db);
        EXPECT_EQ(schmidt_rank(psi, sh.da, sh.db), expected) << sh.da << "(x)" << sh.db;

        const auto decomp = schmidt_decomposition(psi, sh.da, sh.db);
        for (std::size_t k = static_cast<std::size_t>(expected); k < decomp.coefficients.size(); ++k)
            EXPECT_EQ(decomp.coefficients[k], 0.0) << sh.da << "(x)" << sh.db << " k=" << k;

        double norm_sq = 0.0;
        for (double sigma : decomp.coefficients) norm_sq += sigma * sigma;
        EXPECT_NEAR(norm_sq, 1.0, 1e-12) << sh.da << "(x)" << sh.db;
    }
}

// A spurious coefficient carries a spurious (arbitrary, unit-normalised) basis
// vector with it, so it also broke the reconstruction of |psi> at the 1e-9
// level for shapes with dim_a > dim_b.
TEST(QuantumSchmidtResolution, RankDeficientShapesReconstructExactly) {
    struct Shape { int da; int db; };
    const std::vector<Shape> shapes = {{3, 2}, {4, 2}, {4, 3}, {2, 3}};
    for (const auto& sh : shapes) {
        const Ket psi = deterministic_ket(sh.da * sh.db, 0.61, 0.37);
        const auto decomp = schmidt_decomposition(psi, sh.da, sh.db);
        ASSERT_FALSE(decomp.coefficients.empty());

        Ket rec(psi.size(), C(0.0));
        for (std::size_t k = 0; k < decomp.coefficients.size(); ++k) {
            if (decomp.coefficients[k] == 0.0) continue;
            const auto term = tensor_product_states(decomp.basis_a[k], decomp.basis_b[k]);
            for (std::size_t i = 0; i < term.size(); ++i)
                rec[i] += decomp.coefficients[k] * term[i];
        }
        for (std::size_t i = 0; i < psi.size(); ++i)
            EXPECT_NEAR(std::abs(rec[i] - psi[i]), 0.0, 1e-13)
                << sh.da << "(x)" << sh.db << " component " << i;
    }
}

// Entanglement entropy must agree with the reduced-state entropy for these
// same rank-deficient shapes -- the Schmidt coefficients feed it directly.
TEST(QuantumSchmidtResolution, EntanglementEntropyAgreesWithReducedState) {
    struct Shape { int da; int db; };
    const std::vector<Shape> shapes = {{3, 2}, {2, 3}, {4, 3}, {3, 4}};
    for (const auto& sh : shapes) {
        const Ket psi = deterministic_ket(sh.da * sh.db, 0.47, 0.83);
        const auto rho = density_matrix(psi);
        const double sa = von_neumann_entropy(partial_trace(rho, sh.da, sh.db, 0));
        const double sb = von_neumann_entropy(partial_trace(rho, sh.da, sh.db, 1));
        EXPECT_NEAR(sa, sb, 1e-11) << sh.da << "(x)" << sh.db;
        EXPECT_NEAR(entanglement_entropy(psi, sh.da, sh.db), sa, 1e-11)
            << sh.da << "(x)" << sh.db;
        EXPECT_LE(entanglement_entropy(psi, sh.da, sh.db),
                  std::log(static_cast<double>(std::min(sh.da, sh.db))) + 1e-12);
    }
}
