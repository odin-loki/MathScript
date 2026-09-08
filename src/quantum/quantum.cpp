#define _USE_MATH_DEFINES
#include "ms/quantum/quantum.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace quantum {

// ---- State construction ----

Ket ket_basis(int dim, int index) {
    Ket psi(dim, C(0.0));
    if (index < dim) psi[index] = C(1.0);
    return psi;
}

Ket ket_superposition(const std::vector<double>& amp) {
    Ket psi(amp.size());
    for (size_t i = 0; i < amp.size(); ++i) psi[i] = C(amp[i]);
    return ket_normalise(psi);
}

Ket ket_normalise(const Ket& psi) {
    double norm = 0.0;
    for (auto& c : psi) norm += std::norm(c);
    norm = std::sqrt(norm);
    if (norm < 1e-15) return psi;
    Ket out(psi.size());
    for (size_t i = 0; i < psi.size(); ++i) out[i] = psi[i] / norm;
    return out;
}

// ---- Inner / outer products ----

C inner(const Ket& bra, const Ket& ket) {
    C sum = 0.0;
    for (size_t i = 0; i < std::min(bra.size(), ket.size()); ++i)
        sum += std::conj(bra[i]) * ket[i];
    return sum;
}

DensityMatrix outer(const Ket& ket, const Ket& bra) {
    int m = static_cast<int>(ket.size());
    int n = static_cast<int>(bra.size());
    DensityMatrix rho(m, std::vector<C>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            rho[i][j] = ket[i] * std::conj(bra[j]);
    return rho;
}

DensityMatrix density_matrix(const Ket& psi) { return outer(psi, psi); }

// ---- Operator application ----

Ket op_apply(const DensityMatrix& op, const Ket& psi) {
    int m = static_cast<int>(op.size());
    Ket result(m, C(0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < (int)psi.size(); ++j) {
            if (std::norm(op[i][j]) < 1e-30) continue;
            result[i] += op[i][j] * psi[j];
        }
    return result;
}

DensityMatrix matmul_dm(const DensityMatrix& A, const DensityMatrix& B) {
    int m = static_cast<int>(A.size());
    int n = static_cast<int>(B[0].size());
    int k = static_cast<int>(B.size());
    DensityMatrix C(m, std::vector<::ms::quantum::C>(n, 0.0));
    for (int i = 0; i < m; ++i)
        for (int l = 0; l < k; ++l) {
            if (std::norm(A[i][l]) < 1e-30) continue;
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][l] * B[l][j];
        }
    return C;
}

DensityMatrix dagger(const DensityMatrix& op) {
    int m = static_cast<int>(op.size());
    int n = static_cast<int>(op[0].size());
    DensityMatrix D(n, std::vector<::ms::quantum::C>(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            D[j][i] = std::conj(op[i][j]);
    return D;
}

// ---- Measurement ----

double expectation(const Ket& psi, const DensityMatrix& A) {
    auto Apsi = op_apply(A, psi);
    return inner(psi, Apsi).real();
}

double expectation_dm(const DensityMatrix& rho, const DensityMatrix& A) {
    // Tr(rho A)
    auto rhoA = matmul_dm(rho, A);
    C trace = 0.0;
    for (int i = 0; i < (int)rhoA.size(); ++i) trace += rhoA[i][i];
    return trace.real();
}

namespace {

double observable_stddev(const Ket& psi, const DensityMatrix& A) {
    const double mean = expectation(psi, A);
    const auto A2 = matmul_dm(A, A);
    const double mean_sq = expectation(psi, A2);
    const double variance = mean_sq - mean * mean;
    return variance > 0.0 ? std::sqrt(variance) : 0.0;
}

} // namespace

double uncertainty(const Ket& psi, const DensityMatrix& A, const DensityMatrix& B) {
    return observable_stddev(psi, A) * observable_stddev(psi, B);
}

// ---- Pauli matrices ----

DensityMatrix pauli_x() { return {{{0,0},{1,0}}, {{1,0},{0,0}}}; }
DensityMatrix pauli_y() { return {{{0,0},{0,-1}}, {{0,1},{0,0}}}; }
DensityMatrix pauli_z() { return {{{1,0},{0,0}}, {{0,0},{-1,0}}}; }
DensityMatrix pauli_plus()  { return {{{0,0},{1,0}}, {{0,0},{0,0}}}; }
DensityMatrix pauli_minus() { return {{{0,0},{0,0}}, {{1,0},{0,0}}}; }

// ---- Standard gates ----

DensityMatrix hadamard() {
    double h = 1.0 / std::sqrt(2.0);
    return {{{h,0},{h,0}}, {{h,0},{-h,0}}};
}

DensityMatrix phase_gate(double theta) {
    return {{{1,0},{0,0}}, {{0,0},{std::cos(theta), std::sin(theta)}}};
}

DensityMatrix rotation_x(double theta) {
    C ct(std::cos(theta / 2.0), 0.0);
    C ist(0.0, -std::sin(theta / 2.0));
    return {{ct, ist}, {ist, ct}};
}

DensityMatrix rotation_y(double theta) {
    C ct(std::cos(theta / 2.0), 0.0);
    C st(-std::sin(theta / 2.0), 0.0);
    C stp(std::sin(theta / 2.0), 0.0);
    return {{ct, st}, {stp, ct}};
}

DensityMatrix rotation_z(double theta) {
    return {{{std::cos(-theta/2), std::sin(-theta/2)}, {0,0}},
            {{0,0}, {std::cos(theta/2), std::sin(theta/2)}}};
}

DensityMatrix identity(int dim) {
    DensityMatrix I(dim, std::vector<C>(dim, C(0.0)));
    for (int i = 0; i < dim; ++i) I[i][i] = C(1.0);
    return I;
}

DensityMatrix cnot_gate() {
    return {{{1,0},{0,0},{0,0},{0,0}},
            {{0,0},{1,0},{0,0},{0,0}},
            {{0,0},{0,0},{0,0},{1,0}},
            {{0,0},{0,0},{1,0},{0,0}}};
}

DensityMatrix swap_gate() {
    return {{{1,0},{0,0},{0,0},{0,0}},
            {{0,0},{0,0},{1,0},{0,0}},
            {{0,0},{1,0},{0,0},{0,0}},
            {{0,0},{0,0},{0,0},{1,0}}};
}

DensityMatrix toffoli_gate() {
    DensityMatrix T = identity(8);
    // Flip last two elements
    std::swap(T[6][6], T[6][7]);
    std::swap(T[7][6], T[7][7]);
    T[6][6] = C(0.0); T[6][7] = C(1.0);
    T[7][6] = C(1.0); T[7][7] = C(0.0);
    return T;
}

// ---- Tensor product ----

DensityMatrix tensor_product(const DensityMatrix& A, const DensityMatrix& B) {
    int ma = static_cast<int>(A.size()), na = static_cast<int>(A[0].size());
    int mb = static_cast<int>(B.size()), nb = static_cast<int>(B[0].size());
    DensityMatrix C(ma * mb, std::vector<::ms::quantum::C>(na * nb, 0.0));
    for (int i = 0; i < ma; ++i)
        for (int j = 0; j < na; ++j)
            for (int k = 0; k < mb; ++k)
                for (int l = 0; l < nb; ++l)
                    C[i * mb + k][j * nb + l] = A[i][j] * B[k][l];
    return C;
}

Ket tensor_product_states(const Ket& psi1, const Ket& psi2) {
    Ket result;
    result.reserve(psi1.size() * psi2.size());
    for (auto& a : psi1) for (auto& b : psi2) result.push_back(a * b);
    return result;
}

// ---- Commutator / anti-commutator ----

DensityMatrix commutator(const DensityMatrix& A, const DensityMatrix& B) {
    int n = static_cast<int>(A.size());
    DensityMatrix result(n, std::vector<::ms::quantum::C>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int l = 0; l < n; ++l) {
            const C a_il = A[i][l];
            const C b_il = B[i][l];
            const bool a_nz = std::norm(a_il) >= 1e-30;
            const bool b_nz = std::norm(b_il) >= 1e-30;
            if (!a_nz && !b_nz) continue;
            for (int j = 0; j < n; ++j) {
                if (a_nz) result[i][j] += a_il * B[l][j];
                if (b_nz) result[i][j] -= b_il * A[l][j];
            }
        }
    return result;
}

DensityMatrix anticommutator(const DensityMatrix& A, const DensityMatrix& B) {
    int n = static_cast<int>(A.size());
    DensityMatrix result(n, std::vector<::ms::quantum::C>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int l = 0; l < n; ++l) {
            const C a_il = A[i][l];
            const C b_il = B[i][l];
            const bool a_nz = std::norm(a_il) >= 1e-30;
            const bool b_nz = std::norm(b_il) >= 1e-30;
            if (!a_nz && !b_nz) continue;
            for (int j = 0; j < n; ++j) {
                if (a_nz) result[i][j] += a_il * B[l][j];
                if (b_nz) result[i][j] += b_il * A[l][j];
            }
        }
    return result;
}

// ---- QFT gate ----

DensityMatrix qft_gate(int n_qubits) {
    int N = 1 << n_qubits;
    DensityMatrix Q(N, std::vector<::ms::quantum::C>(N));
    double inv_sqN = 1.0 / std::sqrt(N);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k) {
            double angle = 2.0 * M_PI * j * k / N;
            Q[j][k] = C(inv_sqN * std::cos(angle), inv_sqN * std::sin(angle));
        }
    return Q;
}

// ---- Grover's search algorithm ----

namespace {

// n-fold Kronecker power of a single-qubit operator: op^{\otimes n}, using the
// existing tensor_product() Kronecker-product-for-operators helper.
DensityMatrix kronecker_power(const DensityMatrix& op, int n) {
    DensityMatrix result = identity(1);
    for (int i = 0; i < n; ++i) result = tensor_product(result, op);
    return result;
}

} // namespace

Ket grover_search(int n_qubits, const std::vector<int>& marked_indices, int n_iterations) {
    if (n_qubits <= 0) return {};
    const int N = 1 << n_qubits;

    // Uniform superposition H^{\otimes n}|0> == closed-form 1/sqrt(N) for every entry.
    const double inv_sqN = 1.0 / std::sqrt(static_cast<double>(N));
    Ket psi(N, C(inv_sqN));

    if (n_iterations <= 0) return psi;

    // Oracle: diagonal +/-1 matrix, -1 at valid marked indices.
    DensityMatrix O = identity(N);
    for (int idx : marked_indices) {
        if (idx >= 0 && idx < N) O[idx][idx] = C(-1.0);
    }

    // Diffusion: D = H^{\otimes n} (2|0><0| - I) H^{\otimes n}.
    DensityMatrix Hn = kronecker_power(hadamard(), n_qubits);
    DensityMatrix refl0 = identity(N);
    refl0[0][0] = C(1.0); // 2|0><0| - I has (2-1)=1 at [0][0], -1 elsewhere on diagonal.
    for (int i = 1; i < N; ++i) refl0[i][i] = C(-1.0);
    DensityMatrix D = matmul_dm(Hn, matmul_dm(refl0, Hn));

    for (int it = 0; it < n_iterations; ++it) {
        psi = op_apply(O, psi);
        psi = op_apply(D, psi);
    }
    return psi;
}

int grover_optimal_iterations(int n_qubits, int n_marked) {
    if (n_qubits <= 0 || n_marked <= 0) return 0;
    const int N = 1 << n_qubits;
    if (n_marked >= N) return 0;
    const double ratio = static_cast<double>(N) / static_cast<double>(n_marked);
    return static_cast<int>(std::floor(M_PI / 4.0 * std::sqrt(ratio)));
}

// ---- Entropy & information ----

namespace {

// Jacobi diagonalisation for Hermitian matrices of any order n.
//
// Each step annihilates one off-diagonal pair (p, q) by a unitary similarity
// H <- G^H H G, where G is a diagonal phase matrix (making the pivot real and
// positive) followed by a real Givens rotation.  Cyclic sweeps over every pair
// converge quadratically, so the routine is accurate to round-off for the
// dimensions a density matrix or a small Hamiltonian actually reaches -- there
// is no n <= 8 restriction.
//
// On return the diagonal of H holds the eigenvalues (the off-diagonal entries
// are zero to round-off).  When evecs is non-null, column j of *evecs is a
// unit eigenvector for eigenvalue H[j][j], i.e. H_in = V diag(H) V^H.
void hermitian_jacobi_diagonalize(DensityMatrix& H, int n, DensityMatrix* evecs = nullptr,
                                  double tol = 1e-14, int max_sweeps = 60) {
    if (evecs) {
        *evecs = identity(n);
    }
    if (n <= 1) return;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        double frob = 0.0;
        for (int i = 0; i < n; ++i) {
            frob += std::norm(H[i][i]);
            for (int j = i + 1; j < n; ++j) {
                const double m = std::norm(H[i][j]);
                off += 2.0 * m;
                frob += 2.0 * m;
            }
        }
        const double scale = std::max(1.0, std::sqrt(frob));
        if (std::sqrt(off) <= tol * scale) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const C apq0 = H[p][q];
                const double mag = std::abs(apq0);
                // Negligible relative to the two diagonal entries it couples:
                // annihilate it outright rather than rotating on round-off.
                if (mag <= 1e-18 * (std::abs(H[p][p]) + std::abs(H[q][q]))) {
                    H[p][q] = C(0.0, 0.0);
                    H[q][p] = C(0.0, 0.0);
                    continue;
                }

                // Phase elimination.  With D = diag(.., dp, .., dq, ..) the
                // similarity H <- D^H H D maps H[k][l] to conj(d_k) H[k][l] d_l,
                // so choosing dp = exp(+i*arg(apq)/2), dq = exp(-i*arg(apq)/2)
                // leaves H[p][q] = |apq| real and positive.  Eigenvectors
                // accumulate as V <- V D.
                const double half_phase = 0.5 * std::arg(apq0);
                const C dp = std::exp(C(0.0, half_phase));
                const C dq = std::exp(C(0.0, -half_phase));
                for (int k = 0; k < n; ++k) {
                    H[k][p] *= dp;
                    H[p][k] *= std::conj(dp);
                    H[k][q] *= dq;
                    H[q][k] *= std::conj(dq);
                }
                if (evecs) {
                    for (int k = 0; k < n; ++k) {
                        (*evecs)[k][p] *= dp;
                        (*evecs)[k][q] *= dq;
                    }
                }

                // Real Givens rotation G (G[p][p]=c, G[p][q]=s, G[q][p]=-s,
                // G[q][q]=c) with tan(2*phi) = 2*|apq| / (aqq - app), which
                // zeroes the (now real) pivot.  H <- G^T H G, V <- V G.
                const double app = H[p][p].real();
                const double aqq = H[q][q].real();
                const double apq = mag;
                const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double c = std::cos(phi);
                const double s = std::sin(phi);

                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    const C hkp = H[k][p];
                    const C hkq = H[k][q];
                    const C vp = c * hkp - s * hkq;
                    const C vq = s * hkp + c * hkq;
                    // H stays Hermitian: the transposed slot takes the
                    // conjugate, not the same value.
                    H[k][p] = vp;
                    H[p][k] = std::conj(vp);
                    H[k][q] = vq;
                    H[q][k] = std::conj(vq);
                }
                H[p][p] = C(c * c * app - 2.0 * s * c * apq + s * s * aqq, 0.0);
                H[q][q] = C(s * s * app + 2.0 * s * c * apq + c * c * aqq, 0.0);
                H[p][q] = C(0.0, 0.0);
                H[q][p] = C(0.0, 0.0);

                if (evecs) {
                    for (int k = 0; k < n; ++k) {
                        const C wp = (*evecs)[k][p];
                        const C wq = (*evecs)[k][q];
                        (*evecs)[k][p] = c * wp - s * wq;
                        (*evecs)[k][q] = s * wp + c * wq;
                    }
                }
            }
        }
    }
}

// Average a matrix with its conjugate transpose.  Products such as
// sqrt(rho) sigma sqrt(rho) are Hermitian in exact arithmetic; this removes the
// round-off-sized skew part before the Jacobi solver sees them.
DensityMatrix hermitian_part(const DensityMatrix& A) {
    const int n = static_cast<int>(A.size());
    DensityMatrix S(n, std::vector<C>(n, C(0.0)));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            S[i][j] = 0.5 * (A[i][j] + std::conj(A[j][i]));
    return S;
}

std::vector<double> hermitian_eigenvalues(const DensityMatrix& rho) {
    const int n = static_cast<int>(rho.size());
    DensityMatrix H = rho;
    hermitian_jacobi_diagonalize(H, n);
    std::vector<double> evals(n);
    for (int i = 0; i < n; ++i)
        evals[i] = H[i][i].real();
    return evals;
}

void hermitian_eigendecomposition(const DensityMatrix& H_in, int n,
                                   std::vector<double>& evals, DensityMatrix& evecs) {
    DensityMatrix H = H_in;
    hermitian_jacobi_diagonalize(H, n, &evecs);
    evals.resize(n);
    for (int i = 0; i < n; ++i)
        evals[i] = H[i][i].real();
}

// Positive-semidefinite square root of a Hermitian matrix:
// A = U diag(lambda) U^H  ->  sqrt(A) = U diag(sqrt(max(lambda, 0))) U^H.
// Eigenvalues that come out slightly negative through round-off are clamped to
// zero, which is the correct projection for a density matrix.
// Square roots of the eigenvalues of a Hermitian positive-semi-definite matrix,
// with the round-off floor removed.  A Jacobi sweep resolves eigenvalues only to
// O(n * eps * ||M||); the square root maps that noise up to O(sqrt(eps)) ~ 1e-8,
// which would otherwise show up as a 1e-8 shortfall in the concurrence of a Bell
// state.  Anything at or below the floor is therefore taken to be exactly zero.
std::vector<double> psd_eigenvalue_roots(const DensityMatrix& M) {
    const std::vector<double> mu = hermitian_eigenvalues(M);
    double mu_max = 0.0;
    for (double value : mu) mu_max = std::max(mu_max, std::abs(value));
    const double floor_mu = 8.0 * static_cast<double>(mu.size()) *
                            std::numeric_limits<double>::epsilon() * mu_max;

    std::vector<double> roots(mu.size(), 0.0);
    for (std::size_t i = 0; i < mu.size(); ++i)
        if (mu[i] > floor_mu) roots[i] = std::sqrt(mu[i]);
    return roots;
}

DensityMatrix hermitian_psd_sqrt(const DensityMatrix& A) {
    const int n = static_cast<int>(A.size());
    std::vector<double> evals;
    DensityMatrix evecs;
    hermitian_eigendecomposition(A, n, evals, evecs);

    DensityMatrix S(n, std::vector<C>(n, C(0.0)));
    for (int k = 0; k < n; ++k) {
        const double root = std::sqrt(std::max(0.0, evals[k]));
        if (root <= 0.0) continue;
        for (int i = 0; i < n; ++i) {
            const C uik = evecs[i][k];
            if (std::norm(uik) < 1e-30) continue;
            for (int j = 0; j < n; ++j)
                S[i][j] += root * uik * std::conj(evecs[j][k]);
        }
    }
    return S;
}

} // namespace

std::vector<double> eigenspectrum(const DensityMatrix& H) {
    auto evals = hermitian_eigenvalues(H);
    std::sort(evals.begin(), evals.end());
    return evals;
}

Ket ground_state(const DensityMatrix& H) {
    const int n = static_cast<int>(H.size());
    std::vector<double> evals;
    DensityMatrix evecs;
    hermitian_eigendecomposition(H, n, evals, evecs);

    int min_idx = 0;
    for (int i = 1; i < n; ++i)
        if (evals[i] < evals[min_idx]) min_idx = i;

    Ket psi(n);
    for (int i = 0; i < n; ++i)
        psi[i] = evecs[i][min_idx];
    return ket_normalise(psi);
}

// von Neumann entropy: S(rho) = -Tr(rho log rho) = -sum_i lambda_i log(lambda_i)
// (natural logarithm), computed from the full Hermitian eigenspectrum of rho at
// every dimension.  Eigenvalues at or below the round-off floor contribute
// nothing (0 log 0 == 0 by continuity).
double von_neumann_entropy(const DensityMatrix& rho) {
    const int n = static_cast<int>(rho.size());
    if (n <= 0) return 0.0;

    double S = 0.0;
    for (double lambda : hermitian_eigenvalues(hermitian_part(rho))) {
        if (lambda > 1e-15) S -= lambda * std::log(lambda);
    }
    return S;
}

double purity(const DensityMatrix& rho) {
    auto rho2 = matmul_dm(rho, rho);
    C trace = 0.0;
    for (int i = 0; i < static_cast<int>(rho2.size()); ++i)
        trace += rho2[i][i];
    return trace.real();
}

double fidelity(const DensityMatrix& rho, const DensityMatrix& sigma) {
    // Uhlmann fidelity, square-root convention:
    //   F(rho, sigma) = Tr sqrt( sqrt(rho) sigma sqrt(rho) )
    //                 = sum_i sqrt(mu_i),   mu_i = eig( sqrt(rho) sigma sqrt(rho) ).
    // The bracketed matrix is Hermitian positive semi-definite, so its
    // eigenvalues are real and non-negative and the sum of their square roots
    // is exactly the trace norm of sqrt(rho) sqrt(sigma).
    const int n = static_cast<int>(rho.size());
    if (n <= 0 || static_cast<int>(sigma.size()) != n) return 0.0;

    const DensityMatrix root_rho = hermitian_psd_sqrt(hermitian_part(rho));
    const DensityMatrix inner_m =
        hermitian_part(matmul_dm(root_rho, matmul_dm(sigma, root_rho)));

    double F = 0.0;
    for (double root : psd_eigenvalue_roots(inner_m)) F += root;
    return F;
}

double trace_distance(const DensityMatrix& rho, const DensityMatrix& sigma) {
    // T = (1/2) Tr|rho - sigma|, the trace (nuclear) norm of the difference.
    // D = rho - sigma is Hermitian, so |D| has the same eigenvectors as D with
    // eigenvalues |lambda_i| and Tr|D| = sum_i |lambda_i|.  This is NOT the
    // Frobenius norm: the two agree only when D has a single non-zero
    // eigenvalue.
    const int n = static_cast<int>(rho.size());
    if (n <= 0 || static_cast<int>(sigma.size()) != n) return 0.0;

    DensityMatrix diff(n, std::vector<C>(n, C(0.0)));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            diff[i][j] = rho[i][j] - sigma[i][j];

    double sum = 0.0;
    for (double lambda : hermitian_eigenvalues(hermitian_part(diff)))
        sum += std::abs(lambda);
    return 0.5 * sum;
}

double concurrence(const DensityMatrix& rho) {
    // Wootters' concurrence of a two-qubit state:
    //   rho_tilde = (sigma_y (x) sigma_y) conj(rho) (sigma_y (x) sigma_y)
    //   lambda_1 >= .. >= lambda_4 are the square roots of the eigenvalues of
    //   rho * rho_tilde, and C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4).
    //
    // rho * rho_tilde is not Hermitian, so instead diagonalise the similar
    // matrix sqrt(rho) rho_tilde sqrt(rho): it is Hermitian positive
    // semi-definite and has the same spectrum (X Y and Y X share eigenvalues,
    // with X = sqrt(rho), Y = rho_tilde sqrt(rho)).
    //
    // Defined only for 4x4 (two-qubit) density matrices; anything else returns
    // NaN rather than a fabricated 0, which would read as "separable".
    if (rho.size() != 4) return std::numeric_limits<double>::quiet_NaN();
    for (const auto& row : rho)
        if (row.size() != 4) return std::numeric_limits<double>::quiet_NaN();

    const DensityMatrix spin_flip = tensor_product(pauli_y(), pauli_y());
    DensityMatrix rho_conj(4, std::vector<C>(4, C(0.0)));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            rho_conj[i][j] = std::conj(rho[i][j]);

    const DensityMatrix rho_tilde =
        matmul_dm(spin_flip, matmul_dm(rho_conj, spin_flip));
    const DensityMatrix root_rho = hermitian_psd_sqrt(hermitian_part(rho));
    const DensityMatrix product =
        hermitian_part(matmul_dm(root_rho, matmul_dm(rho_tilde, root_rho)));

    std::vector<double> lambda = psd_eigenvalue_roots(product);
    std::sort(lambda.begin(), lambda.end());  // ascending: lambda_1 is last

    return std::max(0.0, lambda[3] - lambda[2] - lambda[1] - lambda[0]);
}

// ---- Partial trace ----

DensityMatrix partial_trace(const DensityMatrix& rho, int d1, int d2, int subsystem) {
    // subsystem: 0 → trace out system B (d2), return d1 × d1
    //            1 → trace out system A (d1), return d2 × d2
    if (subsystem == 0) {
        // Trace out B: rhoA[i][j] = sum_k rho[i*d2+k][j*d2+k]
        DensityMatrix rhoA(d1, std::vector<C>(d1, C(0.0)));
        for (int i = 0; i < d1; ++i)
            for (int j = 0; j < d1; ++j)
                for (int k = 0; k < d2; ++k)
                    rhoA[i][j] += rho[i * d2 + k][j * d2 + k];
        return rhoA;
    } else {
        // Trace out A: rhoB[k][l] = sum_i rho[i*d2+k][i*d2+l]
        DensityMatrix rhoB(d2, std::vector<C>(d2, C(0.0)));
        for (int k = 0; k < d2; ++k)
            for (int l = 0; l < d2; ++l)
                for (int i = 0; i < d1; ++i)
                    rhoB[k][l] += rho[i * d2 + k][i * d2 + l];
        return rhoB;
    }
}

// ---- Schmidt decomposition ----

namespace {

// Reshape |psi> into coefficient matrix M[i][j] = psi[i*dim_b + j] (subsystem A ⊗ B).
DensityMatrix reshape_bipartite_coefficients(const Ket& psi, int dim_a, int dim_b) {
    DensityMatrix M(dim_a, std::vector<C>(dim_b, C(0.0)));
    for (int i = 0; i < dim_a; ++i)
        for (int j = 0; j < dim_b; ++j)
            M[i][j] = psi[static_cast<size_t>(i * dim_b + j)];
    return M;
}

// Hermitian Gram matrix M M† for the left Schmidt spectrum.
DensityMatrix gram_left(const DensityMatrix& M, int dim_a, int dim_b) {
    DensityMatrix G(dim_a, std::vector<C>(dim_a, C(0.0)));
    for (int i = 0; i < dim_a; ++i)
        for (int j = 0; j < dim_a; ++j)
            for (int k = 0; k < dim_b; ++k)
                G[i][j] += M[i][k] * std::conj(M[j][k]);
    return G;
}

// Right Schmidt vector for singular value sigma > 0.
//
// With M = U S V^dagger the reshaped coefficient matrix,
//   |psi> = sum_ij M[i][j] |i>|j> = sum_k sigma_k (sum_i u_k[i]|i>) (x) (sum_j conj(v_k[j])|j>),
// so the Schmidt vector on B is conj(v_k), NOT v_k -- equivalently it is the
// eigenvector of rho_B = conj(M^dagger M), not of M^dagger M.  Hence
// b_k = conj(M^dagger u_k) / sigma_k = (M^T conj(u_k)) / sigma_k.
// For a real |psi> the conjugation is a no-op, which is why a real Bell state
// reconstructs either way.
Ket right_schmidt_vector(const DensityMatrix& M, const Ket& u, int dim_b, double sigma) {
    Ket v(dim_b, C(0.0));
    const int dim_a = static_cast<int>(M.size());
    for (int k = 0; k < dim_b; ++k)
        for (int i = 0; i < dim_a; ++i)
            v[k] += M[i][k] * std::conj(u[i]);
    if (sigma > 1e-15) {
        for (int k = 0; k < dim_b; ++k) v[k] /= sigma;
    }
    return ket_normalise(v);
}

double entropy_from_schmidt_coefficients(const std::vector<double>& coeffs) {
    double S = 0.0;
    for (double lambda : coeffs) {
        const double p = lambda * lambda;
        if (p > 1e-15) S -= p * std::log(p);
    }
    return S;
}

} // namespace

SchmidtDecomposition schmidt_decomposition(const Ket& psi, int dim_a, int dim_b) {
    SchmidtDecomposition result;
    if (dim_a <= 0 || dim_b <= 0) return result;
    const int expected = dim_a * dim_b;
    if (static_cast<int>(psi.size()) != expected) return result;

    const Ket psi_norm = ket_normalise(psi);
    const DensityMatrix M = reshape_bipartite_coefficients(psi_norm, dim_a, dim_b);
    const DensityMatrix G = gram_left(M, dim_a, dim_b);

    std::vector<double> evals;
    DensityMatrix evecs;
    hermitian_eigendecomposition(G, dim_a, evals, evecs);

    // Sort eigenvalues (Schmidt coefficient squares) descending.
    std::vector<int> order(dim_a);
    for (int i = 0; i < dim_a; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return evals[a] > evals[b];
    });

    result.coefficients.reserve(static_cast<size_t>(dim_a));
    result.basis_a.reserve(static_cast<size_t>(dim_a));
    result.basis_b.reserve(static_cast<size_t>(dim_a));

    for (int idx : order) {
        const double ev = std::max(0.0, evals[idx]);
        const double sigma = std::sqrt(ev);
        result.coefficients.push_back(sigma);

        Ket u(dim_a);
        for (int i = 0; i < dim_a; ++i) u[i] = evecs[i][idx];
        result.basis_a.push_back(ket_normalise(u));
        result.basis_b.push_back(right_schmidt_vector(M, result.basis_a.back(), dim_b, sigma));
    }

    return result;
}

int schmidt_rank(const Ket& psi, int dim_a, int dim_b, double tol) {
    const auto decomp = schmidt_decomposition(psi, dim_a, dim_b);
    int rank = 0;
    for (double lambda : decomp.coefficients)
        if (lambda > tol) ++rank;
    return rank;
}

int schmidt_number(const Ket& psi, int dim_a, int dim_b, double tol) {
    // For pure bipartite states the Schmidt number equals the Schmidt rank.
    return schmidt_rank(psi, dim_a, dim_b, tol);
}

// ---- Entanglement entropy ----

double entanglement_entropy(const Ket& psi, int dim_a, int dim_b) {
    // Schmidt coefficients squared are reduced-density-matrix eigenvalues.
    const auto decomp = schmidt_decomposition(psi, dim_a, dim_b);
    if (!decomp.coefficients.empty())
        return entropy_from_schmidt_coefficients(decomp.coefficients);

    // Fallback for invalid subsystem dimensions.
    auto rho = density_matrix(psi);
    auto rhoA = partial_trace(rho, dim_a, dim_b, 0);
    return von_neumann_entropy(rhoA);
}

// ---- Quantum states ----

std::vector<Ket> bell_states() {
    double h = 1.0 / std::sqrt(2.0);
    // |Φ+> = (|00> + |11>) / sqrt(2)
    Ket phi_plus  = {h, 0, 0, h};
    // |Φ-> = (|00> - |11>) / sqrt(2)
    Ket phi_minus = {h, 0, 0, -h};
    // |Ψ+> = (|01> + |10>) / sqrt(2)
    Ket psi_plus  = {0, h, h, 0};
    // |Ψ-> = (|01> - |10>) / sqrt(2)
    Ket psi_minus = {0, h, -h, 0};
    return {phi_plus, phi_minus, psi_plus, psi_minus};
}

Ket ghz_state(int n_qubits) {
    int dim = 1 << n_qubits;
    Ket psi(dim, C(0.0));
    double h = 1.0 / std::sqrt(2.0);
    psi[0] = C(h);
    psi[dim - 1] = C(h);
    return psi;
}

Ket w_state(int n_qubits) {
    int dim = 1 << n_qubits;
    Ket psi(dim, C(0.0));
    double amp = 1.0 / std::sqrt(n_qubits);
    for (int i = 0; i < n_qubits; ++i)
        psi[1 << (n_qubits - 1 - i)] = C(amp);
    return psi;
}

Ket coherent_state(C alpha, int n_max) {
    // |alpha> = exp(-|alpha|^2/2) sum_{n=0}^{n_max} alpha^n/sqrt(n!) |n>
    Ket psi(n_max + 1, C(0.0));
    double norm_factor = std::exp(-std::norm(alpha) / 2.0);
    C alpha_pow = 1.0;
    double fact = 1.0;
    for (int n = 0; n <= n_max; ++n) {
        if (n > 0) { alpha_pow *= alpha; fact *= n; }
        psi[n] = C(norm_factor) * alpha_pow / C(std::sqrt(fact));
    }
    return ket_normalise(psi);
}

Ket fock_state(int n, int n_max) {
    return ket_basis(n_max + 1, n);
}

// ---- Time evolution ----

DensityMatrix time_evolution_operator(const DensityMatrix& H, double t) {
    // U(t) = exp(-i H t) via Taylor series sum_{k=0}^{K} (-iHt)^k / k!
    int n = static_cast<int>(H.size());
    DensityMatrix U = identity(n);
    DensityMatrix term = identity(n);
    C factor = C(0.0, -t);
    for (int k = 1; k <= 30; ++k) {
        term = matmul_dm(term, H);
        // multiply by (-it)^k / k!
        C coeff = 1.0;
        for (int j = 0; j < k; ++j) coeff *= factor / C(j + 1);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                U[i][j] += coeff * term[i][j];
    }
    return U;
}

std::vector<Ket> schrodinger(const DensityMatrix& H, const Ket& psi0,
                              double t0, double t1, int n_steps) {
    // |psi(t+dt)> ≈ exp(-i H dt) |psi(t)>
    std::vector<Ket> trajectory;
    trajectory.reserve(n_steps + 1);
    trajectory.push_back(psi0);
    Ket psi = psi0;
    double dt = (t1 - t0) / n_steps;
    auto U = time_evolution_operator(H, dt);
    for (int step = 0; step < n_steps; ++step) {
        psi = op_apply(U, psi);
        trajectory.push_back(psi);
    }
    return trajectory;
}

// ---- Phase-space quasi-probability distributions ----

namespace {

// Truncated annihilation operator on Fock levels 0..dim-1: a|n> = sqrt(n)|n-1>.
DensityMatrix annihilation_operator(int dim) {
    DensityMatrix a(dim, std::vector<C>(dim, C(0.0)));
    for (int n = 1; n < dim; ++n) a[n - 1][n] = C(std::sqrt(static_cast<double>(n)));
    return a;
}

// Parity operator on Fock levels: Parity|n> = (-1)^n |n>.
DensityMatrix parity_operator(int dim) {
    DensityMatrix P(dim, std::vector<C>(dim, C(0.0)));
    for (int n = 0; n < dim; ++n) P[n][n] = C(n % 2 == 0 ? 1.0 : -1.0);
    return P;
}

// exp(X) for an arbitrary square complex matrix via scaling-and-squaring:
// halve X until its (row-sum) norm is small, Taylor-expand, then square back.
DensityMatrix matrix_exp(const DensityMatrix& X) {
    const int n = static_cast<int>(X.size());
    double row_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += std::abs(X[i][j]);
        row_norm = std::max(row_norm, s);
    }

    int squarings = 0;
    double scale = 1.0;
    while (row_norm * scale > 0.5) {
        scale *= 0.5;
        ++squarings;
    }

    DensityMatrix Xs(n, std::vector<C>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Xs[i][j] = X[i][j] * scale;

    DensityMatrix result = identity(n);
    DensityMatrix term = identity(n);
    for (int k = 1; k <= 25; ++k) {
        term = matmul_dm(term, Xs);
        const C inv_k = C(1.0 / static_cast<double>(k));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                term[i][j] *= inv_k;
                result[i][j] += term[i][j];
            }
    }

    for (int s = 0; s < squarings; ++s) result = matmul_dm(result, result);
    return result;
}

// Displacement operator D(alpha) = exp(alpha a^dagger - conj(alpha) a) truncated
// to the same Fock dimension as rho.
DensityMatrix displacement_operator(C alpha, int dim) {
    DensityMatrix a = annihilation_operator(dim);
    DensityMatrix adag = dagger(a);
    DensityMatrix X(dim, std::vector<C>(dim));
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            X[i][j] = alpha * adag[i][j] - std::conj(alpha) * a[i][j];
    return matrix_exp(X);
}

} // namespace

double wigner_function(const DensityMatrix& rho, double x, double p) {
    const int dim = static_cast<int>(rho.size());
    if (dim <= 0) return 0.0;

    const C alpha(x / std::sqrt(2.0), p / std::sqrt(2.0));
    DensityMatrix D = displacement_operator(alpha, dim);
    DensityMatrix Ddag = dagger(D);
    DensityMatrix P = parity_operator(dim);

    DensityMatrix DP = matmul_dm(D, P);
    DensityMatrix DPDdag = matmul_dm(DP, Ddag);
    DensityMatrix M = matmul_dm(rho, DPDdag);

    C trace = 0.0;
    for (int i = 0; i < dim; ++i) trace += M[i][i];
    return trace.real() / M_PI;
}

double husimi_Q(const DensityMatrix& rho, C alpha) {
    const int dim = static_cast<int>(rho.size());
    if (dim <= 0) return 0.0;

    const int n_max = dim - 1;
    Ket alpha_ket = coherent_state(alpha, n_max);
    Ket rho_alpha = op_apply(rho, alpha_ket);
    const C overlap = inner(alpha_ket, rho_alpha);   // <alpha|rho|alpha>
    return overlap.real() / M_PI;
}

} // namespace quantum
} // namespace ms
