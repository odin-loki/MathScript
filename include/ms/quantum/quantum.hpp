#pragma once
#include <complex>
#include <functional>
#include <vector>

namespace ms {
namespace quantum {

using C = std::complex<double>;
using Ket   = std::vector<C>;           // column state vector
using DensityMatrix = std::vector<std::vector<C>>;

// ---- State construction ----
Ket ket_basis(int dim, int index);      // |index> in dim-dimensional space
Ket ket_superposition(const std::vector<double>& amplitudes);
Ket ket_normalise(const Ket& psi);

// ---- Inner / outer products ----
C inner(const Ket& bra, const Ket& ket);        // <bra|ket>
DensityMatrix outer(const Ket& ket, const Ket& bra);    // |ket><bra|
DensityMatrix density_matrix(const Ket& psi);   // |psi><psi|

// ---- Operator application ----
Ket op_apply(const DensityMatrix& op, const Ket& psi);  // op * psi
DensityMatrix matmul_dm(const DensityMatrix& A, const DensityMatrix& B);
DensityMatrix dagger(const DensityMatrix& op);   // conjugate transpose

// ---- Measurement ----
double expectation(const Ket& psi, const DensityMatrix& A);   // <psi|A|psi>
double expectation_dm(const DensityMatrix& rho, const DensityMatrix& A);
// Standard-deviation product Δ(A)·Δ(B) for state |psi>
double uncertainty(const Ket& psi, const DensityMatrix& A, const DensityMatrix& B);

// ---- Pauli matrices ----
DensityMatrix pauli_x();
DensityMatrix pauli_y();
DensityMatrix pauli_z();
DensityMatrix pauli_plus();
DensityMatrix pauli_minus();

// ---- Standard gates ----
DensityMatrix hadamard();
DensityMatrix phase_gate(double theta);
DensityMatrix rotation_x(double theta);
DensityMatrix rotation_y(double theta);
DensityMatrix rotation_z(double theta);
DensityMatrix identity(int dim = 2);
// 4x4 two-qubit gates
DensityMatrix cnot_gate();
DensityMatrix swap_gate();
DensityMatrix toffoli_gate();  // 8x8

// ---- Tensor product (Kronecker for operators) ----
DensityMatrix tensor_product(const DensityMatrix& A, const DensityMatrix& B);
Ket           tensor_product_states(const Ket& psi1, const Ket& psi2);

// ---- Commutator / anti-commutator ----
DensityMatrix commutator(const DensityMatrix& A, const DensityMatrix& B);
DensityMatrix anticommutator(const DensityMatrix& A, const DensityMatrix& B);

// ---- Quantum Fourier transform gate (2^n × 2^n) ----
DensityMatrix qft_gate(int n_qubits);

// ---- Grover's search algorithm ----
// Amplifies the amplitude of one or more "marked" basis states within an n-qubit
// search space of size N = 2^n_qubits, via repeated application of the Grover
// iterate G = D * O, where:
//   O (oracle): diagonal operator flipping the sign of amplitudes at marked
//               indices (O|x> = -|x> for x in marked_indices, O|x> = |x> otherwise).
//   D (diffusion): D = H^{\otimes n} (2|0><0| - I) H^{\otimes n}, the standard
//               Grover diffusion operator that reflects amplitudes about their mean.
// Starting from the uniform superposition H^{\otimes n}|0>, applies n_iterations
// Grover iterates and returns the resulting state.
// @param n_qubits number of qubits; N = 2^n_qubits. n_qubits <= 0 returns {}.
// @param marked_indices target basis state index/indices (each in [0, N)) to
//        amplify. Empty or out-of-range indices are ignored defensively; if ALL
//        indices are invalid/empty, the oracle becomes a no-op identity and the
//        function still returns a valid (just unamplified) uniform superposition.
// @param n_iterations number of Grover iterations to apply. n_iterations <= 0
//        returns the initial uniform superposition state.
// @note Explicit dense N x N matrices are constructed internally (same
//       explicit-matrix scalability envelope as qft_gate), so this is only
//       practical for small n_qubits.
Ket grover_search(int n_qubits, const std::vector<int>& marked_indices, int n_iterations);

// Helper: theoretically optimal number of Grover iterations to maximise the
// marked-state measurement probability for N = 2^n_qubits with n_marked marked
// states: floor(pi/4 * sqrt(N/M)). Returns 0 for n_marked <= 0 or n_marked >= N.
int grover_optimal_iterations(int n_qubits, int n_marked);

// ---- Entropy & information ----

// von Neumann entropy S(rho) = -Tr(rho log rho) = -sum_i lambda_i log lambda_i,
// in nats (natural logarithm), over the eigenvalues of rho.  Computed from the
// full Hermitian eigendecomposition at every dimension -- there is no
// dimension cap and no diagonal approximation.  S = 0 for a pure state and
// log(d) for the maximally mixed state of dimension d.
double von_neumann_entropy(const DensityMatrix& rho);

double purity(const DensityMatrix& rho);  // Tr(rho^2)

// Uhlmann fidelity in the square-root convention (Nielsen & Chuang / Uhlmann):
//     F(rho, sigma) = Tr sqrt( sqrt(rho) sigma sqrt(rho) ).
// Symmetric, F in [0, 1], with F(rho, rho) = 1 for mixed states as well as
// pure ones, and F = 0 exactly when the supports are orthogonal.  For pure
// states F = |<psi|phi>| -- the MODULUS of the overlap; the squared (Jozsa)
// convention often written F_Jozsa = |<psi|phi>|^2 is the square of the value
// returned here.  With this convention the Fuchs-van de Graaf inequalities
//     1 - F <= trace_distance(rho, sigma) <= sqrt(1 - F^2)
// hold against trace_distance() below.  Mismatched dimensions return 0.
double fidelity(const DensityMatrix& rho, const DensityMatrix& sigma);

// Trace distance T(rho, sigma) = (1/2) Tr|rho - sigma|, i.e. half the trace
// (nuclear) norm -- half the sum of the absolute eigenvalues of rho - sigma,
// not half the Frobenius norm.  T in [0, 1]; 0 for identical states, 1 for
// states with orthogonal supports, and sqrt(1 - |<psi|phi>|^2) for pure states.
// Mismatched dimensions return 0.
double trace_distance(const DensityMatrix& rho, const DensityMatrix& sigma);

// Wootters concurrence of a TWO-QUBIT state: with
// rho_tilde = (sigma_y (x) sigma_y) conj(rho) (sigma_y (x) sigma_y) and
// lambda_1 >= ... >= lambda_4 the square roots of the eigenvalues of
// rho * rho_tilde,  C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4).
// C = 0 for separable states, 1 for any maximally entangled (Bell) state, and
// 2|ad - bc| for a pure state a|00> + b|01> + c|10> + d|11>.
// Defined only for 4x4 input; any other shape returns NaN (a 0 would be
// indistinguishable from a genuine "separable" answer).
double concurrence(const DensityMatrix& rho);   // for 2-qubit states

// ---- Partial trace ----
// Trace out subsystem of dimension d2 from (d1*d2) x (d1*d2) density matrix
DensityMatrix partial_trace(const DensityMatrix& rho, int d1, int d2, int subsystem);

// ---- Entanglement entropy ----
double entanglement_entropy(const Ket& psi, int dim_a, int dim_b);

// ---- Schmidt decomposition ----
// For bipartite pure state |psi> in A (dim_a) ⊗ B (dim_b): reshape psi into a
// dim_a × dim_b coefficient matrix and take its complex SVD.  The Schmidt
// coefficients (singular values) satisfy sum_i lambda_i^2 = 1 for a normalised
// |psi>; their squares are the eigenvalues of either reduced density matrix.
// dim_a coefficients are always returned; the ones beyond min(dim_a, dim_b),
// and any that fall at the eigensolver's resolution floor, are exactly 0.
// The singular values are obtained as square roots of the eigenvalues of
// M M^dagger, so coefficients below roughly sqrt(eps) ~ 1e-8 are not resolvable
// and are reported as exactly zero rather than as sqrt(round-off).
struct SchmidtDecomposition {
    std::vector<double> coefficients;  // Schmidt coefficients (singular values), descending
    std::vector<Ket> basis_a;            // Left Schmidt vectors on subsystem A
    std::vector<Ket> basis_b;            // Right Schmidt vectors on subsystem B
};

SchmidtDecomposition schmidt_decomposition(const Ket& psi, int dim_a, int dim_b);
// Number of Schmidt coefficients above tol (Schmidt rank / Schmidt number for
// pure states).  Coefficients that are unresolvable (see above) are reported as
// exactly zero, so any tol in (0, ~1e-8) gives the same count; a tol below that
// window does NOT buy extra discrimination.
int schmidt_rank(const Ket& psi, int dim_a, int dim_b, double tol = 1e-10);
int schmidt_number(const Ket& psi, int dim_a, int dim_b, double tol = 1e-10);

// ---- Quantum states ----
std::vector<Ket> bell_states();
Ket ghz_state(int n_qubits);
Ket w_state(int n_qubits);
Ket coherent_state(C alpha, int n_max = 30);
Ket fock_state(int n, int n_max);

// ---- Hamiltonian spectrum ----
// Eigenvalues of Hermitian operator H (sorted ascending)
std::vector<double> eigenspectrum(const DensityMatrix& H);
// Ground-state eigenvector (smallest eigenvalue), normalised
Ket ground_state(const DensityMatrix& H);

// ---- Time evolution ----
// Solve Schrödinger equation: i dpsi/dt = H psi  (ℏ = 1)
// Returns psi(t) for t in linspace(t0, t1, n_steps)
std::vector<Ket> schrodinger(const DensityMatrix& H, const Ket& psi0,
                              double t0, double t1, int n_steps = 100);

// Matrix exponential exp(-i H t) for small/medium H
DensityMatrix time_evolution_operator(const DensityMatrix& H, double t);

// ---- Phase-space quasi-probability distributions ----
// Both functions operate on a density matrix expressed in a truncated Fock
// (number) basis of dimension N = rho.size() (the same convention used by
// coherent_state()/fock_state(), i.e. Fock levels 0..N-1).

// Wigner quasi-probability distribution at phase-space point (x, p), using
// the Cahill-Glauber displaced-parity trace formula
//   W(x,p) = (1/pi) * Tr[rho * D(alpha) * Parity * D(alpha)^dagger]
// with alpha = (x + i p) / sqrt(2) (so that alpha is the eigenvalue of the
// annihilation operator a = (x+ip)/sqrt(2), matching coherent_state's
// convention) and Parity|n> = (-1)^n |n>. Normalised so that
// integral over x,p of W(x,p) dx dp == 1. Unlike the Husimi Q-function,
// W can be negative for non-classical states (e.g. Fock states n >= 1).
double wigner_function(const DensityMatrix& rho, double x, double p);

// Husimi Q quasi-probability distribution: Q(alpha) = (1/pi) * <alpha|rho|alpha>,
// where |alpha> is the coherent state from coherent_state(). Always
// non-negative, and normalised so that the integral over the complex plane
// of Q(alpha) d^2(alpha) == 1.
double husimi_Q(const DensityMatrix& rho, C alpha);

} // namespace quantum
} // namespace ms
