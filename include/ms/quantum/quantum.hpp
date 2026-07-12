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
double von_neumann_entropy(const DensityMatrix& rho);
double fidelity(const DensityMatrix& rho, const DensityMatrix& sigma);
double trace_distance(const DensityMatrix& rho, const DensityMatrix& sigma);
double concurrence(const DensityMatrix& rho);   // for 2-qubit states

// ---- Partial trace ----
// Trace out subsystem of dimension d2 from (d1*d2) x (d1*d2) density matrix
DensityMatrix partial_trace(const DensityMatrix& rho, int d1, int d2, int subsystem);

// ---- Entanglement entropy ----
double entanglement_entropy(const Ket& psi, int dim_a, int dim_b);

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
