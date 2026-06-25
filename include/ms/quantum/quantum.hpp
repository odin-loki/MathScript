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

// ---- Time evolution ----
// Solve Schrödinger equation: i hbar dpsi/dt = H psi
// Returns psi(t) for t in linspace(t0, t1, n_steps)
std::vector<Ket> schrodinger(const DensityMatrix& H, const Ket& psi0,
                              double t0, double t1, int n_steps = 100);

// Matrix exponential exp(-i H t) for small/medium H
DensityMatrix time_evolution_operator(const DensityMatrix& H, double t);

} // namespace quantum
} // namespace ms
