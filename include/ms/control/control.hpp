#pragma once
#include "ms/error/error_types.hpp"
#include <complex>
#include <functional>
#include <vector>

namespace ms {
namespace control {

// --- Transfer function: H(s) = num(s) / den(s) ---
// Coefficients in descending power order: [a_n, a_{n-1}, ..., a_0]
struct TransferFunction {
    std::vector<double> num;  // numerator coefficients
    std::vector<double> den;  // denominator coefficients

    TransferFunction(std::vector<double> n, std::vector<double> d)
        : num(std::move(n)), den(std::move(d)) {}
};

// --- State space: dx/dt = A*x + B*u, y = C*x + D*u ---
struct StateSpace {
    std::vector<std::vector<double>> A, B, C, D;
    int n;  // state dimension
    int m;  // input dimension
    int p;  // output dimension

    StateSpace(std::vector<std::vector<double>> a,
               std::vector<std::vector<double>> b,
               std::vector<std::vector<double>> c,
               std::vector<std::vector<double>> d);
};

// --- Bode data ---
struct BodeData {
    std::vector<double> w;          // frequencies (rad/s)
    std::vector<double> magnitude;  // magnitude (dB)
    std::vector<double> phase;      // phase (degrees)
};

// --- Step/impulse response data ---
struct StepData {
    std::vector<double> t;
    std::vector<double> y;
};

// --- Stability margins ---
struct Margins {
    double gain_margin_db;
    double phase_margin_deg;
    double gain_crossover_freq;
    double phase_crossover_freq;
};

// ---- Construction ----
TransferFunction tf(std::vector<double> num, std::vector<double> den);
StateSpace       ss(std::vector<std::vector<double>> A,
                    std::vector<std::vector<double>> B,
                    std::vector<std::vector<double>> C,
                    std::vector<std::vector<double>> D);
// Convert TF to SS (canonical controllable form)
StateSpace tf2ss(const TransferFunction& sys);
// Convert SS to TF
TransferFunction ss2tf(const StateSpace& sys);

// ---- Discretization ----
enum class DiscretizationMethod { ZOH, Tustin, Euler };

/// @note ZOH is exact for piecewise-constant inputs; uses matrix exponential on an
///       augmented state matrix (handles singular A, e.g. integrators).
/// @note Tustin (bilinear) preserves stability via the s↔z map; good frequency response.
/// @note Euler is a first-order approximation — not recommended for stiff systems.
StateSpace c2d(const StateSpace& sys, double Ts,
               DiscretizationMethod method = DiscretizationMethod::ZOH);
StateSpace d2c(const StateSpace& sys, double Ts,
               DiscretizationMethod method = DiscretizationMethod::ZOH);

TransferFunction c2d(const TransferFunction& sys, double Ts,
                     DiscretizationMethod method = DiscretizationMethod::ZOH);
TransferFunction d2c(const TransferFunction& sys, double Ts,
                     DiscretizationMethod method = DiscretizationMethod::ZOH);

// ---- Interconnections ----
TransferFunction series  (const TransferFunction& g, const TransferFunction& h);
TransferFunction parallel(const TransferFunction& g, const TransferFunction& h);
TransferFunction feedback(const TransferFunction& g, const TransferFunction& h,
                          int sign = -1);

// ---- Analysis ----
std::vector<std::complex<double>> poles(const TransferFunction& sys);
std::vector<std::complex<double>> zeros(const TransferFunction& sys);
double dcgain(const TransferFunction& sys);
bool   is_stable(const TransferFunction& sys);  // all poles in LHP

// ---- Frequency response ----
BodeData bode(const TransferFunction& sys,
              double w_lo = 0.01, double w_hi = 1000.0, int n_pts = 200);
std::vector<std::pair<double,double>> nyquist(const TransferFunction& sys,
              double w_lo = 0.01, double w_hi = 1000.0, int n_pts = 300);
Margins  margin(const TransferFunction& sys);

// ---- Time response ----
StepData step_response   (const TransferFunction& sys, double t_end = 10.0, int n_pts = 500);
StepData impulse_response(const TransferFunction& sys, double t_end = 10.0, int n_pts = 500);

// ---- Lyapunov equations ----
// Solve A*X + X*A^T + Q = 0 (continuous Lyapunov)
Result<std::vector<std::vector<double>>>
    lyap(const std::vector<std::vector<double>>& A,
         const std::vector<std::vector<double>>& Q);

// Solve A*X*A^T - X + Q = 0 (discrete Lyapunov)
Result<std::vector<std::vector<double>>>
    dlyap(const std::vector<std::vector<double>>& A,
          const std::vector<std::vector<double>>& Q);

// ---- Riccati equations ----
// Solve A^T*X + X*A - X*B*R^{-1}*B^T*X + Q = 0 (continuous ARE)
Result<std::vector<std::vector<double>>>
    riccati(const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            const std::vector<std::vector<double>>& Q,
            const std::vector<std::vector<double>>& R);

// Discrete ARE
Result<std::vector<std::vector<double>>>
    dare(const std::vector<std::vector<double>>& A,
         const std::vector<std::vector<double>>& B,
         const std::vector<std::vector<double>>& Q,
         const std::vector<std::vector<double>>& R);

// ---- LQR ----
// Returns K s.t. u = -K*x minimises integral(x^T Q x + u^T R u)dt
Result<std::vector<std::vector<double>>>
    lqr(const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B,
        const std::vector<std::vector<double>>& Q,
        const std::vector<std::vector<double>>& R);

// ---- Observability / Controllability ----
// Returns controllability matrix [B, AB, A^2 B, ...]
std::vector<std::vector<double>>
    ctrb(const std::vector<std::vector<double>>& A,
         const std::vector<std::vector<double>>& B);

std::vector<std::vector<double>>
    obsv(const std::vector<std::vector<double>>& A,
         const std::vector<std::vector<double>>& C);

bool is_controllable(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B);
bool is_observable(const std::vector<std::vector<double>>& A,
                   const std::vector<std::vector<double>>& C);

// ---- Gramians ----
enum class GramianType { Controllability, Observability };

/// @brief Controllability/observability Gramian of a continuous-time LTI system.
///
/// The controllability Gramian solves A*Wc + Wc*A^T + B*B^T = 0.
/// The observability  Gramian solves A^T*Wo + Wo*A + C^T*C = 0.
/// Both equations are solved by delegating to lyap(), which solves
/// A*X + X*A^T + Q = 0 (Wo is obtained via lyap(A^T, C^T*C), which — since
/// lyap treats its first argument as "A" — solves A^T*X + X*A + Q = 0, exactly
/// the observability equation).
///
/// @param sys  State-space system. Only (A, B) are used for the controllability
///             Gramian; only (A, C) are used for the observability Gramian.
/// @param type GramianType::Controllability or GramianType::Observability.
/// @return The n×n Gramian matrix, or an error if the underlying Lyapunov solve
///         fails (see @note below).
/// @note The Gramian is only mathematically well-defined (finite) when A is
///       asymptotically stable (all eigenvalues strictly in the open left
///       half-plane). For an unstable or marginally stable A (e.g. eigenvalues
///       on or to the right of the imaginary axis), the Lyapunov equation may
///       have no solution, an unbounded solution, or a non-unique solution;
///       lyap()'s Kronecker-product linear system becomes singular or
///       ill-conditioned in that case and this function propagates that
///       failure through the returned Result rather than throwing or
///       returning a nonsensical matrix.
/// @note For a stable system, the returned Gramian is symmetric by
///       construction and positive semi-definite; it is positive *definite*
///       iff the system is fully controllable (resp. observable).
Result<std::vector<std::vector<double>>>
    gram(const StateSpace& sys, GramianType type);

/// @brief Convenience wrapper for gram(sys, GramianType::Controllability).
Result<std::vector<std::vector<double>>> ctrb_gram(const StateSpace& sys);

/// @brief Convenience wrapper for gram(sys, GramianType::Observability).
Result<std::vector<std::vector<double>>> obsv_gram(const StateSpace& sys);

// ---- Pole placement ----
// Returns state feedback gain K s.t. eigenvalues of (A - B*K) = desired_poles
Result<std::vector<double>>
    place(const std::vector<std::vector<double>>& A,
          const std::vector<std::vector<double>>& B,
          const std::vector<double>& desired_poles);

// ---- PID ----
struct PIDGains { double Kp, Ki, Kd; };
PIDGains pidtune(const TransferFunction& plant, double bandwidth_rad_s = 1.0);

} // namespace control
} // namespace ms
