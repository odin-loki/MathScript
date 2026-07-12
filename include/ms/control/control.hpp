#pragma once
#include "ms/error/error_types.hpp"
#include <complex>
#include <functional>
#include <limits>
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

// --- Step-response characteristics ---
struct StepInfo {
    double rise_time;       // 10%→90% of step size (s); NaN if undefined
    double settling_time;   // time to enter ±tol band and stay (s); +inf if never
    double overshoot_pct;   // peak above final value, percent of |final| (0 if none)
    double peak_time;       // time of peak response (s)
    double peak_value;      // peak response value
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

/// @brief Extract standard step-response metrics from a simulated trace.
///
/// Rise time is the elapsed time between the first crossings of 10% and 90%
/// of the step size (y_final − y_initial), with linear interpolation between
/// samples.  Settling time is the earliest time after which the response
/// remains within ±`settling_tol_pct`% of the final value; if the trace never
/// settles, `settling_time` is +∞.  Overshoot is
/// 100·(y_peak − y_final)/|y_final| when the peak exceeds the final value,
/// otherwise 0.
///
/// @param data              Output of step_response() (or compatible trace).
/// @param settling_tol_pct  Settling-band half-width as a percent of |y_final|
///                          (default 2%, the usual control-engineering value).
/// @return StepInfo with rise/settling time, overshoot, and peak data.
/// @note Final value is taken as the mean of the last 5% of samples (at least
///       one point) so that traces truncated before full settling still yield a
///       consistent steady-state estimate.
/// @note Degenerate input (empty trace, mismatched lengths, or |step size| <
///       1e-12): returns zeros for overshoot/peak fields and NaN for
///       rise_time; settling_time is 0 when already flat, otherwise +∞.
StepInfo step_info(const StepData& data, double settling_tol_pct = 2.0);

/// @brief Same as step_info(StepData) but with an explicit final value.
///
/// @param t                 Time samples (monotone non-decreasing).
/// @param y                 Response samples, same length as @p t.
/// @param final_value       Steady-state (final) response value.
/// @param settling_tol_pct  Settling tolerance in percent (default 2%).
/// @return StepInfo; see step_info(StepData) for metric definitions and edge
///         cases.
StepInfo step_info(const std::vector<double>& t,
                   const std::vector<double>& y,
                   double final_value,
                   double settling_tol_pct = 2.0);

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

// ---- Kalman filter ----
// Kalman filter state: mean estimate and error covariance.
struct KalmanState {
    std::vector<double> x;                    // state mean estimate
    std::vector<std::vector<double>> P;       // state error covariance
};

// Kalman filter predict step (time update): propagates the state estimate and covariance
// forward one step under the linear model x_{k+1} = A*x_k + w_k, w_k ~ N(0, Q).
//   x_pred = A * x
//   P_pred = A * P * A^T + Q
// @param state current (posterior) state estimate {x, P}
// @param A state transition matrix (n x n)
// @param Q process noise covariance (n x n)
// @return predicted (prior) state estimate for the next time step
// @note Defensive: if A/Q/P dimensions are inconsistent with state.x, returns
//       state unchanged rather than crashing (this module never throws).
KalmanState kalman_predict(const KalmanState& state,
                           const std::vector<std::vector<double>>& A,
                           const std::vector<std::vector<double>>& Q);

// Kalman filter update step (measurement update): incorporates a new observation z under the
// linear measurement model z = H*x + v, v ~ N(0, R), producing the posterior estimate.
//   y = z - H*x_pred                          (innovation / residual)
//   S = H*P_pred*H^T + R                      (innovation covariance)
//   K = P_pred*H^T*S^{-1}                     (optimal Kalman gain)
//   x_post = x_pred + K*y
//   P_post = (I - K*H) * P_pred
// @param state predicted (prior) state estimate {x_pred, P_pred}, e.g. from kalman_predict
// @param z observation vector (m-dimensional)
// @param H observation/measurement matrix (m x n)
// @param R measurement noise covariance (m x m)
// @return posterior (updated) state estimate after incorporating the observation
// @note Defensive: if H/R/z dimensions are inconsistent with state.x, or if the
//       innovation covariance S is singular, returns state unchanged rather than
//       crashing (this module never throws).
KalmanState kalman_update(const KalmanState& state,
                          const std::vector<double>& z,
                          const std::vector<std::vector<double>>& H,
                          const std::vector<std::vector<double>>& R);

} // namespace control
} // namespace ms
