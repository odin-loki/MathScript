#pragma once
#include <complex>
#include <functional>
#include <vector>

namespace ms {
namespace cplx {

using C = std::complex<double>;
using CFunc = std::function<C(C)>;
using RealFunc = std::function<double(double)>;

// --- Residue at a simple pole z0 ---
C residue(CFunc f, C z0, double eps = 1e-6);

// --- Winding number of closed curve gamma around z0 ---
// gamma: list of points on the curve (closed, so gamma[0] == gamma[-1])
int winding_number(const std::vector<C>& gamma, C z0);

// --- Contour integral: integral of f along piecewise-linear path ---
C contour_integral(CFunc f, const std::vector<C>& path, int n_pts = 100);

// --- Cauchy integral formula: f(z0) = (1/2πi) ∮ f(z)/(z-z0) dz ---
C cauchy_integral(CFunc f, C z0, const std::vector<C>& contour, int n_pts = 100);

// --- Argument principle: returns (zeros - poles) inside contour ---
int argument_principle(CFunc f, const std::vector<C>& contour, int n_pts = 300);

// --- Möbius (linear fractional) transformation: w = (az+b)/(cz+d) ---
struct Mobius {
    C a, b, c, d;
    Mobius(C a, C b, C c, C d) : a(a), b(b), c(c), d(d) {}
    C operator()(C z) const { return (a * z + b) / (c * z + d); }
    Mobius compose(const Mobius& other) const;
    Mobius inverse() const;
    // Fixed points
    std::vector<C> fixed_points() const;
};

// Standard Möbius transforms
Mobius mobius(C a, C b, C c, C d);
Mobius inversion(C center, double r);           // z → r^2/(conj(z)-conj(c)) + c
double cross_ratio(C z1, C z2, C z3, C z4);     // real when 4 points cocircular

// --- Joukowski transform: z → z + c^2/z ---
C joukowski(C z, double c = 1.0);
// Inverse Joukowski
std::vector<C> joukowski_inv(C w, double c = 1.0);

// --- Poisson kernel: P_r(theta-phi) ---
double poisson_kernel(double theta, double phi, double r);

// --- Harmonic conjugate (numerical, Hilbert transform approach) ---
// Given real part u on circle of radius r (n equally spaced points),
// returns harmonic conjugate v
std::vector<double> harmonic_conjugate(const std::vector<double>& u);

// --- Hyperbolic distance (Poincaré disk model) ---
double hyperbolic_distance(C z1, C z2);

// --- Laurent series coefficients (numerical) ---
// f on circle |z-z0|=r, returns {a_{-n}, ..., a_{-1}, a_0, a_1, ..., a_m}
std::vector<C> laurent_coeffs(CFunc f, C z0, double r,
                                int neg_terms = 5, int pos_terms = 5);

// --- Complex integration (Gaussian quadrature on [a,b] parameterised path) ---
C line_integral(CFunc f, C a, C b, int n_pts = 50);

// --- Power series: sum coefficients[k] * (z-z0)^k ---
C power_series_eval(const std::vector<C>& coeffs, C z0, C z);

// --- Blashcke product ---
C blaschke_product(C z, const std::vector<C>& zeros);

} // namespace cplx
} // namespace ms
