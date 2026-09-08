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
/// Circle inversion (geometric reflection) in the circle |z - center| = r,
/// returned as a Möbius transformation **of the conjugate of z**.
///
/// The geometric map is anti-holomorphic:
///   sigma(z) = center + r^2 / conj(z - center)
///            = r^2 / (conj(z) - conj(center)) + center.
/// Because it conjugates, sigma reverses orientation and therefore CANNOT be
/// written as (a*z + b)/(c*z + d) for any quadruple (a,b,c,d). Substituting
/// u = conj(z) removes the conjugation, and what remains is an ordinary Möbius
/// transformation of u:
///   sigma = (center*u + (r^2 - |center|^2)) / (u - conj(center)),
/// and this function returns exactly that quadruple:
///   a = center,  b = r^2 - |center|^2,  c = 1,  d = -conj(center).
/// Its determinant is a*d - b*c = -r^2, so the quadruple is a non-degenerate
/// Möbius transformation exactly when r != 0.
///
/// CONTRACT - the result must be applied to conj(z), never to z:
/// @code
///   const Mobius m = inversion(center, r);
///   const C w = m(std::conj(z));   // == apply_inversion(z, center, r)
/// @endcode
/// Applying it to z computes a different, holomorphic map and is almost
/// certainly a caller bug. Prefer apply_inversion() / circle_reflect(), which
/// conjugate internally and additionally guard the pole.
///
/// Properties (all exercised by tests/unit/cplx/test_cplx_inversion.cpp):
///  - every point of the circle |z - center| = r is fixed;
///  - |sigma(z) - center| * |z - center| == r^2, and sigma(z) - center lies on
///    the same ray out of the centre as z - center;
///  - sigma is an involution, sigma(sigma(z)) == z; equivalently, for all w
///    conj(inversion(c,r).inverse()(w)) == inversion(c,r)(conj(w));
///  - points strictly inside the circle map strictly outside, and vice versa.
///
/// @param center centre of the circle of inversion
/// @param r radius of the circle. Only r*r enters, so inversion(c, -r) and
///   inversion(c, r) are coefficient-for-coefficient identical. r == 0 yields
///   the degenerate quadruple {c, -|c|^2, 1, -conj(c)} of determinant 0, which
///   is the constant map u -> center for every u != conj(center) (the correct
///   r -> 0 limit); at u == conj(center) it is 0/0. The direct entry point
///   apply_inversion() has no such hole - see its own contract.
/// @return the Möbius transformation of u = conj(z) described above
/// @note No exceptions are thrown. As a bare coefficient quadruple the result
///   cannot special-case its own pole u == conj(center); evaluating it there
///   divides by zero. apply_inversion() guards that point explicitly.
Mobius inversion(C center, double r);

/// Circle inversion applied directly: the complete anti-holomorphic map
///   z -> center + r^2 / conj(z - center),
/// with the conjugation performed internally. This is the entry point callers
/// should normally use; inversion() exposes the same map in Möbius-coefficient
/// form, for composition with genuine (holomorphic) Möbius transformations.
///
/// Equivalent to inversion(center, r)(std::conj(z)) for every z that is not the
/// pole, agreeing with it to within a few ulp.
///
/// Conventions on the extended complex plane (a total function - it never
/// throws and never returns an undefined value for a finite in-domain input):
///  - r == 0: the circle degenerates to a point and the whole sphere collapses
///    onto it; returns center for every z, including z == center;
///  - r < 0: |r| denotes the same circle, and only r*r enters, so
///    apply_inversion(z, c, -r) == apply_inversion(z, c, r) exactly;
///  - z == center (exact equality in both components) is the pole; the image is
///    the point at infinity, returned as C(+infinity, +infinity). The testable
///    invariant is std::abs(result) == +infinity; no directional meaning should
///    be read into the components. This mirrors the module's existing sentinel
///    convention (hyperbolic_distance returns +infinity and
///    green_function_disk returns -infinity at their singularities) rather than
///    producing a NaN;
///  - a z with an infinite component is treated as the point at infinity and
///    maps to center, which makes the involution total: applying the map twice
///    returns the original point even through the pole;
///  - a z containing NaN propagates NaN.
///
/// @param z the point to reflect
/// @param center centre of the circle of inversion
/// @param r radius of the circle of inversion (only r*r is used)
/// @return the reflected point, or the documented sentinel in the degenerate
///   cases above
/// @note No exceptions are thrown.
C apply_inversion(C z, C center, double r);

/// Reflection of z in the circle |z - center| = r. An exact alias for
/// apply_inversion(), provided because "reflection in a circle" is the more
/// common name for the map in geometry: identical contract, identical
/// degenerate conventions, bit-identical results.
C circle_reflect(C z, C center, double r);

/// Real part of the complex cross ratio of four points:
///   (z1, z2; z3, z4) = ((z1-z3)*(z2-z4)) / ((z1-z4)*(z2-z3)).
///
/// The cross ratio is a genuinely COMPLEX quantity. It is real - and therefore
/// equal to the value this function returns - exactly when the four points lie
/// on one common circle or one common line; for four points in general
/// position the imaginary part is non-zero and this function discards it.
/// Use cross_ratio_c() whenever the full value is wanted, and read this
/// function as "the concyclicity coordinate" rather than "the cross ratio".
/// Example of the loss: (0, 1, i, -1) has cross ratio 1 - i, and this function
/// returns 1.0.
///
/// The value is invariant under every Möbius transformation of the four
/// arguments, and is 0 when z1 == z3 or z2 == z4.
///
/// @return cross_ratio_c(z1, z2, z3, z4).real(); +infinity when the cross ratio
///   is the point at infinity and NaN when it is undefined - see cross_ratio_c
/// @note The signature and the returned value for non-degenerate inputs are
///   unchanged from previous releases; only the degenerate cases, which used to
///   inherit whatever std::complex division produced, became deterministic.
double cross_ratio(C z1, C z2, C z3, C z4);

/// Complex cross ratio of four points:
///   (z1, z2; z3, z4) = ((z1-z3)*(z2-z4)) / ((z1-z4)*(z2-z3)).
///
/// This is the full-information companion of cross_ratio(). It is the unique
/// Möbius invariant of four ordered points: for any Möbius transformation m,
/// cross_ratio_c(m(z1), m(z2), m(z3), m(z4)) == cross_ratio_c(z1, z2, z3, z4).
/// Its imaginary part vanishes exactly when the four points are concyclic or
/// collinear, which is the standard test for cocircularity.
///
/// Degenerate inputs (the denominator (z1-z4)*(z2-z3) is exactly zero):
///  - if the numerator is non-zero the cross ratio is the point at infinity;
///    returns C(+infinity, +infinity), matching apply_inversion()'s sentinel,
///    so that std::abs(result) == +infinity;
///  - if the numerator is also exactly zero the value is a true 0/0 and is
///    genuinely undefined; returns C(NaN, NaN).
/// Both branches are deterministic and identical on every platform, unlike the
/// raw std::complex division they replace.
///
/// @return the complex cross ratio, or the documented sentinel above
/// @note No exceptions are thrown.
C cross_ratio_c(C z1, C z2, C z3, C z4);

// --- Joukowski transform: z → z + c^2/z ---
C joukowski(C z, double c = 1.0);
// Inverse Joukowski
std::vector<C> joukowski_inv(C w, double c = 1.0);

// --- Poisson kernel: P_r(theta-phi) ---
double poisson_kernel(double theta, double phi, double r);

/// Green's function for the Dirichlet Laplacian on the disk of radius `radius`
/// centred at the origin:
///   G(z, z0) = (1/(2*pi)) * ln | (z - z0) / (radius - conj(z0)*z/radius) |
/// (equivalently, on the unit disk, G(z,z0) = (1/(2*pi)) * ln |(z-z0)/(1-conj(z0)*z)|,
/// obtained via the rescaling z -> z/radius, z0 -> z0/radius).
/// Symmetric: green_function_disk(z, z0, radius) == green_function_disk(z0, z, radius).
/// Vanishes as |z| -> radius (Dirichlet boundary condition) and has a logarithmic
/// singularity as z -> z0. With this sign convention (G <= 0 inside the disk,
/// G -> -infinity at the source), the outward normal derivative of G in its
/// first argument, taken at the boundary, reproduces the Poisson kernel:
/// dG/dn(e^{i theta}, r*e^{i phi}) = (1/(2*pi)) * poisson_kernel(theta, phi, r),
/// where n is the outward radial direction at the boundary point e^{i theta}.
/// @param z point inside the disk at which G is evaluated
/// @param z0 source point inside the disk
/// @param radius disk radius (default 1.0, the unit disk)
/// @return the value of G(z, z0); 0.0 if z or z0 lies outside/on the boundary of
///   the disk (defensive out-of-domain convention, matching poisson_kernel's
///   behaviour of returning a well-defined finite value for all inputs rather
///   than throwing); -infinity if z == z0 (true logarithmic singularity).
/// @note No exceptions are thrown; degenerate/out-of-domain inputs return a
///   documented sentinel instead.
double green_function_disk(C z, C z0, double radius = 1.0);

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

/// Cauchy principal value of a real definite integral with a simple pole:
///   PV ∫[a,b] f(x)/(x-c) dx
/// where the pole at x=c lies strictly inside (a,b). The caller supplies only
/// the numerator f; the singular factor 1/(x-c) is applied internally.
///
/// Algorithm: decompose f(x)/(x-c) = f(c)/(x-c) + (f(x)-f(c))/(x-c). The
/// principal value of the singular term is evaluated analytically in the
/// symmetric-exclusion limit as f(c)*ln((b-c)/(c-a)). The regular remainder
/// (f(x)-f(c))/(x-c) has a removable singularity at x=c and is integrated
/// numerically over [a,b] with composite Simpson's rule (n_pts subdivisions).
/// Two resolutions (n_pts and 2*n_pts) are compared for consistency.
///
/// @param f numerator function (RealFunc = std::function<double(double)>)
/// @param a lower limit of integration
/// @param c location of the simple pole (must be in (a,b) for a true PV)
/// @param b upper limit of integration
/// @param n_pts Simpson subdivisions on each side of the pole (forced even, >= 2)
/// @return the principal-value estimate; if c is not strictly inside (a,b) the
///   integrand has no singularity in [a,b] and an ordinary Simpson integral of
///   f(x)/(x-c) over [a,b] is returned instead (well-defined finite answer).
/// @note No exceptions are thrown; if a >= b, returns 0.0.
double cauchy_principal_value(RealFunc f, double a, double c, double b, int n_pts = 200);

// --- Power series: sum coefficients[k] * (z-z0)^k ---
C power_series_eval(const std::vector<C>& coeffs, C z0, C z);

// --- Blashcke product ---
C blaschke_product(C z, const std::vector<C>& zeros);

} // namespace cplx
} // namespace ms
