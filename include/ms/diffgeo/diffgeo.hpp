#pragma once
#include <array>
#include <functional>
#include <vector>

namespace ms {
namespace diffgeo {

// Coordinate system: dimension n, coordinates as vector<double>
using Coords   = std::vector<double>;
using MetricFn = std::function<std::vector<std::vector<double>>(const Coords&)>;

// ---- Metric tensor ----
// g: coords → n×n symmetric positive-definite matrix
std::vector<std::vector<double>>
    metric_tensor(MetricFn g, const Coords& x);

// Inverse of metric tensor
std::vector<std::vector<double>>
    metric_inv(const std::vector<std::vector<double>>& g);

// ---- Christoffel symbols Γ^k_{ij} ----
// Returns 3D array [k][i][j] of size n×n×n
std::vector<std::vector<std::vector<double>>>
    christoffel(MetricFn g, const Coords& x, double h = 1e-5);

// ---- Riemann curvature tensor R^l_{ijk} ----
// Returns 4D array [l][i][j][k]
std::vector<std::vector<std::vector<std::vector<double>>>>
    riemann_tensor(MetricFn g, const Coords& x, double h = 1e-5);

// ---- Ricci tensor R_{ij} = R^k_{ikj} ----
std::vector<std::vector<double>>
    ricci_tensor(MetricFn g, const Coords& x, double h = 1e-5);

// ---- Ricci scalar R = g^{ij} R_{ij} ----
double ricci_scalar(MetricFn g, const Coords& x, double h = 1e-5);

// ---- Einstein tensor G_{ij} = R_{ij} - (R/2) g_{ij} ----
std::vector<std::vector<double>>
    einstein_tensor(MetricFn g, const Coords& x, double h = 1e-5);

// ---- Geodesic equation: d²x^k/ds² + Γ^k_{ij} dx^i/ds dx^j/ds = 0 ----
// Returns trajectory as list of (x, dx/ds) states
struct GeodesicState { Coords x; Coords v; };
std::vector<GeodesicState>
    geodesic(MetricFn g, const Coords& x0, const Coords& v0,
             double s_end = 1.0, int n_steps = 100, double h = 1e-5);

// ---- Parallel transport of a vector along a curve ----
// Integrates the parallel transport equation
//   dV^k/ds + Γ^k_{ij} (dx^i/ds) V^j = 0     (sum over i,j)
// which keeps V "as constant as possible" (zero covariant derivative) along
// the curve x(s), given the manifold's intrinsic curvature. This is the
// direct generalization of "keeping a vector pointing the same direction
// while walking along a path" to curved manifolds/coordinates, and is the
// same underlying equation as the geodesic equation (which parallel-
// transports the curve's OWN tangent vector dx/ds) but for an arbitrary
// initial vector V0 that need not be tangent to the curve.
//
// The curve is supplied explicitly as x(s) together with its velocity
// dx/ds, rather than reconstructing the velocity via finite differences of
// `curve` internally: this is more numerically accurate/robust (it avoids
// stacking an extra finite-difference layer on top of christoffel()'s own
// finite-difference derivatives of the metric) and lets callers supply an
// exact closed-form velocity when they have one (e.g. a great-circle or a
// coordinate line on a torus).
//
// Uses the same explicit-Euler integration scheme and step-size convention
// (h_step = s_end / n_steps) as geodesic(), evaluating christoffel() at the
// current point of each step, for consistency with geodesic()'s numerical
// behaviour/accuracy characteristics.
//
// @param g the metric function (same convention as christoffel/geodesic).
// @param curve the curve x(s), s in [0, s_end], with curve(0) the transport's
//        starting point (must match where V0 lives).
// @param curve_velocity the curve's velocity dx/ds as a function of s.
// @param V0 initial vector to transport, in the coordinate basis at curve(0).
// @param s_end final curve parameter value.
// @param n_steps number of integration steps (matching geodesic's n_steps
//        convention); step size is s_end/n_steps.
// @param h finite-difference step passed through to the internal
//        christoffel() calls (same convention/default as christoffel's h).
// @return the transported vector's trajectory, one Coords per step
//         (n_steps+1 points total, including V0 at s=0), analogous to
//         geodesic()'s trajectory convention but for V alone -- the position
//         x(s) is already fully determined by the given `curve` and is not
//         duplicated in the return value.
// @note Degenerate inputs (n_steps <= 0 or s_end <= 0) return the trivial
//       single-element trajectory {V0} rather than throwing, consistent with
//       this module's defensive error handling elsewhere.
std::vector<Coords>
    parallel_transport(MetricFn g, std::function<Coords(double)> curve,
                        std::function<Coords(double)> curve_velocity,
                        const Coords& V0, double s_end, int n_steps, double h = 1e-5);

// ---- Surface differential geometry ----
// Surface parameterised as r(u,v) via a function
using SurfaceFn = std::function<std::array<double,3>(double, double)>;

// First fundamental form coefficients E, F, G at (u,v)
struct FFF { double E, F, G; };
FFF first_fundamental_form(SurfaceFn r, double u, double v, double h = 1e-5);

// Second fundamental form coefficients e, f, g at (u,v)
struct SFF { double e, f, g; };
SFF second_fundamental_form(SurfaceFn r, double u, double v, double h = 1e-5);

// Gaussian curvature K = (eg - f²) / (EG - F²)
double gaussian_curvature(SurfaceFn r, double u, double v, double h = 1e-5);

// Mean curvature H = (eG - 2fF + gE) / (2(EG - F²))
double mean_curvature(SurfaceFn r, double u, double v, double h = 1e-5);

// Principal curvatures k1, k2 (eigenvalues of shape operator)
std::pair<double,double> principal_curvatures(SurfaceFn r, double u, double v, double h = 1e-5);

// Unit normal vector at (u,v)
std::array<double,3> surface_normal(SurfaceFn r, double u, double v, double h = 1e-5);

// ---- Gauss-Bonnet theorem (closed-surface / no-boundary case) ----
// Theorem: ∬_S K dA = 2π·χ(S), where χ is the Euler characteristic of the
// closed surface S, K is the Gaussian curvature, and dA = sqrt(EG - F²) du dv
// is the area element from the first fundamental form.
//
// Computes the left-hand side ∬ K dA numerically via the composite trapezoid
// rule over the (u,v) parameter grid, reusing first_fundamental_form (for the
// area Jacobian) and gaussian_curvature (for K) at each grid node.
//
// @param r    surface parameterisation r(u,v) -> (x,y,z)
// @param u0   lower bound of the u parameter domain
// @param u1   upper bound of the u parameter domain (must be > u0)
// @param v0   lower bound of the v parameter domain
// @param v1   upper bound of the v parameter domain (must be > v0)
// @param n_u  number of trapezoid subintervals in u (>= 1)
// @param n_v  number of trapezoid subintervals in v (>= 1)
// @return numerical estimate of ∬_S K dA over the given (u,v) patch
// @note Trapezoid-rule error is O(1/n_u² + 1/n_v²); doubling the grid
//       resolution in each direction roughly quarters the error. For a full
//       closed surface (e.g. a sphere swept over its whole parameter domain)
//       the result should approach 2π·χ(S) as n_u, n_v grow.
// @note Assumes S is closed (or that the patch boundary contribution is zero,
//       e.g. by periodicity of r); no geodesic-curvature boundary term is
//       added, so applying this to an open patch with a non-geodesic boundary
//       will not satisfy the theorem exactly.
// @note Degenerate/invalid inputs (n_u < 1, n_v < 1, u1 <= u0, or v1 <= v0)
//       return 0.0 rather than throwing, consistent with this module's
//       defensive error handling elsewhere.
double gauss_bonnet_integral(SurfaceFn r, double u0, double u1, double v0, double v1,
                              int n_u, int n_v, double h = 1e-5);

// Convenience wrapper: residual of the Gauss-Bonnet theorem against an
// expected Euler characteristic, i.e. gauss_bonnet_integral(...) - 2π·χ.
// A residual near zero (relative to 4π) confirms the theorem numerically.
//
// @param euler_characteristic expected χ(S) of the closed surface (e.g. 2 for
//        a topological sphere, 0 for a torus)
// @return gauss_bonnet_integral(r, u0, u1, v0, v1, n_u, n_v, h) - 2π·euler_characteristic
double gauss_bonnet_residual(SurfaceFn r, double u0, double u1, double v0, double v1,
                              int n_u, int n_v, int euler_characteristic, double h = 1e-5);

// ---- Lie bracket of vector fields ----
// Vector fields X, Y: Coords → Coords
using VectorField = std::function<Coords(const Coords&)>;
Coords lie_bracket(VectorField X, VectorField Y, const Coords& x, double h = 1e-6);

} // namespace diffgeo
} // namespace ms
