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

// ---- Lie bracket of vector fields ----
// Vector fields X, Y: Coords → Coords
using VectorField = std::function<Coords(const Coords&)>;
Coords lie_bracket(VectorField X, VectorField Y, const Coords& x, double h = 1e-6);

} // namespace diffgeo
} // namespace ms
