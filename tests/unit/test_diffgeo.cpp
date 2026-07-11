#define _USE_MATH_DEFINES
#include "ms/diffgeo/diffgeo.hpp"
#include <cmath>
#include <gtest/gtest.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms::diffgeo;

// ---- Flat Euclidean metric ----

static MetricFn euclidean_2d() {
    return [](const Coords& x) -> std::vector<std::vector<double>> {
        (void)x;
        return {{1,0},{0,1}};
    };
}

// 2-sphere metric (θ,φ): g = diag(1, sin²θ)
static MetricFn sphere_metric() {
    return [](const Coords& x) -> std::vector<std::vector<double>> {
        double theta = x[0];
        double s2 = std::sin(theta) * std::sin(theta);
        return {{1,0},{0,s2}};
    };
}

TEST(DiffGeoMetric, EuclideanInverse) {
    auto g = euclidean_2d();
    auto gv = g({0.0, 0.0});
    auto gi = metric_inv(gv);
    EXPECT_NEAR(gi[0][0], 1.0, 1e-10);
    EXPECT_NEAR(gi[1][1], 1.0, 1e-10);
    EXPECT_NEAR(gi[0][1], 0.0, 1e-10);
}

TEST(DiffGeoChristoffel, FlatSpace) {
    // All Christoffel symbols vanish in flat Euclidean space
    auto g = euclidean_2d();
    auto Chr = christoffel(g, {1.0, 1.0});
    for (auto& k : Chr) for (auto& row : k) for (double v : row)
        EXPECT_NEAR(v, 0.0, 1e-6);
}

TEST(DiffGeoChristoffel, SphereNonZero) {
    // Sphere has non-zero Christoffel symbols
    auto g = sphere_metric();
    Coords x = {M_PI/4, 0.5};  // θ = π/4
    auto Chr = christoffel(g, x);
    // Γ^θ_{φφ} = -sinθcosθ, Γ^φ_{θφ} = cosθ/sinθ
    double theta = x[0];
    double expected_0_11 = -std::sin(theta)*std::cos(theta);
    double expected_1_01 = std::cos(theta)/std::sin(theta);
    EXPECT_NEAR(Chr[0][1][1], expected_0_11, 1e-4);
    EXPECT_NEAR(Chr[1][0][1], expected_1_01, 1e-4);
}

TEST(DiffGeoRiemann, FlatZero) {
    auto g = euclidean_2d();
    auto R = riemann_tensor(g, {0.5, 0.5});
    for (auto& a : R) for (auto& b : a) for (auto& c : b) for (double v : c)
        EXPECT_NEAR(v, 0.0, 1e-4);
}

TEST(DiffGeoRicciScalar, Sphere) {
    // Ricci scalar of unit 2-sphere = 2
    auto g = sphere_metric();
    Coords x = {M_PI/3, 0.0};
    double R = ricci_scalar(g, x);
    EXPECT_NEAR(R, 2.0, 0.02);  // numerical, so relaxed tolerance
}

TEST(DiffGeoGeodesic, FlatLine) {
    // In flat space, geodesic is a straight line
    auto g = euclidean_2d();
    Coords x0 = {0.0, 0.0};
    Coords v0 = {1.0, 0.5};
    auto traj = geodesic(g, x0, v0, 1.0, 50);
    EXPECT_EQ(traj.size(), 51u);
    // At s=1: x should be ~ (1, 0.5)
    EXPECT_NEAR(traj.back().x[0], 1.0, 1e-6);
    EXPECT_NEAR(traj.back().x[1], 0.5, 1e-6);
}

// ---- Surface geometry (sphere) ----

static SurfaceFn unit_sphere() {
    return [](double u, double v) -> std::array<double,3> {
        return {std::cos(u)*std::cos(v), std::cos(u)*std::sin(v), std::sin(u)};
    };
}

TEST(DiffGeoSurface, GaussianCurvatureSphere) {
    auto sphere = unit_sphere();
    double K = gaussian_curvature(sphere, 0.3, 0.7);
    EXPECT_NEAR(K, 1.0, 0.02);  // unit sphere K=1
}

TEST(DiffGeoSurface, MeanCurvatureSphere) {
    auto sphere = unit_sphere();
    double H = mean_curvature(sphere, 0.3, 0.7);
    // Unit sphere mean curvature = 1 (both principal curvatures = 1)
    EXPECT_NEAR(std::abs(H), 1.0, 0.05);
}

TEST(DiffGeoSurface, PrincipalCurvatures) {
    auto sphere = unit_sphere();
    auto [k1, k2] = principal_curvatures(sphere, 0.3, 0.7);
    EXPECT_NEAR(k1, k2, 0.05);  // sphere is umbilic: k1 = k2
}

TEST(DiffGeoSurface, NormalSphere) {
    auto sphere = unit_sphere();
    double u=0.5, v=0.3;
    auto N = surface_normal(sphere, u, v);
    // Normal should have unit length
    double len = std::sqrt(N[0]*N[0]+N[1]*N[1]+N[2]*N[2]);
    EXPECT_NEAR(len, 1.0, 1e-6);
}

// ---- Gauss-Bonnet theorem ----

// Full closed unit sphere: r(u,v) = (sin u cos v, sin u sin v, cos u),
// u in [0, pi] (colatitude), v in [0, 2pi] (longitude). chi = 2.
static SurfaceFn sphere_colatitude(double R = 1.0) {
    return [R](double u, double v) -> std::array<double,3> {
        return {R*std::sin(u)*std::cos(v), R*std::sin(u)*std::sin(v), R*std::cos(u)};
    };
}

TEST(DiffGeoGaussBonnet, UnitSphereApproachesFourPi) {
    auto sphere = sphere_colatitude(1.0);
    double integral = gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 200, 200);
    // chi(sphere) = 2, so ∬K dA should approach 2*pi*2 = 4*pi
    EXPECT_NEAR(integral, 4*M_PI, 0.01 * 4*M_PI);  // within 1%
}

TEST(DiffGeoGaussBonnet, ScaleInvarianceNonUnitRadius) {
    // K = 1/R^2, area = 4*pi*R^2, so the product is 4*pi regardless of R.
    auto sphere = sphere_colatitude(3.5);
    double integral = gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 200, 200);
    EXPECT_NEAR(integral, 4*M_PI, 0.01 * 4*M_PI);
}

TEST(DiffGeoGaussBonnet, FlatPatchVanishes) {
    // Trivial planar parameterisation: K = 0 everywhere, regardless of bounds.
    SurfaceFn plane = [](double u, double v) -> std::array<double,3> { return {u, v, 0.0}; };
    double integral = gauss_bonnet_integral(plane, -2.0, 3.0, -1.0, 5.0, 40, 40);
    EXPECT_NEAR(integral, 0.0, 1e-6);
}

TEST(DiffGeoGaussBonnet, ConvergenceWithGridResolution) {
    auto sphere = sphere_colatitude(1.0);
    double coarse = gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 20, 20);
    double fine   = gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 200, 200);
    double err_coarse = std::abs(coarse - 4*M_PI);
    double err_fine   = std::abs(fine - 4*M_PI);
    EXPECT_LT(err_fine, err_coarse);
}

TEST(DiffGeoGaussBonnet, ResidualSphereNearZero) {
    auto sphere = sphere_colatitude(1.0);
    double residual = gauss_bonnet_residual(sphere, 0.0, M_PI, 0.0, 2*M_PI, 200, 200, 2);
    EXPECT_NEAR(residual, 0.0, 0.01 * 4*M_PI);
}

TEST(DiffGeoGaussBonnet, DegenerateGridResolutionNoCrash) {
    auto sphere = sphere_colatitude(1.0);
    // n_u/n_v of 1 or 2 are valid-but-coarse grids: should not crash and
    // should return a finite (if inaccurate) number.
    double r1 = gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 1, 1);
    double r2 = gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 2, 2);
    EXPECT_TRUE(std::isfinite(r1));
    EXPECT_TRUE(std::isfinite(r2));
}

TEST(DiffGeoGaussBonnet, InvalidGridResolutionReturnsZero) {
    auto sphere = sphere_colatitude(1.0);
    EXPECT_NEAR(gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 0, 10), 0.0, 1e-12);
    EXPECT_NEAR(gauss_bonnet_integral(sphere, 0.0, M_PI, 0.0, 2*M_PI, 10, -1), 0.0, 1e-12);
}

TEST(DiffGeoGaussBonnet, InvalidBoundsReturnsZero) {
    auto sphere = sphere_colatitude(1.0);
    // u1 < u0 and v1 < v0 are invalid domains; should not crash.
    EXPECT_NEAR(gauss_bonnet_integral(sphere, M_PI, 0.0, 0.0, 2*M_PI, 50, 50), 0.0, 1e-12);
    EXPECT_NEAR(gauss_bonnet_integral(sphere, 0.0, M_PI, 2*M_PI, 0.0, 50, 50), 0.0, 1e-12);
    EXPECT_NEAR(gauss_bonnet_integral(sphere, 1.0, 1.0, 0.0, 2*M_PI, 50, 50), 0.0, 1e-12);
}

// ---- Lie bracket ----
TEST(DiffGeoLieBracket, XYcommute) {
    // [∂/∂x, ∂/∂y] = 0 in Euclidean space
    VectorField X = [](const Coords& p) -> Coords { (void)p; return {1, 0}; };
    VectorField Y = [](const Coords& p) -> Coords { (void)p; return {0, 1}; };
    Coords x = {1.0, 1.0};
    auto lb = lie_bracket(X, Y, x);
    EXPECT_NEAR(lb[0], 0.0, 1e-6);
    EXPECT_NEAR(lb[1], 0.0, 1e-6);
}

TEST(DiffGeoLieBracket, NonZero) {
    // [x ∂/∂y, y ∂/∂x] should be non-zero at non-origin
    VectorField X = [](const Coords& p) -> Coords { return {0, p[0]}; };
    VectorField Y = [](const Coords& p) -> Coords { return {p[1], 0}; };
    Coords x = {1.0, 1.0};
    auto lb = lie_bracket(X, Y, x);
    // [X,Y] = X(Y) - Y(X) != 0 generally
    double mag = std::abs(lb[0]) + std::abs(lb[1]);
    EXPECT_GT(mag, 1e-6);
}

// ---- Ricci scalar (additional) ----

TEST(DiffGeoRicciScalar, FlatZero) {
    auto g = euclidean_2d();
    double R = ricci_scalar(g, {1.0, 2.0});
    EXPECT_NEAR(R, 0.0, 1e-3);
}

TEST(DiffGeoRicciScalar, SphereEquator) {
    auto g = sphere_metric();
    double R = ricci_scalar(g, {M_PI / 2, 0.0});
    EXPECT_NEAR(R, 2.0, 0.03);
}

TEST(DiffGeoRicciScalar, SpherePositive) {
    auto g = sphere_metric();
    double R = ricci_scalar(g, {M_PI / 4, M_PI / 3});
    EXPECT_GT(R, 0.5);
}

TEST(DiffGeoRicciScalar, SpherePoleRegion) {
    auto g = sphere_metric();
    double R = ricci_scalar(g, {0.3, 1.2});
    EXPECT_NEAR(R, 2.0, 0.05);
}

// ---- Geodesic (additional) ----

TEST(DiffGeoGeodesic, InitialState) {
    auto g = euclidean_2d();
    Coords x0 = {2.0, -1.0};
    Coords v0 = {0.3, 0.7};
    auto traj = geodesic(g, x0, v0, 1.0, 20);
    EXPECT_NEAR(traj.front().x[0], x0[0], 1e-10);
    EXPECT_NEAR(traj.front().x[1], x0[1], 1e-10);
    EXPECT_NEAR(traj.front().v[0], v0[0], 1e-10);
    EXPECT_NEAR(traj.front().v[1], v0[1], 1e-10);
}

TEST(DiffGeoGeodesic, MidpointFlat) {
    auto g = euclidean_2d();
    Coords x0 = {0.0, 0.0};
    Coords v0 = {2.0, 0.0};
    auto traj = geodesic(g, x0, v0, 1.0, 100);
    EXPECT_EQ(traj.size(), 101u);
    EXPECT_NEAR(traj[50].x[0], 1.0, 1e-2);
    EXPECT_NEAR(traj[50].x[1], 0.0, 1e-2);
}

TEST(DiffGeoGeodesic, StepCount) {
    auto g = euclidean_2d();
    Coords x0 = {0.0, 0.0};
    Coords v0 = {1.0, 1.0};
    auto traj = geodesic(g, x0, v0, 2.0, 40);
    EXPECT_EQ(traj.size(), 41u);
}

TEST(DiffGeoGeodesic, SphereMoves) {
    auto g = sphere_metric();
    Coords x0 = {M_PI / 4, 0.0};
    Coords v0 = {0.1, 0.2};
    auto traj = geodesic(g, x0, v0, 0.5, 30);
    double disp = std::abs(traj.back().x[0] - x0[0]) + std::abs(traj.back().x[1] - x0[1]);
    EXPECT_GT(disp, 1e-4);
}

// ---- Einstein tensor (additional) ----

TEST(DiffGeoEinstein, FlatVanishes) {
    auto g = euclidean_2d();
    auto G = einstein_tensor(g, {0.5, 0.5});
    for (auto& row : G)
        for (double v : row)
            EXPECT_NEAR(v, 0.0, 1e-3);
}

TEST(DiffGeoEinstein, Symmetric) {
    auto g = sphere_metric();
    auto G = einstein_tensor(g, {M_PI / 3, 0.2});
    EXPECT_NEAR(G[0][1], G[1][0], 1e-3);
}

TEST(DiffGeoEinstein, TraceRelation) {
    auto g = sphere_metric();
    Coords x = {M_PI / 4, 0.5};
    auto G = einstein_tensor(g, x);
    auto gv = g(x);
    auto ginv = metric_inv(gv);
    double trG = 0.0;
    int n = static_cast<int>(G.size());
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            trG += ginv[i][j] * G[i][j];
    double R = ricci_scalar(g, x);
    // g^{ij} G_{ij} = R - (n/2) R
    EXPECT_NEAR(trG, R * (1.0 - n / 2.0), 0.05);
}

TEST(DiffGeoEinstein, SphereVanishes) {
    // In 2D, the Einstein tensor vanishes on maximally symmetric backgrounds.
    auto g = sphere_metric();
    auto G = einstein_tensor(g, {M_PI / 3, M_PI / 6});
    for (auto& row : G)
        for (double v : row)
            EXPECT_NEAR(v, 0.0, 0.05);
}

// ---- Lie bracket (additional) ----

TEST(DiffGeoLieBracket, Antisymmetric) {
    VectorField X = [](const Coords& p) -> Coords { return {p[0], p[1]}; };
    VectorField Y = [](const Coords& p) -> Coords { return {p[1], -p[0]}; };
    Coords x = {0.7, -0.4};
    auto lb_xy = lie_bracket(X, Y, x);
    auto lb_yx = lie_bracket(Y, X, x);
    EXPECT_NEAR(lb_xy[0], -lb_yx[0], 1e-5);
    EXPECT_NEAR(lb_xy[1], -lb_yx[1], 1e-5);
}

TEST(DiffGeoLieBracket, SelfZero) {
    VectorField X = [](const Coords& p) -> Coords { return {p[0] * p[0], p[1]}; };
    Coords x = {1.5, -0.5};
    auto lb = lie_bracket(X, X, x);
    EXPECT_NEAR(lb[0], 0.0, 1e-6);
    EXPECT_NEAR(lb[1], 0.0, 1e-6);
}

TEST(DiffGeoLieBracket, KnownLinear) {
    // [∂/∂x, x ∂/∂x + y ∂/∂y] = ∂/∂x
    VectorField X = [](const Coords& p) -> Coords { (void)p; return {1.0, 0.0}; };
    VectorField Y = [](const Coords& p) -> Coords { return {p[0], p[1]}; };
    Coords x = {2.0, 3.0};
    auto lb = lie_bracket(X, Y, x);
    EXPECT_NEAR(lb[0], 1.0, 1e-5);
    EXPECT_NEAR(lb[1], 0.0, 1e-5);
}

TEST(DiffGeoLieBracket, OriginCommutingFields) {
    VectorField X = [](const Coords& p) -> Coords { return {p[0], 0.0}; };
    VectorField Y = [](const Coords& p) -> Coords { return {0.0, p[1]}; };
    Coords x = {0.0, 0.0};
    auto lb = lie_bracket(X, Y, x);
    EXPECT_NEAR(lb[0], 0.0, 1e-6);
    EXPECT_NEAR(lb[1], 0.0, 1e-6);
}
