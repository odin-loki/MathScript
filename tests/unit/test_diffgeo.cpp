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

// ---- Parallel transport ----

// Metric inner product g_ij V^i W^j at coordinate point x.
static double metric_inner(MetricFn g, const Coords& x, const Coords& V, const Coords& W) {
    auto gv = g(x);
    double sum = 0.0;
    for (size_t i = 0; i < V.size(); ++i)
        for (size_t j = 0; j < W.size(); ++j)
            sum += gv[i][j] * V[i] * W[j];
    return sum;
}

TEST(DiffGeoParallelTransport, FlatSpaceStraightLineUnchanged) {
    // In flat space Christoffel symbols vanish everywhere, so parallel
    // transport along ANY curve must leave the vector exactly unchanged.
    auto g = euclidean_2d();
    std::function<Coords(double)> curve = [](double s) -> Coords { return {s, 2.0*s}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {1.0, 2.0}; };
    Coords V0 = {0.5, -1.3};
    auto traj = parallel_transport(g, curve, vel, V0, 1.0, 50);
    EXPECT_NEAR(traj.back()[0], V0[0], 1e-10);
    EXPECT_NEAR(traj.back()[1], V0[1], 1e-10);
}

TEST(DiffGeoParallelTransport, FlatSpaceArbitraryCurveUnchanged) {
    // A curved (non-straight) path in flat space still yields zero Christoffel
    // symbols, so V must still come back exactly unchanged.
    auto g = euclidean_2d();
    std::function<Coords(double)> curve = [](double s) -> Coords {
        return {std::cos(s), std::sin(s)};
    };
    std::function<Coords(double)> vel = [](double s) -> Coords {
        return {-std::sin(s), std::cos(s)};
    };
    Coords V0 = {1.0, 0.0};
    auto traj = parallel_transport(g, curve, vel, V0, 2*M_PI, 200);
    EXPECT_NEAR(traj.back()[0], V0[0], 1e-9);
    EXPECT_NEAR(traj.back()[1], V0[1], 1e-9);
}

TEST(DiffGeoParallelTransport, ZeroVectorStaysZero) {
    auto g = sphere_metric();
    std::function<Coords(double)> curve = [](double s) -> Coords { return {M_PI/3, s}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {0.0, 1.0}; };
    Coords V0 = {0.0, 0.0};
    auto traj = parallel_transport(g, curve, vel, V0, 2*M_PI, 100);
    EXPECT_NEAR(traj.back()[0], 0.0, 1e-12);
    EXPECT_NEAR(traj.back()[1], 0.0, 1e-12);
}

TEST(DiffGeoParallelTransport, InitialVectorIsV0) {
    auto g = sphere_metric();
    std::function<Coords(double)> curve = [](double s) -> Coords { return {M_PI/4, s}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {0.0, 1.0}; };
    Coords V0 = {0.3, 0.9};
    auto traj = parallel_transport(g, curve, vel, V0, 1.0, 30);
    EXPECT_NEAR(traj.front()[0], V0[0], 1e-12);
    EXPECT_NEAR(traj.front()[1], V0[1], 1e-12);
}

TEST(DiffGeoParallelTransport, TrajectorySize) {
    auto g = euclidean_2d();
    std::function<Coords(double)> curve = [](double s) -> Coords { return {s, 0.0}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {1.0, 0.0}; };
    auto traj = parallel_transport(g, curve, vel, Coords{1.0, 0.0}, 3.0, 60);
    EXPECT_EQ(traj.size(), 61u);
}

// Headline test: parallel transport around a FULL closed loop of latitude
// (colatitude theta0 constant, phi: 0 -> 2*pi) on the unit sphere produces a
// holonomy -- the returned vector differs from V0 -- unlike flat space.
TEST(DiffGeoParallelTransport, SphereLoopHolonomyNonzero) {
    auto g = sphere_metric();
    double theta0 = M_PI / 3.0;
    std::function<Coords(double)> curve = [theta0](double s) -> Coords { return {theta0, s}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {0.0, 1.0}; };
    Coords V0 = {1.0, 0.0};
    auto traj = parallel_transport(g, curve, vel, V0, 2*M_PI, 2000);
    Coords Vf = traj.back();
    double diff = std::abs(Vf[0]-V0[0]) + std::abs(Vf[1]-V0[1]);
    EXPECT_GT(diff, 0.1);  // clearly different from the flat-space (zero) case
}

// Same loop, checked quantitatively: the rotation angle of the transported
// vector (measured in the local orthonormal frame e_theta, sin(theta0)*e_phi,
// where parallel transport acts as a pure rotation) should match the
// analytic holonomy angle 2*pi*cos(theta0) predicted by the Gauss-Bonnet
// theorem for the unit sphere (K=1).
TEST(DiffGeoParallelTransport, SphereLoopHolonomyAngleMatchesTheory) {
    auto g = sphere_metric();
    double theta0 = M_PI / 3.0;
    std::function<Coords(double)> curve = [theta0](double s) -> Coords { return {theta0, s}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {0.0, 1.0}; };
    Coords V0 = {1.0, 0.0};
    auto traj = parallel_transport(g, curve, vel, V0, 2*M_PI, 4000);
    Coords Vf = traj.back();
    // Orthonormal-frame components: a = V^theta, b = sin(theta0) * V^phi.
    double a0 = V0[0], b0 = std::sin(theta0) * V0[1];
    double af = Vf[0], bf = std::sin(theta0) * Vf[1];
    double angle = std::atan2(a0*bf - b0*af, a0*af + b0*bf);
    double expected = 2*M_PI*std::cos(theta0);
    // Normalize expected into (-pi, pi] for comparison with atan2's range.
    while (expected > M_PI) expected -= 2*M_PI;
    while (expected <= -M_PI) expected += 2*M_PI;
    EXPECT_NEAR(std::abs(angle), std::abs(expected), 0.05);
}

// Sanity check that the holonomy angle grows as the enclosed latitude
// approaches the pole (smaller theta0 -> larger enclosed curvature integral).
TEST(DiffGeoParallelTransport, SphereLoopHolonomyScalesWithLatitude) {
    auto g = sphere_metric();
    Coords V0 = {1.0, 0.0};
    std::function<Coords(double)> vel = [](double) -> Coords { return {0.0, 1.0}; };

    double theta_small = M_PI / 6.0;   // closer to pole -> bigger enclosed cap
    double theta_large = M_PI / 2.2;   // closer to equator -> smaller enclosed cap

    std::function<Coords(double)> curve_small = [theta_small](double s) -> Coords {
        return {theta_small, s};
    };
    std::function<Coords(double)> curve_large = [theta_large](double s) -> Coords {
        return {theta_large, s};
    };

    auto traj_small = parallel_transport(g, curve_small, vel, V0, 2*M_PI, 2000);
    auto traj_large = parallel_transport(g, curve_large, vel, V0, 2*M_PI, 2000);

    auto diff_mag = [&](const Coords& Vf) {
        return std::abs(Vf[0]-V0[0]) + std::abs(Vf[1]-V0[1]);
    };
    EXPECT_GT(diff_mag(traj_small.back()), diff_mag(traj_large.back()));
}

TEST(DiffGeoParallelTransport, MagnitudePreservedSphere) {
    // Parallel transport is an isometry: g_ij V^i V^j must stay constant
    // along the transport, at every step, even though the raw coordinate
    // components of V change. Explicit Euler (matching geodesic()'s scheme)
    // has a small, step-size-dependent energy drift on this rotational ODE,
    // so we use enough steps to keep that drift comfortably within tolerance.
    auto g = sphere_metric();
    double theta0 = M_PI / 4.0;
    std::function<Coords(double)> curve = [theta0](double s) -> Coords { return {theta0, s}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {0.0, 1.0}; };
    Coords V0 = {0.7, 0.4};
    int n_steps = 5000;
    auto traj = parallel_transport(g, curve, vel, V0, 2*M_PI, n_steps);
    double mag0 = metric_inner(g, curve(0.0), V0, V0);
    double ds = (2*M_PI) / n_steps;
    for (size_t i = 0; i < traj.size(); ++i) {
        double s = static_cast<double>(i) * ds;
        double mag = metric_inner(g, curve(s), traj[i], traj[i]);
        EXPECT_NEAR(mag, mag0, 0.01 * std::abs(mag0) + 1e-6);
    }
}

TEST(DiffGeoParallelTransport, MagnitudePreservedFlat) {
    auto g = euclidean_2d();
    std::function<Coords(double)> curve = [](double s) -> Coords {
        return {std::cos(s), std::sin(s)};
    };
    std::function<Coords(double)> vel = [](double s) -> Coords {
        return {-std::sin(s), std::cos(s)};
    };
    Coords V0 = {2.0, -1.0};
    auto traj = parallel_transport(g, curve, vel, V0, 2*M_PI, 200);
    double mag0 = metric_inner(g, curve(0.0), V0, V0);
    for (auto& V : traj) {
        double mag = metric_inner(g, {0,0}, V, V);  // flat metric is x-independent
        EXPECT_NEAR(mag, mag0, 1e-9);
    }
}

// Cross-check with geodesic(): the sphere's equator (theta = pi/2 constant)
// is itself a geodesic, so parallel-transporting its own tangent vector along
// it should reproduce exactly the velocity trajectory that geodesic() computes
// when started from the same initial position/velocity (both use the same
// Euler scheme, and the equator's Euler-stepped position matches x0 + s*v0
// exactly since the geodesic acceleration is identically zero along it).
TEST(DiffGeoParallelTransport, MatchesGeodesicForGeodesicCurve) {
    auto g = sphere_metric();
    Coords x0 = {M_PI/2, 0.0};
    Coords v0 = {0.0, 1.0};
    std::function<Coords(double)> curve = [x0, v0](double s) -> Coords {
        return {x0[0] + s*v0[0], x0[1] + s*v0[1]};
    };
    std::function<Coords(double)> vel = [v0](double) -> Coords { return v0; };

    auto pt_traj  = parallel_transport(g, curve, vel, v0, 1.0, 100);
    auto geo_traj = geodesic(g, x0, v0, 1.0, 100);

    ASSERT_EQ(pt_traj.size(), geo_traj.size());
    for (size_t i = 0; i < pt_traj.size(); ++i) {
        EXPECT_NEAR(pt_traj[i][0], geo_traj[i].v[0], 1e-10);
        EXPECT_NEAR(pt_traj[i][1], geo_traj[i].v[1], 1e-10);
    }
}

TEST(DiffGeoParallelTransport, TangentStaysTangentAlongGeodesic) {
    // Along a geodesic, parallel-transporting the tangent vector keeps it
    // proportional to the curve's own velocity at every step (tangent stays
    // tangent).
    auto g = sphere_metric();
    Coords x0 = {M_PI/2, 0.0};
    Coords v0 = {0.0, 1.0};
    std::function<Coords(double)> curve = [x0, v0](double s) -> Coords {
        return {x0[0] + s*v0[0], x0[1] + s*v0[1]};
    };
    std::function<Coords(double)> vel = [v0](double) -> Coords { return v0; };

    auto traj = parallel_transport(g, curve, vel, v0, 1.0, 100);
    for (size_t i = 0; i < traj.size(); ++i) {
        // Velocity along this curve is constant (0,1); V should stay parallel
        // to it, i.e. its theta-component (v0's zero component) stays ~0.
        EXPECT_NEAR(traj[i][0], 0.0, 1e-9);
        EXPECT_GT(traj[i][1], 0.0);
    }
}

TEST(DiffGeoParallelTransport, DegenerateNStepsReturnsV0) {
    auto g = euclidean_2d();
    std::function<Coords(double)> curve = [](double s) -> Coords { return {s, 0.0}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {1.0, 0.0}; };
    Coords V0 = {3.0, -2.0};
    auto traj_zero = parallel_transport(g, curve, vel, V0, 1.0, 0);
    auto traj_neg  = parallel_transport(g, curve, vel, V0, 1.0, -5);
    ASSERT_EQ(traj_zero.size(), 1u);
    ASSERT_EQ(traj_neg.size(), 1u);
    EXPECT_NEAR(traj_zero[0][0], V0[0], 1e-12);
    EXPECT_NEAR(traj_zero[0][1], V0[1], 1e-12);
    EXPECT_NEAR(traj_neg[0][0], V0[0], 1e-12);
    EXPECT_NEAR(traj_neg[0][1], V0[1], 1e-12);
}

TEST(DiffGeoParallelTransport, DegenerateSEndReturnsV0) {
    auto g = euclidean_2d();
    std::function<Coords(double)> curve = [](double s) -> Coords { return {s, 0.0}; };
    std::function<Coords(double)> vel   = [](double) -> Coords { return {1.0, 0.0}; };
    Coords V0 = {1.5, 0.5};
    auto traj_zero = parallel_transport(g, curve, vel, V0, 0.0, 50);
    auto traj_neg  = parallel_transport(g, curve, vel, V0, -1.0, 50);
    ASSERT_EQ(traj_zero.size(), 1u);
    ASSERT_EQ(traj_neg.size(), 1u);
    EXPECT_NEAR(traj_zero[0][0], V0[0], 1e-12);
    EXPECT_NEAR(traj_neg[0][0], V0[0], 1e-12);
}
