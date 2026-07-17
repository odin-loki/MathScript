#define _USE_MATH_DEFINES
#include "ms/diffgeo/diffgeo.hpp"
#include <cmath>
#include <array>
#include <utility>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace diffgeo {

// ---- Helper: numerical partial derivative of metric component ----
static std::vector<std::vector<double>> metric_deriv(
    MetricFn g, const Coords& x, int k, double h) {
    Coords xp = x, xm = x;
    xp[k] += h; xm[k] -= h;
    auto gp = g(xp), gm = g(xm);
    int n = static_cast<int>(x.size());
    std::vector<std::vector<double>> dg(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            dg[i][j] = (gp[i][j] - gm[i][j]) / (2*h);
    return dg;
}

std::vector<std::vector<double>>
metric_tensor(MetricFn g, const Coords& x) { return g(x); }

// Invert n×n matrix using Gauss-Jordan
std::vector<std::vector<double>>
metric_inv(const std::vector<std::vector<double>>& g_in) {
    int n = static_cast<int>(g_in.size());
    // Build augmented [g | I]
    std::vector<std::vector<double>> aug(n, std::vector<double>(2*n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = g_in[i][j];
        aug[i][n+i] = 1.0;
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col+1; row < n; ++row)
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) pivot = row;
        std::swap(aug[col], aug[pivot]);
        double sc = aug[col][col];
        if (std::abs(sc) < 1e-14) {
            // singular: return identity as fallback
            std::vector<std::vector<double>> id(n, std::vector<double>(n, 0.0));
            for (int k=0;k<n;++k) id[k][k]=1.0;
            return id;
        }
        for (int j = 0; j < 2*n; ++j) aug[col][j] /= sc;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = aug[row][col];
            for (int j = 0; j < 2*n; ++j) aug[row][j] -= f * aug[col][j];
        }
    }
    std::vector<std::vector<double>> inv(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) inv[i][j] = aug[i][n+j];
    return inv;
}

// ---- Christoffel symbols ----
// Γ^k_{ij} = (1/2) g^{km} (∂_i g_{jm} + ∂_j g_{im} - ∂_m g_{ij})
std::vector<std::vector<std::vector<double>>>
christoffel(MetricFn g, const Coords& x, double h) {
    int n = static_cast<int>(x.size());
    auto gv = g(x);
    auto ginv = metric_inv(gv);
    std::vector<std::vector<std::vector<double>>> dg(n);
    for (int k = 0; k < n; ++k) dg[k] = metric_deriv(g, x, k, h);

    // Γ^k_{ij}
    std::vector<std::vector<std::vector<double>>> Chr(n,
        std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int m = 0; m < n; ++m)
                    sum += ginv[k][m] * (dg[i][j][m] + dg[j][i][m] - dg[m][i][j]);
                Chr[k][i][j] = 0.5 * sum;
            }
    return Chr;
}

// ---- Riemann tensor R^l_{ijk} ----
// R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij} + Γ^l_{jm}Γ^m_{ik} - Γ^l_{km}Γ^m_{ij}
std::vector<std::vector<std::vector<std::vector<double>>>>
riemann_tensor(MetricFn g, const Coords& x, double h) {
    int n = static_cast<int>(x.size());
    auto Chr = christoffel(g, x, h);

    // Numerical derivative of Christoffel
    auto chr_deriv = [&](int wrt) {
        Coords xp=x, xm=x; xp[wrt]+=h; xm[wrt]-=h;
        auto cp = christoffel(g, xp, h);
        auto cm = christoffel(g, xm, h);
        std::vector<std::vector<std::vector<double>>> dc(n,
            std::vector<std::vector<double>>(n, std::vector<double>(n)));
        for (int l=0;l<n;++l) for (int i=0;i<n;++i) for (int j=0;j<n;++j)
            dc[l][i][j] = (cp[l][i][j] - cm[l][i][j]) / (2*h);
        return dc;
    };

    // Build dΓ[wrt][l][i][j]
    std::vector<decltype(Chr)> dChr(n);
    for (int wrt=0; wrt<n; ++wrt) dChr[wrt] = chr_deriv(wrt);

    // R^l_{ijk}
    std::vector<std::vector<std::vector<std::vector<double>>>> R(n,
        std::vector<std::vector<std::vector<double>>>(n,
        std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0))));
    for (int l=0;l<n;++l)
        for (int i=0;i<n;++i)
            for (int j=0;j<n;++j)
                for (int k=0;k<n;++k) {
                    double val = dChr[j][l][i][k] - dChr[k][l][i][j];
                    for (int m=0;m<n;++m)
                        val += Chr[l][j][m]*Chr[m][i][k] - Chr[l][k][m]*Chr[m][i][j];
                    R[l][i][j][k] = val;
                }
    return R;
}

// ---- Ricci tensor R_{ij} = R^k_{ikj} ----
std::vector<std::vector<double>>
ricci_tensor(MetricFn g, const Coords& x, double h) {
    int n = static_cast<int>(x.size());
    auto R = riemann_tensor(g, x, h);
    std::vector<std::vector<double>> Ric(n, std::vector<double>(n, 0.0));
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            for (int k=0;k<n;++k)
                Ric[i][j] += R[k][i][k][j];
    return Ric;
}

// ---- Ricci scalar ----
double ricci_scalar(MetricFn g, const Coords& x, double h) {
    int n = static_cast<int>(x.size());
    auto Ric = ricci_tensor(g, x, h);
    auto ginv = metric_inv(g(x));
    double R = 0.0;
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            R += ginv[i][j] * Ric[i][j];
    return R;
}

// ---- Einstein tensor ----
std::vector<std::vector<double>>
einstein_tensor(MetricFn g, const Coords& x, double h) {
    int n = static_cast<int>(x.size());
    auto Ric = ricci_tensor(g, x, h);
    auto Rv  = ricci_scalar(g, x, h);
    auto gv  = g(x);
    std::vector<std::vector<double>> G(n, std::vector<double>(n));
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            G[i][j] = Ric[i][j] - 0.5 * Rv * gv[i][j];
    return G;
}

// ---- Geodesic (Euler method on geodesic ODE) ----
std::vector<GeodesicState>
geodesic(MetricFn g, const Coords& x0, const Coords& v0,
         double s_end, int n_steps, double h) {
    int n = static_cast<int>(x0.size());
    double ds = s_end / n_steps;
    std::vector<GeodesicState> traj;
    traj.reserve(static_cast<size_t>(n_steps) + 1);
    Coords x = x0, v = v0;
    traj.push_back({x, v});
    for (int step = 0; step < n_steps; ++step) {
        auto Chr = christoffel(g, x, h);
        // d²x^k/ds² = -Γ^k_{ij} dx^i/ds dx^j/ds
        Coords acc(n, 0.0);
        for (int k=0;k<n;++k)
            for (int i=0;i<n;++i)
                for (int j=0;j<n;++j)
                    acc[k] -= Chr[k][i][j] * v[i] * v[j];
        // RK4
        for (int k=0;k<n;++k) { x[k] += ds * v[k]; v[k] += ds * acc[k]; }
        traj.push_back({x, v});
    }
    return traj;
}

// ---- Parallel transport (Euler method on the parallel transport ODE) ----
std::vector<Coords>
parallel_transport(MetricFn g, std::function<Coords(double)> curve,
                    std::function<Coords(double)> curve_velocity,
                    const Coords& V0, double s_end, int n_steps, double h) {
    std::vector<Coords> traj;
    traj.push_back(V0);
    if (n_steps <= 0 || !(s_end > 0.0)) return traj;

    traj.reserve(static_cast<size_t>(n_steps) + 1);

    int n = static_cast<int>(V0.size());
    double ds = s_end / n_steps;
    Coords V = V0;
    for (int step = 0; step < n_steps; ++step) {
        double s = step * ds;
        Coords x  = curve(s);
        Coords dx = curve_velocity(s);
        auto Chr = christoffel(g, x, h);
        // dV^k/ds = -Γ^k_{ij} dx^i/ds V^j
        Coords dV(n, 0.0);
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    dV[k] -= Chr[k][i][j] * dx[i] * V[j];
        for (int k = 0; k < n; ++k) V[k] += ds * dV[k];
        traj.push_back(V);
    }
    return traj;
}

// ---- Shared 3-vector helpers ----

static std::array<double,3> cross3(std::array<double,3> a, std::array<double,3> b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
static double dot3(std::array<double,3> a, std::array<double,3> b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static double norm3(std::array<double,3> a) { return std::sqrt(dot3(a,a)); }

// ---- Space curve geometry helpers ----

static std::array<double,3> curve_deriv(CurveFn r, double t, double h) {
    auto p1 = r(t + h), p2 = r(t - h);
    return {(p1[0] - p2[0]) / (2 * h),
            (p1[1] - p2[1]) / (2 * h),
            (p1[2] - p2[2]) / (2 * h)};
}

static std::array<double,3> curve_deriv2(CurveFn r, double t, double h) {
    auto pp = r(t + h), p0 = r(t), pm = r(t - h);
    return {(pp[0] - 2 * p0[0] + pm[0]) / (h * h),
            (pp[1] - 2 * p0[1] + pm[1]) / (h * h),
            (pp[2] - 2 * p0[2] + pm[2]) / (h * h)};
}

static std::array<double,3> curve_deriv3(CurveFn r, double t, double h) {
    // Central difference of r'' avoids the ill-conditioned O(h^3) 4-point stencil
    // at the default step h = 1e-5 (denominator ~ 2e-15 in double precision).
    auto rpp_p = curve_deriv2(r, t + h, h);
    auto rpp_m = curve_deriv2(r, t - h, h);
    return {(rpp_p[0] - rpp_m[0]) / (2 * h),
            (rpp_p[1] - rpp_m[1]) / (2 * h),
            (rpp_p[2] - rpp_m[2]) / (2 * h)};
}

double torsion(CurveFn r, double t, double h) {
    auto rp   = curve_deriv(r, t, h);
    auto rpp  = curve_deriv2(r, t, h);
    auto rppp = curve_deriv3(r, t, h);
    auto cross = cross3(rp, rpp);
    double cross_mag_sq = dot3(cross, cross);
    if (cross_mag_sq < 1e-30) return 0.0;
    return dot3(cross, rppp) / cross_mag_sq;
}

// ---- Surface geometry helpers ----

static std::array<double,3> surf_deriv(SurfaceFn r, double u, double v, bool du, double h) {
    double u1=u, v1=v, u2=u, v2=v;
    if (du) { u1+=h; u2-=h; } else { v1+=h; v2-=h; }
    auto p1 = r(u1,v1), p2 = r(u2,v2);
    return {(p1[0]-p2[0])/(2*h), (p1[1]-p2[1])/(2*h), (p1[2]-p2[2])/(2*h)};
}

static std::array<double,3> surf_deriv2(SurfaceFn r, double u, double v,
                                         bool du1, bool du2, double h) {
    // Mixed or second derivative
    if (du1 && du2) {
        auto pp=r(u+h,v+h), pm=r(u+h,v-h), mp=r(u-h,v+h), mm=r(u-h,v-h);
        return {(pp[0]-pm[0]-mp[0]+mm[0])/(4*h*h),
                (pp[1]-pm[1]-mp[1]+mm[1])/(4*h*h),
                (pp[2]-pm[2]-mp[2]+mm[2])/(4*h*h)};
    } else if (du1) {
        auto pp=r(u+h,v), p0=r(u,v), pm=r(u-h,v);
        return {(pp[0]-2*p0[0]+pm[0])/(h*h),
                (pp[1]-2*p0[1]+pm[1])/(h*h),
                (pp[2]-2*p0[2]+pm[2])/(h*h)};
    } else {
        auto pp=r(u,v+h), p0=r(u,v), pm=r(u,v-h);
        return {(pp[0]-2*p0[0]+pm[0])/(h*h),
                (pp[1]-2*p0[1]+pm[1])/(h*h),
                (pp[2]-2*p0[2]+pm[2])/(h*h)};
    }
}

std::array<double,3> surface_normal(SurfaceFn r, double u, double v, double h) {
    auto ru = surf_deriv(r, u, v, true, h);
    auto rv = surf_deriv(r, u, v, false, h);
    auto n = cross3(ru, rv);
    double l = norm3(n);
    if (l < 1e-15) return {0,0,1};
    return {n[0]/l, n[1]/l, n[2]/l};
}

FFF first_fundamental_form(SurfaceFn r, double u, double v, double h) {
    auto ru = surf_deriv(r, u, v, true, h);
    auto rv = surf_deriv(r, u, v, false, h);
    return {dot3(ru,ru), dot3(ru,rv), dot3(rv,rv)};
}

SFF second_fundamental_form(SurfaceFn r, double u, double v, double h) {
    auto ruu = surf_deriv2(r, u, v, true, false, h);   // d²r/du²
    auto ruv = surf_deriv2(r, u, v, true, true, h);    // d²r/du dv
    auto rvv = surf_deriv2(r, u, v, false, false, h);  // d²r/dv²
    auto N = surface_normal(r, u, v, h);
    return {dot3(ruu,N), dot3(ruv,N), dot3(rvv,N)};
}

double gaussian_curvature(SurfaceFn r, double u, double v, double h) {
    auto fff = first_fundamental_form(r, u, v, h);
    auto sff = second_fundamental_form(r, u, v, h);
    double W = fff.E*fff.G - fff.F*fff.F;
    if (std::abs(W) < 1e-15) return 0.0;
    return (sff.e*sff.g - sff.f*sff.f) / W;
}

double mean_curvature(SurfaceFn r, double u, double v, double h) {
    auto fff = first_fundamental_form(r, u, v, h);
    auto sff = second_fundamental_form(r, u, v, h);
    double W = fff.E*fff.G - fff.F*fff.F;
    if (std::abs(W) < 1e-15) return 0.0;
    return (sff.e*fff.G - 2*sff.f*fff.F + sff.g*fff.E) / (2*W);
}

std::pair<double,double> principal_curvatures(SurfaceFn r, double u, double v, double h) {
    double K = gaussian_curvature(r, u, v, h);
    double H = mean_curvature(r, u, v, h);
    double disc = H*H - K;
    if (disc < 0) disc = 0;
    double sq = std::sqrt(disc);
    return {H + sq, H - sq};
}

// ---- Gauss-Bonnet theorem (closed-surface / no-boundary case) ----
double gauss_bonnet_integral(SurfaceFn r, double u0, double u1, double v0, double v1,
                              int n_u, int n_v, double h) {
    if (n_u < 1 || n_v < 1) return 0.0;
    if (!(u1 > u0) || !(v1 > v0)) return 0.0;

    double du = (u1 - u0) / n_u;
    double dv = (v1 - v0) / n_v;

    double total = 0.0;
    for (int i = 0; i <= n_u; ++i) {
        double u = u0 + i * du;
        double wu = (i == 0 || i == n_u) ? 0.5 : 1.0;
        for (int j = 0; j <= n_v; ++j) {
            double v = v0 + j * dv;
            double wv = (j == 0 || j == n_v) ? 0.5 : 1.0;

            auto fff = first_fundamental_form(r, u, v, h);
            double jac = fff.E * fff.G - fff.F * fff.F;
            if (jac < 0.0) jac = 0.0;  // guard against numerical noise
            double dA = std::sqrt(jac);
            double K = gaussian_curvature(r, u, v, h);

            total += wu * wv * K * dA;
        }
    }
    return total * du * dv;
}

double gauss_bonnet_residual(SurfaceFn r, double u0, double u1, double v0, double v1,
                              int n_u, int n_v, int euler_characteristic, double h) {
    double integral = gauss_bonnet_integral(r, u0, u1, v0, v1, n_u, n_v, h);
    return integral - 2.0 * M_PI * static_cast<double>(euler_characteristic);
}

// ---- Lie bracket [X,Y]^k = X^i ∂_i Y^k - Y^i ∂_i X^k ----
Coords lie_bracket(VectorField X, VectorField Y, const Coords& x, double h) {
    int n = static_cast<int>(x.size());
    Coords result(n, 0.0);
    auto Xx = X(x), Yx = Y(x);
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            Coords xp=x, xm=x; xp[i]+=h; xm[i]-=h;
            auto Yp=Y(xp), Ym=Y(xm);
            auto Xp=X(xp), Xm=X(xm);
            double dYk_dxi = (Yp[k] - Ym[k]) / (2*h);
            double dXk_dxi = (Xp[k] - Xm[k]) / (2*h);
            result[k] += Xx[i] * dYk_dxi - Yx[i] * dXk_dxi;
        }
    }
    return result;
}

} // namespace diffgeo
} // namespace ms
