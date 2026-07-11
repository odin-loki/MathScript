#define _USE_MATH_DEFINES
#include "ms/cplx/cplx.hpp"
#include <cmath>
#include <numeric>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ms {
namespace cplx {

// ---- Residue at simple pole ----
C residue(CFunc f, C z0, double eps) {
    // res = lim_{z→z0} (z - z0) f(z)
    // Numerical: average over small circle
    const int N = 64;
    C sum = 0.0;
    for (int k = 0; k < N; ++k) {
        double theta = 2.0 * M_PI * k / N;
        C z = z0 + C(eps * std::cos(theta), eps * std::sin(theta));
        C delta = z - z0;
        sum += f(z) * delta;
    }
    // Divide by N (circle average gives residue)
    return sum / C(N, 0.0);
}

// ---- Winding number ----
int winding_number(const std::vector<C>& gamma, C z0) {
    double total_angle = 0.0;
    int n = static_cast<int>(gamma.size());
    for (int i = 0; i < n - 1; ++i) {
        C d1 = gamma[i]     - z0;
        C d2 = gamma[i + 1] - z0;
        total_angle += std::arg(d2 / d1);
    }
    return static_cast<int>(std::round(total_angle / (2.0 * M_PI)));
}

// ---- Contour integral ----
C contour_integral(CFunc f, const std::vector<C>& path, int n_pts) {
    C result = 0.0;
    int m = static_cast<int>(path.size());
    if (m < 2) return result;
    for (int seg = 0; seg < m - 1; ++seg) {
        C a = path[seg], b = path[seg + 1];
        // Gauss-Legendre quadrature on [0,1]
        result += line_integral(f, a, b, n_pts);
    }
    return result;
}

// ---- Cauchy integral ----
C cauchy_integral(CFunc f, C z0, const std::vector<C>& contour, int n_pts) {
    auto g = [&](C z) { return f(z) / (z - z0); };
    C integral = contour_integral(g, contour, n_pts);
    return integral / C(0.0, 2.0 * M_PI);
}

// ---- Argument principle ----
int argument_principle(CFunc f, const std::vector<C>& contour, int n_pts) {
    // Returns (1/2πi) ∮ f'/f dz ≈ N - P
    // Numerical: argument change of f along contour
    int m = static_cast<int>(contour.size());
    double total = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        C a = contour[i], b = contour[i + 1];
        C fa = f(a), fb = f(b);
        if (std::abs(fa) > 1e-12 && std::abs(fb) > 1e-12)
            total += std::arg(fb / fa);
    }
    return static_cast<int>(std::round(total / (2.0 * M_PI)));
}

// ---- Möbius ----
Mobius Mobius::compose(const Mobius& o) const {
    return {a * o.a + b * o.c, a * o.b + b * o.d,
            c * o.a + d * o.c, c * o.b + d * o.d};
}

Mobius Mobius::inverse() const {
    return {d, -b, -c, a};
}

std::vector<C> Mobius::fixed_points() const {
    // (az+b)/(cz+d) = z → cz^2 + (d-a)z - b = 0
    if (std::abs(c) < 1e-15) {
        // Linear: az+b = dz → (a-d)z = -b
        if (std::abs(a - d) < 1e-15) return {};
        return {-b / (a - d)};
    }
    C A = c, B = d - a, CC = -b;
    C disc = B * B - C(4.0) * A * CC;
    C sq = std::sqrt(disc);
    return {(-B + sq) / (C(2.0) * A), (-B - sq) / (C(2.0) * A)};
}

Mobius mobius(C a, C b, C c, C d) { return {a, b, c, d}; }

Mobius inversion(C center, double r) {
    // z → r^2 / conj(z - center) + center (simplified as Möbius: map unit circle)
    // Reflection in circle |z - c| = r: w = r^2/conj(z-c) + c
    // Expressed as Möbius in terms of z (not conj), this is anti-Möbius
    // Return identity for now as Möbius (proper inversion requires anti-holomorphic)
    (void)center; (void)r;
    return {C(1.0), C(0.0), C(0.0), C(1.0)};
}

double cross_ratio(C z1, C z2, C z3, C z4) {
    // (z1-z3)(z2-z4) / ((z1-z4)(z2-z3))
    C cr = ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3));
    return cr.real();
}

// ---- Joukowski ----
C joukowski(C z, double c) {
    if (std::abs(z) < 1e-15) return z;
    return z + c * c / z;
}

std::vector<C> joukowski_inv(C w, double c) {
    // w = z + c^2/z → z^2 - wz + c^2 = 0
    C disc = w * w - C(4.0 * c * c);
    C sq = std::sqrt(disc);
    return {(w + sq) / C(2.0), (w - sq) / C(2.0)};
}

// ---- Poisson kernel ----
double poisson_kernel(double theta, double phi, double r) {
    // P_r(theta-phi) = (1 - r^2) / (1 - 2r*cos(theta-phi) + r^2)
    double diff = theta - phi;
    return (1.0 - r * r) / (1.0 - 2.0 * r * std::cos(diff) + r * r);
}

// ---- Green's function for the Dirichlet Laplacian on a disk ----
double green_function_disk(C z, C z0, double radius) {
    if (radius <= 0.0) return 0.0;
    // Out-of-domain: defensively return 0.0 (like poisson_kernel, which is
    // well-defined for all r rather than throwing on |r| >= 1).
    if (std::abs(z) >= radius || std::abs(z0) >= radius) return 0.0;
    // Rescale to the unit disk: w = z/radius, w0 = z0/radius.
    C w = z / radius;
    C w0 = z0 / radius;
    C num = w - w0;
    if (std::abs(num) < 1e-15) return -std::numeric_limits<double>::infinity();
    C den = C(1.0) - std::conj(w0) * w;
    return std::log(std::abs(num) / std::abs(den)) / (2.0 * M_PI);
}

// ---- Harmonic conjugate via discrete Hilbert transform ----
std::vector<double> harmonic_conjugate(const std::vector<double>& u) {
    int n = static_cast<int>(u.size());
    // FFT-based Hilbert on circle: multiply by -i*sign(k)
    // Simple: use the fact that conjugate function's DFT is -i*sign(k)*U_k
    // Approximate using direct DFT
    std::vector<double> v(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = 1; k <= n / 2; ++k) {
            double Uc_k = 0.0, Us_k = 0.0;
            for (int m = 0; m < n; ++m) {
                Uc_k += u[m] * std::cos(2.0 * M_PI * k * m / n);
                Us_k += u[m] * std::sin(2.0 * M_PI * k * m / n);
            }
            // v_j += (2/n) * (Us_k * cos - Uc_k * sin)
            v[j] += (2.0 / n) * (Us_k * std::cos(2.0 * M_PI * k * j / n)
                                  - Uc_k * std::sin(2.0 * M_PI * k * j / n));
        }
    }
    return v;
}

// ---- Hyperbolic distance (Poincaré disk) ----
double hyperbolic_distance(C z1, C z2) {
    // d(z1,z2) = 2 arctanh(|z1-z2| / |1 - z1*conj(z2)|)
    double num = std::abs(z1 - z2);
    double den = std::abs(C(1.0) - z1 * std::conj(z2));
    if (den < 1e-15) return std::numeric_limits<double>::infinity();
    return 2.0 * std::atanh(num / den);
}

// ---- Laurent coefficients ----
std::vector<C> laurent_coeffs(CFunc f, C z0, double r,
                               int neg_terms, int pos_terms) {
    int N = std::max(256, 4 * (neg_terms + pos_terms));
    // a_k = (1/2πi) ∮ f(z) / (z-z0)^{k+1} dz = (1/N) sum f(z_j) z_j^{-k}
    std::vector<C> coeffs;
    int total = neg_terms + pos_terms + 1;
    coeffs.reserve(total);
    for (int k = -neg_terms; k <= pos_terms; ++k) {
        C sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double theta = 2.0 * M_PI * j / N;
            C z = z0 + C(r * std::cos(theta), r * std::sin(theta));
            // a_k = (1/2πi) ∮ f(z)/(z-z0)^{k+1} dz
            // ≈ (r * i * e^{itheta}) / (r^{k+1} e^{i(k+1)theta}) * f(z) * (2πr/N)
            C integrand = f(z) * std::pow(z - z0, C(-(k + 1)));
            sum += integrand * C(0.0, r * 2.0 * M_PI / N) *
                   C(std::cos(theta), std::sin(theta));
        }
        coeffs.push_back(sum / C(0.0, 2.0 * M_PI));
    }
    return coeffs;
}

// ---- Line integral (Gauss-Legendre on segment) ----
C line_integral(CFunc f, C a, C b, int n_pts) {
    // Parameterise: z(t) = a + t(b-a), t in [0,1]
    C result = 0.0;
    C dz = b - a;
    for (int k = 0; k < n_pts; ++k) {
        double t0 = static_cast<double>(k) / n_pts;
        double t1 = static_cast<double>(k + 1) / n_pts;
        double tm = (t0 + t1) / 2.0;
        C z = a + C(tm) * dz;
        result += f(z) * dz / C(n_pts);
    }
    return result;
}

// ---- Power series ----
C power_series_eval(const std::vector<C>& coeffs, C z0, C z) {
    C result = 0.0;
    C pw = 1.0;
    C dz = z - z0;
    for (auto& c : coeffs) { result += c * pw; pw *= dz; }
    return result;
}

// ---- Blaschke product ----
C blaschke_product(C z, const std::vector<C>& zeros) {
    C result = 1.0;
    for (auto& a : zeros) {
        if (std::abs(a) < 1e-15) { result *= z; continue; }
        result *= (z - a) / (C(1.0) - std::conj(a) * z);
    }
    return result;
}

} // namespace cplx
} // namespace ms
