// MathScript Integration Test: Waves 57–60 full cross-module pipelines (Wave 66)

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/combo/combo.hpp"
#include "ms/compress/compress.hpp"
#include "ms/control/control.hpp"
#include "ms/cplx/cplx.hpp"
#include "ms/diffgeo/diffgeo.hpp"
#include "ms/finance/finance.hpp"
#include "ms/geo/geo.hpp"
#include "ms/graph/graph.hpp"
#include "ms/image/image.hpp"
#include "ms/info/info.hpp"
#include "ms/ml/ml.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/stats/stats.hpp"
#include "ms/tensorops/tensorops.hpp"
#include "ms/topo/topo.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

namespace {

bool is_finite(double v) { return std::isfinite(v); }

image::Image make_checkerboard_rgb(int rows, int cols, int block = 4) {
    image::Image img(rows, cols, 3);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const bool light = ((r / block) + (c / block)) % 2 == 0;
            const float v = light ? 0.9f : 0.1f;
            img.at(r, c, 0) = v;
            img.at(r, c, 1) = light ? 0.2f : 0.8f;
            img.at(r, c, 2) = light ? 0.2f : 0.2f;
        }
    }
    return img;
}

ml::Mat extract_patches(const image::Image& gray, int patch, int stride) {
    ml::Mat patches;
    for (int r = 0; r + patch <= gray.rows; r += stride) {
        for (int c = 0; c + patch <= gray.cols; c += stride) {
            ml::Vec feat;
            feat.reserve(static_cast<size_t>(patch * patch));
            for (int dr = 0; dr < patch; ++dr) {
                for (int dc = 0; dc < patch; ++dc) {
                    feat.push_back(static_cast<double>(gray.at(r + dr, c + dc, 0)));
                }
            }
            patches.push_back(std::move(feat));
        }
    }
    return patches;
}

std::vector<std::vector<double>> point_cloud_from_geo(const std::vector<geo::Point2D>& pts) {
    std::vector<std::vector<double>> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        out.push_back({p.x, p.y});
    }
    return out;
}

std::vector<double> normalize_abs(std::span<const double> values) {
    double total = 0.0;
    for (double v : values) {
        total += std::abs(v);
    }
    std::vector<double> probs;
    probs.reserve(values.size());
    if (total <= 0.0) {
        const double uniform = 1.0 / static_cast<double>(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            (void)i;
            probs.push_back(uniform);
        }
        return probs;
    }
    for (double v : values) {
        probs.push_back(std::abs(v) / total);
    }
    return probs;
}

compress::Bytes probs_to_bytes(const std::vector<double>& probs) {
    compress::Bytes out;
    for (double p : probs) {
        const int q = static_cast<int>(std::lround(p * 10000.0));
        out.push_back(static_cast<uint8_t>((q >> 8) & 0xFF));
        out.push_back(static_cast<uint8_t>(q & 0xFF));
    }
    return out;
}

graph::Graph graph_from_state_matrix(const std::vector<std::vector<double>>& A,
                                     double threshold = 1e-12) {
    const int n = static_cast<int>(A.size());
    graph::Graph G(n, true);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::abs(A[i][j]) > threshold) {
                G.add_edge(i, j, std::abs(A[i][j]));
            }
        }
    }
    return G;
}

std::vector<cplx::C> unit_circle_contour(int n_pts = 64) {
    std::vector<cplx::C> path;
    path.reserve(static_cast<size_t>(n_pts) + 1u);
    for (int i = 0; i <= n_pts; ++i) {
        const double t = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n_pts);
        path.push_back(cplx::C(std::cos(t), std::sin(t)));
    }
    return path;
}

diffgeo::SurfaceFn unit_sphere() {
    return [](double u, double v) -> std::array<double, 3> {
        return {std::cos(u) * std::cos(v), std::cos(u) * std::sin(v), std::sin(u)};
    };
}

bignum::BigInt count_cluster_assignments(const ml::Vec& labels) {
    bignum::BigInt count(0LL);
    for (double label : labels) {
        if (label >= 0.5) {
            count = count + bignum::BigInt(1LL);
        }
    }
    return count;
}

} // namespace

// ---------------------------------------------------------------------------
// Finance + Info + Compress: bond cashflows → entropy → compress roundtrip
// ---------------------------------------------------------------------------
TEST(Wave66FullPipeline, Finance_Info_Compress_BondEntropyCompress) {
    const double coupon_rate = 0.05;
    const double yield = 0.04;
    const int periods = 8;
    const double face = 100.0;
    const double price = finance::bond_price(coupon_rate, yield, periods, face);
    EXPECT_GT(price, 0.0);
    EXPECT_TRUE(is_finite(price));

    const double coupon_pay = coupon_rate * face;
    std::vector<double> cashflows(static_cast<size_t>(periods + 1), coupon_pay);
    cashflows.back() += face;
    cashflows.front() = -price;

    const auto probs = normalize_abs(cashflows);
    ASSERT_EQ(probs.size(), cashflows.size());
    double prob_sum = 0.0;
    for (double p : probs) {
        EXPECT_GE(p, 0.0);
        prob_sum += p;
    }
    EXPECT_NEAR(prob_sum, 1.0, 1e-10);

    const double H_bits = info::entropy(probs, 2.0);
    EXPECT_TRUE(is_finite(H_bits));
    EXPECT_GT(H_bits, 0.0);
    EXPECT_LT(H_bits, std::log2(static_cast<double>(probs.size())) + 0.01);

    const auto raw = probs_to_bytes(probs);
    ASSERT_FALSE(raw.empty());

    const auto rle_back = compress::rle_decode(compress::rle_encode(raw));
    EXPECT_EQ(rle_back, raw);

    const auto hr = compress::huffman_encode(raw);
    const auto huff_back = compress::huffman_decode(hr, raw.size());
    EXPECT_EQ(huff_back, raw);

    const double npv_val = finance::npv(yield, std::span<const double>(cashflows));
    EXPECT_NEAR(npv_val, 0.0, 0.05);
}

// ---------------------------------------------------------------------------
// Control + Graph + Quantum: TF step → path weights → amplitude encoding
// ---------------------------------------------------------------------------
TEST(Wave66FullPipeline, Control_Graph_Quantum_StepPathAmplitudes) {
    const auto sys = control::tf({5.0}, {1.0, 2.0, 1.0});
    const auto step = control::step_response(sys, 3.0, 50);
    ASSERT_FALSE(step.t.empty());
    ASSERT_EQ(step.t.size(), step.y.size());
    for (double y : step.y) {
        EXPECT_TRUE(is_finite(y));
    }

    const auto ss = control::tf2ss(sys);
    ASSERT_GT(ss.n, 0);
    const auto G = graph_from_state_matrix(ss.A);
    EXPECT_EQ(G.n_vertices(), ss.n);

    const auto [dist, parent] = graph::dijkstra(G, 0);
    (void)parent;
    ASSERT_EQ(dist.size(), static_cast<size_t>(ss.n));

    std::vector<double> path_weights;
    path_weights.reserve(dist.size());
    for (double d : dist) {
        path_weights.push_back(d < graph::INF ? d : 0.0);
    }
    ASSERT_FALSE(path_weights.empty());

    double weight_sum = 0.0;
    for (double w : path_weights) {
        weight_sum += w;
    }
    if (weight_sum <= 0.0) {
        weight_sum = static_cast<double>(path_weights.size());
        for (double& w : path_weights) {
            w = 1.0;
        }
    } else {
        for (double& w : path_weights) {
            w /= weight_sum;
        }
    }

    const auto psi_raw = quantum::ket_superposition(path_weights);
    const auto psi = quantum::ket_normalise(psi_raw);
    ASSERT_EQ(psi.size(), path_weights.size());

    double amp_sum = 0.0;
    for (const auto& amp : psi) {
        EXPECT_TRUE(is_finite(amp.real()));
        EXPECT_TRUE(is_finite(amp.imag()));
        amp_sum += std::norm(amp);
    }
    EXPECT_NEAR(amp_sum, 1.0, 1e-10);

    const auto H = quantum::hadamard();
    if (psi.size() >= 2) {
        const auto psi0 = quantum::ket_basis(2, 0);
        const auto mixed = quantum::op_apply(H, psi0);
        EXPECT_NEAR(std::norm(mixed[0]) + std::norm(mixed[1]), 1.0, 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Geo + Diffgeo + Topo: hull area → sphere curvature → Rips Euler
// ---------------------------------------------------------------------------
TEST(Wave66FullPipeline, Geo_Diffgeo_Topo_HullCurvatureEuler) {
    const std::vector<geo::Point2D> geo_pts = {
        {0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}, {1.0, 1.0},
    };
    const auto hull = geo::convex_hull_2d(geo_pts);
    ASSERT_GE(hull.size(), 3u);
    const double hull_area = geo::area(hull);
    EXPECT_NEAR(hull_area, 4.0, 1e-10);

    const auto sphere = unit_sphere();
    const double u = 0.35;
    const double v = 0.9;
    const double K = diffgeo::gaussian_curvature(sphere, u, v);
    const double H = diffgeo::mean_curvature(sphere, u, v);
    EXPECT_TRUE(is_finite(K));
    EXPECT_TRUE(is_finite(H));
    EXPECT_NEAR(K, 1.0, 0.05);
    EXPECT_NEAR(std::abs(H), 1.0, 0.08);

    const auto pts = point_cloud_from_geo(geo_pts);
    const auto dist_mat = topo::pairwise_distances(pts);
    ASSERT_EQ(dist_mat.size(), geo_pts.size());
    EXPECT_NEAR(dist_mat[0][1], geo::dist(geo_pts[0], geo_pts[1]), 1e-10);

    const auto complex = topo::vietoris_rips(dist_mat, 2.5, 2);
    const int chi = complex.euler_characteristic();
    EXPECT_TRUE(is_finite(static_cast<double>(chi)));
    EXPECT_GE(complex.betti_numbers()[0], 1);
}

// ---------------------------------------------------------------------------
// ML + Image + Bignum: patches → PCA → bigint cluster assignment count
// ---------------------------------------------------------------------------
TEST(Wave66FullPipeline, ML_Image_Bignum_PCAClusterCount) {
    const auto rgb = make_checkerboard_rgb(20, 20, 2);
    const auto gray = image::rgb2gray(rgb);
    EXPECT_EQ(gray.channels, 1);

    const auto patches = extract_patches(gray, 3, 3);
    ASSERT_GE(patches.size(), 4u);

    ml::PCA pca(2);
    const auto Z = pca.fit_transform(patches);
    ASSERT_EQ(Z.size(), patches.size());
    ASSERT_EQ(Z[0].size(), 2u);
    EXPECT_TRUE(is_finite(Z.front()[0]));

    ml::KMeans km(2, 50, 1e-3);
    km.fit(Z);
    const auto labels = km.predict(Z);
    ASSERT_EQ(labels.size(), Z.size());

    const auto cluster_one = count_cluster_assignments(labels);
    EXPECT_GT(cluster_one.to_ll(), 0LL);
    EXPECT_LE(cluster_one.to_ll(), static_cast<long long>(labels.size()));

    const bignum::BigInt total(static_cast<long long>(labels.size()));
    const bignum::BigInt cluster_zero = total - cluster_one;
    EXPECT_EQ((cluster_one + cluster_zero).to_ll(), static_cast<long long>(labels.size()));
    EXPECT_LT(km.inertia(Z), 1e6);
}

// ---------------------------------------------------------------------------
// Cplx + Numthy + Combo: contour/Möbius finite → binomial path count
// ---------------------------------------------------------------------------
TEST(Wave66FullPipeline, Cplx_Numthy_Combo_ContourMobiusPathCount) {
    const auto contour = unit_circle_contour(72);

    const auto f_const = [](cplx::C z) {
        (void)z;
        return cplx::C(3.0, 1.0);
    };
    const cplx::C cauchy_val = cplx::cauchy_integral(f_const, cplx::C(0.0, 0.0), contour, 60);
    EXPECT_TRUE(is_finite(cauchy_val.real()));
    EXPECT_TRUE(is_finite(cauchy_val.imag()));
    EXPECT_NEAR(cauchy_val.real(), 3.0, 0.15);

    const double theta = M_PI / 5.0;
    const cplx::C z(std::cos(theta), std::sin(theta));
    const cplx::C w = cplx::joukowski(z, 1.0);
    EXPECT_TRUE(is_finite(w.real()));
    EXPECT_TRUE(is_finite(w.imag()));

    const cplx::Mobius rot_map(cplx::C(1.0, 0.0), cplx::C(0.0, 0.0),
                               cplx::C(0.0, 0.0), cplx::C(std::exp(cplx::C(0.0, theta))));
    const cplx::C mapped = rot_map(z);
    EXPECT_NEAR(std::abs(mapped), 1.0, 1e-10);

    const auto f_z = [](cplx::C z_in) { return z_in; };
    const cplx::C contour_val = cplx::contour_integral(f_z, contour, 40);
    EXPECT_TRUE(is_finite(contour_val.real()));
    EXPECT_TRUE(is_finite(contour_val.imag()));

    const uint32_t steps = static_cast<uint32_t>(numthy::partition(5));
    EXPECT_EQ(steps, 7u);
    const uint32_t right = 3u;
    const auto path_count = combo::binomial(steps, right);
    EXPECT_EQ(path_count, 35u);
    EXPECT_EQ(path_count, combo::combinations(steps, right));
}

// ---------------------------------------------------------------------------
// Tensorops + Stats + Finance: einsum weights → Sharpe on synthetic returns
// ---------------------------------------------------------------------------
TEST(Wave66FullPipeline, Tensorops_Stats_Finance_EinsumSharpe) {
    tensorops::Tensor cov({3, 3}, 0.0);
    for (int i = 0; i < 3; ++i) {
        cov.at({i, i}) = 0.04 + 0.01 * i;
        for (int j = i + 1; j < 3; ++j) {
            cov.at({i, j}) = 0.005 * (i + j + 1);
            cov.at({j, i}) = cov.at({i, j});
        }
    }

    tensorops::Tensor ones({3, 1}, 1.0);
    const auto inv_diag = tensorops::einsum("ij,jk->ik", cov, ones);
    ASSERT_EQ(inv_diag.shape[0], 3);
    ASSERT_EQ(inv_diag.shape[1], 1);

    std::vector<double> raw_weights;
    raw_weights.reserve(3);
    for (int i = 0; i < 3; ++i) {
        raw_weights.push_back(1.0 / inv_diag.at({i, 0}));
    }
    double w_sum = 0.0;
    for (double w : raw_weights) {
        w_sum += w;
    }
    for (double& w : raw_weights) {
        w /= w_sum;
    }

    const std::vector<double> asset_returns = {0.10, 0.06, 0.04};
    const double port_ret = finance::portfolio_return(raw_weights, asset_returns);
    EXPECT_TRUE(is_finite(port_ret));
    EXPECT_GT(port_ret, 0.04);

    std::vector<double> synthetic_returns;
    synthetic_returns.reserve(12);
    for (int t = 0; t < 12; ++t) {
        const double drift = port_ret / 12.0;
        const double noise = 0.002 * std::sin(0.7 * t) + 0.001 * (t % 3);
        synthetic_returns.push_back(drift + noise);
    }

    const double mu = mean(std::span<const double>(synthetic_returns));
    const double sd = stddev(std::span<const double>(synthetic_returns));
    EXPECT_TRUE(is_finite(mu));
    EXPECT_GT(sd, 0.0);

    const double sr = finance::sharpe_ratio(synthetic_returns, 0.0);
    EXPECT_TRUE(is_finite(sr));
    EXPECT_GT(sr, 0.0);
}
