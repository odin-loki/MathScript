// MathScript Integration Test: Geo/Graph/Topo/Tensorops/ML/Diffgeo/Control/Quantum (Wave 57-59)

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "ms/combo/combo.hpp"
#include "ms/control/control.hpp"
#include "ms/diffgeo/diffgeo.hpp"
#include "ms/geo/geo.hpp"
#include "ms/graph/graph.hpp"
#include "ms/ml/ml.hpp"
#include "ms/numthy/numthy.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/tensorops/tensorops.hpp"
#include "ms/topo/topo.hpp"

using namespace ms;

namespace {

bool is_finite(double v) { return std::isfinite(v); }

std::vector<std::vector<double>> point_cloud_from_geo(const std::vector<geo::Point2D>& pts) {
    std::vector<std::vector<double>> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        out.push_back({p.x, p.y});
    }
    return out;
}

ml::Mat tensor_to_mat(const tensorops::Tensor& T) {
    ml::Mat out;
    if (T.ndim() == 1) {
        ml::Vec row;
        row.reserve(static_cast<size_t>(T.shape[0]));
        for (int i = 0; i < T.shape[0]; ++i) {
            row.push_back(T.at({i}));
        }
        out.push_back(std::move(row));
        return out;
    }
    if (T.ndim() == 2) {
        for (int r = 0; r < T.shape[0]; ++r) {
            ml::Vec row;
            row.reserve(static_cast<size_t>(T.shape[1]));
            for (int c = 0; c < T.shape[1]; ++c) {
                row.push_back(T.at({r, c}));
            }
            out.push_back(std::move(row));
        }
    }
    return out;
}

diffgeo::SurfaceFn unit_sphere() {
    return [](double u, double v) -> std::array<double, 3> {
        return {std::cos(u) * std::cos(v), std::cos(u) * std::sin(v), std::sin(u)};
    };
}

} // namespace

// ---------------------------------------------------------------------------
// Geo → Topo: point cloud → pairwise distances → Vietoris-Rips complex
// ---------------------------------------------------------------------------
TEST(Wave59Pipeline, Geo_Topo_VietorisRips) {
    const std::vector<geo::Point2D> geo_pts = {
        {0.0, 0.0}, {1.0, 0.0}, {0.5, 0.866}, {0.5, 0.3},
    };
    const auto pts = point_cloud_from_geo(geo_pts);
    const auto dist_mat = topo::pairwise_distances(pts);
    ASSERT_EQ(dist_mat.size(), geo_pts.size());
    EXPECT_NEAR(dist_mat[0][0], 0.0, 1e-10);
    EXPECT_NEAR(dist_mat[0][1], geo::dist(geo_pts[0], geo_pts[1]), 1e-10);

    const auto complex = topo::vietoris_rips(dist_mat, 1.2, 2);
    const int chi = complex.euler_characteristic();
    EXPECT_TRUE(std::isfinite(static_cast<double>(chi)));

    const auto betti = complex.betti_numbers();
    ASSERT_FALSE(betti.empty());
    for (int b : betti) {
        EXPECT_GE(b, 0);
    }
    EXPECT_EQ(betti[0], 1);
}

// ---------------------------------------------------------------------------
// Graph → Numthy/Combo: shortest paths + combinatorial path count
// ---------------------------------------------------------------------------
TEST(Wave59Pipeline, Graph_Numthy_ShortestPathCount) {
    // Diamond: two distinct shortest paths 0→1→3 and 0→2→3
    graph::Graph G(4, true);
    G.add_edge(0, 1, 1.0);
    G.add_edge(0, 2, 1.0);
    G.add_edge(1, 3, 1.0);
    G.add_edge(2, 3, 1.0);

    const auto [dist, parent] = graph::dijkstra(G, 0);
    ASSERT_EQ(dist.size(), 4u);
    EXPECT_NEAR(dist[3], 2.0, 1e-10);

    const auto pr = graph::pagerank(G, 0.85, 100);
    ASSERT_EQ(pr.size(), 4u);
    for (double r : pr) {
        EXPECT_TRUE(is_finite(r));
        EXPECT_GE(r, 0.0);
    }

    // Cycle graph for normalized PageRank (matches unit test pattern)
    graph::Graph cycle(3, true);
    cycle.add_edge(0, 1);
    cycle.add_edge(1, 2);
    cycle.add_edge(2, 0);
    const auto pr_cycle = graph::pagerank(cycle, 0.85, 100);
    double pr_sum = 0.0;
    for (double r : pr_cycle) {
        pr_sum += r;
    }
    EXPECT_NEAR(pr_sum, 1.0, 1e-5);

    // Grid path count: 3 right + 2 down steps → C(5,3) = 10 routes
    const uint32_t steps = 5;
    const uint32_t right = 3;
    const auto path_count = combo::binomial(steps, right);
    EXPECT_EQ(path_count, 10u);
    EXPECT_EQ(path_count, combo::combinations(steps, right));

    // numthy sanity: partition(5) = 7 distinct integer partitions
    EXPECT_EQ(numthy::partition(5), 7u);
}

// ---------------------------------------------------------------------------
// Tensorops → ML: einsum matmul feeds PCA / KMeans
// ---------------------------------------------------------------------------
TEST(Wave59Pipeline, Tensorops_ML_EinsumToPCA) {
    tensorops::Tensor A({4, 3}, 0.0);
    tensorops::Tensor B({3, 2}, 0.0);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            A.at({i, j}) = 0.1 * i + 0.05 * j + 1.0;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            B.at({i, j}) = (i == j) ? 1.0 : 0.2 * j;
        }
    }

    const auto C = tensorops::einsum("ij,jk->ik", A, B);
    ASSERT_EQ(C.shape[0], 4);
    ASSERT_EQ(C.shape[1], 2);

    const auto samples = tensor_to_mat(C);
    ASSERT_GE(samples.size(), 3u);
    ASSERT_EQ(samples[0].size(), 2u);

    ml::PCA pca(2);
    const auto Z = pca.fit_transform(samples);
    ASSERT_EQ(Z.size(), samples.size());
    ASSERT_EQ(Z[0].size(), 2u);
    EXPECT_TRUE(is_finite(Z.front()[0]));
    EXPECT_TRUE(is_finite(Z.back()[1]));
}

TEST(Wave59Pipeline, Tensorops_ML_EinsumToKMeans) {
    tensorops::Tensor U({6, 2}, 0.0);
    for (int i = 0; i < 6; ++i) {
        U.at({i, 0}) = (i < 3) ? 0.2 * i : 2.0 + 0.1 * i;
        U.at({i, 1}) = (i < 3) ? 0.5 * i : 1.5 + 0.2 * i;
    }
    tensorops::Tensor V({2, 2}, 0.0);
    V.at({0, 0}) = 1.0;
    V.at({1, 1}) = 1.0;

    const auto projected = tensorops::einsum("ij,jk->ik", U, V);
    const auto features = tensor_to_mat(projected);
    ASSERT_GE(features.size(), 4u);

    ml::KMeans km(2, 50, 1e-3);
    km.fit(features);
    const auto labels = km.predict(features);
    ASSERT_EQ(labels.size(), features.size());
    EXPECT_TRUE(labels[0] == 0.0 || labels[0] == 1.0);
    EXPECT_LT(km.inertia(features), 1e6);
}

// ---------------------------------------------------------------------------
// Diffgeo: Gaussian / mean curvature on parametric surface
// ---------------------------------------------------------------------------
TEST(Wave59Pipeline, Diffgeo_SurfaceCurvaturesFinite) {
    const auto sphere = unit_sphere();
    const double u = 0.4;
    const double v = 1.1;

    const double K = diffgeo::gaussian_curvature(sphere, u, v);
    const double H = diffgeo::mean_curvature(sphere, u, v);
    EXPECT_TRUE(is_finite(K));
    EXPECT_TRUE(is_finite(H));
    EXPECT_NEAR(K, 1.0, 0.05);
    EXPECT_NEAR(std::abs(H), 1.0, 0.08);

    const auto [k1, k2] = diffgeo::principal_curvatures(sphere, u, v);
    EXPECT_TRUE(is_finite(k1));
    EXPECT_TRUE(is_finite(k2));
}

// ---------------------------------------------------------------------------
// Control / Quantum smoke: transfer-function analysis + gate application
// ---------------------------------------------------------------------------
TEST(Wave59Pipeline, Control_Quantum_SmokeFinite) {
    const auto sys = control::tf({5.0}, {1.0, 2.0, 1.0});
    const double gain = control::dcgain(sys);
    EXPECT_TRUE(is_finite(gain));
    EXPECT_NEAR(gain, 5.0, 1e-8);

    const auto step = control::step_response(sys, 2.0, 40);
    ASSERT_FALSE(step.t.empty());
    ASSERT_EQ(step.t.size(), step.y.size());
    for (size_t i = 0; i < step.y.size(); ++i) {
        EXPECT_TRUE(is_finite(step.t[i]));
        EXPECT_TRUE(is_finite(step.y[i]));
    }

    const auto H = quantum::hadamard();
    const auto psi0 = quantum::ket_basis(2, 0);
    const auto psi1 = quantum::op_apply(H, psi0);
    ASSERT_EQ(psi1.size(), 2u);
    EXPECT_TRUE(is_finite(psi1[0].real()));
    EXPECT_TRUE(is_finite(psi1[0].imag()));
    EXPECT_TRUE(is_finite(psi1[1].real()));
    EXPECT_TRUE(is_finite(psi1[1].imag()));
    EXPECT_NEAR(std::abs(psi1[0]), 1.0 / std::sqrt(2.0), 1e-10);
}
