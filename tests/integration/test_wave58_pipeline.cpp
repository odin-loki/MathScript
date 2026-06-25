// MathScript Integration Test: Control/Graph/Cplx/Quantum/Finance/Info/Stats (Wave 58)

#define _USE_MATH_DEFINES
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "ms/control/control.hpp"
#include "ms/cplx/cplx.hpp"
#include "ms/finance/finance.hpp"
#include "ms/graph/graph.hpp"
#include "ms/info/info.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/stats/stats.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ms;

namespace {

bool is_finite(double v) { return std::isfinite(v); }

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

std::vector<double> probability_amplitudes_squared(const quantum::Ket& psi) {
    std::vector<double> probs;
    probs.reserve(psi.size());
    for (const auto& amp : psi) {
        probs.push_back(std::norm(amp));
    }
    return probs;
}

} // namespace

// ---------------------------------------------------------------------------
// Control → Graph: TF step response, SS adjacency graph path analysis
// ---------------------------------------------------------------------------
TEST(Wave58Pipeline, Control_Graph_StateFlowPath) {
    const auto sys = control::tf({1.0}, {1.0, 3.0, 2.0});  // 1/((s+1)(s+2))
    EXPECT_TRUE(control::is_stable(sys));

    const auto step = control::step_response(sys, 5.0, 80);
    ASSERT_FALSE(step.t.empty());
    ASSERT_EQ(step.t.size(), step.y.size());
    for (size_t i = 0; i < step.y.size(); ++i) {
        EXPECT_TRUE(is_finite(step.t[i]));
        EXPECT_TRUE(is_finite(step.y[i]));
    }
    EXPECT_NEAR(step.y.back(), control::dcgain(sys), 0.05);

    const auto ss = control::tf2ss(sys);
    ASSERT_GT(ss.n, 0);
    const auto G = graph_from_state_matrix(ss.A);
    EXPECT_EQ(G.n_vertices(), ss.n);

    const auto [dist, parent] = graph::dijkstra(G, 0);
    (void)parent;
    ASSERT_EQ(dist.size(), static_cast<size_t>(ss.n));
    for (double d : dist) {
        if (d < graph::INF) {
            EXPECT_TRUE(is_finite(d));
            EXPECT_GE(d, 0.0);
        }
    }

    const auto order = graph::bfs(G, 0);
    ASSERT_FALSE(order.empty());
    EXPECT_EQ(order.front(), 0);
}

// ---------------------------------------------------------------------------
// Cplx → Quantum: Möbius/Joukowski phase maps to rotation gate
// ---------------------------------------------------------------------------
TEST(Wave58Pipeline, Cplx_Quantum_MobiusRotationPhase) {
    const double theta = M_PI / 6.0;
    const cplx::C z(std::cos(theta), std::sin(theta));

    const cplx::C w = cplx::joukowski(z, 1.0);
    EXPECT_TRUE(is_finite(w.real()));
    EXPECT_TRUE(is_finite(w.imag()));

    const cplx::Mobius rot_map(cplx::C(1.0, 0.0), cplx::C(0.0, 0.0),
                               cplx::C(0.0, 0.0), cplx::C(std::exp(cplx::C(0.0, theta))));
    const cplx::C mapped = rot_map(z);
    EXPECT_NEAR(std::abs(mapped), 1.0, 1e-10);

    const double gate_angle = std::arg(mapped);
    const auto Rz = quantum::rotation_z(gate_angle);
    const auto psi0 = quantum::ket_basis(2, 0);
    const auto psi1 = quantum::op_apply(Rz, psi0);

    ASSERT_EQ(psi1.size(), 2u);
    EXPECT_TRUE(is_finite(psi1[0].real()));
    EXPECT_TRUE(is_finite(psi1[0].imag()));
    EXPECT_NEAR(std::norm(psi1[0]) + std::norm(psi1[1]), 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Finance → Stats: BS call strip, return statistics and Sharpe
// ---------------------------------------------------------------------------
TEST(Wave58Pipeline, Finance_Stats_CallReturnsSharpe) {
    const double K = 100.0;
    const double T = 1.0;
    const double r = 0.05;
    const double sigma = 0.2;
    const std::vector<double> spots = {90.0, 95.0, 100.0, 105.0, 110.0};

    std::vector<double> call_prices;
    call_prices.reserve(spots.size());
    for (double S : spots) {
        const double price = finance::bs_call(S, K, T, r, sigma);
        EXPECT_GT(price, 0.0);
        EXPECT_TRUE(is_finite(price));
        call_prices.push_back(price);
    }

    std::vector<double> log_returns;
    log_returns.reserve(call_prices.size() - 1);
    for (size_t i = 1; i < call_prices.size(); ++i) {
        log_returns.push_back(std::log(call_prices[i] / call_prices[i - 1]));
    }
    ASSERT_GE(log_returns.size(), 2u);

    const double mu = mean(log_returns);
    const double sd = stddev(log_returns);
    const double sr = finance::sharpe_ratio(log_returns, 0.0);

    EXPECT_TRUE(is_finite(mu));
    EXPECT_GT(sd, 0.0);
    EXPECT_TRUE(is_finite(sr));
    EXPECT_GT(sr, 0.0);

    const std::vector<double> weights = {0.6, 0.4};
    const std::vector<double> asset_returns = {0.08, 0.04};
    const double port_ret = finance::portfolio_return(weights, asset_returns);
    EXPECT_NEAR(port_ret, 0.6 * 0.08 + 0.4 * 0.04, 1e-10);
}

// ---------------------------------------------------------------------------
// Graph → Control: MST / shortest path on network, finite control gains
// ---------------------------------------------------------------------------
TEST(Wave58Pipeline, Graph_Control_NetworkToTransferFunction) {
    graph::Graph G(5, false);
    G.add_edge(0, 1, 1.0);
    G.add_edge(1, 2, 2.0);
    G.add_edge(2, 3, 1.5);
    G.add_edge(3, 4, 1.0);
    G.add_edge(0, 2, 4.0);
    G.add_edge(1, 3, 5.0);

    const auto mst = graph::mst_kruskal(G);
    ASSERT_EQ(mst.size(), 4u);
    double mst_weight = 0.0;
    for (const auto& e : mst) {
        mst_weight += e.weight;
        EXPECT_TRUE(is_finite(e.weight));
    }
    EXPECT_NEAR(mst_weight, 5.5, 1e-10);

    const auto [dist, parent] = graph::dijkstra(G, 0);
    (void)parent;
    ASSERT_EQ(dist.size(), 5u);
    EXPECT_NEAR(dist[4], 5.5, 1e-10);
    for (double d : dist) {
        EXPECT_LT(d, graph::INF);
        EXPECT_TRUE(is_finite(d));
    }

    // Shortest-path weight parametrizes a stable first-order plant gain
    const auto plant = control::tf({dist[4]}, {1.0, 1.0});
    EXPECT_TRUE(control::is_stable(plant));
    EXPECT_NEAR(control::dcgain(plant), dist[4], 1e-10);

    const auto step = control::step_response(plant, 4.0, 50);
    ASSERT_EQ(step.y.size(), 50u);
    for (double y : step.y) {
        EXPECT_TRUE(is_finite(y));
    }
    EXPECT_NEAR(step.y.back(), dist[4], 0.15);
}

// ---------------------------------------------------------------------------
// Quantum → Info: |ψ|² probabilities feed Shannon entropy
// ---------------------------------------------------------------------------
TEST(Wave58Pipeline, Quantum_Info_AmplitudeEntropy) {
    const auto psi_raw = quantum::ket_superposition({0.6, 0.8});
    const auto psi = quantum::ket_normalise(psi_raw);

    const auto probs = probability_amplitudes_squared(psi);
    ASSERT_EQ(probs.size(), 2u);
    double prob_sum = 0.0;
    for (double p : probs) {
        EXPECT_GE(p, 0.0);
        prob_sum += p;
    }
    EXPECT_NEAR(prob_sum, 1.0, 1e-10);

    const double H_bits = info::entropy(probs, 2.0);
    EXPECT_TRUE(is_finite(H_bits));
    EXPECT_GT(H_bits, 0.0);
    EXPECT_LT(H_bits, 1.0);

    const auto rho = quantum::density_matrix(psi);
    const double S_vn = quantum::von_neumann_entropy(rho);
    EXPECT_TRUE(is_finite(S_vn));
    // Pure-state density matrix: von Neumann entropy should be ~0 (nats)
    EXPECT_NEAR(S_vn, 0.0, 1e-10);
    // Shannon entropy of measurement outcomes differs for non-diagonal pure states
    const double S_diag = -probs[0] * std::log(probs[0]) - probs[1] * std::log(probs[1]);
    EXPECT_GT(S_diag, 0.0);
    EXPECT_NEAR(S_diag, H_bits * std::log(2.0), 1e-6);

    const auto plus = quantum::op_apply(quantum::hadamard(), quantum::ket_basis(2, 0));
    const auto plus_probs = probability_amplitudes_squared(plus);
    const double H_plus = info::entropy(plus_probs, 2.0);
    EXPECT_NEAR(H_plus, 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Cplx contour smoke: Cauchy integral recovers analytic value
// ---------------------------------------------------------------------------
TEST(Wave58Pipeline, Cplx_ContourIntegralFinite) {
    const auto contour = unit_circle_contour(80);

    const auto f_const = [](cplx::C z) {
        (void)z;
        return cplx::C(2.0, -1.0);
    };
    const cplx::C cauchy_val = cplx::cauchy_integral(f_const, cplx::C(0.0, 0.0), contour, 60);
    EXPECT_TRUE(is_finite(cauchy_val.real()));
    EXPECT_TRUE(is_finite(cauchy_val.imag()));
    EXPECT_NEAR(cauchy_val.real(), 2.0, 0.15);
    EXPECT_NEAR(cauchy_val.imag(), -1.0, 0.15);

    const auto f_z = [](cplx::C z) { return z; };
    const cplx::C contour_val = cplx::contour_integral(f_z, contour, 40);
    EXPECT_TRUE(is_finite(contour_val.real()));
    EXPECT_TRUE(is_finite(contour_val.imag()));
    EXPECT_NEAR(std::abs(contour_val), 0.0, 0.2);
}
