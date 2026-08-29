#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, info_kl_divergence) {
    Interpreter interp;
    expect_contains(interp, "help", "info_kl_divergence(p,q)");

    expect_ok(interp, "p = [0.3; 0.4; 0.3]");
    expect_ok(interp, "kl0 = info_kl_divergence(p, p)");
    EXPECT_NEAR(interp.state().scalars.at("kl0"), 0.0, 1e-9);

    expect_ok(interp, "kl = info_kl_divergence([0.4; 0.6], [0.5; 0.5])");
    EXPECT_GT(interp.state().scalars.at("kl"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("kl"), 0.029049, 1e-4);

    expect_contains(interp, "info_kl_divergence([0.4, 0.6], [0.5, 0.5])", "0.029");
}

TEST(ReplCommandsTest, info_cross_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_cross_entropy(p,q)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "q = [0.25; 0.75]");
    expect_ok(interp, "ce = info_cross_entropy(p, q)");
    EXPECT_GT(interp.state().scalars.at("ce"), 0.0);

    expect_contains(interp, "info_cross_entropy([0.5, 0.5], [0.25, 0.75])", "1.");
}

TEST(ReplCommandsTest, info_mutual_info) {
    Interpreter interp;
    expect_contains(interp, "help", "info_mutual_info(joint)");

    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "mi = info_mutual_info(J)");
    EXPECT_NEAR(interp.state().scalars.at("mi"), 0.0, 1e-9);

    expect_contains(interp, "info_mutual_info([0.25, 0; 0, 0.25])", "1");
}

TEST(ReplCommandsTest, info_js_divergence) {
    Interpreter interp;
    expect_contains(interp, "help", "info_js_divergence(p,q)");

    expect_ok(interp, "p = [1; 0]");
    expect_ok(interp, "q = [0; 1]");
    expect_ok(interp, "js = info_js_divergence(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("js"), 1.0, 1e-9);

    expect_contains(interp, "info_js_divergence([1, 0], [0, 1])", "1");
}

TEST(ReplCommandsTest, info_tv_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "info_tv_distance(p,q)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "q = [1; 0]");
    expect_ok(interp, "tv = info_tv_distance(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("tv"), 0.5, 1e-9);

    expect_contains(interp, "info_tv_distance([0.5; 0.5], [1; 0])", "0.5");
}

TEST(ReplCommandsTest, info_hellinger_dist) {
    Interpreter interp;
    expect_contains(interp, "help", "info_hellinger_dist(p,q)");

    expect_ok(interp, "p = [1; 0]");
    expect_ok(interp, "q = [0; 1]");
    expect_ok(interp, "hd = info_hellinger_dist(p, q)");
    EXPECT_NEAR(interp.state().scalars.at("hd"), 1.0, 1e-9);

    expect_contains(interp, "info_hellinger_dist([1; 0], [0; 1])", "1");
}

TEST(ReplCommandsTest, info_renyi_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_renyi_entropy(alpha,p)");

    expect_ok(interp, "p = [0.25; 0.25; 0.25; 0.25]");
    expect_ok(interp, "r2 = info_renyi_entropy(2.0, p)");
    EXPECT_NEAR(interp.state().scalars.at("r2"), 2.0, 1e-9);

    expect_contains(interp, "info_renyi_entropy(2.0, [0.25; 0.25; 0.25; 0.25])", "2");
}

TEST(ReplCommandsTest, info_redundancy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_redundancy(p)");

    expect_ok(interp, "p = [0.9; 0.1]");
    expect_ok(interp, "red = info_redundancy(p)");
    EXPECT_GT(interp.state().scalars.at("red"), 0.0);

    expect_contains(interp, "info_redundancy([0.9; 0.1])", "0.5");
}

TEST(ReplCommandsTest, info_efficiency) {
    Interpreter interp;
    expect_contains(interp, "help", "info_efficiency(p)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "eff = info_efficiency(p)");
    EXPECT_NEAR(interp.state().scalars.at("eff"), 1.0, 1e-9);

    expect_contains(interp, "info_efficiency([0.5; 0.5])", "1");
}

TEST(ReplCommandsTest, info_channel_capacity_bsc) {
    Interpreter interp;
    expect_contains(interp, "help", "info_channel_capacity_bsc(p_error)");

    expect_ok(interp, "cap = info_channel_capacity_bsc(0)");
    EXPECT_NEAR(interp.state().scalars.at("cap"), 1.0, 1e-9);

    expect_contains(interp, "info_channel_capacity_bsc(0)", "1");
}

TEST(ReplCommandsTest, info_channel_capacity_bec) {
    Interpreter interp;
    expect_contains(interp, "help", "info_channel_capacity_bec(epsilon)");

    expect_ok(interp, "capbec = info_channel_capacity_bec(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("capbec"), 0.5, 1e-9);

    expect_contains(interp, "info_channel_capacity_bec(0.5)", "0.5");
}

TEST(ReplCommandsTest, info_shannon_hartley) {
    Interpreter interp;
    expect_contains(interp, "help", "info_shannon_hartley(bandwidth_hz,snr_linear)");

    expect_ok(interp, "sh = info_shannon_hartley(1000000, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sh"), 1e6, 1.0);

    expect_contains(interp, "info_shannon_hartley(1000000, 1)", "1000000");
}

TEST(ReplCommandsTest, info_differential_entropy_gaussian) {
    Interpreter interp;
    expect_contains(interp, "help", "info_differential_entropy_gaussian(sigma)");

    expect_ok(interp, "hgauss = info_differential_entropy_gaussian(1)");
    EXPECT_NEAR(interp.state().scalars.at("hgauss"), 1.4189385332046727, 1e-3);

    expect_contains(interp, "info_differential_entropy_gaussian(1)", "1.418");
}

TEST(ReplCommandsTest, info_differential_entropy_uniform) {
    Interpreter interp;
    expect_contains(interp, "help", "info_differential_entropy_uniform(a,b)");

    expect_ok(interp, "hu = info_differential_entropy_uniform(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hu"), 0.0, 1e-9);

    expect_contains(interp, "info_differential_entropy_uniform(0, 1)", "0");
}

TEST(ReplCommandsTest, info_rate_distortion_gaussian) {
    Interpreter interp;
    expect_contains(interp, "help", "info_rate_distortion_gaussian(variance,distortion)");

    expect_ok(interp, "rd = info_rate_distortion_gaussian(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("rd"), 1.0, 1e-9);

    expect_contains(interp, "info_rate_distortion_gaussian(1, 0.25)", "1");
}

TEST(ReplCommandsTest, info_source_coding_rate) {
    Interpreter interp;
    expect_contains(interp, "help", "info_source_coding_rate(p)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "scr = info_source_coding_rate(p)");
    EXPECT_NEAR(interp.state().scalars.at("scr"), 1.0, 1e-9);

    expect_contains(interp, "info_source_coding_rate([0.5; 0.5])", "1");
}

TEST(ReplCommandsTest, info_tsallis_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_tsallis_entropy(q,p)");

    expect_ok(interp, "p = [0.5; 0.5]");
    expect_ok(interp, "ts = info_tsallis_entropy(2.0, p)");
    EXPECT_NEAR(interp.state().scalars.at("ts"), 0.5, 1e-9);

    expect_contains(interp, "info_tsallis_entropy(2.0, [0.5; 0.5])", "0.5");
}

TEST(ReplCommandsTest, info_joint_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_joint_entropy(joint,rows,cols)");

    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "hj = info_joint_entropy(J, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("hj"), 2.0, 1e-9);

    expect_contains(interp, "info_joint_entropy([0.25, 0.25; 0.25, 0.25], 2, 2)", "2");
}

TEST(ReplCommandsTest, info_conditional_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_conditional_entropy(joint,rows,cols)");

    expect_ok(interp, "J = [0.25, 0.25; 0.25, 0.25]");
    expect_ok(interp, "hc = info_conditional_entropy(J, 2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("hc"), 1.0, 1e-9);

    expect_contains(interp, "info_conditional_entropy([0.25, 0.25; 0.25, 0.25], 2, 2)", "1");
}

TEST(ReplCommandsTest, info_sample_entropy) {
    Interpreter interp;
    expect_contains(interp, "help", "info_sample_entropy(x,m,r)");

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "se = info_sample_entropy(x, 2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("se"), 0.0, 1e-9);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("se")));

    expect_contains(interp, "info_sample_entropy([1; 2; 3; 4; 5], 2, 0.5)", "0");
}

TEST(ReplCommandsTest, info_lz_complexity) {
    Interpreter interp;
    expect_contains(interp, "help", "info_lz_complexity(seq)");

    expect_ok(interp, "seq = [0; 1; 0; 1; 1; 0]");
    expect_ok(interp, "lz = info_lz_complexity(seq)");
    EXPECT_GE(interp.state().scalars.at("lz"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("lz")));

    expect_contains(interp, "info_lz_complexity([0; 1; 0; 1; 1; 0])", "1.");
}

TEST(ReplCommandsTest, info_channel_capacity) {
    Interpreter interp;
    expect_contains(interp, "help", "info_blahut_arimoto(W)");
    expect_contains(interp, "help", "info_channel_capacity(W)");

    expect_ok(interp, "W = [1, 0; 0, 1]");
    expect_ok(interp, "c = info_blahut_arimoto(W)");
    EXPECT_NEAR(interp.state().scalars.at("c"), 1.0, 1e-6);
    expect_contains(interp, "info_blahut_arimoto([1, 0; 0, 1])", "1");

    expect_ok(interp, "Wbsc = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "cap = info_channel_capacity(Wbsc)");
    const double p = 0.2;
    const double h = -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
    EXPECT_NEAR(interp.state().scalars.at("cap"), 1.0 - h, 1e-4);
    expect_contains(interp, "info_channel_capacity([0.8, 0.2; 0.2, 0.8])", "0.278");
}

TEST(ReplCommandsTest, graph_info) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_connected_components(A)");
    expect_contains(interp, "help", "graph_is_planar(A)");
    expect_contains(interp, "help", "info_normalized_entropy(p)");
    expect_contains(interp, "help", "info_channel_capacity_input(W)");

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    ASSERT_GT(interp.state().matrices.count("cc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "P = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "pl = graph_is_planar(P)");
    EXPECT_NEAR(interp.state().scalars.at("pl"), 1.0, 1e-9);
    expect_contains(interp, "graph_is_planar([0,1,0; 1,0,1; 0,1,0])", "1");

    expect_ok(interp, "h = info_normalized_entropy([0.25; 0.25; 0.25; 0.25])");
    EXPECT_NEAR(interp.state().scalars.at("h"), 1.0, 1e-6);
    expect_contains(interp, "info_normalized_entropy([0.25; 0.25; 0.25; 0.25])", "1");

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    ASSERT_GT(interp.state().matrices.count("pin"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("pin")(0, 0), 0.5, 1e-3);
    EXPECT_NEAR(interp.state().matrices.at("pin")(1, 0), 0.5, 1e-3);
}

TEST(ReplCommandsTest, graph_info_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_info_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_2) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_3) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_4) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_5) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_6) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_7) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_8) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_9) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_10) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_11) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_12) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_13) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_14) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_15) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_16) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_17) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_18) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_19) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_20) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_21) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_22) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_23) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_24) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_25) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_26) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_27) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_28) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_29) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_30) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, graph_connected_components_info_channel_capacity_input_31) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0,0; 1,0,0,0; 0,0,0,1; 0,0,1,0]");
    expect_ok(interp, "cc = graph_connected_components(A)");
    EXPECT_EQ(interp.state().matrices.at("cc").rows(), 2u);

    expect_ok(interp, "W = [0.8, 0.2; 0.2, 0.8]");
    expect_ok(interp, "pin = info_channel_capacity_input(W)");
    EXPECT_EQ(interp.state().matrices.at("pin").rows(), 2u);
}

TEST(ReplCommandsTest, info_permutation_entropy) {
    Interpreter interp;

    expect_ok(interp, "info_permutation_entropy([1;2;3;4;5])");
    expect_contains(interp, "info_permutation_entropy([1; 2; 3; 4; 5], 3, 1)", "0");

    expect_error_contains(interp, "info_permutation_entropy()",
                          "expected info_permutation_entropy(x[, order[, delay]])");
    expect_error_contains(interp, "info_permutation_entropy([1;2;3;4;5], 0)",
                          "expected positive integer order");
    expect_error_contains(interp, "info_permutation_entropy([1;2;3;4;5], 3, 0)",
                          "expected positive integer delay");
    expect_error_contains(interp, "info_permutation_entropy([1;2;3;4;5], notnum)",
                          "expected info_permutation_entropy(x[, order[, delay]])");
    expect_error_contains(interp, "info_permutation_entropy(missing)", "unknown matrix");
    expect_error_contains(interp, "pe = info_permutation_entropy()",
                          "expected info_permutation_entropy(x[, order[, delay]])");
    expect_error_contains(interp, "pe = info_permutation_entropy([1;2;3;4;5], 0)",
                          "expected positive integer order");
    expect_error_contains(interp, "pe = info_permutation_entropy([1;2;3;4;5], 1.5)",
                          "expected positive integer order");
}

TEST(ReplCommandsTest, info_sample_entropy_errors) {
    Interpreter interp;

    expect_error_contains(interp, "info_sample_entropy(missing, 2, 0.5)", "unknown matrix");
    expect_error_contains(interp, "info_sample_entropy([1; 2; 3; 4; 5], 0, 0.5)",
                          "expected positive integer m");
    expect_error_contains(interp, "info_sample_entropy([1; 2; 3; 4; 5], 1.5, 0.5)",
                          "expected positive integer m");
    expect_error_contains(interp, "info_sample_entropy([1; 2; 3; 4; 5], notnum, 0.5)",
                          "expected info_sample_entropy(x, m, r)");
    expect_error_contains(interp, "se = info_sample_entropy()",
                          "expected info_sample_entropy(x, m, r)");
    expect_error_contains(interp, "se = info_sample_entropy(missing, 2, 0.5)", "unknown matrix");
    expect_error_contains(interp, "se = info_sample_entropy([1; 2; 3; 4; 5], 0, 0.5)",
                          "expected positive integer m");
}

TEST(ReplCommandsTest, info_transfer_entropy_errors) {
    Interpreter interp;

    expect_ok(interp, "info_transfer_entropy([1; 2; 3; 4; 5], [2; 3; 4; 5; 6])");

    expect_error_contains(interp, "info_transfer_entropy()",
                          "expected info_transfer_entropy(x, y[, bins[, lag]])");
    expect_error_contains(interp, "info_transfer_entropy(missing, [1; 2; 3; 4; 5])",
                          "unknown matrix");
    expect_error_contains(interp, "info_transfer_entropy([1; 2; 3; 4; 5], [2; 3; 4; 5; 6], 0)",
                          "expected positive integer bins");
    expect_error_contains(interp, "info_transfer_entropy([1; 2; 3; 4; 5], [2; 3; 4; 5; 6], 8, 0)",
                          "expected positive integer lag");
    expect_error_contains(interp, "info_transfer_entropy([1; 2; 3; 4; 5], [2; 3; 4; 5; 6], notnum)",
                          "expected info_transfer_entropy(x, y[, bins[, lag]])");
    expect_error_contains(interp, "te = info_transfer_entropy([1; 2; 3; 4; 5])",
                          "expected info_transfer_entropy(x, y[, bins[, lag]])");
    expect_error_contains(interp, "te = info_transfer_entropy(missing, [1; 2; 3; 4; 5])",
                          "unknown matrix");
}

TEST(ReplCommandsTest, info_channel_capacity_input_noassign) {
    Interpreter interp;
    expect_contains(interp, "info_channel_capacity_input([0.8, 0.2; 0.2, 0.8])", "p_in");
    expect_error_contains(interp, "info_channel_capacity_input(no_such_matrix)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, info_entropy_noassign) {
    Interpreter interp;
    expect_contains(interp, "info_entropy([0.5; 0.5])", "1");
    expect_error_contains(interp, "info_entropy(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, info_redundancy_noassign) {
    Interpreter interp;
    expect_contains(interp, "info_redundancy([0.9; 0.1])", "0.5");
    expect_error_contains(interp, "info_redundancy(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, info_efficiency_noassign) {
    Interpreter interp;
    expect_contains(interp, "info_efficiency([0.5; 0.5])", "1");
    expect_error_contains(interp, "info_efficiency(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, info_channel_capacity_noassign) {
    Interpreter interp;
    expect_contains(interp, "info_channel_capacity([0.8, 0.2; 0.2, 0.8])", "0.278");
    expect_error_contains(interp, "info_channel_capacity(no_such_matrix)", "unknown matrix");
}

TEST(ReplCommandsTest, info_normalized_entropy_noassign) {
    Interpreter interp;
    expect_contains(interp, "info_normalized_entropy([0.5; 0.5])", "1");
    expect_error_contains(interp, "info_normalized_entropy(no_such_matrix)", "unknown matrix");
}
