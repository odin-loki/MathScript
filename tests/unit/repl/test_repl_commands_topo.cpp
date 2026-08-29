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

TEST(ReplCommandsTest, topo_euler_tetrahedron) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_euler_tetrahedron()");

    expect_ok(interp, "chi = topo_euler_tetrahedron()");
    EXPECT_NEAR(interp.state().scalars.at("chi"), 1.0, 1e-9);
    expect_contains(interp, "topo_euler_tetrahedron()", "1");
}

TEST(ReplCommandsTest, topo_euler_sphere_surface) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_euler_sphere_surface()");

    expect_ok(interp, "chi = topo_euler_sphere_surface()");
    EXPECT_NEAR(interp.state().scalars.at("chi"), 2.0, 1e-9);

    expect_contains(interp, "topo_euler_sphere_surface()", "2");
}

TEST(ReplCommandsTest, topo_vietoris_rips_betti0) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_vietoris_rips_betti0(D,r,max_dim)");

    expect_ok(interp, "Dtri = [0, 1, 1; 1, 0, 1; 1, 1, 0]");
    expect_ok(interp, "b0t = topo_vietoris_rips_betti0(Dtri, 1.1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("b0t"), 1.0, 1e-9);

    expect_ok(interp, "Dfar = [0, 10; 10, 0]");
    expect_ok(interp, "b0d = topo_vietoris_rips_betti0(Dfar, 1.0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("b0d"), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_betti_curve) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_betti_curve(D,thresholds,max_dim)");

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, topo_bottleneck_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_bottleneck_distance(dgm1,dgm2,dim)");

    expect_ok(interp, "dgm1 = [0, 0, 2; 1, 1, 3]");
    expect_ok(interp, "dgm2 = [0, 0.2, 2.2; 1, 1.2, 3.2]");
    expect_ok(interp, "bn = topo_bottleneck_distance(dgm1, dgm2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("bn"), 0.2, 0.05);
}

TEST(ReplCommandsTest, topo_persistence_diagram) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_persistence_diagram(S,births)");

    expect_ok(interp, "S = [0, -1; 1, -1; 0, 1]");
    expect_ok(interp, "births = [0; 0; 1]");
    expect_ok(interp, "pd = topo_persistence_diagram(S, births)");
    ASSERT_GT(interp.state().matrices.count("pd"), 0u);
    EXPECT_GT(interp.state().matrices.at("pd").rows(), 0u);
}

TEST(ReplCommandsTest, topo_wasserstein_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_wasserstein_distance(dgm1,dgm2,dim)");

    expect_ok(interp, "dgm = [0, 0, 2; 1, 1, 3]");
    expect_ok(interp, "ws = topo_wasserstein_distance(dgm, dgm, 0)");
    EXPECT_NEAR(interp.state().scalars.at("ws"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, topo_pairwise_distances) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_pairwise_distances(P)");

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
}

TEST(ReplCommandsTest, topo_complex) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_alpha_complex(P,alpha[,max_dim])");
    expect_contains(interp, "help", "topo_select_landmarks(P,n[,seed])");
    expect_contains(interp, "help", "topo_witness_complex(P,landmarks,eps[,max_dim])");
    expect_contains(interp, "help", "topo_persistence_landscape(dgm,n_layers,n_samples[,t_min,t_max])");

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
    EXPECT_GE(interp.state().matrices.at("pl")(0, 4), 0.0);
}

TEST(ReplCommandsTest, crypto_bloom_topo) {
    Interpreter interp;
    ms::izaac::clear_session();
    expect_contains(interp, "help", "crypto_from_hex");
    expect_contains(interp, "help", "bloom_hash_count");

    expect_ok(interp, "bytes = crypto_from_hex(\"00ff\")");
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "bloom_new(bf277, 50, 0.05)");
    expect_ok(interp, "bloom_bit_count(bf277)");

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    expect_ok(interp, "dim = topo_simplicial_dimension(vr)");
}

TEST(ReplCommandsTest, topo_quantum_cfd) {
    Interpreter interp;

    expect_ok(interp, "pts = [0, 0; 1, 0; 0, 1]");
    expect_ok(interp, "lm = topo_select_landmarks(pts, 2)");
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "rho = [1, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "r = quantum_partial_trace(rho, 2, 2, 1)");
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "f = [0, 1, 0; 0, 1, 0; 0, 1, 0]");
    expect_ok(interp, "f1 = cfd_upwind_step_2d(f, 1, 0, 0.05, 0.2, 0.2)");
    EXPECT_EQ(interp.state().matrices.at("f1").rows(), 3u);
}

TEST(ReplCommandsTest, image_topo_stats) {
    Interpreter interp;

    expect_ok(interp, "G = [10, 20, 30, 40; 50, 60, 70, 80]");
    expect_ok(interp, "B = imgaussfilt(G, 0.5)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 2u);

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "M = [1, 2, 3, 4, 5; 6, 7, 8, 9, 10; 11, 12, 13, 14, 15]");
    expect_ok(interp, "C = imcrop(M, 1, 1, 3, 4)");
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 2u);
}

TEST(ReplCommandsTest, cellai_topo) {
    Interpreter interp;

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_witness_landscape) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, watershed_pairwise) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
}

TEST(ReplCommandsTest, topo_complex_2) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, watershed_pairwise_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0.2, 0.3, 0.4, 0.5; 0.25, 0.35, 0.45, 0.55; 0.3, 0.4, 0.5, 0.6; 0.35, 0.45, 0.55, 0.65]");
    expect_ok(interp, "Mk = [0, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 0]");
    expect_ok(interp, "W = watershed(G, Mk)");
    EXPECT_EQ(interp.state().matrices.at("W").rows(), 4u);

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
}

TEST(ReplCommandsTest, topo_complex_3) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_2) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_2) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_4) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_3) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_2) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_5) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_2) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_4) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_3) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_6) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_3) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_5) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_4) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_7) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_4) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_6) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_5) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_8) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_5) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_7) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_6) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_9) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_6) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_8) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_7) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_10) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_7) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_9) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_8) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_11) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_8) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_10) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_9) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_12) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_9) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_11) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_10) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_13) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_10) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_12) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_11) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_14) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_11) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_13) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_12) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_15) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_12) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_14) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_13) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_16) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_13) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_15) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_14) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_17) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_14) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_16) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_15) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_18) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_15) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_17) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_16) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_19) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_16) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_18) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_17) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_20) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_17) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_19) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_18) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_21) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_18) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_20) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_19) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_22) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_19) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_21) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_20) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_23) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_20) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_22) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_21) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_24) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_21) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_23) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_22) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_25) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_22) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_24) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_23) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_26) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_23) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_25) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_24) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_27) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_24) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_26) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_25) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_28) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_25) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_27) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_26) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_29) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_26) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_28) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_27) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_30) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_27) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_29) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_28) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_31) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_28) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_30) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_29) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_32) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_29) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_31) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_30) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, topo_complex_33) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1, 1.414; 1, 0, 1; 1.414, 1, 0]");
    expect_ok(interp, "cech = topo_cech_complex(D, 1.0)");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "betti = topo_simplicial_betti(vr)");
    ASSERT_GT(interp.state().matrices.count("cech"), 0u);
    ASSERT_GT(interp.state().matrices.count("vr"), 0u);
    ASSERT_GT(interp.state().matrices.count("betti"), 0u);
}

TEST(ReplCommandsTest, topo_alpha_landscape_30) {
    Interpreter interp;

    expect_ok(interp, "P = [0, 0; 1, 0; 0, 1; 0.5, 0.5]");
    expect_ok(interp, "ac = topo_alpha_complex(P, 1.0, 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    EXPECT_GE(interp.state().matrices.at("ac").rows(), 4u);

    expect_ok(interp, "lm = topo_select_landmarks(P, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("lm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lm").rows(), 2u);

    expect_ok(interp, "wc = topo_witness_complex(P, lm, 1.5, 2)");
    ASSERT_GT(interp.state().matrices.count("wc"), 0u);
    EXPECT_GE(interp.state().matrices.at("wc").rows(), 2u);

    expect_ok(interp, "dgm = [0, 0, 1; 1, 0.2, 0.8]");
    expect_ok(interp, "pl = topo_persistence_landscape(dgm, 2, 8)");
    ASSERT_GT(interp.state().matrices.count("pl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("pl").cols(), 8u);
}

TEST(ReplCommandsTest, huffman_lz77_betti_32) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("HE"), 0u);
    ASSERT_GT(interp.state().matrices.at("HE").rows(), 0u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    ASSERT_GT(interp.state().matrices.count("bc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, pairwise_nextperm_31) {
    Interpreter interp;

    expect_ok(interp, "D = topo_pairwise_distances([0, 0; 1, 0; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 1.0, 1e-9);

    expect_ok(interp, "np = combo_next_perm([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("np"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("np")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("np")(2, 0), 2.0, 1e-9);
}
