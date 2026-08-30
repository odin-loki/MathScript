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

TEST(ReplCommandsTest, gria_gf2n_lfsr) {
    Interpreter interp;
    expect_contains(interp, "help", "gria_gf2n_mul(a, b, poly)");
    expect_contains(interp, "help", "gria_gf2n_pow(a, exp, poly)");
    expect_contains(interp, "help", "gria_gf2n_inv(a, poly)");
    expect_contains(interp, "help", "gria_lfsr_step(state, poly)");
    expect_contains(interp, "help", "gria_alpha_lfsr(poly, steps)");
    expect_contains(interp, "help", "gria_lfsr_is_maximal(poly, n)");

    expect_ok(interp, "sq = gria_gf2n_pow(3, 2, 283)");
    expect_ok(interp, "m = gria_gf2n_mul(3, 3, 283)");
    EXPECT_EQ(interp.state().scalars.at("sq"), interp.state().scalars.at("m"));

    expect_ok(interp, "p = gria_gf2n_mul(3, 5, 283)");
    EXPECT_GT(interp.state().scalars.at("p"), 0.0);

    expect_ok(interp, "inv3 = gria_gf2n_inv(3, 283)");
    expect_ok(interp, "one = gria_gf2n_mul(3, inv3, 283)");
    EXPECT_EQ(interp.state().scalars.at("one"), 1.0);

    expect_ok(interp, "hx = gria_gf2n_mul(0x53, 0xCA, 0x11B)");
    EXPECT_GT(interp.state().scalars.at("hx"), 0.0);

    expect_ok(interp, "next = gria_lfsr_step(5, 11)");
    EXPECT_EQ(interp.state().scalars.at("next"), 9.0);

    expect_contains(interp, "gria_lfsr_is_maximal(6, 3)", "true");
    expect_contains(interp, "gria_lfsr_is_maximal(3, 4)", "false");

    expect_ok(interp, "alpha = gria_alpha_lfsr(184, 100)");
    EXPECT_GE(interp.state().scalars.at("alpha"), 0.0);
    EXPECT_LE(interp.state().scalars.at("alpha"), 1.0);
}

TEST(ReplCommandsTest, gria_ca) {
    Interpreter interp;
    expect_contains(interp, "help", "gria_ca_step(state,rule)");
    expect_contains(interp, "help", "gria_langton_lambda(rule)");
    expect_contains(interp, "help", "gria_alpha_ca(rule,steps,width)");
    expect_contains(interp, "help", "gria_hamming_distance(a,b)");
    expect_contains(interp, "help", "gria_divergence_trajectory(a,b,rule,n_steps)");
    expect_contains(interp, "help", "gria_settling_time(a,b,rule,n_steps)");

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("s1")(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("s1")(3, 0), 1.0, 1e-9);

    expect_ok(interp, "lam = gria_langton_lambda(110)");
    EXPECT_GE(interp.state().scalars.at("lam"), 0.0);
    EXPECT_LE(interp.state().scalars.at("lam"), 1.0);

    expect_ok(interp, "ac = gria_alpha_ca(110, 10, 20)");
    EXPECT_GE(interp.state().scalars.at("ac"), 0.0);
    EXPECT_LE(interp.state().scalars.at("ac"), 1.0);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "hd = gria_hamming_distance(a, b)");
    EXPECT_NEAR(interp.state().scalars.at("hd"), 3.0, 1e-9);

    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("dt")(0, 0), 3.0, 1e-9);

    expect_ok(interp, "st = gria_settling_time(a, a, 110, 10)");
    EXPECT_NEAR(interp.state().scalars.at("st"), 0.0, 1e-9);

    expect_ok(interp, "st_none = gria_settling_time(a, b, 110, 0)");
    EXPECT_NEAR(interp.state().scalars.at("st_none"), -1.0, 1e-9);
}

TEST(ReplCommandsTest, cellai_ext) {
    Interpreter interp;
    expect_contains(interp, "help", "cellai_boltzmann_weights");
    expect_contains(interp, "help", "cellai_cell_to_cypha_features");

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    const auto& bw = interp.state().matrices.at("bw");
    EXPECT_EQ(bw.rows(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(bw(i, 0), 0.2, 1e-9);
    }

    expect_ok(interp, "E = [5; 1; -3; 10]");
    expect_ok(interp, "bw2 = cellai_boltzmann_weights(E, 1)");
    const std::vector<double> energies_ref{5.0, 1.0, -3.0, 10.0};
    const auto ref_weights = ms::cellai::boltzmann_weights(energies_ref, 1.0);
    const auto& bw2 = interp.state().matrices.at("bw2");
    ASSERT_EQ(bw2.rows(), ref_weights.size());
    for (size_t i = 0; i < ref_weights.size(); ++i) {
        EXPECT_NEAR(bw2(i, 0), ref_weights[i], 1e-9);
    }
    EXPECT_GT(bw2(2, 0), bw2(1, 0));
    EXPECT_GT(bw2(1, 0), bw2(0, 0));

    const auto boltz_out = interp.execute("cellai_boltzmann_weights([0, 1, 2], 0.5)");
    ASSERT_TRUE(boltz_out.has_value());
    EXPECT_NE(boltz_out->find("weights ="), std::string::npos);

    expect_ok(interp, "cellmemory_new(cm, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm, [1;0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cf").rows(), 3u);

    ms::cellai::CellMemory memory(2, 4, {0.1, 1.0, 10.0});
    ASSERT_TRUE(memory.step(ms::Matrix<double>{{1.0}, {0.0}}).has_value());
    const auto ref_features =
        ms::cellai::cell_to_cypha_features(memory, {0.1, 1.0, 5.0});
    const auto& cf = interp.state().matrices.at("cf");
    ASSERT_EQ(cf.rows(), ref_features.rows());
    for (size_t i = 0; i < cf.rows(); ++i) {
        EXPECT_NEAR(cf(i, 0), ref_features(i, 0), 1e-9);
    }

    const auto cypha_out =
        interp.execute("cellai_cell_to_cypha_features(cm, [0.1, 1, 5])");
    ASSERT_TRUE(cypha_out.has_value());
    EXPECT_NE(cypha_out->find("features ="), std::string::npos);

    expect_error(interp, "cellai_boltzmann_weights()");
    expect_error(interp, "cellai_cell_to_cypha_features(missing, [1])");
}

TEST(ReplCommandsTest, axiom_matrix) {
    Interpreter interp;
    expect_contains(interp, "help", "axiom_evaluate");
    expect_contains(interp, "help", "axiom_mse_fitness");
    expect_contains(interp, "help", "axiom_rmse_fitness");

    expect_ok(interp, "Y = axiom_evaluate(\"x0\", [1;2;3])");
    ASSERT_GT(interp.state().matrices.count("Y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Y")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Y")(1, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Y")(2, 0), 3.0, 1e-9);

    expect_ok(interp, "mse = axiom_mse_fitness(\"x0\", [1;2;3], [1,2,3])");
    EXPECT_NEAR(interp.state().scalars.at("mse"), 0.0, 1e-9);

    expect_ok(interp, "rmse = axiom_rmse_fitness(\"x0\", [1;2;3], [1,2,3])");
    EXPECT_NEAR(interp.state().scalars.at("rmse"), 0.0, 1e-9);

    expect_ok(interp, "mse2 = axiom_mse_fitness(\"x0\", [1;2;3;4], [4,5,6,7])");
    EXPECT_NEAR(interp.state().scalars.at("mse2"), 9.0, 1e-6);

    expect_ok(interp, "rmse2 = axiom_rmse_fitness(\"x0\", [1;2;3;4], [4,5,6,7])");
    EXPECT_NEAR(interp.state().scalars.at("rmse2"), 3.0, 1e-6);

    expect_contains(interp, "axiom_mse_fitness(\"x0\", [1;2], [1,2])", "0");
}

TEST(ReplCommandsTest, gria_gf2n_field) {
    Interpreter interp;
    expect_contains(interp, "help", "gria_gf2n_generate_field(n)");

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
    const auto ref = ms::gria::gf2n::generate_field(4);
    ASSERT_EQ(interp.state().matrices.at("fld").rows(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_EQ(static_cast<uint64_t>(interp.state().matrices.at("fld")(i, 0)), ref[i]);
    }
}

TEST(ReplCommandsTest, izaac_mpc_backtest) {
    Interpreter interp;
    expect_ok(interp, "izaac seed 42");
    expect_contains(interp, "help", "izaac_exponential_mechanism");

    expect_ok(interp, "idx = izaac_exponential_mechanism([1, 2, 3], 1, 1)");
    EXPECT_GE(interp.state().scalars.at("idx"), 0.0);
    EXPECT_LE(interp.state().scalars.at("idx"), 2.0);

    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_EQ(interp.state().matrices.at("sh").cols(), 3u);
    expect_ok(interp, "sec = mpc_reconstruct(sh)");
    EXPECT_NEAR(interp.state().scalars.at("sec"), 42.0, 1e-9);

    expect_ok(interp, "path = simulate_gbm_path(100, 0.05, 0.2, 0.01, 20)");
    EXPECT_EQ(interp.state().matrices.at("path").rows(), 21u);

    expect_ok(interp, "prices = [100; 101; 102; 101]");
    expect_ok(interp, "pos = [1; 1; 1; 0]");
    expect_ok(interp, "bt = run_backtest(prices, pos, 10000)");
    ASSERT_EQ(interp.state().matrices.at("bt").cols(), 4u);

    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, axiom_gria_dispatch) {
    Interpreter interp;
    expect_contains(interp, "help", "axiom_gria_fitness");
    expect_contains(interp, "help", "axiom_evolve(data");

    expect_ok(interp, "fit = axiom_gria_fitness(\"x0\", [1, 2; 3, 4])");
    EXPECT_GE(interp.state().scalars.at("fit"), 0.0);
    EXPECT_LE(interp.state().scalars.at("fit"), 1.0);

    expect_ok(interp, "best = axiom_evolve([1, 0; 0, 1], 8, 5)");
    EXPECT_GT(interp.state().scalars.at("best"), 0.0);

    expect_ok(interp, "a = gria_dispatch_hint_register(\"wave272_op\", 0.37)");
    EXPECT_NEAR(interp.state().scalars.at("a"), 0.37, 1e-12);
    expect_ok(interp, "ha = gria_dispatch_hint_alpha(\"wave272_op\")");
    EXPECT_NEAR(interp.state().scalars.at("ha"), 0.37, 1e-12);
}

TEST(ReplCommandsTest, cypha_nig_moments) {
    Interpreter interp;
    expect_contains(interp, "help", "cypha_nig_mean");
    expect_contains(interp, "help", "cypha_nig_variance");

    expect_ok(interp, "cypha_nig_mean(0.5, 1.2, 0.3, 0.8)");
    expect_ok(interp, "cypha_nig_variance(0.5, 1.2, 0.3, 0.8)");
}

TEST(ReplCommandsTest, izaac_vrf_crypto_randn) {
    Interpreter interp;
    expect_contains(interp, "help", "izaac_vrf_prove");
    expect_contains(interp, "help", "izaac_encrypt");
    expect_contains(interp, "help", "izaac_randn_matrix");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream pub_cmd;
    pub_cmd << "pub = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            pub_cmd << ", ";
        }
        pub_cmd << vrf(1, j);
    }
    pub_cmd << "]";
    expect_ok(interp, pub_cmd.str());
    expect_ok(interp, "vok = izaac_vrf_verify(pub, msg, proof)");
    EXPECT_NEAR(interp.state().scalars.at("vok"), 1.0, 1e-12);

    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);

    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, izaac_rand_matrix) {
    Interpreter interp;
    expect_contains(interp, "help", "izaac_rand_matrix");
    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("rm").cols(), 2u);
}

TEST(ReplCommandsTest, cell_backtest_special) {
    Interpreter interp;
    ms::izaac::clear_session();
    expect_contains(interp, "help", "cellmemory_memory_dim");
    expect_contains(interp, "help", "run_backtest_max_drawdown");

    expect_ok(interp, "cellmemory_new(cm, 2, 2, [1, 5])");
    expect_ok(interp, "cellmemory_input_dim(cm)");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm)");
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 2u);

    expect_ok(interp, "p = [10, 11, 12]");
    expect_ok(interp, "q = [0, 1, 0]");
    expect_ok(interp, "sh = run_backtest_sharpe(p, q, 1000)");
    expect_ok(interp, "sj = spherical_jn(1, 2)");
    EXPECT_TRUE(interp.state().scalars.count("sh") > 0);
    EXPECT_TRUE(interp.state().scalars.count("sj") > 0);
}

TEST(ReplCommandsTest, backtest_cell_energy) {
    Interpreter interp;
    ms::izaac::clear_session();
    expect_contains(interp, "help", "run_backtest_total_return");

    expect_ok(interp, "p = [10, 11, 12]");
    expect_ok(interp, "q = [0, 1, 0]");
    expect_ok(interp, "tr = run_backtest_total_return(p, q, 100)");

    expect_ok(interp, "cellmemory_new(cm, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm, 1)");
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "W = [0, 0; 0, 0]");
    expect_ok(interp, "e = cellai_energy(W, [0; 0], [0; 0])");
    EXPECT_TRUE(interp.state().scalars.count("e") > 0);
}

TEST(ReplCommandsTest, gria_quantum) {
    Interpreter interp;

    expect_ok(interp, "s = [1; 1; 0; 0]");
    expect_ok(interp, "s2 = gria_ca_step(s, 30)");
    EXPECT_EQ(interp.state().matrices.at("s2").rows(), 4u);

    expect_ok(interp, "lam = gria_langton_lambda(110)");
    EXPECT_TRUE(interp.state().scalars.count("lam") > 0);

    expect_ok(interp, "H = [2, 1; 1, 2]");
    expect_ok(interp, "ev = quantum_eigenspectrum(H)");
    EXPECT_EQ(interp.state().matrices.at("ev").rows(), 2u);
}

TEST(ReplCommandsTest, fem_cfd_cell) {
    Interpreter interp;
    ms::izaac::clear_session();

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);

    expect_ok(interp, "cellmemory_new(cm, 2, 2)");
    expect_ok(interp, "cellmemory_reset(cm)");
}

TEST(ReplCommandsTest, izaac_vrf_crypto_randn_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.at("mut").rows(), 0u);

    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);

    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_gria) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm332, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm332, 1)");
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm334, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm334, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm334, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, vrf_fuzz) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
}

TEST(ReplCommandsTest, vrf_prove) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);
}

TEST(ReplCommandsTest, encrypt_decrypt) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, randn) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmem_ptrace) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm489, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm489)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, recall_gria) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm491, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm491, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, boltzmann_cypha) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm493, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm493, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm493, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, vrf_fuzz_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
}

TEST(ReplCommandsTest, vrf_encrypt) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, randn_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmem_ptrace_2) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm687, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm687)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, recall_gria_2) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm689, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm689, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, boltzmann_cypha_2) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm691, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm691, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm691, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_2) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_2) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_2) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_2) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_3) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_3) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_3) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_3) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_3) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_3) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_4) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_4) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_4) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_4) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_4) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_4) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_5) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_5) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_5) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_5) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_5) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_5) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_6) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_6) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_6) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_6) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_6) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_6) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_7) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_7) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_7) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_7) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_7) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_7) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_8) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_8) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_8) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_8) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_8) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_8) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_9) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_9) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_9) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_9) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_9) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_9) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_10) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_10) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_10) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_10) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_10) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_10) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_11) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_11) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_11) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_11) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_11) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_11) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_12) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_12) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_12) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_12) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_12) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_12) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_13) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_13) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_13) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_13) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_13) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_13) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_14) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_14) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_14) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_14) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_14) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_14) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_15) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_15) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_15) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_15) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_15) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_15) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_16) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_16) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_16) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_16) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_16) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_16) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_17) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_17) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_17) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_17) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_17) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_17) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_18) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_18) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_18) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_18) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_18) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_18) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_19) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_19) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_19) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_19) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_19) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_19) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_20) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_20) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_20) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_20) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_20) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_20) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_21) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_21) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_21) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_21) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_21) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_21) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_22) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_22) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_22) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_22) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_22) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_22) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_23) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_23) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_23) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_23) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_23) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_23) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_24) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_24) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_24) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_24) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_24) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_24) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_25) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_25) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_25) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_25) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_25) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_25) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_26) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_26) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_26) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_26) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_26) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_26) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_27) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_27) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_27) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_27) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_27) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_27) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_28) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_28) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_28) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_28) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_28) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_28) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_29) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_29) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_29) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_29) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_29) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_29) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_30) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_30) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_keygen_izaac_fuzz_mutate_30) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    ASSERT_GT(interp.state().matrices.count("vrf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.count("mut"), 0u);
    EXPECT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(ReplCommandsTest, izaac_vrf_prove_izaac_encrypt_izaac_decrypt_30) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    ASSERT_GT(interp.state().matrices.count("pt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(ReplCommandsTest, izaac_randn_matrix_30) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    ASSERT_GT(interp.state().matrices.count("rn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(ReplCommandsTest, cellmemory_ptrace_30) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm890, 2, 3, [0.5, 2])");
    expect_ok(interp, "lt = cellmemory_long_term_state(cm890)");
    ASSERT_GT(interp.state().matrices.count("lt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lt").rows(), 3u);

    expect_ok(interp, "rho = [0.25, 0, 0, 0.25; 0, 0, 0, 0; 0, 0, 0, 0; 0.25, 0, 0, 0.25]");
    expect_ok(interp, "ptr = quantum_partial_trace(rho, 2, 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ptr"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ptr").rows(), 2u);
}

TEST(ReplCommandsTest, cellmemory_gria_31) {
    Interpreter interp;

    expect_ok(interp, "cellmemory_new(cm892, 2, 2)");
    expect_ok(interp, "r = cellmemory_recall(cm892, 1)");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);
    EXPECT_EQ(interp.state().matrices.at("r").rows(), 2u);

    expect_ok(interp, "s0 = [0; 0; 1; 0; 0]");
    expect_ok(interp, "s1 = gria_ca_step(s0, 90)");
    ASSERT_GT(interp.state().matrices.count("s1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("s1").rows(), 5u);

    expect_ok(interp, "a = [0; 0; 1; 1; 0; 1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 1; 1; 1; 1; 1; 1]");
    expect_ok(interp, "dt = gria_divergence_trajectory(a, b, 110, 4)");
    ASSERT_GT(interp.state().matrices.count("dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dt").rows(), 5u);

    expect_ok(interp, "fld = gria_gf2n_generate_field(4)");
    ASSERT_GT(interp.state().matrices.count("fld"), 0u);
}

TEST(ReplCommandsTest, cellai_boltzmann_cypha_31) {
    Interpreter interp;

    expect_ok(interp, "bw = cellai_boltzmann_weights([3, 3, 3, 3, 3], 1)");
    ASSERT_GT(interp.state().matrices.count("bw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bw").rows(), 5u);

    expect_ok(interp, "cellmemory_new(cm894, 2, 4, [0.1, 1, 10])");
    expect_ok(interp, "cellmemory_step(cm894, [1; 0])");
    expect_ok(interp, "cf = cellai_cell_to_cypha_features(cm894, [0.1, 1, 5])");
    ASSERT_GT(interp.state().matrices.count("cf"), 0u);
}

TEST(ReplCommandsTest, gria_hamming_distance_noassign) {
    Interpreter interp;
    expect_contains(interp, "gria_hamming_distance([0; 0; 1; 1], [0; 1; 1; 1])", "1");
    expect_error_contains(interp, "gria_hamming_distance([0; 1], missing)", "unknown matrix");
}

TEST(ReplCommandsTest, izaac_vrf_verify_noassign) {
    Interpreter interp;
    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream pub_cmd;
    pub_cmd << "pub = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            pub_cmd << ", ";
        }
        pub_cmd << vrf(1, j);
    }
    pub_cmd << "]";
    expect_ok(interp, pub_cmd.str());
    expect_contains(interp, "izaac_vrf_verify(pub, msg, proof)", "1");
    expect_error_contains(interp, "izaac_vrf_verify(missing, msg, proof)", "unknown matrix");
}
