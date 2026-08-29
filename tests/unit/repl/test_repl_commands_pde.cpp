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

TEST(ReplCommandsTest, pde_cn_adi) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_heat_1d_cn(x0,alpha,dx,dt,steps)");
    expect_contains(interp, "help", "pde_heat_2d_cn_adi(u0,alpha,dx,dy,dt,steps)");

    expect_ok(interp, "x0 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "h1cn = pde_heat_1d_cn(x0, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("h1cn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h1cn").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h1cn").cols(), 1u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(interp.state().matrices.at("h1cn")(i, 0)));
    }
    expect_ok(interp, "h1cn2 = pde_heat_1d_cn(x0, 0.1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("h1cn2"), 0u);
    expect_error(interp, "pde_heat_1d_cn([0; 1], 0.1, 0.1, 0.01, 5)");

    expect_ok(interp,
              "u0 = [0,0,0,0,0; 0,0,0,0,0; 0,0,10,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 5; ++c) {
            EXPECT_TRUE(std::isfinite(interp.state().matrices.at("h2adi")(r, c)));
        }
    }
    expect_error(interp, "pde_heat_2d_cn_adi(zeros(2, 2), 0.1, 0.1, 0.1, 0.01, 5)");
}

TEST(ReplCommandsTest, pde_elliptic) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_poisson_1d(f,dx,ua,ub)");
    expect_contains(interp, "help", "pde_laplace_2d(nx,ny,boundary)");
    expect_contains(interp, "help", "pde_helmholtz_2d(f,k,dx,dy");

    expect_ok(interp, "f = zeros(6,1)");
    expect_ok(interp, "u1 = pde_poisson_1d(f, 0.2, 1, 3)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 6u);
    EXPECT_NEAR(interp.state().matrices.at("u1")(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(interp.state().matrices.at("u1")(5, 0), 3.0, 1e-10);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_LE(interp.state().matrices.at("u1")(i, 0), interp.state().matrices.at("u1")(i + 1, 0));
    }
    EXPECT_GT(interp.state().matrices.at("u1")(3, 0), 1.0);
    EXPECT_LT(interp.state().matrices.at("u1")(3, 0), 3.0);

    expect_ok(interp, "B = [1,1,1,1,1; 0,0,0,0,0; 0,0,0,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("u2")(4, 2), 0.0, 1e-6);
    EXPECT_GT(interp.state().matrices.at("u2")(2, 2), 0.0);
    EXPECT_LT(interp.state().matrices.at("u2")(2, 2), 1.0);

    expect_ok(interp, "F = zeros(11,11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
    EXPECT_NEAR(interp.state().matrices.at("u3")(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(interp.state().matrices.at("u3")(10, 10), 0.0, 1e-10);

    ms::Poisson1DResult ref =
        ms::pde_poisson_1d({0, 0, 0, 0, 0, 0}, 0.2, 1.0, 3.0);
    ASSERT_EQ(ref.u.size(), 6u);
    EXPECT_NEAR(interp.state().matrices.at("u1")(3, 0), ref.u[3], 1e-10);
}

TEST(ReplCommandsTest, pde_hyperbolic_adv) {
    Interpreter interp;
    expect_contains(interp, "help", "pde_wave_2d(u0,v0,c,dx,dy,dt,steps)");
    expect_contains(interp, "help", "pde_advection_1d_lax_wendroff(u0,v,dx,dt,steps)");
    expect_contains(interp, "help", "pde_reaction_diffusion_1d(u0,D,r,dx,dt,steps)");

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
    EXPECT_GT(interp.state().matrices.at("rd")(10, 0), 0.39);
    EXPECT_LT(interp.state().matrices.at("rd")(10, 0), 0.95);
}

TEST(ReplCommandsTest, compress_golomb_wavelet) {
    Interpreter interp;
    expect_contains(interp, "help", "golomb_rice_encode_vec(V,m_bits)");
    expect_contains(interp, "help", "golomb_rice_decode_vec(E,m_bits,count)");
    expect_contains(interp, "help", "wavelet_compress_vec(M[,threshold])");
    expect_contains(interp, "help", "wavelet_decompress_vec(C)");

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    ASSERT_GT(interp.state().matrices.count("GE"), 0u);
    EXPECT_GE(interp.state().matrices.at("GE").rows(), 1u);

    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    ASSERT_GT(interp.state().matrices.count("GR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
    const double golomb_expected[] = {0, 1, 2, 5, 10};
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("GR")(i, 0), golomb_expected[i], 1e-9);
    }

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);
    EXPECT_GE(interp.state().matrices.at("WC").rows(), 1u);

    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("WR").rows(), 8u);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("WR")(i, 0),
                    interp.state().matrices.at("D")(i, 0), 1e-9);
    }

    expect_ok(interp, "WL = wavelet_compress_vec(D)");
    ASSERT_GT(interp.state().matrices.count("WL"), 0u);
    expect_ok(interp, "WLR = wavelet_decompress_vec(WL)");
    ASSERT_GT(interp.state().matrices.count("WLR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("WLR").rows(), 8u);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("WLR")(i, 0),
                    interp.state().matrices.at("D")(i, 0), 1e-9);
    }
}

TEST(ReplCommandsTest, pde_control) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    EXPECT_GT(interp.state().matrices.at("Co").rows(), 0u);
}

TEST(ReplCommandsTest, hadamard_heat1d) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_cn_adi) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv_wavelet_decode) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);
}

TEST(ReplCommandsTest, poisson2d_burgers) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
}

TEST(ReplCommandsTest, hadamard_heat) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_2) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_2) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_2) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_2) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_2) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_3) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_2) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_2) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_3) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_3) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_3) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_3) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_4) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_3) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_3) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_4) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_4) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_4) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_4) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_5) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_4) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_4) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_5) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_5) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_5) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_5) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_6) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_5) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_5) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_6) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_6) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_6) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_6) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_7) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_6) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_6) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_7) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_7) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_7) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_7) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_8) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_7) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_7) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_8) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_8) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_8) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_8) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_9) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_8) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_8) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_9) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_9) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_9) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_9) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_10) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_9) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_9) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_10) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_10) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_10) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_10) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_11) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_10) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_10) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_11) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_11) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_11) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_11) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_12) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_11) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_11) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_12) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_12) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_12) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_12) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_13) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_12) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_12) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_13) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_13) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_13) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_13) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_14) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_13) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_13) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_14) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_14) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_14) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_14) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_15) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_14) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_14) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_15) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_15) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_15) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_15) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_16) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_15) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_15) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_16) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_16) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_16) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_16) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_17) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_16) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_16) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_17) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_17) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_17) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_17) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_18) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_17) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_17) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_18) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_18) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_18) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_18) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_19) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_18) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_18) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_19) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_19) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_19) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_19) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_20) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_19) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_19) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_20) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_20) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_20) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_20) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_21) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_20) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_20) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_21) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_21) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_21) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_21) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_22) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_21) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_21) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_22) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_22) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_22) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_22) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_23) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_22) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_22) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_23) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_23) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_23) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_23) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_24) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_23) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_23) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_24) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_24) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_24) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_24) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_25) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_24) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_24) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_25) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_25) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_25) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_25) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_26) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_25) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_25) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_26) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_26) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_26) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_26) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_27) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_26) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_26) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_27) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_27) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_27) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_27) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_28) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_27) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_27) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_28) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_28) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_28) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_28) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_29) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_28) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_28) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_29) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_29) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_29) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_29) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_30) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_29) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_29) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_30) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_30) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_30) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_30) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_31) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_30) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_30) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_31) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_31) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_31) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_31) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_32) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_31) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_31) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_32) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_32) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_32) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, hadamard_heat_32) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "hpsi = quantum_hadamard(psi)");
    ASSERT_GT(interp.state().matrices.count("hpsi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hpsi").rows(), 2u);

    expect_ok(interp, "x0 = [0; 1; 0]");
    expect_ok(interp, "h = pde_heat_1d(x0, 0.1, 0.1, 0.01, 5)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 3u);

    expect_ok(interp, "x1 = [0; 0.5; 1; 0.5; 0]");
    expect_ok(interp, "hcn = pde_heat_1d_cn(x1, 0.1, 0.1, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("hcn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hcn").rows(), 5u);
}

TEST(ReplCommandsTest, wavelet_wave1d_33) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    ASSERT_GT(interp.state().matrices.count("WC"), 0u);

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    ASSERT_GT(interp.state().matrices.count("w1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);
}

TEST(ReplCommandsTest, heat2d_32) {
    Interpreter interp;

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    ASSERT_GT(interp.state().matrices.count("h2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "u0 = [0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 10, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "h2adi = pde_heat_2d_cn_adi(u0, 0.05, 0.1, 0.1, 0.5, 10)");
    ASSERT_GT(interp.state().matrices.count("h2adi"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("h2adi").cols(), 5u);
}

TEST(ReplCommandsTest, poisson_adv1d_32) {
    Interpreter interp;

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    ASSERT_GT(interp.state().matrices.count("p1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    ASSERT_GT(interp.state().matrices.count("adv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);
}

TEST(ReplCommandsTest, poisson2d_burgers_33) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    ASSERT_GT(interp.state().matrices.count("p2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "u0b = linspace(0, 0, 11)");
    expect_ok(interp, "b1 = pde_burgers_1d(u0b, 0.01, 0.1, 0.001, 20)");
    ASSERT_GT(interp.state().matrices.count("b1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b1").rows(), 11u);
}

TEST(ReplCommandsTest, laplace_helmholtz_33) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 1, 1, 1; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0]");
    expect_ok(interp, "u2 = pde_laplace_2d(5, 5, B)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("u2").cols(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("u2")(0, 2), 1.0, 1e-6);

    expect_ok(interp, "F = zeros(11, 11)");
    expect_ok(interp, "u3 = pde_helmholtz_2d(F, 1, 0.1, 0.1)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 11u);
}

TEST(ReplCommandsTest, lax_wendroff_rd_33) {
    Interpreter interp;

    expect_ok(interp, "u0a = [1; 0; 0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "lw = pde_advection_1d_lax_wendroff(u0a, 1.0, 0.1, 0.05, 4)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lw").rows(), 10u);

    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_ok(interp, "rd = pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)");
    ASSERT_GT(interp.state().matrices.count("rd"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rd").rows(), 21u);
}

TEST(ReplCommandsTest, pde_reaction_diffusion_1d_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "u0r = linspace(0.4, 0.4, 21)");
    expect_contains(interp, "pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 30)", "u =");
    expect_error_contains(interp, "pde_reaction_diffusion_1d(u0r, 0.05, 2.0, 0.1, 0.01, 1.5)",
                          "non-negative integer steps");
}
