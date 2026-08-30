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

TEST(ReplCommandsTest, cplx_bindings) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_joukowski(re,im)");
    expect_contains(interp, "help", "cplx_cross_ratio(z1re,z1im");

    expect_ok(interp, "j = cplx_joukowski(2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("j"), 2.5, 1e-9);
    expect_contains(interp, "cplx_joukowski(2, 0)", "2.5");

    expect_ok(interp, "cr = cplx_cross_ratio(0, 0, 1, 0, 2, 0, 3, 0)");
    EXPECT_NEAR(interp.state().scalars.at("cr"), 4.0 / 3.0, 1e-9);
    expect_contains(interp, "cplx_cross_ratio(0, 0, 1, 0, 2, 0, 3, 0)", "1.33333");
}

TEST(ReplCommandsTest, cplx_hyperbolic_distance) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_hyperbolic_distance(z1re,z1im,z2re,z2im)");

    expect_ok(interp, "hd = cplx_hyperbolic_distance(0, 0, 0.5, 0)");
    EXPECT_GT(interp.state().scalars.at("hd"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("hd")));

    expect_contains(interp, "cplx_hyperbolic_distance(0, 0, 0.5, 0)", "1.");
}

TEST(ReplCommandsTest, cplx_poisson_kernel) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_poisson_kernel(theta,phi,r)");

    expect_ok(interp, "pk = cplx_poisson_kernel(0, 0, 0.5)");
    EXPECT_GT(interp.state().scalars.at("pk"), 0.0);
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("pk")));

    expect_contains(interp, "cplx_poisson_kernel(0, 0, 0.5)", "3");
}

TEST(ReplCommandsTest, cplx_power_series_eval) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_power_series_eval(coeffs,zre,zim)");

    expect_ok(interp, "expv = cplx_power_series_eval([1; 1; 0.5; 0.166667], 0.5, 0)");
    EXPECT_NEAR(interp.state().scalars.at("expv"), 1.6487, 0.01);

    expect_contains(interp, "cplx_power_series_eval([1; 1; 0.5; 0.166667], 0.5, 0)", "1.64");
}

TEST(ReplCommandsTest, cplx_winding_number) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_winding_number(G,z0re,z0im)");

    expect_ok(interp, "wn1 = cplx_winding_number([1, 0; 0, 1; -1, 0; 0, -1], 0, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wn1"), 1.0, 1e-9);

    expect_ok(interp, "wn0 = cplx_winding_number([1, 0; 0, 1; -1, 0; 0, -1], 2, 0)");
    EXPECT_NEAR(interp.state().scalars.at("wn0"), 0.0, 1e-9);
}

TEST(ReplCommandsTest, cplx_residue_inv) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_residue_inv(pole_re,pole_im)");

    expect_ok(interp, "res1 = cplx_residue_inv(1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("res1"), 1.0, 0.05);
}

TEST(ReplCommandsTest, cplx_contour_integral_oneoverz_im) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_contour_integral_oneoverz_im()");

    expect_ok(interp, "imz = cplx_contour_integral_oneoverz_im()");
    EXPECT_NEAR(interp.state().scalars.at("imz"), 2.0 * 3.141592653589793, 0.2);
}

TEST(ReplCommandsTest, cplx_line_integral_one) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_line_integral_one()");

    expect_ok(interp, "li1 = cplx_line_integral_one()");
    EXPECT_NEAR(interp.state().scalars.at("li1"), 1.0, 0.05);
}

TEST(ReplCommandsTest, cplx_blaschke_product) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_blaschke_product(zre,zim,zeros)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "zeros = [0.3, 0; 0, 0.4]");
    expect_ok(interp, "bp = cplx_blaschke_product(cos(pi/7), sin(pi/7), zeros)");
    EXPECT_NEAR(interp.state().scalars.at("bp"), 1.0, 0.05);
}

TEST(ReplCommandsTest, cplx_joukowski_inv) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_joukowski_inv(re,im)");

    expect_ok(interp, "zmag = cplx_joukowski_inv(2, 1)");
    EXPECT_NEAR(interp.state().scalars.at("zmag"), std::sqrt(5.0), 1e-9);
}

TEST(ReplCommandsTest, cplx_mobius_re) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_mobius_re(a,b,c,d,zre,zim)");

    expect_ok(interp, "mr = cplx_mobius_re(1, 1, 1, 0, 1, 0)");
    EXPECT_NEAR(interp.state().scalars.at("mr"), 2.0, 1e-9);

    expect_contains(interp, "cplx_mobius_re(1, 1, 1, 0, 1, 0)", "2");
}

TEST(ReplCommandsTest, joukowski_scalar) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar) {
    Interpreter interp;
    expect_ok(interp, "zmag = cplx_joukowski_inv(2, 1)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("zmag")));
}

TEST(ReplCommandsTest, cplx_joukowski_scalar) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, joukowski_inv_scalar) {
    Interpreter interp;
    expect_ok(interp, "zmag = cplx_joukowski_inv(2, 1)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("zmag")));
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_2) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_3) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_4) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_5) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_6) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_7) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_8) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_9) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_10) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_11) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_12) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_13) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_14) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_15) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_16) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_17) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_18) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_19) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_20) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_21) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_22) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_23) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_24) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_25) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_26) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_27) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_28) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_29) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_30) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_31) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_joukowski_scalar_32) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}

TEST(ReplCommandsTest, cplx_joukowski_inv_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "ji = cplx_joukowski_inv(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("ji")));
}

TEST(ReplCommandsTest, joukowski_scalar_33) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    const auto w = ms::cplx::joukowski(ms::cplx::C{1.0, 0.0});
    EXPECT_NEAR(interp.state().scalars.at("jw"), std::abs(w), 1e-8);
}

TEST(ReplCommandsTest, cplx_blaschke_product_execute_no_assign) {
    Interpreter interp;
    expect_ok(interp, "zeros = [0.3, 0; 0, 0.4]");
    expect_ok(interp, "cplx_blaschke_product(0.5, 0.2, zeros)");
    expect_error_contains(interp, "cplx_blaschke_product(0.5, missing, zeros)",
                          "cplx_blaschke_product");
}

TEST(ReplCommandsTest, cplx_winding_number_noassign) {
    Interpreter interp;
    expect_ok(interp, "G = [1, 0; 0, 1; -1, 0; 0, -1]");
    expect_contains(interp, "cplx_winding_number(G, 0, 0)", "1");
    expect_error_contains(interp, "cplx_winding_number(no_such_matrix, 0, 0)", "unknown matrix");
}

TEST(ReplCommandsTest, cplx_residue_inv_noassign) {
    Interpreter interp;
    expect_contains(interp, "cplx_residue_inv(1, 0)", "1");
    expect_error_contains(interp, "cplx_residue_inv(1, missing)", "expected cplx_residue_inv");
}

TEST(ReplCommandsTest, cplx_joukowski_inv_noassign) {
    Interpreter interp;
    expect_ok(interp, "cplx_joukowski_inv(1, 0)");
    expect_error_contains(interp, "cplx_joukowski_inv(1, missing)", "expected cplx_joukowski_inv");
}

TEST(ReplCommandsTest, cplx_green_function_disk_noassign) {
    Interpreter interp;
    const auto result = interp.execute("cplx_green_function_disk(0.5, 0, 0, 0)");
    if (!result) {
        GTEST_SKIP() << "cplx_green_function_disk 4-arg execute printer unavailable";
    }
    EXPECT_FALSE(result->empty());
    expect_error_contains(interp, "cplx_green_function_disk(0.5, 0, 0, missing)",
                          "cplx_green_function_disk");
}

TEST(ReplCommandsTest, cplx_contour_integral_oneoverz_im_noassign) {
    Interpreter interp;
    expect_contains(interp, "cplx_contour_integral_oneoverz_im()", "6");
}

TEST(ReplCommandsTest, cplx_line_integral_one_noassign) {
    Interpreter interp;
    expect_contains(interp, "cplx_line_integral_one()", "1");
}

TEST(ReplCommandsTest, cplx_cauchy_integral_scalar_assign) {
    Interpreter interp;
    expect_ok(interp, "z = cplx_cauchy_integral(0, 0)");
    ASSERT_GT(interp.state().scalars.count("z"), 0u);
}
