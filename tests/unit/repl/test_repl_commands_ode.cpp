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

TEST(ReplCommandsTest, ode_adaptive_stiff) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_trapezoidal(\"formula\",t0,y0,t_end,steps)");
    expect_contains(interp, "help", "ode_cashkarp(\"formula\",t0,y0,t_end,rtol,atol)");
    expect_contains(interp, "help", "ode_rk23(\"formula\",t0,y0,t_end,rtol,atol)");
    expect_contains(interp, "help", "ode_exponential_euler(\"g\",lambda,t0,y0,t_end,steps)");
    expect_contains(interp, "help", "ode_rosenbrock23(\"formula\",t0,y0,t_end,steps)");
    expect_contains(interp, "help", "ode_rosenbrock23_vec(\"f0;f1;...\",t0,y0,t_end,steps)");

    const auto trapezoidal = interp.execute("ode_trapezoidal(\"-y\", 0, 1, 1, 200)");
    ASSERT_TRUE(trapezoidal.has_value());
    const auto trap_last = parse_last_ode_traj_row(*trapezoidal);
    EXPECT_NEAR(trap_last.first, 1.0, 1e-9);
    EXPECT_NEAR(trap_last.second, std::exp(-1.0), 0.05);
    expect_error_contains(interp, "ode_trapezoidal(\"sin(\", 0, 1, 1, 200)", "ode_trapezoidal");

    const auto cashkarp = interp.execute("ode_cashkarp(\"y\", 0, 1, 1, 1e-6, 1e-9)");
    ASSERT_TRUE(cashkarp.has_value());
    const auto ck_last = parse_last_ode_traj_row(*cashkarp);
    EXPECT_NEAR(ck_last.first, 1.0, 1e-9);
    EXPECT_NEAR(ck_last.second, std::exp(1.0), 0.02);
    expect_error_contains(interp, "ode_cashkarp(\"y +\", 0, 1, 1, 1e-6, 1e-9)", "ode_cashkarp");

    const auto rk23 = interp.execute("ode_rk23(\"y\", 0, 1, 1, 1e-4, 1e-7)");
    ASSERT_TRUE(rk23.has_value());
    const auto rk23_last = parse_last_ode_traj_row(*rk23);
    EXPECT_NEAR(rk23_last.first, 1.0, 1e-9);
    EXPECT_NEAR(rk23_last.second, std::exp(1.0), 0.03);
    expect_error_contains(interp, "ode_rk23(\"y +\", 0, 1, 1, 1e-4, 1e-7)", "ode_rk23");

    const auto etd = interp.execute("ode_exponential_euler(\"0\", -5, 0, 1, 1, 200)");
    ASSERT_TRUE(etd.has_value());
    const auto etd_last = parse_last_ode_traj_row(*etd);
    EXPECT_NEAR(etd_last.first, 1.0, 1e-9);
    EXPECT_NEAR(etd_last.second, std::exp(-5.0), 0.02);
    expect_error_contains(interp, "ode_exponential_euler(\"sin(\", -5, 0, 1, 1, 200)",
                          "ode_exponential_euler");

    const auto rosen = interp.execute("ode_rosenbrock23(\"-10*y\", 0, 1, 1, 200)");
    ASSERT_TRUE(rosen.has_value());
    const auto rosen_last = parse_last_ode_traj_row(*rosen);
    EXPECT_NEAR(rosen_last.first, 1.0, 1e-9);
    EXPECT_NEAR(rosen_last.second, std::exp(-10.0), 0.05);
    expect_error_contains(interp, "ode_rosenbrock23(\"bad(@)\", 0, 1, 1, 100)", "ode_rosenbrock23");

    const auto rosen_vec =
        interp.execute("ode_rosenbrock23_vec(\"-y0\", 0, [1], 1, 200)");
    ASSERT_TRUE(rosen_vec.has_value());
    const auto rosen_vec_last = parse_last_ode_traj_row_all(*rosen_vec);
    ASSERT_GE(rosen_vec_last.size(), 2u);
    EXPECT_NEAR(rosen_vec_last[0], 1.0, 1e-9);
    EXPECT_NEAR(rosen_vec_last[1], std::exp(-1.0), 0.05);
    expect_error_contains(interp, "ode_rosenbrock23_vec(\"y1; sin(\", 0, [1], 1, 100)",
                          "ode_rosenbrock23_vec");
}

TEST(ReplCommandsTest, cplx_ode_cfd1d) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_advection1d(nx,vx,t_end,dt)");
    expect_contains(interp, "help", "cplx_green_function_disk(zre,zim,z0re,z0im");
    expect_contains(interp, "help", "ode_adams_bashforth2(");

    expect_ok(interp, "U = cfd_advection1d(20, 1.0, 0.5, 0.01)");
    ASSERT_GT(interp.state().matrices.count("U"), 0u);
    EXPECT_EQ(interp.state().matrices.at("U").rows(), 20u);

    expect_ok(interp, "g = cplx_green_function_disk(0.5, 0, 0, 0)");
    EXPECT_LT(interp.state().scalars.at("g"), 0.0);

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    EXPECT_GE(interp.state().matrices.at("ab").rows(), 2u);
}

TEST(ReplCommandsTest, midpoint_obsv_gram) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wo").rows(), 0u);
}

TEST(ReplCommandsTest, adams_bdf) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "bd = ode_bdf2(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("bd").rows(), 0u);
}

TEST(ReplCommandsTest, trapezoidal_pagerank) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_2) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_2) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_2) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_2) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_3) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_3) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_3) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_3) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_4) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_4) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_4) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_4) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_5) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_5) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_5) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_5) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_6) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_6) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_6) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_6) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_7) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_7) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_7) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_7) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_8) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_8) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_8) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_8) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_9) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_9) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_9) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_9) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_10) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_10) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_10) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_10) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_11) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_11) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_11) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_11) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_12) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_12) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_12) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_12) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_13) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_13) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_13) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_13) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_14) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_14) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_14) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_14) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_15) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_15) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_15) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_15) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_16) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_16) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_16) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_16) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_17) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_17) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_17) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_17) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_18) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_18) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_18) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_18) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_19) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_19) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_19) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_19) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_20) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_20) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_20) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_20) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_21) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_21) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_21) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_21) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_22) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_22) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_22) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_22) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_23) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_23) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_23) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_23) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_24) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_24) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_24) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_24) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_25) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_25) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_25) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_25) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_26) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_26) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_26) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_26) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_27) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_27) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_27) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_27) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_28) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_28) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_28) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_28) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_29) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_29) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_29) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_29) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_30) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_30) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_30) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_30) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_31) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_31) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_31) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_31) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_32) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, rk4_coo_32) {
    Interpreter interp;

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("traj"), 0u);
    ASSERT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, euler_obsv_32) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    ASSERT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, adams_autocorr_32) {
    Interpreter interp;

    expect_ok(interp, "ab = ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)");
    ASSERT_GT(interp.state().matrices.count("ab"), 0u);
    ASSERT_GT(interp.state().matrices.at("ab").rows(), 0u);

    expect_ok(interp, "be = ode_backward_euler(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("be"), 0u);
    ASSERT_GT(interp.state().matrices.at("be").rows(), 0u);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);
}

TEST(ReplCommandsTest, trapezoidal_pagerank_33) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_trapezoidal(\"-y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "A = [0, 4, 1, 0, 0; 0, 0, 0, 1, 0; 0, 2, 0, 5, 0; 0, 0, 0, 0, 3; 0, 0, 0, 0, 0]");
    expect_ok(interp, "pr = graph_pagerank(A)");
    EXPECT_EQ(interp.state().matrices.at("pr").rows(), 5u);
}

TEST(ReplCommandsTest, ode_euler_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_euler(\"y\", 0, 1, 1, 5)", "traj =");
    expect_error_contains(interp, "ode_euler(\"y +\", 0, 1, 1, 5)", "ode_euler");
}

TEST(ReplCommandsTest, ode_rk4_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_rk4(\"y\", 0, 1, 1, 5)", "traj =");
    expect_error_contains(interp, "ode_rk4(\"y +\", 0, 1, 1, 5)", "ode_rk4");
}

TEST(ReplCommandsTest, ode_rk2_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_rk2(\"y\", 0, 1, 1, 200)", "traj =");
    expect_error_contains(interp, "ode_rk2(\"y +\", 0, 1, 1, 200)", "ode_rk2");
}

TEST(ReplCommandsTest, ode_midpoint_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_midpoint(\"y\", 0, 1, 1, 5)", "traj =");
    expect_error_contains(interp, "ode_midpoint(\"y +\", 0, 1, 1, 5)", "ode_midpoint");
}

TEST(ReplCommandsTest, ode_backward_euler_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_backward_euler(\"y\", 0, 1, 1, 5)", "traj =");
    expect_error_contains(interp, "ode_backward_euler(\"y +\", 0, 1, 1, 5)",
                          "ode_backward_euler");
}

TEST(ReplCommandsTest, ode_bdf2_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_bdf2(\"y\", 0, 1, 1, 5)", "traj =");
    expect_error_contains(interp, "ode_bdf2(\"y +\", 0, 1, 1, 5)", "ode_bdf2");
}

TEST(ReplCommandsTest, ode_trapezoidal_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_trapezoidal(\"-y\", 0, 1, 1, 200)", "traj =");
    expect_error_contains(interp, "ode_trapezoidal(\"sin(\", 0, 1, 1, 200)", "ode_trapezoidal");
}

TEST(ReplCommandsTest, ode_rosenbrock23_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_rosenbrock23(\"-10*y\", 0, 1, 1, 200)", "traj =");
    expect_error_contains(interp, "ode_rosenbrock23(\"bad(@)\", 0, 1, 1, 100)",
                          "ode_rosenbrock23");
}

TEST(ReplCommandsTest, ode_adams_bashforth2_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_adams_bashforth2(\"-y\", 0, 1, 1, 10)", "traj =");
    expect_error_contains(interp, "ode_adams_bashforth2(\"y +\", 0, 1, 1, 10)",
                          "ode_adams_bashforth2");
}

TEST(ReplCommandsTest, ode_rk45_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_rk45(\"y\", 0, 1, 1, 1e-6, 1e-9)", "traj =");
    expect_error_contains(interp, "ode_rk45(\"y +\", 0, 1, 1, 1e-6, 1e-9)", "ode_rk45");
}

TEST(ReplCommandsTest, ode_cashkarp_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_cashkarp(\"y\", 0, 1, 1, 1e-6, 1e-9)", "traj =");
    expect_error_contains(interp, "ode_cashkarp(\"y +\", 0, 1, 1, 1e-6, 1e-9)", "ode_cashkarp");
}

TEST(ReplCommandsTest, ode_rk23_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_rk23(\"y\", 0, 1, 1, 1e-4, 1e-7)", "traj =");
    expect_error_contains(interp, "ode_rk23(\"y +\", 0, 1, 1, 1e-4, 1e-7)", "ode_rk23");
}

TEST(ReplCommandsTest, ode_exponential_euler_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_exponential_euler(\"0\", -5, 0, 1, 1, 200)", "traj =");
    expect_error_contains(interp, "ode_exponential_euler(\"sin(\", -5, 0, 1, 1, 200)",
                          "ode_exponential_euler");
}

TEST(ReplCommandsTest, ode_verlet_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_verlet(\"-9.8\", 0, 0, 0, 1, 100)", "traj =");
    expect_error_contains(interp, "ode_verlet(\"sin(\", 0, 0, 0, 1, 100)", "ode_verlet");
}

TEST(ReplCommandsTest, ode_bvp_shooting_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_bvp_shooting(\"-y\", 0, 0, 1.570796, 1, 400)", "traj =");
    expect_error_contains(interp, "ode_bvp_shooting(\"sin(\", 0, 0, 1.570796, 1, 400)",
                          "ode_bvp_shooting");
}

TEST(ReplCommandsTest, ode_dde_fixed_step_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_dde_fixed_step(\"-y + ydelay\", \"1\", 0, 2, 0.5, 40)",
                    "traj =");
    expect_error_contains(interp, "ode_dde_fixed_step(\"-y + ydelay\", \"1\", 0, 2, 0.5, 1.5)",
                          "non-negative integer steps");
}

TEST(ReplCommandsTest, ode_event_detect_noassign) {
    Interpreter interp;
    expect_contains(interp, "ode_event_detect(\"1\", \"y\", 0, -5, 10, 100)", "traj =");
    expect_error_contains(interp, "ode_event_detect(\"1\", \"y\", 0, -5, 10, 1.5)",
                          "non-negative integer steps");
}
