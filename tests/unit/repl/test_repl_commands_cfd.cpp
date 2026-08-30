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

TEST(ReplCommandsTest, cfd2d_primitives) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_grid2d(x0,x1,y0,y1,nx,ny)");
    expect_contains(interp, "help", "cfd_square_pulse_2d(grid,xc,yc,width_x,width_y");
    expect_contains(interp, "help", "cfd_upwind_step_2d(u,vx,vy,dt,dx,dy");
    expect_contains(interp, "help", "cfd_integrated_mass_2d(u,dx,dy)");

    expect_ok(interp, "g = cfd_grid2d(0, 1, 0, 1, 50, 40)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g").rows(), 3u);

    expect_ok(interp, "u0 = cfd_square_pulse_2d(g, 0.5, 0.5, 0.2, 0.2, 2.0)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 40u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 50u);

    expect_ok(interp, "m = cfd_integrated_mass_2d(u0, 0.02, 0.025)");
    EXPECT_NEAR(interp.state().scalars.at("m"), 0.2 * 0.2 * 2.0, 1e-6);

    expect_ok(interp, "u1 = cfd_upwind_step_2d(u0, 1.0, 0.0, 0.005, 0.02, 0.025)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 40u);

    expect_ok(interp, "m1 = cfd_integrated_mass_2d(u1, 0.02, 0.025)");
    EXPECT_NEAR(interp.state().scalars.at("m1"), interp.state().scalars.at("m"), 1e-6);
}

TEST(ReplCommandsTest, cfd3d_primitives) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_grid3d(x0,x1,y0,y1,z0,z1,nx,ny,nz)");
    expect_contains(interp, "help", "cfd_square_pulse_3d(grid,xc,yc,zc,wx,wy,wz[,amp])");

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);

    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_integrated_mass_3d) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_integrated_mass_3d(grid,u)");

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "mass3 = cfd_integrated_mass_3d(g3, u0)");
    EXPECT_GT(interp.state().scalars.at("mass3"), 0.0);
}

TEST(ReplCommandsTest, cfd_run_advection_3d) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_run_advection_3d");

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_run_advection_2d) {
    Interpreter interp;
    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u0, 0.2, 0, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, cfd_mass_composable) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_integrated_mass_1d");
    expect_contains(interp, "help", "cfd_constant_velocity");

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    expect_ok(interp, "m1 = cfd_integrated_mass_1d(g1, u1)");
    EXPECT_GT(interp.state().scalars.at("m1"), 0.0);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    expect_ok(interp, "m2 = cfd_integrated_mass_2d(g2, u3)");
    EXPECT_GT(interp.state().scalars.at("m2"), 0.0);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, cfd_1d2d_advection) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, cfd_upwind1d_constvel) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, cfd_upwind2d) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);
}

TEST(ReplCommandsTest, cfd3d_grid_pulse) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);
}

TEST(ReplCommandsTest, cfd_upwind3d) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_pulse_dagger) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
}

TEST(ReplCommandsTest, cfd3d) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, adv2d) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_pulse_dagger_2) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
}

TEST(ReplCommandsTest, cfd3d_2) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_2) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_2) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_2) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_2) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_2) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, adv2d_2) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_2) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_3) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_3) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_3) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_3) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_3) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_2) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_2) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_3) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_4) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_4) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_4) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_4) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_4) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_3) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_3) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_4) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_5) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_5) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_5) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_5) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_5) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_4) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_4) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_5) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_6) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_6) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_6) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_6) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_6) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_5) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_5) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_6) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_7) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_7) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_7) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_7) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_7) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_6) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_6) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_7) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_8) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_8) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_8) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_8) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_8) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_7) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_7) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_8) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_9) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_9) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_9) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_9) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_9) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_8) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_8) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_9) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_10) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_10) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_10) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_10) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_10) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_9) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_9) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_10) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_11) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_11) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_11) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_11) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_11) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_10) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_10) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_11) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_12) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_12) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_12) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_12) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_12) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_11) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_11) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_12) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_13) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_13) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_13) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_13) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_13) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_12) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_12) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_13) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_14) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_14) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_14) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_14) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_14) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_13) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_13) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_14) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_15) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_15) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_15) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_15) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_15) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_14) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_14) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_15) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_16) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_16) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_16) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_16) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_16) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_15) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_15) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_16) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_17) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_17) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_17) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_17) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_17) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_16) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_16) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_17) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_18) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_18) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_18) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_18) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_18) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_17) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_17) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_18) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_19) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_19) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_19) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_19) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_19) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_18) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_18) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_19) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_20) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_20) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_20) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_20) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_20) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_19) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_19) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_20) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_21) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_21) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_21) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_21) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_21) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_20) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_20) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_21) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_22) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_22) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_22) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_22) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_22) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_21) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_21) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_22) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_23) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_23) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_23) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_23) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_23) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_22) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_22) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_23) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_24) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_24) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_24) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_24) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_24) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_23) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_23) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_24) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_25) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_25) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_25) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_25) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_25) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_24) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_24) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_25) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_26) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_26) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_26) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_26) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_26) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_25) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_25) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_26) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_27) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_27) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_27) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_27) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_27) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_26) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_26) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_27) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_28) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_28) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_28) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_28) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_28) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_27) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_27) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_28) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_29) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_29) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_29) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_29) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_29) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_28) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_28) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_29) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_30) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_30) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_30) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_30) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_30) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_29) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_29) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_30) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_31) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_31) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_31) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_31) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_31) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_30) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_30) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_run_advection_3d_31) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);
}

TEST(ReplCommandsTest, cfd1d_pulse_32) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advect_32) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_run_advection(g1, u0, 0.2, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "uf = cfd_run_advection_2d(g2, u2, 0.2, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 4u);
}

TEST(ReplCommandsTest, upwind1d_vel_32) {
    Interpreter interp;

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1 = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);

    expect_ok(interp, "vc = cfd_constant_velocity(4, 2.0)");
    ASSERT_GT(interp.state().matrices.count("vc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("vc").rows(), 4u);
}

TEST(ReplCommandsTest, upwind2d_hebbian_32) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u2 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    expect_ok(interp, "u3 = cfd_upwind_step_2d(g2, u2, 0.2, 0, 0.001)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("u3").cols(), 4u);

    expect_ok(interp, "W = [0.2; 0.3]");
    expect_ok(interp, "X = [1; 0]");
    expect_ok(interp, "Y = [0.9; 0.1]");
    expect_ok(interp, "Wn = cellai_hebbian_update(W, X, Y, 0.05)");
    ASSERT_GT(interp.state().matrices.count("Wn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Wn").rows(), 2u);
}

TEST(ReplCommandsTest, cfd3d_pulse_upwind_32) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u0").rows(), 16u);
    EXPECT_EQ(interp.state().matrices.at("u0").cols(), 4u);

    expect_ok(interp, "u1 = cfd_upwind_step_3d(g3, u0, 0.2, 0, 0, 0.01)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 16u);
}

TEST(ReplCommandsTest, cfd_advection2d_31) {
    Interpreter interp;

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(ReplCommandsTest, cfd_grid2d_square_pulse_2d_31) {
    Interpreter interp;

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.4, 0.4)");
    ASSERT_GT(interp.state().matrices.count("u0"), 0u);
}

TEST(ReplCommandsTest, cfd_advection2d_noassign) {
    Interpreter interp;
    expect_contains(interp, "cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)", "u =");
    expect_error_contains(interp, "cfd_advection2d(1.5, 8, 1, 0, 0.1, 0.01)",
                          "non-negative integer nx and ny");
}

TEST(ReplCommandsTest, cfd_advection1d_senary_noassign) {
    Interpreter interp;
    expect_contains(interp, "cfd_advection1d(8, 1, 0.5, 0.01, 0, 0)", "u =");
    expect_error_contains(interp, "cfd_advection1d(1.5, 1, 0.5, 0.01, 0, 0)",
                          "expected non-negative integer nx");
}
