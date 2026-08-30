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

TEST(ReplCommandsTest, fem_poisson1d) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_poisson1d(n)");

    expect_ok(interp, "u = fem_poisson1d(4)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);
    EXPECT_GT(interp.state().matrices.at("u")(2, 0), 0.0);
    EXPECT_NEAR(interp.state().matrices.at("u")(2, 0), 0.125, 0.02);
}

TEST(ReplCommandsTest, fem2d_assembly) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_mesh2d_rectangular(x0,y0,x1,y1,nx,ny)");
    expect_contains(interp, "help", "fem_stiffness_2d(mesh)");
    expect_contains(interp, "help", "fem_load_2d(mesh,f)");
    expect_contains(interp, "help", "fem_apply_dirichlet(K,f,node_indices,values)");
    expect_contains(interp, "help", "fem_solve(K,f)");

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("m"), 0u);
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
    EXPECT_EQ(interp.state().matrices.at("m")(0, 1), 9.0);

    expect_ok(interp, "m2 = fem_mesh2d(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m2").rows(), interp.state().matrices.at("m").rows());

    expect_ok(interp, "K = fem_stiffness_2d(m)");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 9u);

    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);

    expect_ok(interp, "f = fem_load_2d(m, 1)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);

    expect_ok(interp, "bc = [0; 1; 2; 3; 5; 6; 7; 8]");
    expect_ok(interp, "bv = [0; 0; 0; 0; 0; 0; 0; 0]");
    expect_ok(interp, "sys = fem_apply_dirichlet(K, f, bc, bv)");
    ASSERT_GT(interp.state().matrices.count("sys"), 0u);
    EXPECT_EQ(interp.state().matrices.at("sys").rows(), 9u);
    EXPECT_EQ(interp.state().matrices.at("sys").cols(), 10u);

    expect_ok(interp, "u = fem_solve(sys)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 9u);
    EXPECT_GT(interp.state().matrices.at("u")(4, 0), 0.0);

    expect_ok(interp, "u_ref = fem_poisson2d(2, 2)");
    EXPECT_NEAR(interp.state().matrices.at("u")(4, 0),
                interp.state().matrices.at("u_ref")(4, 0), 0.05);
}

TEST(ReplCommandsTest, fem3d_assembly) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_mesh3d_box(x0,y0,z0,x1,y1,z1,nx,ny,nz)");
    expect_contains(interp, "help", "fem_stiffness_3d(mesh)");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    ASSERT_GT(interp.state().matrices.count("m3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("m3")(0, 0), 271.0);

    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);

    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("f3").rows(), interp.state().matrices.at("K3").rows());
}

TEST(ReplCommandsTest, fem3d_pipeline) {
    Interpreter interp;
    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "bc = [0]");
    expect_ok(interp, "bv = [0]");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, bc, bv)");
    expect_ok(interp, "u3 = fem_solve(sys3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), interp.state().matrices.at("K3").rows());
}

TEST(ReplCommandsTest, fem_mesh3d) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_mesh3d_box(x0,y0,z0,x1,y1,z1,nx,ny,nz)");

    expect_ok(interp, "m3 = fem_mesh3d(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    ASSERT_GT(interp.state().matrices.count("m3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("m3")(0, 0), 271.0);

    expect_ok(interp, "ref = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    EXPECT_EQ(interp.state().matrices.at("m3").rows(), interp.state().matrices.at("ref").rows());

    expect_error_contains(interp, "bad = fem_mesh3d(0, 0, 0, 1, 1, 1, 0, 1, 1)",
                         "positive integer nx");
    expect_error(interp, "bad = fem_mesh3d(0, 0, 0)");
}

TEST(ReplCommandsTest, assemble_stiffness_3d) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_stiffness_3d(mesh)");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 1, 1, 1)");
    expect_ok(interp, "K = assemble_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_GT(interp.state().matrices.at("K").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), interp.state().matrices.at("K").cols());

    expect_ok(interp, "Kref = fem_stiffness_3d(m3)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), interp.state().matrices.at("Kref").rows());

    expect_error_contains(interp, "bad = assemble_stiffness_3d(missing)", "unknown matrix");
    expect_error_contains(interp, "bad = assemble_stiffness_3d([1, 2; 3, 4])", "fem_stiffness_3d");
}

TEST(ReplCommandsTest, fem1d_cfd_composable) {
    Interpreter interp;
    expect_contains(interp, "help", "fem_stiffness_1d(mesh)");
    expect_contains(interp, "help", "cfd_grid1d(x0,x1,n)");

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    expect_ok(interp, "sys1 = fem_apply_dirichlet(K1, f1, [0, 8], [0, 0])");
    expect_ok(interp, "u1 = fem_solve(sys1)");
    EXPECT_EQ(interp.state().matrices.at("u1").rows(), 9u);

    expect_ok(interp, "g1 = cfd_grid1d(0, 1, 16)");
    expect_ok(interp, "u0 = cfd_square_pulse(g1, 0.5, 0.25)");
    expect_ok(interp, "u1c = cfd_upwind_step_1d(g1, u0, 0.2, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("u1c").rows(), 16u);
}

TEST(ReplCommandsTest, topo_fem_compress) {
    Interpreter interp;
    expect_contains(interp, "help", "topo_vietoris_rips");
    expect_contains(interp, "help", "fem_solve_3d");

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "b = topo_simplicial_betti(vr)");
    expect_ok(interp, "e = topo_simplicial_euler(vr)");

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "v = [1, 1, 2]");
    expect_ok(interp, "e2 = run_length_encode_vec(v)");
    expect_ok(interp, "d2 = run_length_decode_vec(e2)");
    EXPECT_EQ(interp.state().matrices.at("d2").rows(), 3u);
}

TEST(ReplCommandsTest, compress_fem_quantum) {
    Interpreter interp;

    expect_ok(interp, "orig = [10, 20, 30]");
    expect_ok(interp, "aenc = arithmetic_encode_vec(orig)");
    expect_ok(interp, "adec = arithmetic_decode_vec(orig, aenc)");
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 3u);

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys = fem_apply_dirichlet(K3, f3, [0], [0])");
    EXPECT_GT(interp.state().matrices.at("sys").cols(), 1u);

    expect_ok(interp, "H = [1, 0; 0, -1]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.05, 5)");
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem_cfd_quantum) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    EXPECT_GT(interp.state().matrices.at("u2").rows(), 0u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_2d(g2, 0.5, 0.5, 0.2, 0.2)");

    expect_ok(interp, "H = [1, 0; 0, 2]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "tr = quantum_schrodinger(H, psi0, 0, 0.1, 3)");
    EXPECT_EQ(interp.state().matrices.at("tr").rows(), 4u);

    expect_ok(interp, "p = [1, 2, 3]");
    expect_ok(interp, "q = [0, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(p, q, 100)");
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 3u);
}

TEST(ReplCommandsTest, image_fem_cfd) {
    Interpreter interp;

    expect_ok(interp, "M = [0.1, 0.2; 0.3, 0.4]");
    expect_ok(interp, "P = impad(M, 1, 0)");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 4u);

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);

    expect_ok(interp, "g = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    EXPECT_GT(interp.state().matrices.at("g").rows(), 0u);
}

TEST(ReplCommandsTest, cfd3d_fem1d) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 16u);

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);

    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_quantum_rle) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);
}

TEST(ReplCommandsTest, fem_dirichlet_schrodinger_final) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    EXPECT_GT(interp.state().matrices.at("sys3").rows(), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiffness_load) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("f3").rows(), interp.state().matrices.at("K3").rows());
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, fem_stiffness_load) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_cfdgrid) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
}

TEST(ReplCommandsTest, fem_cfd3d) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
}

TEST(ReplCommandsTest, fem1d_stiff) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_lagrange) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, poisson2d_adv1d) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, fem_stiffness_load_2) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_cfdgrid_2) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);

    expect_ok(interp, "g2 = cfd_grid2d(0, 1, 0, 1, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g2"), 0u);
}

TEST(ReplCommandsTest, fem_cfd3d_2) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
}

TEST(ReplCommandsTest, fem1d_stiff_2) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_lagrange_2) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_2) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_2) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_2) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_2) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, poisson2d_adv1d_2) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_3) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_3) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_3) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_3) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_3) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_2) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_2) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_2) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_4) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_2) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_2) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_2) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_2) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_4) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_4) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_4) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_4) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_3) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_3) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_3) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_5) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_3) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_3) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_3) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_3) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_5) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_5) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_5) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_5) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_4) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_4) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_4) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_6) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_4) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_4) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_4) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_4) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_6) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_6) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_6) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_6) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_5) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_5) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_5) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_7) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_5) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_5) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_5) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_5) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_7) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_7) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_7) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_7) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_6) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_6) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_6) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_8) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_6) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_6) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_6) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_6) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_8) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_8) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_8) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_8) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_7) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_7) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_7) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_9) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_7) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_7) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_7) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_7) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_9) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_9) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_9) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_9) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_8) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_8) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_8) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_10) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_8) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_8) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_8) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_8) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_10) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_10) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_10) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_10) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_9) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_9) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_9) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_11) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_9) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_9) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_9) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_9) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_11) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_11) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_11) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_11) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_10) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_10) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_10) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_12) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_10) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_10) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_10) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_10) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_12) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_12) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_12) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_12) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_11) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_11) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_11) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_13) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_11) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_11) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_11) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_11) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_13) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_13) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_13) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_13) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_12) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_12) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_12) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_14) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_12) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_12) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_12) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_12) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_14) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_14) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_14) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_14) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_13) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_13) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_13) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_15) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_13) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_13) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_13) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_13) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_15) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_15) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_15) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_15) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_14) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_14) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_14) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_16) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_14) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_14) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_14) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_14) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_16) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_16) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_16) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_16) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_15) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_15) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_15) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_17) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_15) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_15) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_15) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_15) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_17) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_17) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_17) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_17) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_16) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_16) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_16) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_18) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_16) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_16) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_16) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_16) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_18) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_18) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_18) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_18) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_17) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_17) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_17) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_19) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_17) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_17) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_17) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_17) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_19) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_19) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_19) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_19) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_18) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_18) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_18) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_20) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_18) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_18) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_18) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_18) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_20) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_20) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_20) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_20) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_19) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_19) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_19) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_21) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_19) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_19) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_19) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_19) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_21) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_21) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_21) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_21) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_20) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_20) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_20) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_22) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_20) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_20) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_20) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_20) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_22) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_22) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_22) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_22) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_21) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_21) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_21) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_23) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_21) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_21) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_21) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_21) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_23) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_23) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_23) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_23) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_22) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_22) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_22) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_24) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_22) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_22) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_22) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_22) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_24) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_24) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_24) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_24) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_23) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_23) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_23) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_25) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_23) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_23) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_23) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_23) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_25) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_25) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_25) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_25) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_24) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_24) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_24) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_26) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_24) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_24) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_24) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_24) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_26) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_26) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_26) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_26) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_25) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_25) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_25) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_27) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_25) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_25) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_25) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_25) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_27) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_27) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_27) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_27) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_26) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_26) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_26) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_28) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_26) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_26) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_26) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_26) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_28) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_28) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_28) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_28) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_27) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_27) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_27) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_29) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_27) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_27) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_27) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_27) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_29) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_29) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_29) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_29) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_28) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_28) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_28) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_30) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_28) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_28) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_28) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_28) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_30) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_30) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_30) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_30) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_29) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_29) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_29) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_31) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_29) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_29) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_29) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_29) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_31) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_31) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_31) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_31) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_30) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_30) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_30) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_32) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_30) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_cfd_advection3d_30) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_GT(interp.state().matrices.at("u3").rows(), 0u);

    expect_ok(interp, "uf3 = cfd_advection3d(8, 8, 8, 1, 0, 0, 0.2, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf3"), 0u);
    EXPECT_GT(interp.state().matrices.at("uf3").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh1d_fem_stiffness_1d_30) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "K1 = fem_stiffness_1d(m1)");
    ASSERT_GT(interp.state().matrices.count("K1"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K1").rows(), 9u);
}

TEST(ReplCommandsTest, fem_load_1d_fem_lagrange_eval_30) {
    Interpreter interp;

    expect_ok(interp, "m1 = fem_mesh1d(0, 1, 8)");
    expect_ok(interp, "f1 = fem_load_1d(m1, 1)");
    ASSERT_GT(interp.state().matrices.count("f1"), 0u);
    expect_ok(interp, "lb = fem_lagrange_eval(0.5)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("lb").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_evolve_32) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "u3 = fem_solve_3d(K3, f3)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);

    expect_ok(interp, "ld = fem_lagrange_deriv(0.5)");
    ASSERT_GT(interp.state().matrices.count("ld"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ld").cols(), 2u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "psit = quantum_time_evolve_psi(H, psi, 0.1)");
    ASSERT_GT(interp.state().matrices.count("psit"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psit").rows(), 2u);
}

TEST(ReplCommandsTest, dirichlet_schrodinger_32) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    ASSERT_GT(interp.state().matrices.count("sys3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    ASSERT_GT(interp.state().matrices.count("psif"), 0u);
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}

TEST(ReplCommandsTest, fem3d_mesh_stiff_32) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    ASSERT_GT(interp.state().matrices.count("K3"), 0u);
    EXPECT_GT(interp.state().matrices.at("K3").rows(), 0u);
}

TEST(ReplCommandsTest, load3d_grid3d_32) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    ASSERT_GT(interp.state().matrices.count("f3"), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    ASSERT_GT(interp.state().matrices.count("g3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("g3").rows(), 4u);
}

TEST(ReplCommandsTest, fem_poisson_cfd_adv1d_31) {
    Interpreter interp;

    expect_ok(interp, "u = fem_poisson2d(3, 3)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_GT(interp.state().matrices.at("u").rows(), 0u);

    expect_ok(interp, "f = cfd_advection1d(8, 1, 0.05, 0.01)");
    ASSERT_GT(interp.state().matrices.count("f"), 0u);
    EXPECT_GT(interp.state().matrices.at("f").rows(), 0u);
}

TEST(ReplCommandsTest, poisson1d_31) {
    Interpreter interp;

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, fem_mesh2d_rectangular_31) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("m")(0, 0), 270.0);
}

TEST(ReplCommandsTest, fem_stiffness_load_33) {
    Interpreter interp;

    expect_ok(interp, "m = fem_mesh2d_rectangular(0, 0, 1, 1, 2, 2)");
    expect_ok(interp, "K = fem_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 9u);
    expect_ok(interp, "K2 = assemble_stiffness_2d(m)");
    EXPECT_EQ(interp.state().matrices.at("K2").rows(), 9u);
    expect_ok(interp, "f = fem_load_2d(m, 1)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 9u);
}

TEST(ReplCommandsTest, fem_solve_31) {
    Interpreter interp;

    expect_ok(interp, "m2 = fem_mesh2d_rectangular(0, 0, 1, 1, 3, 3)");
    expect_ok(interp, "K2 = fem_stiffness_2d(m2)");
    expect_ok(interp, "f2 = fem_load_2d(m2, 1)");
    expect_ok(interp, "u2 = fem_solve(K2, f2)");
    ASSERT_GT(interp.state().matrices.count("u2"), 0u);
}

TEST(ReplCommandsTest, fem_poisson3d_senary_noassign) {
    Interpreter interp;
    expect_contains(interp, "fem_poisson3d(2, 2, 2, 0, 0, 0)", "u =");
    expect_error_contains(interp, "fem_poisson3d(1.5, 2, 2, 0, 0, 0)",
                          "expected non-negative integer nx, ny, and nz");
}
