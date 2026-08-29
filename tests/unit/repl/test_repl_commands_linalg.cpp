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

TEST(ReplCommandsTest, imcrop) {
    Interpreter interp;
    expect_contains(interp, "help", "imcrop(M,r0,c0,r1,c1)");

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    ASSERT_GT(interp.state().matrices.count("crop"), 0u);
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("crop").cols(), 4u);
}

TEST(ReplCommandsTest, hess_schur) {
    Interpreter interp;
    expect_contains(interp, "help", "hess(A)");
    expect_contains(interp, "help", "T, Q = schur(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "hess(A)", "H =");
    expect_ok(interp, "H = hess(A)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);

    expect_contains(interp, "schur(A)", "T =");
    expect_ok(interp, "T = schur(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Ts, Qs = schur(A)");
    ASSERT_GT(interp.state().matrices.count("Ts"), 0u);
    ASSERT_GT(interp.state().matrices.count("Qs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ts").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Qs").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Qs").cols(), 3u);
}

TEST(ReplCommandsTest, bidiag) {
    Interpreter interp;
    expect_contains(interp, "help", "U, B, V = bidiag(A)");
    expect_contains(interp, "help", "bidiag(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "bidiag(A)", "B =");
    expect_ok(interp, "B = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("B").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "U, Bd, V = bidiag(A)");
    ASSERT_GT(interp.state().matrices.count("U"), 0u);
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("U").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Bd").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 3u);
}

TEST(ReplCommandsTest, solve_sylvester) {
    Interpreter interp;
    expect_contains(interp, "help", "solve_sylvester(A,B,C)");

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    // A*X + X*B with X=[1,2;3,4] => C=[4,10;15,24]
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 1), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(1, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("X")(1, 1), 4.0, 1e-8);
}

TEST(ReplCommandsTest, minres) {
    Interpreter interp;
    expect_contains(interp, "help", "minres(A");

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xm").cols(), 1u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("xm")(0, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("xm")(1, 0)));
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("xm")(2, 0)));
}

TEST(ReplCommandsTest, cg) {
    Interpreter interp;
    expect_contains(interp, "help", "cg(A");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xc").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xc").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xc")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xc")(2, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, gmres) {
    Interpreter interp;
    expect_contains(interp, "help", "gmres(A");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xg").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xg").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xg")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xg")(2, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, jacobi) {
    Interpreter interp;
    expect_contains(interp, "help", "jacobi(A");

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("xj").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xj")(1, 0), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("xj")(2, 0), 4.0, 1e-6);
}

TEST(ReplCommandsTest, eig) {
    Interpreter interp;
    expect_contains(interp, "help", "D, V = eig(A)");
    expect_contains(interp, "help", "eig(A)");

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_contains(interp, "eig(A)", "eigenvalues:");
    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 1u);

    expect_ok(interp, "De, Ve = eig(A)");
    ASSERT_GT(interp.state().matrices.count("De"), 0u);
    ASSERT_GT(interp.state().matrices.count("Ve"), 0u);
    EXPECT_EQ(interp.state().matrices.at("De").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Ve").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Ve").cols(), 3u);
}

TEST(ReplCommandsTest, ldl) {
    Interpreter interp;
    expect_contains(interp, "help", "L, D = ldl(A)");
    expect_contains(interp, "help", "ldl(A)");

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_contains(interp, "ldl(S)", "L =");
    expect_contains(interp, "ldl(S)", "D =");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 1), 0.0, 1e-10);

    expect_ok(interp, "Ll, Dl = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("Ll"), 0u);
    ASSERT_GT(interp.state().matrices.count("Dl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ll").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Dl").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Dl").cols(), 1u);

    expect_ok(interp, "Lp, Dp, Pp = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Pp").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Pp").cols(), 2u);
}

TEST(ReplCommandsTest, diag) {
    Interpreter interp;
    expect_contains(interp, "help", "diag(v)");

    expect_ok(interp, "v = [2; 3; 5]");
    expect_contains(interp, "diag(v)", "D =");
    expect_ok(interp, "D = diag(v)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("D").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(2, 2), 5.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 0), 0.0, 1e-12);
}

TEST(ReplCommandsTest, linalg_funm_precond) {
    Interpreter interp;
    expect_contains(interp, "help", "matrix_rank(A[, tol])");
    expect_contains(interp, "help", "funm(A, \"sin\"|\"cos\"|\"exp\"|\"sqrt\")");
    expect_contains(interp, "help", "precond_diag(A)");
    expect_contains(interp, "help", "precond_ssor(A[, omega])");

    expect_ok(interp, "A = [1, 2; 2, 4]");
    expect_ok(interp, "r = matrix_rank(A)");
    EXPECT_NEAR(interp.state().scalars.at("r"), 1.0, 1e-12);
    expect_contains(interp, "matrix_rank([1, 2; 2, 4])", "1");

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 1), std::exp(1.0), 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 1), 0.0, 1e-6);

    expect_ok(interp, "A2 = [4, 0; 0, 2]");
    expect_ok(interp, "Pd = precond_diag(A2)");
    ASSERT_EQ(interp.state().matrices.at("Pd").rows(), 2u);
    ASSERT_EQ(interp.state().matrices.at("Pd").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("Pd")(0, 0), 0.25, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Pd")(1, 0), 0.5, 1e-12);

    expect_ok(interp, "M = [4, 1; 1, 3]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_EQ(interp.state().matrices.at("Ps").rows(), 2u);
    ASSERT_EQ(interp.state().matrices.at("Ps").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("Ps")(0, 0), 4.0 / 1.2, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Ps")(1, 1), 3.0 / 1.2, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Ps")(0, 1), 0.0, 1e-12);
}

TEST(ReplCommandsTest, pde_sparse_control) {
    Interpreter interp;

    expect_ok(interp, "u0 = zeros(4, 1)");
    expect_ok(interp, "v0 = zeros(4, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 5)");
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 4u);

    expect_ok(interp, "ri = [0; 1]");
    expect_ok(interp, "ci = [0; 1]");
    expect_ok(interp, "vv = [2; 3]");
    expect_ok(interp, "A = sparse_from_coo(2, 2, ri, ci, vv)");
    expect_ok(interp, "y = sparse_spmv(A, [1; 2])");
    EXPECT_GT(interp.state().matrices.at("y").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    EXPECT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(ReplCommandsTest, linalg_geo) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "H = hess(A)");
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);

    expect_ok(interp, "T = schur(A)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);

    expect_ok(interp, "pt = geo_bezier_eval([0, 0; 1, 2; 2, 0], 0.5)");
    EXPECT_EQ(interp.state().matrices.at("pt").cols(), 2u);
}

TEST(ReplCommandsTest, ml_linalg_graph) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_EQ(interp.state().matrices.at("At").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-9);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-6);

    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);
}

TEST(ReplCommandsTest, wave2d_fem_poisson1d) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    EXPECT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, morph_bilateral) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
}

TEST(ReplCommandsTest, imcrop_triangulate) {
    Interpreter interp;

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);

    expect_ok(interp, "P = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "T = geo_triangulate_polygon(P)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
}

TEST(ReplCommandsTest, hough) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, logm_cosm_sinm) {
    Interpreter interp;

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_sylvester) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, transpose_funm) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, arb_imfilter) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_poisson1d) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);

    expect_ok(interp, "u1 = fem_poisson1d(6)");
    ASSERT_GT(interp.state().matrices.count("u1"), 0u);
    ASSERT_GT(interp.state().matrices.at("u1").rows(), 0u);
}

TEST(ReplCommandsTest, solve_iter) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, morph_bilateral_2) {
    Interpreter interp;

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,1,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "D = imdilate(M, 3)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    expect_ok(interp, "E = imerode(M, 3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);

    expect_ok(interp, "U = ones(5, 5)");
    expect_ok(interp, "B = bilateral(U, 1, 0.1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 5u);
}

TEST(ReplCommandsTest, imcrop_triangulate_2) {
    Interpreter interp;

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);

    expect_ok(interp, "P = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "T = geo_triangulate_polygon(P)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
}

TEST(ReplCommandsTest, hough_2) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, logm_cosm_sinm_2) {
    Interpreter interp;

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_2) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_sylvester_2) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_2) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_2) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, transpose_funm_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, arb_imfilter_2) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, solve_iter_2) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_2) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_2) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_2) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_2) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_3) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_3) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_3) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_3) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_3) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_3) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_3) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_3) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_3) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_4) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_2) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_2) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_4) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_2) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_4) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_4) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_2) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_2) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_4) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_4) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_4) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_4) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_4) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_5) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_3) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_3) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_5) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_3) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_5) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_5) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_3) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_3) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_5) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_5) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_5) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_5) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_5) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_6) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_4) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_4) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_6) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_4) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_6) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_6) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_4) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_4) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_6) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_6) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_6) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_6) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_6) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_7) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_5) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_5) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_7) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_5) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_7) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_7) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_5) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_5) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_7) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_7) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_7) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_7) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_7) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_8) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_6) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_6) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_8) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_6) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_8) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_8) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_6) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_6) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_8) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_8) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_8) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_8) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_8) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_9) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_7) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_7) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_9) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_7) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_9) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_9) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_7) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_7) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_9) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_9) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_9) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_9) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_9) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_10) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_8) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_8) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_10) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_8) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_10) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_10) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_8) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_8) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_10) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_10) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_10) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_10) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_10) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_11) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_9) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_9) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_11) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_9) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_11) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_11) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_9) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_9) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_11) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_11) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_11) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_11) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_11) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_12) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_10) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_10) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_12) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_10) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_12) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_12) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_10) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_10) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_12) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_12) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_12) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_12) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_12) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_13) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_11) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_11) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_13) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_11) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_13) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_13) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_11) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_11) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_13) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_13) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_13) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_13) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_13) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_14) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_12) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_12) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_14) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_12) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_14) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_14) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_12) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_12) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_14) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_14) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_14) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_14) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_14) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_15) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_13) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_13) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_15) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_13) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_15) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_15) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_13) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_13) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_15) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_15) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_15) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_15) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_15) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_16) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_14) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_14) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_16) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_14) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_16) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_16) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_14) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_14) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_16) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_16) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_16) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_16) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_16) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_17) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_15) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_15) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_17) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_15) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_17) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_17) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_15) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_15) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_17) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_17) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_17) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_17) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_17) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_18) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_16) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_16) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_18) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_16) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_18) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_18) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_16) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_16) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_18) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_18) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_18) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_18) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_18) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_19) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_17) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_17) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_19) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_17) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_19) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_19) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_17) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_17) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_19) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_19) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_19) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_19) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_19) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_20) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_18) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_18) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_20) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_18) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_20) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_20) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_18) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_18) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_20) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_20) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_20) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_20) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_20) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_21) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_19) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_19) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_21) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_19) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_21) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_21) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_19) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_19) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_21) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_21) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_21) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_21) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_21) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_22) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_20) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_20) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_22) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_20) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_22) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_22) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_20) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_20) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_22) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_22) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_22) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_22) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_22) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_23) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_21) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_21) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_23) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_21) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_23) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_23) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_21) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_21) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_23) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_23) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_23) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_23) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_23) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_24) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_22) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_22) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_24) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_22) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_24) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_24) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_22) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_22) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_24) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_24) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_24) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_24) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_24) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_25) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_23) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_23) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_25) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_23) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_25) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_25) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_23) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_23) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_25) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_25) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_25) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_25) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_25) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_26) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_24) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_24) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_26) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_24) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_26) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_26) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_24) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_24) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_26) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_26) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_26) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_26) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_26) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_27) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_25) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_25) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_27) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_25) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_27) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_27) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_25) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_25) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_27) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_27) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_27) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_27) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_27) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_28) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_26) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_26) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_28) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_26) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_28) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_28) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_26) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_26) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_28) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_28) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_28) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_28) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_28) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_29) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_27) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_27) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_29) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_27) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_29) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_29) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_27) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_27) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_29) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_29) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_29) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_29) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_29) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_30) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_28) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_28) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_30) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_28) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_30) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_30) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_28) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_28) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_30) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_30) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_30) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_30) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_30) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_31) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_29) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_29) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_31) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_29) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_31) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_31) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_29) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_29) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_31) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_31) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_31) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_31) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_31) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_32) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_30) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_30) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_32) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_32) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_32) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_30) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_32) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_32) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}

TEST(ReplCommandsTest, precond_diag_precond_ssor_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(ReplCommandsTest, graph_min_arborescence_imfilter_30) {
    Interpreter interp;

    expect_ok(interp, "arb = graph_min_arborescence([0, 1, 10; 0, 0, 2; 0, 0, 0], 0)");
    ASSERT_GT(interp.state().matrices.count("arb"), 0u);
    EXPECT_GT(interp.state().matrices.at("arb").rows(), 0u);

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "K = ones(3, 3) / 9");
    expect_ok(interp, "F = imfilter(G, K)");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    EXPECT_EQ(interp.state().matrices.at("F").rows(), 2u);
}

TEST(ReplCommandsTest, wave2d_30) {
    Interpreter interp;

    expect_ok(interp, "u0w = zeros(7, 7)");
    expect_ok(interp, "v0w = zeros(7, 7)");
    expect_ok(interp, "w2 = pde_wave_2d(u0w, v0w, 1.0, 0.1, 0.1, 0.02, 8)");
    ASSERT_GT(interp.state().matrices.count("w2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w2").rows(), 7u);
    EXPECT_EQ(interp.state().matrices.at("w2").cols(), 7u);
}

TEST(ReplCommandsTest, solve_iter_32) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xb = bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xq = qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);
}

TEST(ReplCommandsTest, lsqr_lsq_tfqmr_lsmr_32) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xs = lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xs").rows(), 2u);

    expect_ok(interp, "xl = lsq(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xt = tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);

    expect_ok(interp, "xm = lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 2u);
}

TEST(ReplCommandsTest, transpose_chol_expm_inv_32) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    ASSERT_GT(interp.state().matrices.count("Bi"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(ReplCommandsTest, zeros_eye_ones_rand_32) {
    Interpreter interp;

    expect_ok(interp, "Z = zeros(3)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    ASSERT_GT(interp.state().matrices.count("I"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    ASSERT_GT(interp.state().matrices.count("O"), 0u);
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(ReplCommandsTest, pinv_null_orth_kron_32) {
    Interpreter interp;

    expect_ok(interp, "A = [3, 1; 1, 2]");
    expect_ok(interp, "P = pinv(A)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    expect_ok(interp, "AP = matmul(A, P)");
    ASSERT_GT(interp.state().matrices.count("AP"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(0, 1), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("AP")(1, 0), 0.0, 1e-8);

    expect_ok(interp, "W = [1, 2, 3, 4; 2, 4, 6, 8]");
    expect_ok(interp, "N = null(W)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 4u);
    expect_ok(interp, "WN = matmul(W, N)");
    ASSERT_GT(interp.state().matrices.count("WN"), 0u);
    const auto& wn = interp.state().matrices.at("WN");
    for (size_t i = 0; i < wn.rows(); ++i) {
        for (size_t j = 0; j < wn.cols(); ++j) {
            EXPECT_NEAR(wn(i, j), 0.0, 1e-8);
        }
    }

    expect_ok(interp, "M = [1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 0; 0, 1, 1]");
    expect_ok(interp, "Q = orth(M)");
    ASSERT_GT(interp.state().matrices.count("Q"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Q").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Q").cols(), 3u);

    expect_ok(interp, "K = kron(eye(2), eye(2))");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("K").cols(), 4u);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(1, 1), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("K")(0, 1), 0.0, 1e-8);
}

TEST(ReplCommandsTest, repmat_linspace_rgb_32) {
    Interpreter interp;

    expect_ok(interp, "R = repmat([1, 2; 3, 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);

    expect_ok(interp, "V = linspace(0, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("V"), 0u);
    EXPECT_EQ(interp.state().matrices.at("V").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("V")(0, 0), 0.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("V")(4, 0), 1.0, 1e-8);

    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_GT(interp.state().matrices.count("G"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("G")(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("G")(1, 0), 0.587, 1e-6);

    expect_ok(interp, "RGB3 = [1, 0, 0; 0, 1, 0; 0, 0, 1]");
    expect_ok(interp, "HSV = rgb2hsv(RGB3)");
    ASSERT_GT(interp.state().matrices.count("HSV"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("HSV")(1, 0), 1.0 / 3.0, 1e-6);
}

TEST(ReplCommandsTest, hough_33) {
    Interpreter interp;

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(ReplCommandsTest, sqrtm_logm_31) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);
}

TEST(ReplCommandsTest, cosm_sinm_31) {
    Interpreter interp;

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(ReplCommandsTest, diag_tril_33) {
    Interpreter interp;

    expect_ok(interp, "v = [1; 2; 3]");
    expect_ok(interp, "D = diag(v)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
}

TEST(ReplCommandsTest, triu_hess_33) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Upr = triu(A)");
    EXPECT_NEAR(interp.state().matrices.at("Upr")(1, 0), 0.0, 1e-9);

    expect_ok(interp, "H = hess([1, 2, 3; 4, 5, 6; 7, 8, 9])");
    EXPECT_NEAR(interp.state().matrices.at("H")(2, 0), 0.0, 1e-10);
}

TEST(ReplCommandsTest, bidiag_eig_33) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6; 7, 8, 9]");
    expect_ok(interp, "B = bidiag(A)");
    EXPECT_NEAR(interp.state().matrices.at("B")(2, 0), 0.0, 1e-8);

    expect_ok(interp, "D = eig(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
}

TEST(ReplCommandsTest, ldl_solve_sylvester_31) {
    Interpreter interp;

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(ReplCommandsTest, minres_cg_33) {
    Interpreter interp;

    expect_ok(interp, "Am = [4, 1, 0; 1, 3, 1; 0, 1, 2]");
    expect_ok(interp, "bm = [1; 1; 1]");
    expect_ok(interp, "xm = minres(Am, bm)");
    EXPECT_EQ(interp.state().matrices.at("xm").rows(), 3u);

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xc = cg(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xc")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, gmres_jacobi_33) {
    Interpreter interp;

    expect_ok(interp, "Ai = eye(3)");
    expect_ok(interp, "bi = [2; 3; 4]");
    expect_ok(interp, "xg = gmres(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xg")(0, 0), 2.0, 1e-6);
    expect_ok(interp, "xj = jacobi(Ai, bi)");
    EXPECT_NEAR(interp.state().matrices.at("xj")(0, 0), 2.0, 1e-6);
}

TEST(ReplCommandsTest, ml_mat_transpose_funm_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "At = ml_mat_transpose(A)");
    ASSERT_GT(interp.state().matrices.count("At"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("At")(0, 1), 3.0, 1e-8);

    expect_ok(interp, "I2 = eye(2)");
    expect_ok(interp, "S = funm(I2, \"exp\")");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), std::exp(1.0), 1e-8);
}
