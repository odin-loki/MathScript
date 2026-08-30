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

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_2) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_2) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_2) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_3) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_3) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_3) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_4) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_4) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_4) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_5) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_5) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_5) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_6) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_6) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_6) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_7) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_7) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_7) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_8) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_8) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_8) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_9) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_9) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_9) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_10) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_10) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_10) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_11) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_11) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_11) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_12) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_12) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_12) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_13) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_13) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_13) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_14) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_14) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_14) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_15) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_15) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_15) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_16) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_16) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_16) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_17) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_17) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_17) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_18) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_18) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_18) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_19) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_19) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_19) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_20) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_20) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_20) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_21) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_21) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_21) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_22) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_22) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_22) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_23) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_23) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_23) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_24) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_24) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_24) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_25) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_25) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_25) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_26) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_26) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_26) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_27) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_27) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_27) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_28) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_28) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_28) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_29) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_29) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_29) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_30) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_30) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_30) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_31) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_31) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_31) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_solve_cg_gmres_jacobi_32) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("x")(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("x")(1, 0), 7.0 / 11.0, 1e-6);

    expect_ok(interp, "xcg = dist_cg(A, b)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xcg").rows(), 2u);

    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xgm").rows(), 2u);

    expect_ok(interp, "xj = dist_jacobi(A, b)");
    ASSERT_GT(interp.state().matrices.count("xj"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xj").rows(), 2u);
}

TEST(ReplCommandsTest, dist_bicgstab_minres_qmr_tfqmr_32) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xb = dist_bicgstab(A, b)");
    ASSERT_GT(interp.state().matrices.count("xb"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xb").rows(), 2u);

    expect_ok(interp, "xn = dist_minres(A, b)");
    ASSERT_GT(interp.state().matrices.count("xn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xn").rows(), 2u);

    expect_ok(interp, "xq = dist_qmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "xt = dist_tfqmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xt").rows(), 2u);
}

TEST(ReplCommandsTest, dist_lsmr_lsqr_matmul_32) {
    Interpreter interp;

    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "xl = dist_lsmr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xl"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xl").rows(), 2u);

    expect_ok(interp, "xq = dist_lsqr(A, b)");
    ASSERT_GT(interp.state().matrices.count("xq"), 0u);
    EXPECT_EQ(interp.state().matrices.at("xq").rows(), 2u);

    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("P")(0, 0), 19.0, 1e-6);
}

TEST(ReplCommandsTest, dist_tfqmr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_tfqmr(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_lsmr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_lsmr(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_lsqr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_lsqr(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_matmul_noassign) {
    Interpreter interp;
    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_contains(interp, "dist_matmul(M, N)", "C =");
}

TEST(ReplCommandsTest, dist_solve_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_solve(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_cg_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_cg(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_gmres_noassign) {
    Interpreter interp;
    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_contains(interp, "dist_gmres(G, brhs)", "x =");
}

TEST(ReplCommandsTest, dist_jacobi_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_jacobi(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_bicgstab_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_bicgstab(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_minres_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_minres(A, b)", "x =");
}

TEST(ReplCommandsTest, dist_qmr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "dist_qmr(A, b)", "x =");
}

TEST(ReplCommandsTest, bicgstab_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "bicgstab(A, b)", "x =");
    expect_error_contains(interp, "bicgstab(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, cg_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "cg(A, b)", "x =");
    expect_error_contains(interp, "cg(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, gmres_noassign) {
    Interpreter interp;
    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_contains(interp, "gmres(G, brhs)", "x =");
    expect_error_contains(interp, "gmres(no_such_matrix, brhs)", "unknown matrix");
}

TEST(ReplCommandsTest, jacobi_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "jacobi(A, b)", "x =");
    expect_error_contains(interp, "jacobi(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, qmr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "qmr(A, b)", "x =");
    expect_error_contains(interp, "qmr(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, lsqr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "lsqr(A, b)", "x =");
    expect_error_contains(interp, "lsqr(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, tfqmr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "tfqmr(A, b)", "x =");
    expect_error_contains(interp, "tfqmr(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, lsmr_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "lsmr(A, b)", "x =");
    expect_error_contains(interp, "lsmr(no_such_matrix, b)", "unknown matrix");
}

TEST(ReplCommandsTest, minres_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_contains(interp, "minres(A, b)", "x =");
    expect_error_contains(interp, "minres(no_such_matrix, b)", "unknown matrix");
}
