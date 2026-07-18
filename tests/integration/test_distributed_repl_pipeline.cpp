// MathScript Integration Tests: REPL MPI / distributed bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms::interp;
using ms::ColMatrix;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(DistributedReplPipeline, MpiAndDistSolveBindings) {
    Interpreter interp;

    // mpi command reports stub backend with rank=0 size=1
    expect_contains(interp, "mpi", "backend=stub");
    expect_contains(interp, "mpi", "rank=0");
    expect_contains(interp, "mpi", "size=1");

    // nullary MPI queries (stub-safe)
    expect_ok(interp, "r = mpi_rank()");
    expect_ok(interp, "sz = mpi_size()");
    EXPECT_NEAR(interp.state().scalars.at("r"), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().scalars.at("sz"), 1.0, 1e-9);

    // allreduce_sum identity on single-rank stub
    expect_ok(interp, "s = mpi_allreduce_sum(3.5)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 3.5, 1e-9);
    expect_ok(interp, "z = mpi_allreduce_sum(0)");
    EXPECT_NEAR(interp.state().scalars.at("z"), 0.0, 1e-9);

    // dist_solve wraps ms::distributed::solve (stub path: local gather + solve)
    // 4x + y = 1, x + 3y = 2  => x=1/11, y=7/11
    expect_ok(interp, "A = [4, 1; 1, 3]");
    expect_ok(interp, "b = [1; 2]");
    expect_ok(interp, "x = dist_solve(A, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 2u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(x(1, 0), 7.0 / 11.0, 1e-6);

    // dist_matmul: row-block scatter + local GEMM (stub: gather + matmul)
    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_ok(interp, "N = [5, 6; 7, 8]");
    expect_ok(interp, "P = dist_matmul(M, N)");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    const auto& P = interp.state().matrices.at("P");
    EXPECT_EQ(P.rows(), 2u);
    EXPECT_EQ(P.cols(), 2u);
    EXPECT_NEAR(P(0, 0), 19.0, 1e-6);
    EXPECT_NEAR(P(0, 1), 22.0, 1e-6);
    EXPECT_NEAR(P(1, 0), 43.0, 1e-6);
    EXPECT_NEAR(P(1, 1), 50.0, 1e-6);
    expect_contains(interp, "dist_matmul(M, N)", "19");

    // identity system via direct dist_solve call
    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "rhs = [3; 5]");
    expect_contains(interp, "dist_solve(I, rhs)", "3");
    expect_contains(interp, "dist_solve(I, rhs)", "5");

    // dist_cg: SPD system via distributed conjugate gradient (stub: gather + cg)
    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "rhs = [1; 2]");
    expect_ok(interp, "xcg = dist_cg(S, rhs)");
    expect_contains(interp, "dist_cg(S, rhs)", "0.090909");
    expect_ok(interp, "xcg = dist_cg(S, rhs)");
    ASSERT_GT(interp.state().matrices.count("xcg"), 0u);
    const auto& xcg = interp.state().matrices.at("xcg");
    EXPECT_EQ(xcg.rows(), 2u);
    EXPECT_EQ(xcg.cols(), 1u);
    EXPECT_NEAR(xcg(0, 0), 1.0 / 11.0, 1e-6);
    EXPECT_NEAR(xcg(1, 0), 7.0 / 11.0, 1e-6);

    // dist_gmres: general system via distributed GMRES (stub: gather + gmres)
    expect_ok(interp, "G = [3, 1; 1, 2]");
    expect_ok(interp, "brhs = [5; 5]");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    expect_ok(interp, "xgm = dist_gmres(G, brhs)");
    ASSERT_GT(interp.state().matrices.count("xgm"), 0u);
    const auto& xgm = interp.state().matrices.at("xgm");
    EXPECT_EQ(xgm.rows(), 2u);
    EXPECT_EQ(xgm.cols(), 1u);
    const auto expected_gm = ms::gmres(
        ColMatrix<double>{{3.0, 1.0}, {1.0, 2.0}},
        ColMatrix<double>{{5.0}, {5.0}}).value();
    EXPECT_NEAR(xgm(0, 0), expected_gm(0, 0), 1e-6);
    EXPECT_NEAR(xgm(1, 0), expected_gm(1, 0), 1e-6);
}
