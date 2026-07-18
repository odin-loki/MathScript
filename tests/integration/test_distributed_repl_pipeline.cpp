// MathScript Integration Tests: REPL MPI / distributed bindings pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

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

    // identity system via direct dist_solve call
    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "rhs = [3; 5]");
    expect_contains(interp, "dist_solve(I, rhs)", "3");
    expect_contains(interp, "dist_solve(I, rhs)", "5");
}
