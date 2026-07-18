// MathScript Integration Tests: REPL Interpreter – Wave 259 Pipeline
//
// Smoke: Wave 258 APIs already on main (sqrtm/tril, imtophat/imhist,
// graph_dijkstra, info_permutation_entropy). Deterministic; Wave 259
// features belong in separate tests once merged.

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplWave259Pipeline, SqrtmDiagonal) {
    Interpreter interp;

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 1), 3.0, 1e-6);
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("S")(1, 0), 0.0, 1e-9);
    expect_contains(interp, "help", "sqrtm(A)");
}

TEST(ReplWave259Pipeline, TrilExtract) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Lwr = tril(A)");
    ASSERT_GT(interp.state().matrices.count("Lwr"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Lwr")(1, 1), 4.0, 1e-9);
    expect_contains(interp, "help", "tril(A[, k])");
}

TEST(ReplWave259Pipeline, ImtophatPeak) {
    Interpreter interp;

    expect_ok(interp, "M = [0, 0, 0; 0, 1, 0; 0, 0, 0]");
    expect_ok(interp, "T = imtophat(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);
    EXPECT_GE(interp.state().matrices.at("T")(1, 1), 0.0);
    expect_contains(interp, "help", "imtophat(M[,k])");
}

TEST(ReplWave259Pipeline, ImhistBins) {
    Interpreter interp;

    expect_ok(interp, "Hsrc = [0, 0.25; 0.5, 0.75]");
    expect_ok(interp, "H = imhist(Hsrc, 4)");
    ASSERT_GT(interp.state().matrices.count("H"), 0u);
    EXPECT_EQ(interp.state().matrices.at("H").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 1u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("H")(3, 0), 0.0);
    expect_contains(interp, "help", "imhist(M[,nbins])");
}

TEST(ReplWave259Pipeline, GraphDijkstra) {
    Interpreter interp;

    expect_ok(interp, "A = [0, 1, 0; 0, 0, 2; 0, 0, 0]");
    expect_ok(interp, "D = graph_dijkstra(A, 0)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& dist = interp.state().matrices.at("D");
    EXPECT_EQ(dist.rows(), 3u);
    EXPECT_EQ(dist.cols(), 2u);
    EXPECT_NEAR(dist(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(dist(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(dist(2, 0), 3.0, 1e-9);
    EXPECT_NEAR(dist(0, 1), -1.0, 1e-9);
    EXPECT_NEAR(dist(1, 1), 0.0, 1e-9);
    EXPECT_NEAR(dist(2, 1), 1.0, 1e-9);
    expect_contains(interp, "help", "graph_dijkstra(A,source)");
}

TEST(ReplWave259Pipeline, InfoPermutationEntropy) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]");
    expect_ok(interp, "pe = info_permutation_entropy(x)");
    ASSERT_GT(interp.state().scalars.count("pe"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("pe"), 0.0, 1e-10);

    expect_ok(interp, "y = [1; 3; 2; 5; 4; 6]");
    expect_ok(interp, "pe2 = info_permutation_entropy(y, 3, 1)");
    EXPECT_NEAR(interp.state().scalars.at("pe2"), 1.0 / std::log2(6.0), 1e-10);
    expect_contains(interp, "help", "info_permutation_entropy");
}
