// MathScript Integration Tests: REPL Interpreter – Wave 256 Pipeline
//
// Smoke: Wave 255 APIs already on main (geo kdtree/triangulate, graph spectral,
// crypto_x25519_keypair, mpi_bcast). Deterministic; Wave 256 features belong in
// separate tests once merged.

#include <gtest/gtest.h>
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

TEST(ReplWave256Pipeline, MpiBcastIdentityAssignment) {
    Interpreter interp;

    expect_ok(interp, "bx = mpi_bcast(3.5)");
    ASSERT_GT(interp.state().scalars.count("bx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("bx"), 3.5, 1e-12);
    expect_contains(interp, "help", "mpi_bcast(x)");
}

TEST(ReplWave256Pipeline, GeoKdtreeKnnSmoke) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    ASSERT_GT(interp.state().matrices.count("n"), 0u);
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("n").cols(), 1u);
    expect_contains(interp, "help", "geo_kdtree_knn(P,x,y,k)");
}

TEST(ReplWave256Pipeline, GeoTriangulatePolygonSmoke) {
    Interpreter interp;

    expect_ok(interp, "P = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "T = geo_triangulate_polygon(P)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);
    expect_contains(interp, "help", "geo_triangulate_polygon(P)");
}

TEST(ReplWave256Pipeline, GraphSpectralSmoke) {
    Interpreter interp;

    expect_ok(interp, "A = [0,1,0; 1,0,1; 0,1,0]");
    expect_ok(interp, "k = graph_katz_centrality(A)");
    ASSERT_GT(interp.state().matrices.count("k"), 0u);
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("k").cols(), 1u);

    expect_ok(interp, "ac = graph_algebraic_connectivity(A)");
    EXPECT_GT(interp.state().scalars.at("ac"), 0.0);
    expect_contains(interp, "help", "graph_katz_centrality(A)");
    expect_contains(interp, "help", "graph_algebraic_connectivity(A)");
}

TEST(ReplWave256Pipeline, CryptoX25519KeypairRfc7748Alice) {
    Interpreter interp;

    expect_contains(interp, "help", "crypto_x25519_keypair");
    expect_contains(
        interp,
        R"cmd(crypto_x25519_keypair("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"))cmd",
        "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
}
