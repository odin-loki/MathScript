// MathScript Integration Tests: REPL Interpreter – Wave 236 Bindings Pipeline
//
// Wired: fem_poisson3d, crypto_x25519_shared, dist_matmul.

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave236Pipeline, FemPoisson3dBinding) {
    Interpreter interp;

    expect_ok(interp, "u3 = fem_poisson3d(2, 2, 2)");
    ASSERT_GT(interp.state().matrices.count("u3"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u3").rows(), 27u);
    EXPECT_GT(interp.state().matrices.at("u3")(13, 0), 0.0);

    expect_error(interp, "fem_poisson3d(0, 2, 2)");
    expect_error(interp, "fem_poisson3d(2, 2)");

    expect_contains(interp, "help", "fem_poisson3d");
}

TEST(ReplWave236Pipeline, CryptoX25519SharedBinding) {
    Interpreter interp;

    expect_contains(
        interp,
        R"cmd(crypto_x25519_shared("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4", "e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"))cmd",
        "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");
}

TEST(ReplWave236Pipeline, DistMatmulBinding) {
    Interpreter interp;

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
}
