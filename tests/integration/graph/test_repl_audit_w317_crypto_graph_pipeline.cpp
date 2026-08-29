// MathScript Integration Tests: REPL Interpreter – Audit Wave 317 Mixed Cleanup Pipeline
//
// Inventory: leftover C++ names are already bound under REPL aliases
// (crypto_sha256/sha512/hmac/pbkdf2/chacha20/x25519, fem_*, graph_is_planar,
// stats_levene/bartlett/ks_2sample). Optim leftovers need callbacks. CUDA stays
// gated on MS_ENABLE_CUDA.

#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplAuditW317CryptoGraphPipeline, PathGraphIsPlanar) {
    Interpreter interp;
    expect_contains(interp, "help", "graph_is_planar(A)");

    expect_ok(interp, "A = [0, 1, 0; 1, 0, 1; 0, 1, 0]");
    expect_ok(interp, "p = graph_is_planar(A)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("p"), 1.0);
}

TEST(ReplAuditW317CryptoGraphPipeline, Sha256EmptyHexKnownDigest) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_sha256(hex_data)");

    // SHA-256 of empty input (hex "").
    const auto result = interp.execute("crypto_sha256(\"\")");
    ASSERT_TRUE(result.has_value()) << (result ? *result : "unknown");
    EXPECT_NE(result->find("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
              std::string::npos)
        << "output: " << *result;
}
