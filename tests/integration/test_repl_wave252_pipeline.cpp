// MathScript Integration Tests: REPL Interpreter – Wave 252 Pipeline
//
// Wired: crypto_aes256_decrypt_block (AES-256 ECB single-block decrypt),
// plus Wave 251 smoke APIs already on main:
//   crypto_aes256_encrypt_block, mpi_allreduce_max, geo_convex_hull, signal_filter.
// Deterministic so this pipeline can land before other Wave 252 merges.

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

void expect_error(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    EXPECT_FALSE(result.has_value()) << cmd;
}

} // namespace

TEST(ReplWave252Pipeline, CryptoAes256EncryptDecryptRoundTrip) {
    Interpreter interp;

    constexpr const char* key =
        "603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41";
    constexpr const char* plain = "6bc1bee22e409f96e93d7e117393172a";
    constexpr const char* cipher = "a36452d23436433a516cace8bf319e9c";

    expect_contains(
        interp,
        std::string("crypto_aes256_encrypt_block(\"") + key + "\", \"" + plain + "\")",
        cipher);

    expect_contains(
        interp,
        std::string("crypto_aes256_decrypt_block(\"") + key + "\", \"" + cipher + "\")",
        plain);

    expect_error(interp, R"cmd(crypto_aes256_decrypt_block("00", "a36452d23436433a516cace8bf319e9c"))cmd");
    expect_error(interp, R"cmd(crypto_aes256_decrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "00"))cmd");
    expect_error(interp, "crypto_aes256_decrypt_block()");
    expect_contains(interp, "help", "crypto_aes256_decrypt_block");
}

TEST(ReplWave252Pipeline, Wave251CryptoAes256EncryptSmoke) {
    Interpreter interp;

    expect_contains(
        interp,
        R"cmd(crypto_aes256_encrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "6bc1bee22e409f96e93d7e117393172a"))cmd",
        "a36452d23436433a516cace8bf319e9c");
    expect_contains(interp, "help", "crypto_aes256_encrypt_block");
}

TEST(ReplWave252Pipeline, Wave251MpiAllreduceMaxSmoke) {
    Interpreter interp;

    expect_ok(interp, "mx = mpi_allreduce_max(3.5)");
    ASSERT_GT(interp.state().scalars.count("mx"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("mx"), 3.5, 1e-9);
    // Help collapses sum/max/min into one line: mpi_allreduce_sum/max/min(x)
    expect_contains(interp, "help", "mpi_allreduce_sum/max/min");
}

TEST(ReplWave252Pipeline, Wave251GeoConvexHullSmoke) {
    Interpreter interp;

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    ASSERT_GT(interp.state().matrices.count("hull"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("hull").cols(), 2u);
    expect_contains(interp, "help", "geo_convex_hull(P)");
}

TEST(ReplWave252Pipeline, Wave251SignalFilterSmoke) {
    Interpreter interp;

    expect_ok(interp, "b = [1, -1]");
    expect_ok(interp, "a = [1, -0.5]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_filter(b, a, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("y").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 1.0, 1e-12);
    expect_contains(interp, "help", "signal_filter(b,a,x)");
}
