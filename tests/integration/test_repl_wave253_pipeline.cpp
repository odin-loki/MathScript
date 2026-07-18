// MathScript Integration Tests: REPL Interpreter – Wave 253 Pipeline
//
// Wired: crypto_aes128_decrypt_block (AES-128 ECB single-block decrypt),
// plus Wave 252 smoke APIs already on main:
//   signal_sosfilt OR signal_savgol, crypto_aes256_decrypt_block, mpi_barrier.
// Deterministic so this pipeline can land before other Wave 253 merges.

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

TEST(ReplWave253Pipeline, CryptoAes128EncryptDecryptRoundTrip) {
    Interpreter interp;

    constexpr const char* key = "2b7e151628aed2a6abf7158809cf4f3c";
    constexpr const char* plain = "3243f6a8885a308d313198a2e0370734";
    constexpr const char* cipher = "3925841d02dc09fbdc118597196a0b32";

    expect_contains(
        interp,
        std::string("crypto_aes128_encrypt_block(\"") + key + "\", \"" + plain + "\")",
        cipher);

    expect_contains(
        interp,
        std::string("crypto_aes128_decrypt_block(\"") + key + "\", \"" + cipher + "\")",
        plain);

    expect_error(interp, R"cmd(crypto_aes128_decrypt_block("00", "3925841d02dc09fbdc118597196a0b32"))cmd");
    expect_error(interp, R"cmd(crypto_aes128_decrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "00"))cmd");
    expect_error(interp, "crypto_aes128_decrypt_block()");
    expect_contains(interp, "help", "crypto_aes128_decrypt_block");
}

TEST(ReplWave253Pipeline, Wave252SignalSosfiltSmoke) {
    Interpreter interp;

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("y").cols(), 1u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 1.0, 1e-12);
    expect_contains(interp, "help", "signal_sosfilt(sos,x)");
}

TEST(ReplWave253Pipeline, Wave252CryptoAes256DecryptSmoke) {
    Interpreter interp;

    expect_contains(
        interp,
        R"cmd(crypto_aes256_decrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "a36452d23436433a516cace8bf319e9c"))cmd",
        "6bc1bee22e409f96e93d7e117393172a");
    expect_contains(interp, "help", "crypto_aes256_decrypt_block");
}

TEST(ReplWave253Pipeline, Wave252MpiBarrierSmoke) {
    Interpreter interp;

    expect_contains(interp, "mpi_barrier()", "ok");
    expect_contains(interp, "help", "mpi_barrier()");
}
