// MathScript Integration Tests: REPL Interpreter – Wave 251 Pipeline
//
// Wired: crypto_aes256_encrypt_block (AES-256 ECB single-block),
// plus Wave 250 smoke APIs already on main:
//   crypto_sha512, cuda_nccl_comm_size, signal_filtfilt.
// Optional Wave 251 peers (e.g. signal_filter) are not required here.

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

TEST(ReplWave251Pipeline, CryptoAes256EncryptBlock) {
    Interpreter interp;

    // NIST FIPS-197 Appendix C.3 / unit test vector
    expect_contains(
        interp,
        R"cmd(crypto_aes256_encrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "6bc1bee22e409f96e93d7e117393172a"))cmd",
        "a36452d23436433a516cace8bf319e9c");

    expect_error(interp, R"cmd(crypto_aes256_encrypt_block("00", "6bc1bee22e409f96e93d7e117393172a"))cmd");
    expect_error(interp, R"cmd(crypto_aes256_encrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "00"))cmd");
    expect_error(interp, "crypto_aes256_encrypt_block()");
    expect_contains(interp, "help", "crypto_aes256_encrypt_block");
}

TEST(ReplWave251Pipeline, Wave250CryptoSha512Smoke) {
    Interpreter interp;

    expect_contains(
        interp, "crypto_sha512(\"\")",
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
    expect_contains(interp, "help", "crypto_sha512(hex_data)");
}

TEST(ReplWave251Pipeline, Wave250CudaNcclCommSizeSmoke) {
    Interpreter interp;

    expect_ok(interp, "s = cuda_nccl_comm_size()");
    ASSERT_GT(interp.state().scalars.count("s"), 0u);
    EXPECT_GE(interp.state().scalars.at("s"), 1.0);
    expect_contains(interp, "help", "cuda_nccl_comm_size()");
}

TEST(ReplWave251Pipeline, Wave250SignalFiltfiltSmoke) {
    Interpreter interp;

    expect_ok(interp, "b = [0.5, 0.5]");
    expect_ok(interp, "a = [1]");
    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "y = signal_filtfilt(b, a, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("y").cols(), 1u);
    expect_contains(interp, "help", "signal_filtfilt(b,a,x)");
}
