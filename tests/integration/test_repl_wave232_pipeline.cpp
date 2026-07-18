// MathScript Integration Tests: REPL Interpreter – Wave 231/232 Bindings Pipeline

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

TEST(ReplWave232Pipeline, SymbolicTransformBindings) {
    Interpreter interp;

    expect_contains(interp, R"cmd(sym_laplace("exp(2*t)", "t", "s"))cmd", "s");
    expect_contains(interp, R"cmd(sym_ilaplace("1/(s-2)", "s", "t"))cmd", "exp");
    expect_contains(interp, R"cmd(sym_fourier("exp(-2*t)", "t", "omega"))cmd", "omega");
    expect_contains(interp, R"cmd(sym_ztransform("0.5^n", "n", "z"))cmd", "z");
    expect_error(interp, R"cmd(sym_laplace("exp(2*t)", "t"))cmd");
    expect_contains(interp, "help", "sym_laplace");
    expect_contains(interp, "help", "sym_ztransform");
}

TEST(ReplWave232Pipeline, Wave231_CryptoFemCfdBindings) {
    Interpreter interp;

    // crypto AES-128 ECB block (NIST FIPS-197 Appendix F)
    expect_contains(
        interp,
        R"cmd(crypto_aes128_encrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734"))cmd",
        "3925841d02dc09fbdc118597196a0b32");

    // crypto AES-128 CBC encrypt + decrypt round-trip via known ciphertext
    expect_ok(
        interp,
        R"cmd(crypto_aes128_cbc_encrypt("2b7e151628aedb2a6abf7158809cf40f", "000102030405060708090a0b0c0d0e0f", "6bc1bee22e409f96e93d7e117393172a"))cmd");
    expect_contains(
        interp,
        R"cmd(crypto_aes128_cbc_decrypt("2b7e151628aedb2a6abf7158809cf40f", "000102030405060708090a0b0c0d0e0f", "08f2b59c5efccc8ca069951b94f69d14074b848c41fe11f3ffa61ab5bf3532cb"))cmd",
        "6bc1bee22e409f96e93d7e117393172a");

    // crypto ChaCha20 stream (XOR self-inverse; output differs from plaintext)
    expect_contains(
        interp,
        R"cmd(crypto_chacha20("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "00000000000000000000", 1, "00000000000000000000000000000000"))cmd",
        "18b84231ade6a6d113615c61af434e27");

    // fem 2D Poisson P1 solve on unit square (f = 1, zero Dirichlet)
    expect_ok(interp, "u = fem_poisson2d(4, 4)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 25u);
    EXPECT_GT(interp.state().matrices.at("u")(12, 0), 0.0);

    // cfd 2D FVM upwind advection final field
    expect_ok(interp, "uf = cfd_advection2d(20, 15, 1, 0, 0.4, 0.01)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
    EXPECT_EQ(interp.state().matrices.at("uf").rows(), 15u);
    EXPECT_EQ(interp.state().matrices.at("uf").cols(), 20u);
    double mass = 0.0;
    for (std::size_t i = 0; i < 15; ++i) {
        for (std::size_t j = 0; j < 20; ++j) {
            mass += interp.state().matrices.at("uf")(i, j);
        }
    }
    EXPECT_NEAR(mass, 0.1 * 0.1, 0.05);

    // help lists Wave 231 bindings
    expect_contains(interp, "help", "crypto_aes128_encrypt_block");
    expect_contains(interp, "help", "fem_poisson2d");
    expect_contains(interp, "help", "cfd_advection2d");
}
