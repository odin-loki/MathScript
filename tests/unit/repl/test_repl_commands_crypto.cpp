#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, crypto_sha256_and_hmac_sha256) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_sha256(hex_data)");
    expect_contains(interp, "help", "crypto_hmac_sha256(hex_key,hex_data)");

    // NIST empty-string SHA-256
    expect_contains(interp, "crypto_sha256(\"\")",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    // ASCII "abc" as hex
    expect_contains(interp, "crypto_sha256(616263)",
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // RFC 4231 HMAC-SHA256 test case 1
    expect_contains(
        interp,
        "crypto_hmac_sha256(0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b,4869205468657265)",
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

TEST(ReplCommandsTest, crypto_sha512) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_sha512(hex_data)");

    // Empty-string SHA-512 (same vector as tests/unit/test_crypto.cpp)
    expect_contains(
        interp, "crypto_sha512(\"\")",
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

TEST(ReplCommandsTest, crypto_aes256_encrypt_block) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_aes256_encrypt_block");

    // NIST FIPS-197 AES-256 block (same vector as tests/unit/test_crypto.cpp)
    expect_contains(
        interp,
        R"cmd(crypto_aes256_encrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "6bc1bee22e409f96e93d7e117393172a"))cmd",
        "a36452d23436433a516cace8bf319e9c");
}

TEST(ReplCommandsTest, crypto_aes256_decrypt_block) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_aes256_decrypt_block");

    expect_contains(
        interp,
        R"cmd(crypto_aes256_decrypt_block("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41", "a36452d23436433a516cace8bf319e9c"))cmd",
        "6bc1bee22e409f96e93d7e117393172a");
}

TEST(ReplCommandsTest, crypto_aes128_decrypt_block) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_aes128_decrypt_block");

    // FIPS-197 AES-128 encrypt then decrypt round-trip
    expect_contains(
        interp,
        R"cmd(crypto_aes128_encrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734"))cmd",
        "3925841d02dc09fbdc118597196a0b32");
    expect_contains(
        interp,
        R"cmd(crypto_aes128_decrypt_block("2b7e151628aed2a6abf7158809cf4f3c", "3925841d02dc09fbdc118597196a0b32"))cmd",
        "3243f6a8885a308d313198a2e0370734");
}

TEST(ReplCommandsTest, crypto_x25519_keypair) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_x25519_keypair");

    // RFC 7748 Alice public key (same vector as tests/unit/test_crypto.cpp)
    expect_contains(
        interp,
        R"cmd(crypto_x25519_keypair("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"))cmd",
        "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
}

TEST(ReplCommandsTest, token_crypto_bwt) {
    Interpreter interp;
    ms::izaac::clear_session();
    expect_contains(interp, "help", "tokenbucket_refill_rate");
    expect_contains(interp, "help", "crypto_bytes_to_hex");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "tokenbucket_new(tb278, 5, 1)");
    expect_ok(interp, "tokenbucket_capacity(tb278)");

    expect_ok(interp, "b = crypto_from_hex(\"ff\")");
    expect_ok(interp, "h = crypto_bytes_to_hex(b)");
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 2u);

    expect_ok(interp, "x = [10, 20, 30]");
    expect_ok(interp, "e = bwt_encode_vec(x)");
    expect_ok(interp, "pi = bwt_primary_index(x)");
    expect_ok(interp, "d = bwt_decode_vec(e, pi)");
    EXPECT_EQ(interp.state().matrices.at("d").rows(), 3u);
}

TEST(ReplCommandsTest, crypto_bwt_equity) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);

    expect_ok(interp, "prices = [100, 101, 102, 101]");
    expect_ok(interp, "pos = [0, 1, 1, 0]");
    expect_ok(interp, "eq = run_backtest_equity(prices, pos, 10000)");
    EXPECT_EQ(interp.state().matrices.at("eq").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_2) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_3) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_4) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_5) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_6) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_7) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_8) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_9) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_10) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_11) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_12) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_13) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_14) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_15) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_16) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_17) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_18) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_19) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_20) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_21) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_22) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_23) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_24) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_25) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_26) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_27) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_28) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_29) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_30) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_31) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}

TEST(ReplCommandsTest, hex_bwt_32) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"ab\")");
    expect_ok(interp, "hex = crypto_bytes_to_hex(b)");
    ASSERT_GT(interp.state().matrices.count("hex"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hex").rows(), 2u);

    expect_ok(interp, "raw = [1, 2, 3, 4]");
    expect_ok(interp, "bwt = bwt_encode_vec(raw)");
    expect_ok(interp, "pi = bwt_primary_index(raw)");
    expect_ok(interp, "dec = bwt_decode_vec(bwt, pi)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 4u);
}
