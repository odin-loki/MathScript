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

TEST(ReplCommandsTest, crypto_hmac_sha512_rfc4231) {
    Interpreter interp;
    expect_contains(
        interp,
        "crypto_hmac_sha512(0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b,4869205468657265)",
        "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cd"
        "edaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854");
}

TEST(ReplCommandsTest, crypto_hkdf_sha256_rfc5869) {
    Interpreter interp;
    expect_contains(
        interp,
        "crypto_hkdf_sha256(0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b,"
        "000102030405060708090a0b0c,f0f1f2f3f4f5f6f7f8f9,42)",
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865");
}

TEST(ReplCommandsTest, crypto_hkdf_sha512_rfc5869) {
    Interpreter interp;
    expect_contains(
        interp,
        "crypto_hkdf_sha512(0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b,"
        "000102030405060708090a0b0c,f0f1f2f3f4f5f6f7f8f9,42)",
        "832390086cda71fb47625bb5ceb168e4c8e26a1a16ed34d9fc7fe92c14815793"
        "38da362cb8d9f925d7cb");
}

TEST(ReplCommandsTest, crypto_pbkdf2_sha256_rfc6070) {
    Interpreter interp;
    expect_contains(interp, "crypto_pbkdf2_sha256(70617373776f7264,73616c74,1,20)",
                    "120fb6cffcf8b32c43e7225256c4f837a86548c9");
}

TEST(ReplCommandsTest, crypto_pbkdf2_hmac_sha512_rfc8018) {
    Interpreter interp;
    expect_contains(
        interp, "crypto_pbkdf2_hmac_sha512(70617373776f7264,73616c74,1,64)",
        "867f70cf1ade02cff3752599a3a53dc4af34c7a669815ae5d513554e1c8cf252"
        "c02d470a285a0501bad999bfe943c08f050235d7d68b1da55e63f73b60a57fce");
}

TEST(ReplCommandsTest, crypto_aes128_cbc_roundtrip) {
    Interpreter interp;
    const char* key = "2b7e151628aed2a6abf7158809cf4f3c";
    const char* iv = "000102030405060708090a0b0c0d0e0f";
    const char* plain =
        "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a16ef";
    const auto enc = interp.execute(std::string("crypto_aes128_cbc_encrypt(") + key + "," + iv +
                                    "," + plain + ")");
    ASSERT_TRUE(enc.has_value());
    std::string cipher = *enc;
    while (!cipher.empty() && (cipher.back() == '\n' || cipher.back() == ' ')) {
        cipher.pop_back();
    }
    const auto dec = interp.execute(std::string("crypto_aes128_cbc_decrypt(") + key + "," + iv +
                                    "," + cipher + ")");
    ASSERT_TRUE(dec.has_value());
    EXPECT_NE((*dec).find(plain), std::string::npos);
}

TEST(ReplCommandsTest, crypto_aes256_cbc_roundtrip) {
    Interpreter interp;
    const char* key =
        "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4";
    const char* iv = "000102030405060708090a0b0c0d0e0f";
    const char* plain =
        "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef";
    const auto enc = interp.execute(std::string("crypto_aes256_cbc_encrypt(") + key + "," + iv +
                                    "," + plain + ")");
    ASSERT_TRUE(enc.has_value());
    std::string cipher = *enc;
    while (!cipher.empty() && (cipher.back() == '\n' || cipher.back() == ' ')) {
        cipher.pop_back();
    }
    const auto dec = interp.execute(std::string("crypto_aes256_cbc_decrypt(") + key + "," + iv +
                                    "," + cipher + ")");
    ASSERT_TRUE(dec.has_value());
    EXPECT_NE((*dec).find(plain), std::string::npos);
}

TEST(ReplCommandsTest, crypto_aes128_gcm_nist_case2) {
    Interpreter interp;
    expect_contains(
        interp,
        "crypto_aes128_gcm_encrypt(00000000000000000000000000000000,"
        "000000000000000000000000,\"\",00000000000000000000000000000000)",
        "0388dace60b6a392f328c2b971b2fe78");
    expect_contains(
        interp,
        "crypto_aes128_gcm_decrypt(00000000000000000000000000000000,"
        "000000000000000000000000,\"\",0388dace60b6a392f328c2b971b2fe78,"
        "ab6e47d42cec13bdf53a67b21257bddf)",
        "00000000000000000000000000000000");
}

TEST(ReplCommandsTest, crypto_aes256_gcm_nist_case14) {
    Interpreter interp;
    expect_contains(
        interp,
        "crypto_aes256_gcm_encrypt("
        "0000000000000000000000000000000000000000000000000000000000000000,"
        "000000000000000000000000,\"\",00000000000000000000000000000000)",
        "cea7403d4d606b6e074ec5d3baf39d18");
    expect_contains(
        interp,
        "crypto_aes256_gcm_decrypt("
        "0000000000000000000000000000000000000000000000000000000000000000,"
        "000000000000000000000000,\"\",cea7403d4d606b6e074ec5d3baf39d18,"
        "d0d1c8a799996bf0265b98b5d48ab919)",
        "00000000000000000000000000000000");
}

TEST(ReplCommandsTest, crypto_chacha20_poly1305_rfc8439) {
    Interpreter interp;
    const std::string cmd =
        "crypto_chacha20_poly1305_encrypt("
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f,"
        "070000004041424344454647,50515253c0c1c2c3c4c5c6c7,"
        "4c616469657320616e642047656e746c656d656e206f662074686520636c617373206f66202739393a2049662049"
        "20636f756c64206f6666657220796f75206f6e6c79206f6e652074697020666f7220746865206675747572652c2073"
        "756e73637265656e20776f756c642062652069742e)";
    expect_contains(interp, cmd, "1ae10b594f09e26a7e902ecbd0600691");
    expect_contains(
        interp,
        "crypto_chacha20_poly1305_decrypt("
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f,"
        "070000004041424344454647,50515253c0c1c2c3c4c5c6c7,"
        "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
        "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
        "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b6116,1ae10b594f09e26a7e902ecbd0600691)",
        "4c616469657320616e642047656e746c656d656e");
}

TEST(ReplCommandsTest, crypto_ed25519_rfc8032) {
    Interpreter interp;
    const char* seed =
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
    expect_contains(interp, std::string("crypto_ed25519_keypair(") + seed + ")",
                    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    expect_contains(
        interp, std::string("crypto_ed25519_sign(") + seed + ",\"\")",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b");
    expect_contains(
        interp,
        "crypto_ed25519_verify("
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a,\"\","
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b)",
        "1");
}

TEST(ReplCommandsTest, crypto_random_bytes_and_constant_time_eq) {
    Interpreter interp;
    const auto rand = interp.execute("crypto_random_bytes(16)");
    ASSERT_TRUE(rand.has_value());
    EXPECT_GE(rand->size(), 32u);
    expect_contains(interp, "crypto_constant_time_eq(00ff,00ff)", "1");
    expect_contains(interp, "crypto_constant_time_eq(00ff,00fe)", "0");
}
