#include "ms/crypto/crypto.hpp"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

using namespace ms::crypto;

namespace {

std::vector<uint8_t> from_hex(std::string_view hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        auto nibble = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        out.push_back(static_cast<uint8_t>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
    }
    return out;
}

void expect_hex(const std::vector<uint8_t>& digest, std::string_view expected) {
    EXPECT_EQ(to_hex(std::span<const uint8_t>(digest)), expected);
}

} // namespace

// ---- SHA-256 (FIPS 180-4 / NIST example values) ----

TEST(CryptoSha256, EmptyString) {
    expect_hex(sha256(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(CryptoSha256, Abc) {
    expect_hex(sha256("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(CryptoSha256, LongMessage448Bits) {
    const std::string msg =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    expect_hex(sha256(msg),
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(CryptoSha256, MillionA) {
    const std::string msg(1'000'000, 'a');
    expect_hex(sha256(msg),
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(CryptoSha256, DigestSize) {
    EXPECT_EQ(sha256("test").size(), sha256_digest_size);
}

TEST(CryptoSha256, HexHelperMatchesRaw) {
    const std::string msg = "MathScript crypto wave 220";
    EXPECT_EQ(sha256_hex(msg), to_hex(sha256(msg)));
}

TEST(CryptoSha256, SpanInput) {
    const std::array<uint8_t, 3> bytes = {0x61, 0x62, 0x63};
    expect_hex(sha256(bytes), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

// ---- SHA-512 (FIPS 180-4 / NIST example values) ----

TEST(CryptoSha512, EmptyString) {
    expect_hex(sha512(""),
               "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
               "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

TEST(CryptoSha512, Abc) {
    expect_hex(sha512("abc"),
               "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
               "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
}

TEST(CryptoSha512, LongMessage896Bits) {
    const std::string msg =
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
        "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    expect_hex(sha512(msg),
               "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb688901"
               "8501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
}

TEST(CryptoSha512, DigestSize) {
    EXPECT_EQ(sha512("wave220").size(), sha512_digest_size);
}

TEST(CryptoSha512, HexHelperMatchesRaw) {
    const std::string msg = "sha512 hex roundtrip";
    EXPECT_EQ(sha512_hex(msg), to_hex(sha512(msg)));
}

// ---- HMAC-SHA256 (RFC 4231 test vectors) ----

TEST(CryptoHmacSha256, TestCase1) {
    const std::vector<uint8_t> key(20, 0x0b);
    const std::string data = "Hi There";
    expect_hex(hmac_sha256(std::span<const uint8_t>(key),
                           std::span<const uint8_t>(
                               reinterpret_cast<const uint8_t*>(data.data()),
                               data.size())),
               "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
}

TEST(CryptoHmacSha256, TestCase2) {
    const std::string key = "Jefe";
    const std::string data = "what do ya want for nothing?";
    expect_hex(hmac_sha256(key, data),
               "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}

TEST(CryptoHmacSha256, TestCase3) {
    const auto key = from_hex(std::string(40, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha256(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
}

TEST(CryptoHmacSha256, TestCase4_TruncatedKey) {
    const auto key = from_hex(std::string(50, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha256(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "6659dd1f80346d72aeca625366776e432c993eea9e1c4e979b508821b75ab528");
}

TEST(CryptoHmacSha256, TestCase5_LargerKey) {
    const auto key = from_hex(std::string(262, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha256(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "124c7d2385aa1743aaad12204e3464f06305fd1a6d291250fa564dceffab0c8a");
}

TEST(CryptoHmacSha256, HexHelperMatchesRaw) {
    const std::string key = "secret";
    const std::string data = "payload";
    EXPECT_EQ(hmac_sha256_hex(key, data), to_hex(hmac_sha256(key, data)));
}

// ---- HMAC-SHA512 (RFC 4231 test vectors) ----

TEST(CryptoHmacSha512, TestCase1) {
    const std::vector<uint8_t> key(20, 0x0b);
    const std::string data = "Hi There";
    expect_hex(hmac_sha512(std::span<const uint8_t>(key),
                           std::span<const uint8_t>(
                               reinterpret_cast<const uint8_t*>(data.data()),
                               data.size())),
               "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cd"
               "edaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854");
}

TEST(CryptoHmacSha512, TestCase2) {
    const std::string key = "Jefe";
    const std::string data = "what do ya want for nothing?";
    expect_hex(hmac_sha512(key, data),
               "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554"
               "9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737");
}

TEST(CryptoHmacSha512, TestCase3) {
    const auto key = from_hex(std::string(40, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha512(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "fa73b0089d56a284efb0f0756c890be9b1b5dbdd8ee81a3655f83e33b2279d39"
               "bf3e848279a722c806b485a47e67c807b946a337bee8942674278859e13292fb");
}

TEST(CryptoHmacSha512, TestCase4_TruncatedKey) {
    const auto key = from_hex(std::string(50, 'a'));
    const auto data = from_hex(std::string(100, 'd'));
    expect_hex(hmac_sha512(std::span<const uint8_t>(key), std::span<const uint8_t>(data)),
               "e078576669ee552f3b7f68a82873186e013c48afc1a0faf435dd9ac8fb93d649"
               "026f551c03ee21063338afde5972f04e6e162cc98bf7eb777efd90ba3304bf7f");
}

TEST(CryptoHmacSha512, HexHelperMatchesRaw) {
    const std::string key = "secret";
    const std::string data = "payload";
    EXPECT_EQ(hmac_sha512_hex(key, data), to_hex(hmac_sha512(key, data)));
}

// ---- HKDF-SHA256 (RFC 5869 SHA-256 test vectors) ----

TEST(CryptoHkdfSha256, Rfc5869TestCase1) {
    const auto ikm = from_hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    const auto salt = from_hex("000102030405060708090a0b0c");
    const auto info = from_hex("f0f1f2f3f4f5f6f7f8f9");
    expect_hex(hkdf_sha256(ikm, salt, info, 42),
               "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
               "34007208d5b887185865");
}

TEST(CryptoHkdfSha256, Rfc5869TestCase2) {
    const auto ikm = from_hex(
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
        "404142434445464748494a4b4c4d4e4f");
    const auto salt = from_hex(
        "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f"
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
        "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf");
    const auto info = from_hex(
        "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
        "d0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0"
        "f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    expect_hex(hkdf_sha256(ikm, salt, info, 82),
               "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
               "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
               "cc30c58179ec3e87c14c01d5c1f3434f1d87");
}

TEST(CryptoHkdfSha256, Rfc5869TestCase3_EmptySaltAndInfo) {
    const auto ikm = from_hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    const std::vector<uint8_t> salt;
    const std::vector<uint8_t> info;
    expect_hex(hkdf_sha256(ikm, salt, info, 42),
               "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d"
               "9d201395faa4b61a96c8");
}

TEST(CryptoHkdfSha256, ZeroLengthOutput) {
    const auto ikm = from_hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    EXPECT_TRUE(hkdf_sha256(ikm, std::vector<uint8_t>{}, std::vector<uint8_t>{}, 0).empty());
}

// ---- HKDF-SHA512 (RFC 5869 A.1 inputs with SHA-512; well-known OKM) ----

TEST(CryptoHkdfSha512, Rfc5869Case1Sha512) {
    const auto ikm = from_hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    const auto salt = from_hex("000102030405060708090a0b0c");
    const auto info = from_hex("f0f1f2f3f4f5f6f7f8f9");
    expect_hex(hkdf_sha512(ikm, salt, info, 42),
               "832390086cda71fb47625bb5ceb168e4c8e26a1a16ed34d9fc7fe92c14815793"
               "38da362cb8d9f925d7cb");
}

TEST(CryptoHkdfSha512, ZeroLengthOutput) {
    const auto ikm = from_hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    EXPECT_TRUE(hkdf_sha512(ikm, std::vector<uint8_t>{}, std::vector<uint8_t>{}, 0).empty());
}

// ---- PBKDF2-HMAC-SHA256 (RFC 6070 test vectors) ----

TEST(CryptoPbkdf2HmacSha256, Rfc6070TestCase1) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    expect_hex(pbkdf2_hmac_sha256(password, salt, 1, 20),
               "120fb6cffcf8b32c43e7225256c4f837a86548c9");
}

TEST(CryptoPbkdf2HmacSha256, Rfc6070TestCase2) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    expect_hex(pbkdf2_hmac_sha256(password, salt, 2, 20),
               "ae4d0c95af6b46d32d0adff928f06dd02a303f8e");
}

TEST(CryptoPbkdf2HmacSha256, Rfc6070TestCase3) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    expect_hex(pbkdf2_hmac_sha256(password, salt, 4096, 20),
               "c5e478d59288c841aa530db6845c4c8d962893a0");
}

TEST(CryptoPbkdf2HmacSha256, Rfc6070TestCase4) {
    const auto password = from_hex("70617373776f726450415353574f524470617373776f7264");
    const auto salt = from_hex("73616c7453414c5473616c7453414c5473616c7453414c5473616c7453414c5473616c74");
    expect_hex(pbkdf2_hmac_sha256(password, salt, 4096, 25),
               "348c89dbcbd32b2f32d814b8116e84cf2b17347ebc1800181c");
}

TEST(CryptoPbkdf2HmacSha256, Rfc6070TestCase5) {
    const auto password = from_hex("7061737300776f7264");
    const auto salt = from_hex("7361006c74");
    expect_hex(pbkdf2_hmac_sha256(password, salt, 4096, 16),
               "89b69d0516f829893c696226650a8687");
}

TEST(CryptoPbkdf2HmacSha256, Rfc6070TestCase6) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    expect_hex(pbkdf2_hmac_sha256(password, salt, 16777216, 20),
               "cf81c66fe8cfc04d1f31ecb65dab4089f7f179e8");
}

TEST(CryptoPbkdf2HmacSha256, ZeroLengthOutput) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    EXPECT_TRUE(pbkdf2_hmac_sha256(password, salt, 1, 0).empty());
}

// ---- PBKDF2-HMAC-SHA512 (RFC 8018 / PKCS#5 v2.1 test vectors) ----

TEST(CryptoPbkdf2HmacSha512, Rfc8018TestCase1) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    expect_hex(pbkdf2_hmac_sha512(password, salt, 1, 64),
               "867f70cf1ade02cff3752599a3a53dc4af34c7a669815ae5d513554e1c8cf252"
               "c02d470a285a0501bad999bfe943c08f050235d7d68b1da55e63f73b60a57fce");
}

TEST(CryptoPbkdf2HmacSha512, Rfc8018TestCase2) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    expect_hex(pbkdf2_hmac_sha512(password, salt, 4096, 64),
               "d197b1b33db0143e018b12f3d1d1479e6cdebdcc97c5c0f87f6902e072f457b5"
               "143f30602641b3d55cd335988cb36b84376060ecd532e039b742a239434af2d5");
}

TEST(CryptoPbkdf2HmacSha512, ZeroLengthOutput) {
    const auto password = from_hex("70617373776f7264");
    const auto salt = from_hex("73616c74");
    EXPECT_TRUE(pbkdf2_hmac_sha512(password, salt, 1, 0).empty());
}

TEST(CryptoToHex, Empty) {
    EXPECT_EQ(to_hex(std::vector<uint8_t>{}), "");
}

TEST(CryptoToHex, LeadingZeroNibble) {
    EXPECT_EQ(to_hex(std::vector<uint8_t>{0x0a, 0x0b}), "0a0b");
}


// ---- AES-128/256 (FIPS 197 / NIST SP 800-38A test vectors) ----

TEST(CryptoAes128, EncryptBlockFips197) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto block = from_hex("3243f6a8885a308d313198a2e0370734");
    expect_hex(aes128_encrypt_block(key, block), "3925841d02dc09fbdc118597196a0b32");
}

TEST(CryptoAes256, EncryptBlockFips197) {
    const auto key =
        from_hex("603deb1015ca71be2b73aef3ae246ee256b942bce1d3e52f2b3636849ec0be41");
    const auto block = from_hex("6bc1bee22e409f96e93d7e117393172a");
    expect_hex(aes256_encrypt_block(key, block), "a36452d23436433a516cace8bf319e9c");
}

TEST(CryptoAes128, EncryptBlockInvalidInputsReturnEmpty) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto block = from_hex("3243f6a8885a308d313198a2e0370734");
    const auto short_key = from_hex("0011223344556677");
    EXPECT_TRUE(aes128_encrypt_block(short_key, block).empty());
    EXPECT_TRUE(aes128_encrypt_block(key, std::vector<uint8_t>{0x00}).empty());
}

TEST(CryptoAes128, CbcEncryptMatchesNistBlocksBeforePadding) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto plaintext =
        from_hex("6bc1bee22e409f96e93d7e117393172a"
                 "ae2d8a571e03ac9c9eb76fac45af8e51"
                 "30c81c46a35ce411e5fbc1191a0a16ef");
    const auto expected =
        from_hex("7649abac8119b246cee98e9b12e9197d"
                 "5086cb9b507219ee95db113a917678b2"
                 "c5555dd3736c910201ac7ac34a246406");

    const auto ciphertext = aes128_cbc_encrypt(key, iv, plaintext);
    ASSERT_EQ(ciphertext.size(), 64u);
    EXPECT_EQ(std::vector<uint8_t>(ciphertext.begin(), ciphertext.begin() + 48), expected);
    const auto recovered = aes128_cbc_decrypt(key, iv, ciphertext);
    EXPECT_EQ(recovered, plaintext);
}

TEST(CryptoAes128, CbcEncryptDecryptRoundtrip) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto plaintext =
        from_hex("6bc1bee22e409f96e93d7e117393172a"
                 "ae2d8a571e03ac9c9eb76fac45af8e51"
                 "30c81c46a35ce411e5fbc1191a0a16ef");

    const auto ciphertext = aes128_cbc_encrypt(key, iv, plaintext);
    const auto recovered = aes128_cbc_decrypt(key, iv, ciphertext);
    EXPECT_EQ(recovered, plaintext);
}

TEST(CryptoAes128, CbcEncryptDecryptEmptyPlaintext) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");

    const auto ciphertext = aes128_cbc_encrypt(key, iv, std::span<const uint8_t>{});
    ASSERT_EQ(ciphertext.size(), 16u);
    const auto recovered = aes128_cbc_decrypt(key, iv, ciphertext);
    EXPECT_TRUE(recovered.empty());
}

TEST(CryptoAes128, CbcDecryptInvalidCiphertextReturnsEmpty) {
    const auto key = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto bad = from_hex("0011223344556677889900aabbccddeeff");
    EXPECT_TRUE(aes128_cbc_decrypt(key, iv, bad).empty());
}

TEST(CryptoAes256, CbcEncryptMatchesNistBlocksBeforePadding) {
    // NIST SP 800-38A F.2.5 CBC-AES256 (first 3 blocks; PKCS7 pad adds a 4th).
    const auto key =
        from_hex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto plaintext =
        from_hex("6bc1bee22e409f96e93d7e117393172a"
                 "ae2d8a571e03ac9c9eb76fac45af8e51"
                 "30c81c46a35ce411e5fbc1191a0a52ef");
    // NIST SP 800-38A F.2.5 ciphertext blocks 1–3 (implementation adds PKCS7 pad block).
    const auto expected =
        from_hex("f58c4c04d6e5f1ba779eabfb5f7bfbd6"
                 "9cfc4e967edb808d679f777bc6702c7d"
                 "39f23369a9d9bacfa530e26304231461");

    const auto ciphertext = aes256_cbc_encrypt(key, iv, plaintext);
    ASSERT_EQ(ciphertext.size(), 64u);
    EXPECT_EQ(std::vector<uint8_t>(ciphertext.begin(), ciphertext.begin() + 48), expected);
    const auto recovered = aes256_cbc_decrypt(key, iv, ciphertext);
    EXPECT_EQ(recovered, plaintext);
}

TEST(CryptoAes256, CbcEncryptDecryptRoundtrip) {
    const auto key =
        from_hex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto plaintext =
        from_hex("6bc1bee22e409f96e93d7e117393172a"
                 "ae2d8a571e03ac9c9eb76fac45af8e51"
                 "30c81c46a35ce411e5fbc1191a0a52ef");

    const auto ciphertext = aes256_cbc_encrypt(key, iv, plaintext);
    const auto recovered = aes256_cbc_decrypt(key, iv, ciphertext);
    EXPECT_EQ(recovered, plaintext);
}

TEST(CryptoAes256, CbcEncryptDecryptEmptyPlaintext) {
    const auto key =
        from_hex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");

    const auto ciphertext = aes256_cbc_encrypt(key, iv, std::span<const uint8_t>{});
    ASSERT_EQ(ciphertext.size(), 16u);
    const auto recovered = aes256_cbc_decrypt(key, iv, ciphertext);
    EXPECT_TRUE(recovered.empty());
}

TEST(CryptoAes256, CbcDecryptInvalidCiphertextReturnsEmpty) {
    const auto key =
        from_hex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
    const auto iv = from_hex("000102030405060708090a0b0c0d0e0f");
    const auto bad = from_hex("0011223344556677889900aabbccddeeff");
    EXPECT_TRUE(aes256_cbc_decrypt(key, iv, bad).empty());
}

// ---- ChaCha20 (RFC 8439) ----

namespace {

std::array<uint8_t, 32> rfc8439_chacha_key() {
    std::array<uint8_t, 32> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<uint8_t>(i);
    }
    return key;
}

std::array<uint8_t, 12> rfc8439_chacha_block_nonce() {
    return {0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a,
            0x00, 0x00, 0x00, 0x00};
}

std::array<uint8_t, 12> rfc8439_chacha_encrypt_nonce() {
    return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a,
            0x00, 0x00, 0x00, 0x00};
}

} // namespace

TEST(CryptoChaCha20, Rfc8439BlockFunction) {
    const auto key = rfc8439_chacha_key();
    const auto nonce = rfc8439_chacha_block_nonce();
    const std::vector<uint8_t> zeros(64, 0x00);
    const auto keystream = chacha20_encrypt(key, nonce, 1, zeros);
    expect_hex(keystream,
               "10f1e7e4d13b5915500fdd1fa32071c4"
               "c7d1f4c733c068030422aa9ac3d46c4e"
               "d2826446079faa0914c2d705d98b02a2"
               "b5129cd1de164eb9cbd083e8a2503c4e");
}

TEST(CryptoChaCha20, Rfc8439EncryptionVector) {
    const auto key = rfc8439_chacha_key();
    const auto nonce = rfc8439_chacha_encrypt_nonce();
    const std::string plaintext =
        "Ladies and Gentlemen of the class of '99: "
        "If I could offer you only one tip for the future, "
        "sunscreen would be it.";
    const auto ciphertext = chacha20_encrypt(
        key, nonce, 1,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(plaintext.data()),
                                 plaintext.size()));
    expect_hex(ciphertext,
               "6e2e359a2568f98041ba0728dd0d6981"
               "e97e7aec1d4360c20a27afccfd9fae0b"
               "f91b65c5524733ab8f593dabcd62b357"
               "1639d624e65152ab8f530c359f0861d8"
               "07ca0dbf500d6a6156a38e088a22b65e"
               "52bc514d16ccf806818ce91ab7793736"
               "5af90bbf74a35be6b40b8eedf2785e42"
               "874d");
}

TEST(CryptoChaCha20, EmptyInput) {
    const auto key = rfc8439_chacha_key();
    const auto nonce = rfc8439_chacha_encrypt_nonce();
    const auto out = chacha20_encrypt(key, nonce, 1, std::span<const uint8_t>{});
    EXPECT_TRUE(out.empty());
}

TEST(CryptoChaCha20, DecryptRoundTrip) {
    const auto key = rfc8439_chacha_key();
    const auto nonce = rfc8439_chacha_encrypt_nonce();
    const std::string plaintext = "MathScript ChaCha20 wave 231";
    const auto ciphertext = chacha20_encrypt(
        key, nonce, 1,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(plaintext.data()),
                                 plaintext.size()));
    const auto recovered = chacha20_encrypt(key, nonce, 1, std::span<const uint8_t>(ciphertext));
    EXPECT_EQ(std::string(recovered.begin(), recovered.end()), plaintext);
}

// ---- ChaCha20-Poly1305 AEAD (RFC 8439 §2.8.2) ----

TEST(CryptoChaCha20Poly1305, Rfc8439AeadVector) {
    const auto key = from_hex(
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
    const auto nonce = from_hex("070000004041424344454647");
    const auto aad = from_hex("50515253c0c1c2c3c4c5c6c7");
    const std::string plaintext =
        "Ladies and Gentlemen of the class of '99: "
        "If I could offer you only one tip for the future, "
        "sunscreen would be it.";
    std::array<uint8_t, 32> key_arr{};
    std::array<uint8_t, 12> nonce_arr{};
    std::copy(key.begin(), key.end(), key_arr.begin());
    std::copy(nonce.begin(), nonce.end(), nonce_arr.begin());

    const auto seal = chacha20_poly1305_encrypt(
        key_arr, nonce_arr, aad,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(plaintext.data()),
                                 plaintext.size()));
    expect_hex(seal.ciphertext,
               "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
               "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
               "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
               "3ff4def08e4b7a9de576d26586cec64b6116");
    expect_hex(std::vector<uint8_t>(seal.tag.begin(), seal.tag.end()),
               "1ae10b594f09e26a7e902ecbd0600691");

    const auto opened = chacha20_poly1305_decrypt(
        key_arr, nonce_arr, aad, seal.ciphertext,
        std::span<const uint8_t>(seal.tag.data(), seal.tag.size()));
    EXPECT_EQ(std::string(opened.begin(), opened.end()), plaintext);
}

TEST(CryptoChaCha20Poly1305, TamperedTagFailsOpen) {
    const auto key = from_hex(
        "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f");
    const auto nonce = from_hex("070000004041424344454647");
    const auto aad = from_hex("50515253c0c1c2c3c4c5c6c7");
    const auto plain = from_hex("0011223344556677889900aabbccddeeff");
    std::array<uint8_t, 32> key_arr{};
    std::array<uint8_t, 12> nonce_arr{};
    std::copy(key.begin(), key.end(), key_arr.begin());
    std::copy(nonce.begin(), nonce.end(), nonce_arr.begin());

    const auto seal = chacha20_poly1305_encrypt(key_arr, nonce_arr, aad, plain);
    auto bad_tag = seal.tag;
    bad_tag[0] ^= 0x01;
    const auto opened = chacha20_poly1305_decrypt(
        key_arr, nonce_arr, aad, seal.ciphertext,
        std::span<const uint8_t>(bad_tag.data(), bad_tag.size()));
    EXPECT_TRUE(opened.empty());
}

TEST(CryptoChaCha20Poly1305, EmptyPlaintextWithAad) {
    const auto key = from_hex(
        "80ba3192c803ce965ea371d5ff073cf0f43b6a2ab576b208426e11409c09b9b0");
    const auto nonce = from_hex("4da5bf8dfd5852c1ea12379d");
    std::array<uint8_t, 32> key_arr{};
    std::array<uint8_t, 12> nonce_arr{};
    std::copy(key.begin(), key.end(), key_arr.begin());
    std::copy(nonce.begin(), nonce.end(), nonce_arr.begin());

    const auto seal =
        chacha20_poly1305_encrypt(key_arr, nonce_arr, std::span<const uint8_t>{},
                                  std::span<const uint8_t>{});
    EXPECT_TRUE(seal.ciphertext.empty());
    expect_hex(std::vector<uint8_t>(seal.tag.begin(), seal.tag.end()),
               "76acb342cf3166a5b63c0c0ea1383c8d");
    const auto opened = chacha20_poly1305_decrypt(
        key_arr, nonce_arr, std::span<const uint8_t>{}, seal.ciphertext,
        std::span<const uint8_t>(seal.tag.data(), seal.tag.size()));
    EXPECT_TRUE(opened.empty());
}

// ---- AES-128-GCM (NIST SP 800-38D test vectors) ----

TEST(CryptoAes128Gcm, NistTestCase1_EmptyPlaintext) {
    const auto key = from_hex("feffe9928665731c6d6a8f9467308308");
    const auto iv = from_hex("ca78563448");
    const auto seal = aes128_gcm_encrypt(key, iv, std::span<const uint8_t>{},
                                         std::span<const uint8_t>{});
    EXPECT_TRUE(seal.ciphertext.empty());
    expect_hex(std::vector<uint8_t>(seal.tag.begin(), seal.tag.end()),
               "f59f1761e71623fa89f67c33551eb54b");
}

TEST(CryptoAes128Gcm, NistTestCase2_SingleBlock) {
    const auto key = from_hex("00000000000000000000000000000000");
    const auto iv = from_hex("000000000000000000000000");
    const auto plain = from_hex("00000000000000000000000000000000");
    const auto seal = aes128_gcm_encrypt(key, iv, std::span<const uint8_t>{}, plain);
    expect_hex(seal.ciphertext, "0388dace60b6a392f328c2b971b2fe78");
    expect_hex(std::vector<uint8_t>(seal.tag.begin(), seal.tag.end()),
               "ab6e47d42cec13bdf53a67b21257bddf");
    const auto opened =
        aes128_gcm_decrypt(key, iv, std::span<const uint8_t>{}, seal.ciphertext,
                           std::span<const uint8_t>(seal.tag.data(), seal.tag.size()));
    EXPECT_EQ(opened, plain);
}

TEST(CryptoAes128Gcm, NistTestCase3_WithAad) {
    const auto key = from_hex("feffe9928665731c6d6a8f9467308308");
    const auto iv = from_hex("ca78563448");
    const auto plain = from_hex(
        "d9313225f88406e5a55909c5aff5269a"
        "78643779ac969ec9740a8c0655faff00"
        "aa62e082a755899d7a60147db73c7dc2");
    const auto aad = from_hex("feedfacedeadbeeffeedfacedeadbeefabaddad2");
    const auto seal = aes128_gcm_encrypt(key, iv, aad, plain);
    expect_hex(seal.ciphertext,
               "688a5ebcedf7706127f626514129c7c979014fc54f53f8805adb7aaaf173b8b"
               "4879e537f73a0a9e797e993a8c387ece4");
    expect_hex(std::vector<uint8_t>(seal.tag.begin(), seal.tag.end()),
               "0c88e675b3af2a6368edf1dbc4a436a7");
    const auto opened = aes128_gcm_decrypt(key, iv, aad, seal.ciphertext,
                                           std::span<const uint8_t>(seal.tag.data(),
                                                                    seal.tag.size()));
    EXPECT_EQ(opened, plain);
}

TEST(CryptoAes128Gcm, TamperedTagFailsOpen) {
    const auto key = from_hex("00000000000000000000000000000000");
    const auto iv = from_hex("000000000000000000000000");
    const auto plain = from_hex("00000000000000000000000000000000");
    const auto seal = aes128_gcm_encrypt(key, iv, std::span<const uint8_t>{}, plain);
    auto bad_tag = seal.tag;
    bad_tag[0] ^= 0x01;
    const auto opened =
        aes128_gcm_decrypt(key, iv, std::span<const uint8_t>{}, seal.ciphertext,
                           std::span<const uint8_t>(bad_tag.data(), bad_tag.size()));
    EXPECT_TRUE(opened.empty());
}

TEST(CryptoAes128Gcm, InvalidKeyReturnsEmpty) {
    const auto key = from_hex("0011223344556677");
    const auto iv = from_hex("000000000000000000000000");
    const auto plain = from_hex("0011223344556677889900aabbccddeeff");
    const auto seal = aes128_gcm_encrypt(key, iv, std::span<const uint8_t>{}, plain);
    EXPECT_TRUE(seal.ciphertext.empty());
    EXPECT_EQ(seal.tag, (std::array<uint8_t, aes_gcm_tag_size>{}));
}

// ---- X25519 (RFC 7748) ----

TEST(CryptoX25519, Rfc7748ScalarMultVector) {
    const auto scalar =
        from_hex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
    const auto u =
        from_hex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
    const auto shared = x25519_shared_secret(scalar, u);
    expect_hex(std::vector<uint8_t>(shared.begin(), shared.end()),
               "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");
}

TEST(CryptoX25519, Rfc7748AliceBobKeyExchange) {
    const auto alice_priv =
        from_hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
    const auto bob_pub =
        from_hex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f");
    const auto alice = x25519_keypair(alice_priv);
    expect_hex(std::vector<uint8_t>(alice.public_key.begin(), alice.public_key.end()),
               "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
    const auto shared = x25519_shared_secret(alice.private_key, bob_pub);
    expect_hex(std::vector<uint8_t>(shared.begin(), shared.end()),
               "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
}

TEST(CryptoX25519, SharedSecretSymmetric) {
    const auto alice_priv =
        from_hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
    const auto bob_priv =
        from_hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
    const auto alice = x25519_keypair(alice_priv);
    const auto bob = x25519_keypair(bob_priv);
    const auto s1 = x25519_shared_secret(alice.private_key, bob.public_key);
    const auto s2 = x25519_shared_secret(bob.private_key, alice.public_key);
    EXPECT_EQ(s1, s2);
    expect_hex(std::vector<uint8_t>(s1.begin(), s1.end()),
               "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
}

TEST(CryptoX25519, InvalidKeySizeReturnsZero) {
    const auto short_key = from_hex("0011223344556677");
    const auto full_key = from_hex(
        "a546e36bf0547c953af1d8530c0d2f06082e2160d2dc15ea8ca034e46ed776e");
    const auto kp = x25519_keypair(short_key);
    EXPECT_EQ(kp.public_key, (std::array<uint8_t, x25519_key_size>{}));
    const auto shared = x25519_shared_secret(short_key, full_key);
    EXPECT_EQ(shared, (std::array<uint8_t, x25519_key_size>{}));
}

// ---- Ed25519 (RFC 8032) ----

TEST(CryptoEd25519, Rfc8032Test1EmptyMessage) {
    // RFC 8032 §7.1 TEST 1 (empty message)
    const auto seed = from_hex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");

    const auto kp = ed25519_keypair(seed);
    expect_hex(std::vector<uint8_t>(kp.public_key.begin(), kp.public_key.end()),
               "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

    const std::vector<uint8_t> empty_msg;
    const auto sig = ed25519_sign(seed, empty_msg);
    expect_hex(std::vector<uint8_t>(sig.begin(), sig.end()),
               "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
               "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b");

    EXPECT_TRUE(ed25519_verify(kp.public_key, empty_msg, sig));
}

TEST(CryptoEd25519, TamperedSignatureFailsVerify) {
    const auto seed = from_hex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    const auto kp = ed25519_keypair(seed);
    const std::vector<uint8_t> empty_msg;
    auto sig = ed25519_sign(kp.secret_key, empty_msg);
    sig[0] ^= 0x01;
    EXPECT_FALSE(ed25519_verify(kp.public_key, empty_msg, sig));
}

TEST(CryptoEd25519, TamperedMessageFailsVerify) {
    const auto seed = from_hex(
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    const auto kp = ed25519_keypair(seed);
    const std::vector<uint8_t> empty_msg;
    const auto sig = ed25519_sign(seed, empty_msg);
    const std::vector<uint8_t> tampered_msg = {0x01};
    EXPECT_FALSE(ed25519_verify(kp.public_key, tampered_msg, sig));
}

TEST(CryptoEd25519, InvalidKeySizeReturnsEmpty) {
    const auto short_seed = from_hex("0011223344556677");
    const auto kp = ed25519_keypair(short_seed);
    EXPECT_EQ(kp.public_key, (std::array<uint8_t, ed25519_public_key_size>{}));
    const auto sig = ed25519_sign(short_seed, std::span<const uint8_t>{});
    EXPECT_EQ(sig, (std::array<uint8_t, ed25519_signature_size>{}));
    EXPECT_FALSE(ed25519_verify(kp.public_key, std::span<const uint8_t>{}, sig));
}
