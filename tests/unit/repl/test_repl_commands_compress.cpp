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

TEST(ReplCommandsTest, rgb2gray_and_rle_roundtrip) {
    Interpreter interp;
    expect_contains(interp, "help", "rgb2gray(M)");
    expect_contains(interp, "help", "rle_encode_vec(M)");
    expect_contains(interp, "help", "bigint(\"495\")");

    // Pure red and pure green pixels in [0,1]
    expect_ok(interp, "RGB = [1, 0, 0; 0, 1, 0]");
    expect_ok(interp, "G = rgb2gray(RGB)");
    ASSERT_TRUE(interp.state().matrices.count("G") > 0);
    const auto& gray = interp.state().matrices.at("G");
    EXPECT_EQ(gray.rows(), 2u);
    EXPECT_EQ(gray.cols(), 1u);
    EXPECT_NEAR(gray(0, 0), 0.299, 1e-6);
    EXPECT_NEAR(gray(1, 0), 0.587, 1e-6);

    // RLE roundtrip on a small byte matrix
    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_TRUE(interp.state().matrices.count("R") > 0);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 8u);
    EXPECT_EQ(restored.cols(), 1u);
    const double expected[] = {1, 1, 2, 2, 2, 2, 3, 3};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(restored(i, 0), expected[i]);
    }

    expect_ok(interp, "x = bigint(\"495\")");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("x"), 495.0);
    expect_error_contains(interp, "y = bigint(\"9007199254740993\")", "too large");
}

TEST(ReplCommandsTest, delta_encode_roundtrip) {
    Interpreter interp;
    expect_contains(interp, "help", "delta_encode_vec(M)");

    expect_ok(interp, "B = [10, 12, 15, 20]");
    expect_ok(interp, "E = delta_encode_vec(B)");
    expect_ok(interp, "R = delta_decode_vec(E)");
    ASSERT_TRUE(interp.state().matrices.count("R") > 0);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 4u);
    const double expected[] = {10, 12, 15, 20};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(restored(i, 0), expected[i]);
    }
}

TEST(ReplCommandsTest, bwt_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "bwt_encode_vec(M)");

    expect_ok(interp, "B = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "E = bwt_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_EQ(interp.state().matrices.at("E").rows(), 7u);
}

TEST(ReplCommandsTest, bwt_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "bwt_decode_vec(L,primary_index)");

    expect_ok(interp, "B = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "E = bwt_encode_vec(B)");
    expect_ok(interp, "pi = bwt_primary_index(B)");
    expect_ok(interp, "R = bwt_decode_vec(E, pi)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 6u);
    const double banana[] = {98, 97, 110, 97, 110, 97};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(restored(i, 0), banana[i], 1e-9);
    }
}

TEST(ReplCommandsTest, huffman_encode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "huffman_encode_vec(M)");

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "E = huffman_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    EXPECT_GE(interp.state().matrices.at("E").rows(), 1u);
}

TEST(ReplCommandsTest, huffman_decode_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "huffman_decode_vec(orig_M,E)");

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "E = huffman_encode_vec(M)");
    expect_ok(interp, "R = huffman_decode_vec(M, E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    const auto& restored = interp.state().matrices.at("R");
    EXPECT_EQ(restored.rows(), 6u);
    const double expected[] = {97, 98, 99, 97, 97, 98};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(restored(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, bzip2_compress_decompress_vec) {
    Interpreter interp;
    expect_contains(interp, "help", "bzip2_compress_vec(M)");
    expect_contains(interp, "help", "bzip2_decompress_vec(C)");

    expect_ok(interp, "M = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(M)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);

    expect_ok(interp, "R = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 11u);
}

TEST(ReplCommandsTest, compress_bits_to_bytes) {
    Interpreter interp;
    expect_contains(interp, "help", "compress_bits_to_bytes(bits_vec)");

    expect_ok(interp, "bits = [1;0;1;0;1;0;1;1;1;1;0;0;1;0;1;1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);
}

TEST(ReplCommandsTest, compress_bytes_to_bits) {
    Interpreter interp;
    expect_contains(interp, "help", "compress_bytes_to_bits(bytes_vec)");

    expect_ok(interp, "bytes = [171; 205]");
    expect_ok(interp, "bits = compress_bytes_to_bits(bytes)");
    ASSERT_GT(interp.state().matrices.count("bits"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits").rows(), 16u);
    expect_ok(interp, "bytes2 = compress_bits_to_bytes(bits)");
    EXPECT_EQ(interp.state().matrices.at("bytes2").rows(), 2u);
}

TEST(ReplCommandsTest, compress_arith_ans) {
    Interpreter interp;
    expect_contains(interp, "help", "arithmetic_encode_vec(M)");
    expect_contains(interp, "help", "arithmetic_decode_vec(orig_M,E)");
    expect_contains(interp, "help", "ans_encode_vec(M)");
    expect_contains(interp, "help", "ans_decode_vec(orig_M,E)");

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("AE"), 0u);
    EXPECT_GE(interp.state().matrices.at("AE").rows(), 1u);

    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
    const double expected[] = {97, 98, 99, 97, 97, 98, 99};
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("AR")(i, 0), expected[i], 1e-9);
    }

    expect_ok(interp, "NE = ans_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("NE"), 0u);
    EXPECT_GE(interp.state().matrices.at("NE").rows(), 1u);

    expect_ok(interp, "NR = ans_decode_vec(M, NE)");
    ASSERT_GT(interp.state().matrices.count("NR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("NR").rows(), 7u);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_NEAR(interp.state().matrices.at("NR")(i, 0), expected[i], 1e-9);
    }
}

TEST(ReplCommandsTest, crypto_topo_compress_decode) {
    Interpreter interp;

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, rle_hex) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_2) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_2) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_2) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_2) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_2) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_2) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_2) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_3) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_3) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_3) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_3) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_3) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_3) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_3) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_4) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_4) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_4) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_4) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_4) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_4) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_4) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_5) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_5) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_5) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_5) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_5) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_5) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_5) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_6) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_6) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_6) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_6) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_6) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_6) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_6) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_7) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_7) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_7) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_7) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_7) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_7) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_7) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_8) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_8) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_8) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_8) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_8) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_8) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_8) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_9) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_9) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_9) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_9) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_9) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_9) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_9) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_10) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_10) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_10) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_10) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_10) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_10) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_10) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_11) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_11) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_11) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_11) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_11) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_11) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_11) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_12) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_12) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_12) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_12) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_12) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_12) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_12) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_13) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_13) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_13) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_13) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_13) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_13) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_13) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_14) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_14) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_14) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_14) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_14) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_14) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_14) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_15) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_15) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_15) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_15) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_15) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_15) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_15) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_16) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_16) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_16) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_16) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_16) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_16) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_16) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_17) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_17) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_17) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_17) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_17) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_17) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_17) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_18) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_18) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_18) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_18) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_18) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_18) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_18) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_19) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_19) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_19) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_19) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_19) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_19) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_19) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_20) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_20) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_20) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_20) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_20) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_20) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_20) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_21) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_21) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_21) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_21) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_21) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_21) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_21) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_22) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_22) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_22) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_22) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_22) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_22) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_22) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_23) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_23) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_23) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_23) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_23) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_23) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_23) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_24) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_24) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_24) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_24) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_24) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_24) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_24) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_25) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_25) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_25) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_25) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_25) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_25) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_25) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_26) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_26) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_26) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_26) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_26) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_26) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_26) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_27) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_27) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_27) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_27) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_27) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_27) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_27) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_28) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_28) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_28) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_28) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_28) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_28) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_28) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_29) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_29) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_29) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_29) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_29) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_29) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_29) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_30) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_30) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_30) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_30) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_30) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_30) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_30) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_31) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_31) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_31) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_31) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_31) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_31) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_31) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(ReplCommandsTest, rle_hex_32) {
    Interpreter interp;

    expect_ok(interp, "raw = [1, 2, 2, 3, 3, 3]");
    expect_ok(interp, "enc = run_length_encode_vec(raw)");
    expect_ok(interp, "dec = run_length_decode_vec(enc)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 6u);

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);
}

TEST(ReplCommandsTest, counts_decode_32) {
    Interpreter interp;

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    ASSERT_GT(interp.state().matrices.count("cnt"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    ASSERT_GT(interp.state().matrices.count("hdec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);

    expect_ok(interp, "aenc = ans_encode_vec(orig)");
    expect_ok(interp, "adec = ans_decode_vec(orig, aenc)");
    ASSERT_GT(interp.state().matrices.count("adec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("adec").rows(), 5u);

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98; 99]");
    expect_ok(interp, "AE = arithmetic_encode_vec(M)");
    expect_ok(interp, "AR = arithmetic_decode_vec(M, AE)");
    ASSERT_GT(interp.state().matrices.count("AR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("AR").rows(), 7u);
}

TEST(ReplCommandsTest, sobel_rle_32) {
    Interpreter interp;

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = rle_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("R")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(1, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(2, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(3, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(4, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(5, 0), 2.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(6, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("R")(7, 0), 3.0, 1e-8);
}

TEST(ReplCommandsTest, mtf_bwt_32) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    expect_ok(interp, "R = mtf_decode_vec(E)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    ASSERT_GT(interp.state().matrices.count("BE"), 0u);
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(ReplCommandsTest, delta_lzw_32) {
    Interpreter interp;

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "DE = delta_encode_vec(B)");
    ASSERT_GT(interp.state().matrices.count("DE"), 0u);
    ASSERT_GT(interp.state().matrices.at("DE").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("LE"), 0u);
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    ASSERT_GT(interp.state().matrices.count("LR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("LR").rows(), 6u);
}

TEST(ReplCommandsTest, compress_geodesic_32) {
    Interpreter interp;

    expect_ok(interp, "bits = [1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 0; 0; 1; 0; 1; 1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    ASSERT_GT(interp.state().matrices.count("bytes"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    ASSERT_GT(interp.state().matrices.count("bits2"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, lz77_bzip2_32) {
    Interpreter interp;

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    ASSERT_GT(interp.state().matrices.count("T"), 0u);
    expect_ok(interp, "R = lz77_decode_vec(T)");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    ASSERT_GT(interp.state().matrices.count("DR"), 0u);
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}
