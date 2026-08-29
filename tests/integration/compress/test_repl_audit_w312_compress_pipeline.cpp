// MathScript Integration Tests: REPL Interpreter – Audit Wave 312 Compress Pipeline
//
// Inventory found every user-facing ms::compress encode/decode pair already bound
// as *_vec (rle, run_length, mtf, lzw, lz77, huffman, arithmetic, ans, golomb_rice,
// wavelet, bzip2, bwt, delta, compress_bits_to_bytes / compress_bytes_to_bits).
// No new REPL handlers. Property-test existing rle and lzw round-trips.

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

void expect_roundtrip_bytes(Interpreter& interp, const char* encode_cmd, const char* decode_cmd) {
    expect_ok(interp, "raw = [65; 66; 67; 68; 65; 65; 65; 66]");
    expect_ok(interp, encode_cmd);
    expect_ok(interp, decode_cmd);
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    const auto& restored = interp.state().matrices.at("dec");
    ASSERT_EQ(restored.rows(), 8u);
    ASSERT_EQ(restored.cols(), 1u);
    const double expected[] = {65, 66, 67, 68, 65, 65, 65, 66};
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(restored(i, 0), expected[i]);
    }
}

} // namespace

TEST(ReplAuditW312CompressPipeline, RleRoundtripAlreadyBound) {
    Interpreter interp;
    expect_contains(interp, "help", "rle_encode_vec(M)");
    expect_contains(interp, "help", "rle_decode_vec(M)");
    expect_roundtrip_bytes(interp, "enc = rle_encode_vec(raw)", "dec = rle_decode_vec(enc)");
}

TEST(ReplAuditW312CompressPipeline, LzwRoundtripAlreadyBound) {
    Interpreter interp;
    expect_contains(interp, "help", "lzw_encode_vec(M)");
    expect_contains(interp, "help", "lzw_decode_vec(C)");
    expect_roundtrip_bytes(interp, "enc = lzw_encode_vec(raw)", "dec = lzw_decode_vec(enc)");
}
