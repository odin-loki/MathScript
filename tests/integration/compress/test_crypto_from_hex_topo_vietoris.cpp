
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/special/special.hpp"

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

TEST(IntegrationCompress,  CryptoTopoCompressDecodeTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "crypto_from_hex");
    expect_contains(interp, "help", "topo_simplicial_counts");
    expect_contains(interp, "help", "huffman_decode_vec");

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

TEST(IntegrationCompress,  SphericalJnScalar) {
    Interpreter interp;
    expect_ok(interp, "sj = spherical_jn(1, 1.0)");
    EXPECT_NEAR(interp.state().scalars.at("sj"), ms::spherical_jn(1, 1.0), 1e-8);
}
