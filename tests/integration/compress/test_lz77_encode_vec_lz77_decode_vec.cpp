
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"

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

TEST(IntegrationCompress,  Lz77DecodeBzip2) {
    Interpreter interp;
    expect_contains(interp, "help", "lz77_decode_vec");
    expect_contains(interp, "help", "bzip2_compress_vec");
    expect_contains(interp, "help", "bzip2_decompress_vec");

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(M)");
    expect_ok(interp, "R = lz77_decode_vec(T)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 9u);

    expect_ok(interp, "Bz = [97; 98; 114; 97; 99; 97; 100; 97; 98; 114; 97]");
    expect_ok(interp, "C = bzip2_compress_vec(Bz)");
    expect_ok(interp, "DR = bzip2_decompress_vec(C)");
    EXPECT_EQ(interp.state().matrices.at("DR").rows(), 11u);
}

TEST(IntegrationCompress,  ProbPoisCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "pc = prob_pois_cdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pc"), ms::pois_cdf(2, 2), 1e-8);
}
