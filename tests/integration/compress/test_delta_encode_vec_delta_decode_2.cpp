
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

TEST(IntegrationCompress,  DeltaLzwEncodeDecodeTail18) {
    Interpreter interp;
    expect_contains(interp, "help", "delta_encode_vec(M)");
    expect_contains(interp, "help", "lzw_encode_vec(M)");

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = delta_encode_vec(B)");
    EXPECT_GT(interp.state().matrices.at("E").rows(), 0u);
    expect_ok(interp, "D = delta_decode_vec(E)");
    EXPECT_GT(interp.state().matrices.at("D").rows(), 0u);

    expect_ok(interp, "M = [97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "LE = lzw_encode_vec(M)");
    expect_ok(interp, "LR = lzw_decode_vec(LE)");
    const auto& restored = interp.state().matrices.at("LR");
    EXPECT_EQ(restored.rows(), 6u);
    const double expected[] = {97, 98, 99, 97, 98, 99};
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(restored(i, 0), expected[i], 1e-8);
    }
}

TEST(IntegrationCompress,  ProbTCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "tc = prob_t_cdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tc"), ms::t_cdf(0, 5), 1e-8);
}
