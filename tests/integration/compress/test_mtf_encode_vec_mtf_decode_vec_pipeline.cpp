
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

TEST(IntegrationCompress,  MtfBwtEncodeDecodeTail18) {
    Interpreter interp;
    expect_contains(interp, "help", "mtf_encode_vec(M)");
    expect_contains(interp, "help", "bwt_encode_vec(M)");
    expect_contains(interp, "help", "mtf_decode_vec(M)");

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = mtf_encode_vec(B)");
    expect_ok(interp, "R = mtf_decode_vec(E)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);

    expect_ok(interp, "Bw = [98; 97; 110; 97; 110; 97]");
    expect_ok(interp, "BE = bwt_encode_vec(Bw)");
    EXPECT_EQ(interp.state().matrices.at("BE").rows(), 7u);
}

TEST(IntegrationCompress,  ProbChi2CdfScalar) {
    Interpreter interp;
    expect_ok(interp, "cc = prob_chi2_cdf(1, 2)");
    EXPECT_NEAR(interp.state().scalars.at("cc"), ms::chi2_cdf(1, 2), 1e-8);
}
