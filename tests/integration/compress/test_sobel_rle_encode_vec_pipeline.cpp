
#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>

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

TEST(IntegrationCompress,  SobelRleRoundTrip) {
    Interpreter interp;
    expect_contains(interp, "help", "sobel");
    expect_contains(interp, "help", "rle_encode_vec");
    expect_contains(interp, "help", "rle_decode_vec");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "S = sobel(G)");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 3u);

    expect_ok(interp, "B = [1, 1, 2, 2; 2, 2, 3, 3]");
    expect_ok(interp, "E = rle_encode_vec(B)");
    expect_ok(interp, "R = rle_decode_vec(E)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
    const auto& r = interp.state().matrices.at("R");
    const std::vector<double> want{1, 1, 2, 2, 2, 2, 3, 3};
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(r(i, 0), want[i], 1e-8);
    }
}

TEST(IntegrationCompress,  ProbPoisPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "pp = prob_pois_pdf(2, 2)");
    EXPECT_NEAR(interp.state().scalars.at("pp"), ms::pois_pdf(2, 2), 1e-8);
}
