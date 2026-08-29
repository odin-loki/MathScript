
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

TEST(IntegrationMl,  MlTsneGolombTail21) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_tsne_fit");
    expect_contains(interp, "help", "golomb_rice_encode_vec");
    expect_contains(interp, "help", "golomb_rice_decode_vec");

    expect_ok(interp, "T = [0, 0; 0.1, 0; 0.2, 0; 10, 10; 10.1, 10; 10.2, 10]");
    expect_ok(interp, "Y = ml_tsne_fit(T, 5, 10, 42)");
    EXPECT_EQ(interp.state().matrices.at("Y").rows(), 6u);
    EXPECT_EQ(interp.state().matrices.at("Y").cols(), 2u);

    expect_ok(interp, "V = [0; 1; 2; 5; 10]");
    expect_ok(interp, "GE = golomb_rice_encode_vec(V, 2)");
    expect_ok(interp, "GR = golomb_rice_decode_vec(GE, 2, 5)");
    EXPECT_EQ(interp.state().matrices.at("GR").rows(), 5u);
}

TEST(IntegrationMl,  GeoVec2dLengthScalar) {
    Interpreter interp;
    expect_ok(interp, "vl = geo_vec2d_length(3, 4)");
    EXPECT_NEAR(interp.state().scalars.at("vl"), 5.0, 1e-8);
}
