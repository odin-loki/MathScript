
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

TEST(IntegrationMl,  MlScalerCompressTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_standard_scaler_fit(X)");
    expect_contains(interp, "help", "ml_minmax_scaler_fit(X)");
    expect_contains(interp, "help", "arithmetic_encode_vec");

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    double col0_mean = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        col0_mean += interp.state().matrices.at("Z")(i, 0);
    }
    col0_mean /= 3.0;
    EXPECT_NEAR(col0_mean, 0.0, 1e-9);

    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    EXPECT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    EXPECT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(IntegrationMl,  SpheroidalS2Scalar) {
    Interpreter interp;

    expect_ok(interp, "s2 = spheroidal_s2(1, 0, 0.1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("s2"), ms::spheroidal_s2(1, 0, 0.1, 0.5), 1e-5);
}
