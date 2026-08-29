
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

TEST(IntegrationMl,  MinmaxEncode) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_minmax_scaler_fit");
    expect_contains(interp, "help", "arithmetic_encode_vec");
    expect_contains(interp, "help", "ans_encode_vec");

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "mm_m = ml_minmax_scaler_fit(X)");
    expect_ok(interp, "Zm = ml_minmax_scaler_transform(X, mm_m)");
    EXPECT_EQ(interp.state().matrices.at("Zm").rows(), 3u);

    expect_ok(interp, "bits = arithmetic_encode_vec([1; 0; 1; 0; 1])");
    EXPECT_GT(interp.state().matrices.at("bits").rows(), 0u);

    expect_ok(interp, "ans = ans_encode_vec([1; 2; 3; 1; 2])");
    EXPECT_GT(interp.state().matrices.at("ans").rows(), 0u);
}

TEST(IntegrationMl,  Theta4Scalar) {
    Interpreter interp;
    expect_ok(interp, "th4 = theta4(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th4"), ms::theta4(0.2, 0.1), 1e-8);
}
