
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

TEST(IntegrationLinalg,  ZerosEyeOnesRand) {
    Interpreter interp;
    expect_contains(interp, "help", "cplx_joukowski");

    expect_ok(interp, "Z = zeros(3)");
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("Z")(1, 1), 0.0, 1e-8);

    expect_ok(interp, "I = eye(4)");
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("I")(0, 1), 0.0, 1e-8);

    expect_ok(interp, "O = ones(2, 3)");
    EXPECT_EQ(interp.state().matrices.at("O").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("O").cols(), 3u);

    expect_ok(interp, "R = rand(3, 4)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("R").cols(), 4u);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const double v = interp.state().matrices.at("R")(i, j);
            EXPECT_GE(v, 0.0);
            EXPECT_LT(v, 1.0);
        }
    }

    expect_ok(interp, "N = randn(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("N").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("N").cols(), 2u);
}

TEST(IntegrationLinalg,  CplxJoukowskiScalar) {
    Interpreter interp;
    expect_ok(interp, "jw = cplx_joukowski(1, 0)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("jw")));
}
