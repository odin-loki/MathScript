
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

TEST(IntegrationMl,  PcaFitTransform) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_pca_fit");
    expect_contains(interp, "help", "ml_pca_transform");

    expect_ok(interp, "X = [1, 0; 2, 0; 3, 0; 4, 0; 5, 0]");
    expect_ok(interp, "model = ml_pca_fit(X, 1)");
    EXPECT_EQ(interp.state().matrices.at("model").rows(), 2u);

    expect_ok(interp, "Z = ml_pca_transform(X, model)");
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z").cols(), 1u);

    expect_ok(interp, "Z2 = ml_pca_fit_transform(X, 1)");
    EXPECT_EQ(interp.state().matrices.at("Z2").rows(), 5u);
    EXPECT_EQ(interp.state().matrices.at("Z2").cols(), 1u);
}

TEST(IntegrationMl,  JacobiAmScalar) {
    Interpreter interp;
    expect_ok(interp, "ja = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ja"), ms::jacobi_am(0.5, 0.5), 1e-8);
}
