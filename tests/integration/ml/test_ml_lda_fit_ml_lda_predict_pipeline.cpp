
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

TEST(IntegrationMl,  LdaFitPredictTransform) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_lda_fit");
    expect_contains(interp, "help", "ml_lda_predict");

    expect_ok(interp, "Lx = [-2,-1; -1,-0.5; 2,1; 3,1.5]");
    expect_ok(interp, "Ly = [0; 0; 1; 1]");
    expect_ok(interp, "lda_m = ml_lda_fit(Lx, Ly)");
    expect_ok(interp, "lda_p = ml_lda_predict([0,0; 3,1], lda_m)");
    EXPECT_LT(interp.state().matrices.at("lda_p")(0, 0), 0.5);
    EXPECT_GT(interp.state().matrices.at("lda_p")(1, 0), 0.5);

    expect_ok(interp, "lda_z = ml_lda_transform(Lx, lda_m)");
    EXPECT_EQ(interp.state().matrices.at("lda_z").rows(), 4u);
}

TEST(IntegrationMl,  HermiteHScalar) {
    Interpreter interp;
    expect_ok(interp, "hh = hermite_h(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hh"), ms::hermite_h(2, 0.5), 1e-8);
}
