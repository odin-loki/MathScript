
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

TEST(IntegrationMl,  MlGboostIsolationTail11) {
    Interpreter interp;

    expect_contains(interp, "help", "ml_gradient_boosting_fit(X,y");
    expect_contains(interp, "help", "ml_isolation_forest_fit");

    expect_ok(interp, "X = [0;1;2;3;4]");
    expect_ok(interp, "y = [0;1;2;3;4]");
    expect_ok(interp, "gb_m = ml_gradient_boosting_fit(X, y, 20, 0.1, 3)");
    expect_ok(interp, "gb_p = ml_gradient_boosting_predict(X, gb_m)");
    EXPECT_EQ(interp.state().matrices.at("gb_p").rows(), 5u);

    expect_ok(interp, "IsoX = [0,0; 0.1,0; 0.2,0; 0.3,0; 0.4,0; 10,10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);
    double cluster_max = interp.state().matrices.at("iso_s")(0, 0);
    for (size_t i = 1; i < 5; ++i) {
        if (interp.state().matrices.at("iso_s")(i, 0) > cluster_max) {
            cluster_max = interp.state().matrices.at("iso_s")(i, 0);
        }
    }
    EXPECT_GT(interp.state().matrices.at("iso_s")(5, 0), cluster_max);
}

TEST(IntegrationMl,  ChebyshevTScalar) {
    Interpreter interp;

    expect_ok(interp, "t = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("t"), ms::chebyshev_t(2, 0.5), 1e-8);
}
