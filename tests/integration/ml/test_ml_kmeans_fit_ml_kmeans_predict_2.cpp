
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

TEST(IntegrationMl,  MlKmeansTail18) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_kmeans_fit(X,k)");
    expect_contains(interp, "help", "ml_kmeans_predict(X,model)");

    expect_ok(interp, "X2 = [0, 0; 1, 0; 2, 0; 100, 0; 101, 0; 102, 0]");
    expect_ok(interp, "km = ml_kmeans_fit(X2, 2)");
    EXPECT_EQ(interp.state().matrices.at("km").rows(), 2u);

    expect_ok(interp, "labels = ml_kmeans_predict(X2, km)");
    EXPECT_EQ(interp.state().matrices.at("labels").rows(), 6u);
}

TEST(IntegrationMl,  JacobiNsScalar) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.5, 0.5), 1e-8);
}
