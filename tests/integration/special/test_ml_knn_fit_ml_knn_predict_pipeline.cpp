
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

TEST(IntegrationSpecial,  KnnFitPredict) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_knn_fit");
    expect_contains(interp, "help", "ml_knn_predict");

    expect_ok(interp, "Kx = [0,0; 1,0; 0,1; 3,3; 4,3; 3,4]");
    expect_ok(interp, "Ky = [0; 0; 0; 1; 1; 1]");
    expect_ok(interp, "knn_m = ml_knn_fit(Kx, Ky, 3)");
    expect_ok(interp, "knn_p = ml_knn_predict([0.5,0.5; 3.5,3.5], knn_m)");
    ASSERT_GT(interp.state().matrices.count("knn_p"), 0u);
}

TEST(IntegrationSpecial,  ChebyshevTScalar) {
    Interpreter interp;
    expect_ok(interp, "ct = chebyshev_t(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ct"), ms::chebyshev_t(2, 0.5), 1e-8);
}
