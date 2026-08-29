
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

TEST(IntegrationMl,  DbscanSpectralScaler) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_dbscan_fit");
    expect_contains(interp, "help", "ml_spectral_clustering");
    expect_contains(interp, "help", "ml_standard_scaler_fit");

    expect_ok(interp, "D = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 0; 10.1, 0; 10.2, 0; 10.3, 0; 10.4, 0]");
    expect_ok(interp, "db_l = ml_dbscan_fit(D, 0.5, 2)");
    EXPECT_EQ(interp.state().matrices.at("db_l").rows(), 10u);

    expect_ok(interp, "S = [0, 0; 0.1, 0; 0.2, 0; 0, 0.1; 10, 10; 10.1, 10; 10.2, 10; 10, 10.1]");
    expect_ok(interp, "sp_l = ml_spectral_clustering(S, 2, 1.0)");
    EXPECT_EQ(interp.state().matrices.at("sp_l").rows(), 8u);

    expect_ok(interp, "X = [1, 2; 3, 4; 5, 6]");
    expect_ok(interp, "ss_m = ml_standard_scaler_fit(X)");
    expect_ok(interp, "Z = ml_standard_scaler_transform(X, ss_m)");
    EXPECT_EQ(interp.state().matrices.at("Z").rows(), 3u);
}

TEST(IntegrationMl,  Theta3Scalar) {
    Interpreter interp;
    expect_ok(interp, "th3 = theta3(0.2, 0.1)");
    EXPECT_NEAR(interp.state().scalars.at("th3"), ms::theta3(0.2, 0.1), 1e-8);
}
