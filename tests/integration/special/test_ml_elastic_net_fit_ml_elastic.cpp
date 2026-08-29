
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

TEST(IntegrationSpecial,  ElasticNetFitPredict) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_elastic_net_fit");
    expect_contains(interp, "help", "ml_elastic_net_predict");

    expect_ok(interp, "Xr = [0;1;2;3;4]");
    expect_ok(interp, "yr = [1;3;5;7;9]");
    expect_ok(interp, "enet_m = ml_elastic_net_fit(Xr, yr, 0.001, 0.5)");
    expect_ok(interp, "enet_p = ml_elastic_net_predict([5], enet_m)");
    EXPECT_NEAR(interp.state().matrices.at("enet_p")(0, 0), 11.0, 0.5);
}

TEST(IntegrationSpecial,  BesselKScalar) {
    Interpreter interp;
    expect_ok(interp, "bk = bessel_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bk"), ms::bessel_k(0, 1.0), 1e-8);
}
