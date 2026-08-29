
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

TEST(IntegrationFinance,  LogMinVarianceTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "laplacian_of_gaussian");
    expect_contains(interp, "help", "finance_min_variance_portfolio");

    expect_ok(interp, "G = [1, 2; 3, 4]");
    expect_ok(interp, "L = laplacian_of_gaussian(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("L").rows(), 2u);

    expect_ok(interp, "cov2 = [0.04, 0.01; 0.01, 0.02]");
    expect_ok(interp, "w_min = finance_min_variance_portfolio(cov2)");
    EXPECT_EQ(interp.state().matrices.at("w_min").rows(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("w_min")(0, 0) +
                    interp.state().matrices.at("w_min")(1, 0),
                1.0, 1e-8);
}

TEST(IntegrationFinance,  SphBesselYScalar) {
    Interpreter interp;
    expect_ok(interp, "sy = sph_bessel_y(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sy"), ms::sph_bessel_y(1, 1.0), 1e-8);
}
