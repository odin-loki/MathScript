
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

TEST(IntegrationMl,  IsoAgglo) {
    Interpreter interp;
    expect_contains(interp, "help", "ml_isolation_forest_fit");
    expect_contains(interp, "help", "ml_agglomerative_fit");

    expect_ok(interp, "IsoX = [0, 0; 0.1, 0; 0.2, 0; 0.3, 0; 0.4, 0; 10, 10]");
    expect_ok(interp, "iso_m = ml_isolation_forest_fit(IsoX, 50, 32, 42)");
    expect_ok(interp, "iso_s = ml_isolation_forest_score(IsoX, iso_m)");
    EXPECT_EQ(interp.state().matrices.at("iso_s").rows(), 6u);

    expect_ok(interp, "A = [0, 0; 1, 0; 2, 0; 3, 0; 4, 0; 100, 0; 101, 0; 102, 0; 103, 0; 104, 0]");
    expect_ok(interp, "ac_l = ml_agglomerative_fit(A, 2, \"single\")");
    EXPECT_EQ(interp.state().matrices.at("ac_l").rows(), 10u);
}

TEST(IntegrationMl,  NumthyLcmScalar) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}
