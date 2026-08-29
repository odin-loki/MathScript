
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

TEST(IntegrationStats,  KdeBootstrapTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_kde");
    expect_contains(interp, "help", "stats_bootstrap_ci");

    expect_ok(interp, "k = stats_kde([0; 1; 2; 3; 4], [-1; 0; 1; 2; 3; 4; 5], 1)");
    EXPECT_EQ(interp.state().matrices.at("k").rows(), 7u);

    expect_ok(interp, "ci = stats_bootstrap_ci([1; 2; 3; 4; 5; 6; 7; 8; 9; 10])");
    EXPECT_EQ(interp.state().matrices.at("ci").cols(), 4u);
}

TEST(IntegrationStats,  DawsonScalar) {
    Interpreter interp;
    expect_ok(interp, "dw = dawson(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("dw"), ms::dawson(0.5), 1e-8);
}
