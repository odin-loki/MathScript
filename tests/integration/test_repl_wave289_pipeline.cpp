// MathScript Integration Tests: REPL Interpreter – Wave 289 Pipeline
//
// Wave 289 REPL smoke: stats/geo/image tail11 extensions, chebyshev_u/hermite_he scalar.

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

TEST(ReplWave289Pipeline, StatsGeoImage) {
    Interpreter interp;

    expect_contains(interp, "help", "stats_one_way_anova");
    expect_contains(interp, "help", "geo_voronoi");
    expect_contains(interp, "help", "threshold_otsu");

    expect_ok(interp, "G = [1, 2, 3; 10, 11, 12; 20, 21, 22]");
    expect_ok(interp, "an = stats_one_way_anova(G)");
    EXPECT_EQ(interp.state().matrices.at("an").rows(), 1u);

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_GT(interp.state().matrices.at("V").rows(), 0u);

    expect_ok(interp, "M = [10, 20; 30, 40]");
    expect_ok(interp, "T = threshold_otsu(M)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
}

TEST(ReplWave289Pipeline, SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "cu = chebyshev_u(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("cu"), ms::chebyshev_u(2, 0.5), 1e-9);

    expect_ok(interp, "he = hermite_he(3, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(3, 0.25), 1e-9);
}
