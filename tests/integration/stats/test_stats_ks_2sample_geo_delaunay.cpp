
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

TEST(IntegrationStats,  KsDelaunayTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ks_2sample(a,b)");
    expect_contains(interp, "help", "geo_delaunay_2d(P)");

    expect_ok(interp, "ksa = [0; 1; 2; 3; 4; 5; 6; 7]");
    expect_ok(interp, "ksb = [10; 11; 12; 13; 14; 15; 16; 17]");
    expect_ok(interp, "ks = stats_ks_2sample(ksa, ksb)");
    EXPECT_EQ(interp.state().matrices.at("ks").cols(), 2u);

    expect_ok(interp, "T3 = geo_delaunay_2d([0, 0; 1, 0; 0.5, 0.866])");
    ASSERT_GT(interp.state().matrices.count("T3"), 0u);
}

TEST(IntegrationStats,  EllipPiScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.3, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.3, 0.5), 1e-8);
}
