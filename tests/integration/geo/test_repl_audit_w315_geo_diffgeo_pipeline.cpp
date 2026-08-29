// MathScript Integration Tests: REPL Interpreter – Audit Wave 315 Geo/Diffgeo Pipeline
//
// Inventory: leftover geo names are already bound as geo_* aliases, or Wave 311-descoped
// vec helpers (dot/length/normalise). Diffgeo MetricFn/SurfaceFn/CurveFn APIs need a
// session-object / CFunc design and are deferred.

#include <gtest/gtest.h>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(ReplAuditW315GeoDiffgeoPipeline, Dist2dIs345Triangle) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_dist2d(x1,y1,x2,y2)");

    expect_ok(interp, "d = geo_dist2d(0, 0, 3, 4)");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("d"), 5.0);
}

TEST(ReplAuditW315GeoDiffgeoPipeline, ConvexHullSquareHasFourVertices) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_convex_hull(P)");

    expect_ok(interp, "sq = [0, 0; 1, 0; 1, 1; 0, 1]");
    expect_ok(interp, "hull = geo_convex_hull(sq)");
    EXPECT_EQ(interp.state().matrices.at("hull").rows(), 4u);
}
