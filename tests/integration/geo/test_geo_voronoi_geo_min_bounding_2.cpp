
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"

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

TEST(IntegrationGeo,  VoronoiMinRectTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_voronoi");
    expect_contains(interp, "help", "geo_min_bounding_rect");

    expect_ok(interp, "V = geo_voronoi([0, 0; 1, 0; 1, 1; 0, 1])");
    EXPECT_EQ(interp.state().matrices.at("V").cols(), 2u);

    expect_ok(interp, "R = geo_min_bounding_rect([0, 0; 1, 0; 0.5, 1])");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 5u);
}

TEST(IntegrationGeo,  ProbTPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "tp = prob_t_pdf(0, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tp"), ms::t_pdf(0, 5), 1e-8);
}
