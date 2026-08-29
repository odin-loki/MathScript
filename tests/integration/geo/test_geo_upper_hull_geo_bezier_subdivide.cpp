
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

TEST(IntegrationGeo,  Geo) {
    Interpreter interp;

    expect_contains(interp, "help", "geo_upper_hull");
    expect_contains(interp, "help", "geo_kdtree_3d_knn");

    expect_ok(interp, "sq = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "uh = geo_upper_hull(sq)");
    EXPECT_GE(interp.state().matrices.at("uh").rows(), 2u);

    expect_ok(interp, "ctrl = [0, 0; 1, 2; 2, 0]");
    expect_ok(interp, "sub = geo_bezier_subdivide(ctrl, 0.5)");
    EXPECT_EQ(interp.state().matrices.at("sub").rows(), 6u);

    expect_ok(interp, "P = [0,0,0; 1,0,0; 2,0,0]");
    expect_ok(interp, "n = geo_kdtree_3d_knn(P, 0.9, 0, 0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
}

TEST(IntegrationGeo,  StruveScalar) {
    Interpreter interp;

    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1.0), 1e-9);

    expect_ok(interp, "aj = anger_j(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(1, 1.0), 1e-9);
}
