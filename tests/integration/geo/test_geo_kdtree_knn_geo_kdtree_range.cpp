
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

TEST(IntegrationGeo,  KdtreeKnnRangeTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_knn(P,x,y,k)");
    expect_contains(interp, "help", "geo_kdtree_range(P,x,y,r)");

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);
}

TEST(IntegrationGeo,  JacobiAmScalar) {
    Interpreter interp;
    expect_ok(interp, "jam = jacobi_am(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jam"), ms::jacobi_am(0.5, 0.5), 1e-8);
}
