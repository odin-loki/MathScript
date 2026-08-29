
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"
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

TEST(IntegrationGeo,  KdtreeGaussTail19) {
    Interpreter interp;
    expect_contains(interp, "help", "geo_kdtree_knn(P,x,y,k)");
    expect_contains(interp, "help", "geo_kdtree_range(P,x,y,r)");
    expect_contains(interp, "help", "imgaussfilt(M,s)");

    expect_ok(interp, "P = [0,0; 1,0; 2,0; 3,0; 4,0]");
    expect_ok(interp, "n = geo_kdtree_knn(P, 1.0, 0.0, 2)");
    EXPECT_EQ(interp.state().matrices.at("n").rows(), 2u);
    expect_ok(interp, "r = geo_kdtree_range(P, 2.0, 0.0, 1.5)");
    EXPECT_GE(interp.state().matrices.at("r").rows(), 3u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);
}

TEST(IntegrationGeo,  ProbTPpfScalar) {
    Interpreter interp;
    expect_ok(interp, "tq = prob_t_ppf(0.5, 5)");
    EXPECT_NEAR(interp.state().scalars.at("tq"), ms::t_ppf(0.5, 5), 1e-8);
}
