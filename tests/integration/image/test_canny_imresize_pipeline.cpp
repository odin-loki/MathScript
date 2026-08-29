
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

TEST(IntegrationImage,  CannyResizeTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "canny");
    expect_contains(interp, "help", "imresize");

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "E = canny(M, 0.1, 0.3)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "R = imresize(G, 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 2u);
}

TEST(IntegrationImage,  ZetaHurwitzScalar) {
    Interpreter interp;
    expect_ok(interp, "zh = zeta_hurwitz(2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("zh"), ms::zeta_hurwitz(2.0, 0.3), 1e-8);
}
