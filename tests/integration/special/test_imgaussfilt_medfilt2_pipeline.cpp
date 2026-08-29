
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

TEST(IntegrationSpecial,  ImgaussfiltMedfilt2Tail27) {
    Interpreter interp;
    expect_contains(interp, "help", "imgaussfilt(M,s)");
    expect_contains(interp, "help", "medfilt2(M,k)");

    expect_ok(interp, "G = [0, 0.5, 1; 0.25, 0.75, 0.5; 1, 0, 0.25]");
    expect_ok(interp, "B = imgaussfilt(G, 1)");
    EXPECT_EQ(interp.state().matrices.at("B").rows(), 3u);

    expect_ok(interp, "M = [0,0,0,0,0; 0,0,0,0,0; 0,0,1,0,0; 0,0,0,0,0; 0,0,0,0,0]");
    expect_ok(interp, "F = medfilt2(M, 3)");
    EXPECT_NEAR(interp.state().matrices.at("F")(2, 2), 0.0, 1e-9);
}

TEST(IntegrationSpecial,  JacobiScScalar) {
    Interpreter interp;
    expect_ok(interp, "jsc = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsc"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}
