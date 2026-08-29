
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

TEST(IntegrationImage,  RadonIradonTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "radon(M,theta)");
    expect_contains(interp, "help", "iradon");

    expect_ok(interp,
              "Img = [0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; 0,0,1,1,1,1,0,0; "
              "0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0]");
    expect_ok(interp, "Th = [0, 45, 90]");
    expect_ok(interp, "S = radon(Img, Th)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);

    expect_ok(interp, "R = iradon(S, Th)");
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 8u);
}

TEST(IntegrationImage,  ErfcxScalar) {
    Interpreter interp;
    expect_ok(interp, "ex = erfcx(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ex"), ms::erfcx(0.5), 1e-8);
}
