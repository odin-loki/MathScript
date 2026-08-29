
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

TEST(IntegrationImage,  HarrisShiTomasiTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "harris");
    expect_contains(interp, "help", "shi_tomasi");

    expect_ok(interp,
              "Q = [0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1; "
              "0,0,0,0,0,0,1,1,1,1,1,1; 0,0,0,0,0,0,1,1,1,1,1,1]");
    expect_ok(interp, "H = harris(Q, 0.04, 0.001)");
    EXPECT_EQ(interp.state().matrices.at("H").cols(), 3u);
    expect_ok(interp, "S = shi_tomasi(Q, 5, 0.01)");
    EXPECT_EQ(interp.state().matrices.at("S").cols(), 3u);
}

TEST(IntegrationImage,  EllipEIncScalar) {
    Interpreter interp;
    expect_ok(interp, "ee = ellip_e_inc(0.5, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ee"), ms::ellip_e_inc(0.5, 0.3), 1e-8);
}
