
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

TEST(IntegrationLinalg,  HoughTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "hough_lines(M[,edge])");
    expect_contains(interp, "help", "hough_circles(M[,r_min,r_max])");

    expect_ok(interp,
              "E = [0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 1,1,1,1,1,1,1,1,1,1; "
              "0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; 0,0,0,0,0,0,0,0,0,0; "
              "0,0,0,0,0,0,0,0,0,0]");
    expect_ok(interp, "L = hough_lines(E, 0.5, 180, 200, 5)");
    EXPECT_EQ(interp.state().matrices.at("L").cols(), 3u);

    expect_ok(interp, "B = zeros(30, 30)");
    expect_ok(interp, "C = hough_circles(B, 5, 15)");
    EXPECT_EQ(interp.state().matrices.at("C").cols(), 4u);
}

TEST(IntegrationLinalg,  EllipKScalar) {
    Interpreter interp;
    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ek"), ms::ellip_k(0.5), 1e-8);
}
