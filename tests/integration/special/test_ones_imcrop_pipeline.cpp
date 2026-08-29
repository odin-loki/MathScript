
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

TEST(IntegrationSpecial,  ImcropTriangulateTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "imcrop");
    expect_contains(interp, "help", "geo_triangulate_polygon");

    expect_ok(interp, "M = ones(8, 8)");
    expect_ok(interp, "crop = imcrop(M, 2, 2, 6, 6)");
    EXPECT_EQ(interp.state().matrices.at("crop").rows(), 4u);

    expect_ok(interp, "P = [0,0; 1,0; 1,1; 0,1]");
    expect_ok(interp, "T = geo_triangulate_polygon(P)");
    EXPECT_EQ(interp.state().matrices.at("T").rows(), 2u);
}

TEST(IntegrationSpecial,  JacobiCnScalar) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cn(0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cn(0, 0.5), 1e-8);
}
