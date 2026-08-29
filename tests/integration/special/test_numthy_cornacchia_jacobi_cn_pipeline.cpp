
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

TEST(IntegrationSpecial,  NumthyCornacchia) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_cornacchia(d,p)");

    expect_ok(interp, "xy = numthy_cornacchia(1, 5)");
    const auto& xy = interp.state().matrices.at("xy");
    EXPECT_EQ(xy.rows(), 1u);
    EXPECT_EQ(xy.cols(), 2u);
    EXPECT_NEAR(xy(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(xy(0, 1), 1.0, 1e-9);
}

TEST(IntegrationSpecial,  JacobiCnScalar) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}
