
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

TEST(IntegrationStats,  LjungBoxTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_ljung_box(x,max_lag)");

    expect_ok(interp, "lbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]");
    expect_ok(interp, "lb = stats_ljung_box(lbx, 2)");
    ASSERT_GT(interp.state().matrices.count("lb"), 0u);
}

TEST(IntegrationStats,  JacobiCnScalar) {
    Interpreter interp;
    expect_ok(interp, "jcn = jacobi_cn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jcn"), ms::jacobi_cn(0.5, 0.5), 1e-8);
}
