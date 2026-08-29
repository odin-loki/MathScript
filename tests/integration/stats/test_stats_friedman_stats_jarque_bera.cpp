
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

TEST(IntegrationStats,  FriedmanJarqueBeraTail31) {
    Interpreter interp;
    expect_contains(interp, "help", "stats_friedman(data)");
    expect_contains(interp, "help", "stats_jarque_bera(x)");

    expect_ok(interp, "Fdata = [10, 20, 30; 15, 25, 20; 5, 8, 6; 100, 90, 95]");
    expect_ok(interp, "fr = stats_friedman(Fdata)");
    EXPECT_NEAR(interp.state().matrices.at("fr")(0, 0), 1.5, 1e-9);

    expect_ok(interp, "jbx = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 20; 50]");
    expect_ok(interp, "jb = stats_jarque_bera(jbx)");
    ASSERT_GT(interp.state().matrices.count("jb"), 0u);
}

TEST(IntegrationStats,  JacobiSnScalar) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}
