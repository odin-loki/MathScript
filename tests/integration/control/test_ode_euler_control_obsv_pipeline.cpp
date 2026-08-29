
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

TEST(IntegrationControl,  OdeEulerControlObsv) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_euler");
    expect_contains(interp, "help", "control_obsv(A,C)");

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    EXPECT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(IntegrationControl,  StruveHnScalar) {
    Interpreter interp;
    expect_ok(interp, "hn = struve_hn(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("hn"), ms::struve_hn(1, 1), 1e-8);
}
