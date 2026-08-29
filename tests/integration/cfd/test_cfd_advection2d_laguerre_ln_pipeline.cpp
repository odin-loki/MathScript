
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

TEST(IntegrationCfd,  CfdAdvection2dTail14) {
    Interpreter interp;
    expect_contains(interp, "help", "cfd_advection2d");

    expect_ok(interp, "g = cfd_advection2d(8, 8, 1, 0, 0.1, 0.01)");
    ASSERT_GT(interp.state().matrices.count("g"), 0u);
}

TEST(IntegrationCfd,  LaguerreLnScalar) {
    Interpreter interp;
    expect_ok(interp, "ln = laguerre_ln(2, 1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("ln"), ms::laguerre_ln(2, 1, 0.5), 1e-8);
}
