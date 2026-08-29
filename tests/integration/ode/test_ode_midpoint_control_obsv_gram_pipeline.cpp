
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

TEST(IntegrationOde,  MidpointObsvGramTail15) {
    Interpreter interp;
    expect_contains(interp, "help", "ode_midpoint");
    expect_contains(interp, "help", "control_obsv_gram");

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wo").rows(), 0u);
}

TEST(IntegrationOde,  Hypergeo0f1Scalar) {
    Interpreter interp;
    expect_ok(interp, "h = hypergeo_0f1(1, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("h"), ms::hypergeo_0f1(1, 0.5), 1e-8);
}
