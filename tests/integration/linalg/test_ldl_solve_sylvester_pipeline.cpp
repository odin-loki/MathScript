
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

TEST(IntegrationLinalg,  LdlSylvesterTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "ldl");
    expect_contains(interp, "help", "solve_sylvester");

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = ldl(S)");
    ASSERT_GT(interp.state().matrices.count("L"), 0u);

    expect_ok(interp, "As = [1, 0; 0, 2]");
    expect_ok(interp, "Bs = [3, 0; 0, 4]");
    expect_ok(interp, "Cs = [4, 10; 15, 24]");
    expect_ok(interp, "X = solve_sylvester(As, Bs, Cs)");
    EXPECT_NEAR(interp.state().matrices.at("X")(0, 0), 1.0, 1e-8);
}

TEST(IntegrationLinalg,  EllipPiScalar) {
    Interpreter interp;
    expect_ok(interp, "ep = ellip_pi(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("ep"), ms::ellip_pi(0.2, 0.3), 1e-8);
}
