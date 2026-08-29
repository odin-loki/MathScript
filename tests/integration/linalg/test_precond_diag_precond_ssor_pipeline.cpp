
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

TEST(IntegrationLinalg,  PrecondDiagSsorTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "precond_diag");
    expect_contains(interp, "help", "precond_ssor");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Pd = precond_diag(A)");
    ASSERT_GT(interp.state().matrices.count("Pd"), 0u);

    expect_ok(interp, "M = [4, -1, 0; -1, 4, -1; 0, -1, 4]");
    expect_ok(interp, "Ps = precond_ssor(M, 1.2)");
    ASSERT_GT(interp.state().matrices.count("Ps"), 0u);
}

TEST(IntegrationLinalg,  HermiteHeScalar) {
    Interpreter interp;
    expect_ok(interp, "he = hermite_he(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("he"), ms::hermite_he(2, 0.5), 1e-8);
}
