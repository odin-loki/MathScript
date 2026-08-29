
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

TEST(IntegrationSpecial,  SchmidtMpcSplit) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schmidt_decomposition");
    expect_contains(interp, "help", "mpc_split");

    expect_ok(interp, "psi = [0.5; 0.5; 0.5; 0.5]");
    expect_ok(interp, "sch = quantum_schmidt_decomposition(psi, 2, 2)");
    EXPECT_GT(interp.state().matrices.at("sch").rows(), 0u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "sh = mpc_split(42, 5, 3)");
    ASSERT_EQ(interp.state().matrices.at("sh").cols(), 3u);
}

TEST(IntegrationSpecial,  JacobiScScalar) {
    Interpreter interp;
    expect_ok(interp, "js = jacobi_sc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("js"), ms::jacobi_sc(0.5, 0.5), 1e-8);
}
