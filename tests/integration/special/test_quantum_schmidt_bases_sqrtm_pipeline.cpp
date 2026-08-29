
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

TEST(IntegrationSpecial,  SchmidtSqrtmTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_schmidt_bases");
    expect_contains(interp, "help", "sqrtm");

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);

    expect_ok(interp, "D = [4, 0; 0, 9]");
    expect_ok(interp, "S = sqrtm(D)");
    EXPECT_NEAR(interp.state().matrices.at("S")(0, 0), 2.0, 1e-6);
}

TEST(IntegrationSpecial,  JacobiDsScalar) {
    Interpreter interp;
    expect_ok(interp, "jds = jacobi_ds(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jds"), ms::jacobi_ds(0.2, 0.3), 1e-8);
}
