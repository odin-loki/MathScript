
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

TEST(IntegrationQuantum,  QuantumIzaac) {
    Interpreter interp;

    expect_contains(interp, "help", "quantum_dagger");
    expect_contains(interp, "help", "izaac_rand_matrix");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    EXPECT_EQ(interp.state().matrices.at("Ad").rows(), 2u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rm = izaac_rand_matrix(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 2u);

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sb = quantum_schmidt_bases(psi, 2, 2)");
    EXPECT_GE(interp.state().matrices.at("sb").rows(), 5u);
}

TEST(IntegrationQuantum,  SpecialScalar) {
    Interpreter interp;

    expect_ok(interp, "la = laguerre_la(2, 1.0, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("la"), ms::laguerre_la(2, 1.0, 0.5), 1e-9);
}
