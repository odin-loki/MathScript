
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

TEST(IntegrationSpecial,  MatmulDmIzaacTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "quantum_matmul_dm");
    expect_contains(interp, "help", "izaac_rand_matrix");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "Ad = quantum_dagger(A)");
    expect_ok(interp, "I2 = quantum_matmul_dm(A, Ad)");
    EXPECT_EQ(interp.state().matrices.at("I2").rows(), 2u);

    expect_ok(interp, "izaac seed 99");
    expect_ok(interp, "rm = izaac_rand_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rm").rows(), 3u);
}

TEST(IntegrationSpecial,  JacobiNsScalar) {
    Interpreter interp;
    expect_ok(interp, "jns = jacobi_ns(0.2, 0.3)");
    EXPECT_NEAR(interp.state().scalars.at("jns"), ms::jacobi_ns(0.2, 0.3), 1e-8);
}
