
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

TEST(IntegrationSpecial,  NumthyPrimes) {
    Interpreter interp;
    expect_contains(interp, "help", "numthy_primes(lo,hi)");

    expect_ok(interp, "Pr = numthy_primes(1, 20)");
    const auto& Pr = interp.state().matrices.at("Pr");
    EXPECT_EQ(Pr.rows(), 8u);
    EXPECT_NEAR(Pr(0, 0), 2.0, 1e-9);
}

TEST(IntegrationSpecial,  JacobiSnScalar) {
    Interpreter interp;
    expect_ok(interp, "jsn = jacobi_sn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jsn"), ms::jacobi_sn(0.5, 0.5), 1e-8);
}
