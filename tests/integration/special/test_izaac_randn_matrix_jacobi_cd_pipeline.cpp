
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

TEST(IntegrationSpecial,  IzaacRandnMatrix) {
    Interpreter interp;
    expect_contains(interp, "help", "izaac_randn_matrix");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "rn = izaac_randn_matrix(2, 3)");
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("rn").cols(), 3u);
}

TEST(IntegrationSpecial,  JacobiCdScalar) {
    Interpreter interp;
    expect_ok(interp, "jc = jacobi_cd(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jc"), ms::jacobi_cd(0.5, 0.5), 1e-8);
}
