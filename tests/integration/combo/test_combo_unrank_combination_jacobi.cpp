
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

TEST(IntegrationCombo,  ComboUnrankCombination) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_unrank_combination(n,k,rank)");

    expect_ok(interp, "c0 = combo_unrank_combination(4, 2, 0)");
    EXPECT_NEAR(interp.state().matrices.at("c0")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c0")(1, 0), 1.0, 1e-9);

    expect_ok(interp, "c1 = combo_unrank_combination(4, 2, 1)");
    EXPECT_NEAR(interp.state().matrices.at("c1")(0, 0), 0.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("c1")(1, 0), 2.0, 1e-9);
}

TEST(IntegrationCombo,  JacobiDnScalar) {
    Interpreter interp;
    expect_ok(interp, "jdn = jacobi_dn(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jdn"), ms::jacobi_dn(0.5, 0.5), 1e-8);
}
