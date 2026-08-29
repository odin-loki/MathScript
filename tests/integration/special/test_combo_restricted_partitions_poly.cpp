
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

TEST(IntegrationSpecial,  RestrictedSquarefreeTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_restricted_partitions");
    expect_contains(interp, "help", "poly_squarefree");

    expect_ok(interp, "rp = combo_restricted_partitions(5, 2)");
    ASSERT_GT(interp.state().matrices.count("rp"), 0u);
    expect_ok(interp, "mult = [-12; 16; -7; 1]");
    expect_ok(interp, "sf = poly_squarefree(mult)");
    ASSERT_GT(interp.state().matrices.count("sf"), 0u);
}

TEST(IntegrationSpecial,  StruveLScalar) {
    Interpreter interp;
    expect_ok(interp, "sl = struve_l(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sl"), ms::struve_l(0, 1), 1e-8);
}
