
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

TEST(IntegrationNumthy,  ReverseNumthyTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_reverse");
    expect_contains(interp, "help", "numthy_factor_exp");

    expect_ok(interp, "r = poly_reverse([1; 2; 3])");
    ASSERT_GT(interp.state().matrices.count("r"), 0u);

    expect_ok(interp, "fe = numthy_factor_exp(12)");
    EXPECT_EQ(interp.state().matrices.at("fe").rows(), 2u);
    expect_ok(interp, "f = numthy_farey(4)");
    EXPECT_EQ(interp.state().matrices.at("f").rows(), 7u);
}

TEST(IntegrationNumthy,  AngerJScalar) {
    Interpreter interp;
    expect_ok(interp, "aj = anger_j(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("aj"), ms::anger_j(0, 1), 1e-8);
}
