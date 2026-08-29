
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

TEST(IntegrationSpecial,  LyndonDeBruijnTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "combo_lyndon_words");
    expect_contains(interp, "help", "combo_de_bruijn_sequence");

    expect_ok(interp, "lw = combo_lyndon_words(3, 2)");
    ASSERT_GT(interp.state().matrices.count("lw"), 0u);
    expect_ok(interp, "db = combo_de_bruijn_sequence(2, 2)");
    EXPECT_EQ(interp.state().matrices.at("db").rows(), 4u);
}

TEST(IntegrationSpecial,  BesselZeroYnuScalar) {
    Interpreter interp;
    expect_ok(interp, "bz = bessel_zero_ynu(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("bz"), ms::bessel_zero_ynu(0, 1), 1e-8);
}
