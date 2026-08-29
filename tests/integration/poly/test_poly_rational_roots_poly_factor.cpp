
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

TEST(IntegrationPoly,  RationalFactorTail27) {
    Interpreter interp;
    expect_contains(interp, "help", "poly_rational_roots");
    expect_contains(interp, "help", "poly_factor_rational");

    expect_ok(interp, "p = [6; -5; 1]");
    expect_ok(interp, "rr = poly_rational_roots(p)");
    ASSERT_GT(interp.state().matrices.count("rr"), 0u);
    expect_ok(interp, "fr = poly_factor_rational(p)");
    EXPECT_EQ(interp.state().matrices.at("fr").rows(), 3u);
}

TEST(IntegrationPoly,  StruveKScalar) {
    Interpreter interp;
    expect_ok(interp, "sk = struve_k(0, 1)");
    EXPECT_NEAR(interp.state().scalars.at("sk"), ms::struve_k(0, 1), 1e-8);
}
