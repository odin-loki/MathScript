
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

TEST(IntegrationLinalg,  TransposeCholExpmInv) {
    Interpreter interp;
    expect_contains(interp, "help", "transpose");
    expect_contains(interp, "help", "chol");
    expect_contains(interp, "help", "expm");
    expect_contains(interp, "help", "inv");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "T = transpose(A)");
    EXPECT_NEAR(interp.state().matrices.at("T")(0, 1), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("T")(1, 0), 2.0, 1e-8);

    expect_ok(interp, "S = [4, 1; 1, 3]");
    expect_ok(interp, "L = chol(S)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "E0 = [0, 1; 0, 0]");
    expect_ok(interp, "E = expm(E0)");
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 0), 1.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("E")(0, 1), 1.0, 1e-8);

    expect_ok(interp, "B = [2, 0; 0, 2]");
    expect_ok(interp, "Bi = inv(B)");
    EXPECT_NEAR(interp.state().matrices.at("Bi")(0, 0), 0.5, 1e-8);
}

TEST(IntegrationLinalg,  HermiteHfScalar) {
    Interpreter interp;
    expect_ok(interp, "hf = hermite_hf(2, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("hf"), ms::hermite_hf(2, 0.5), 1e-8);
}
