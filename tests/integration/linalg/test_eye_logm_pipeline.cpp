
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

TEST(IntegrationLinalg,  LogmCosmSinmTail16) {
    Interpreter interp;
    expect_contains(interp, "help", "logm");
    expect_contains(interp, "help", "cosm");
    expect_contains(interp, "help", "sinm");

    expect_ok(interp, "I = eye(2)");
    expect_ok(interp, "L = logm(I)");
    EXPECT_NEAR(interp.state().matrices.at("L")(0, 0), 0.0, 1e-6);

    expect_ok(interp, "T = [0, 0; 0, 1.5707963267948966]");
    expect_ok(interp, "Cs = cosm(T)");
    expect_ok(interp, "Sn = sinm(T)");
    ASSERT_GT(interp.state().matrices.count("Sn"), 0u);
}

TEST(IntegrationLinalg,  Hypergeo1f1Scalar) {
    Interpreter interp;
    expect_ok(interp, "h1 = hypergeo_1f1(1, 0.25)");
    EXPECT_NEAR(interp.state().scalars.at("h1"), ms::hypergeo_1f1(1, 0.25), 1e-8);
}
