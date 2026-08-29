
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

TEST(IntegrationTensorops,  MatmulTensorops) {
    Interpreter interp;
    expect_contains(interp, "help", "matmul");
    expect_contains(interp, "help", "tensorops_matmul");
    expect_contains(interp, "help", "tensorops_einsum");

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    EXPECT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(IntegrationTensorops,  NumthyLcmScalar) {
    Interpreter interp;
    expect_ok(interp, "lc = numthy_lcm(4, 6)");
    EXPECT_NEAR(interp.state().scalars.at("lc"), 12.0, 1e-8);
}
