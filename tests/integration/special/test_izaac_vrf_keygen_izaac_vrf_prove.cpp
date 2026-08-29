
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

TEST(IntegrationSpecial,  IzaacVrfProve) {
    Interpreter interp;
    expect_contains(interp, "help", "izaac_vrf_keygen");
    expect_contains(interp, "help", "izaac_vrf_prove");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);
}

TEST(IntegrationSpecial,  JacobiDcScalar) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}
