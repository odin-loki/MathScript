
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

TEST(IntegrationSpecial,  IzaacVrfKeygenFuzzMutateTail28) {
    Interpreter interp;
    expect_contains(interp, "help", "izaac_vrf_keygen");
    expect_contains(interp, "help", "izaac_fuzz_mutate");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    EXPECT_EQ(interp.state().matrices.at("vrf").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("vrf").cols(), 32u);

    expect_ok(interp, "mut = izaac_fuzz_mutate([65, 66, 67], 2)");
    ASSERT_GT(interp.state().matrices.at("mut").rows(), 0u);
}

TEST(IntegrationSpecial,  JacobiNcScalar) {
    Interpreter interp;
    expect_ok(interp, "jn = jacobi_nc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jn"), ms::jacobi_nc(0.5, 0.5), 1e-8);
}
