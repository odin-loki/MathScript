
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
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

TEST(IntegrationFrameworks,  IzaacVrfProveEncryptDecrypt) {
    Interpreter interp;
    expect_contains(interp, "help", "izaac_vrf_prove");
    expect_contains(interp, "help", "izaac_encrypt");

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [65, 66, 67]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");
    ASSERT_GT(interp.state().matrices.count("proof"), 0u);

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) key_cmd << ", ";
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 3u);
}

TEST(IntegrationFrameworks,  JacobiDcScalar) {
    Interpreter interp;
    expect_ok(interp, "jd = jacobi_dc(0.5, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("jd"), ms::jacobi_dc(0.5, 0.5), 1e-8);
}
