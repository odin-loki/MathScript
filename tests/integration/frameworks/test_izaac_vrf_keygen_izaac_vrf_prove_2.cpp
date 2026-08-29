
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error";
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationFrameworks,  IzaacVrfCrypto) {
    Interpreter interp;

    expect_ok(interp, "izaac seed 7");
    expect_ok(interp, "vrf = izaac_vrf_keygen()");
    expect_ok(interp, "msg = [1, 2, 3, 4]");
    expect_ok(interp, "proof = izaac_vrf_prove(vrf, msg)");

    const auto& vrf = interp.state().matrices.at("vrf");
    std::ostringstream pub_cmd;
    pub_cmd << "pub = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            pub_cmd << ", ";
        }
        pub_cmd << vrf(1, j);
    }
    pub_cmd << "]";
    expect_ok(interp, pub_cmd.str());
    expect_ok(interp, "vok = izaac_vrf_verify(pub, msg, proof)");
    EXPECT_NEAR(interp.state().scalars.at("vok"), 1.0, 1e-12);

    std::ostringstream key_cmd;
    key_cmd << "key32 = [";
    for (size_t j = 0; j < 32; ++j) {
        if (j > 0) {
            key_cmd << ", ";
        }
        key_cmd << vrf(0, j);
    }
    key_cmd << "]";
    expect_ok(interp, key_cmd.str());
    expect_ok(interp, "ct = izaac_encrypt(msg, key32)");
    expect_ok(interp, "pt = izaac_decrypt(ct, key32)");
    EXPECT_EQ(interp.state().matrices.at("pt").rows(), 4u);

    expect_ok(interp, "rn = izaac_randn_matrix(3, 2)");
    EXPECT_EQ(interp.state().matrices.at("rn").rows(), 3u);
}

TEST(IntegrationFrameworks,  QuantumCyphaSpecial) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0; 0; 1]");
    expect_ok(interp, "sn = quantum_schmidt_number(psi, 2, 2)");
    EXPECT_GE(interp.state().scalars.at("sn"), 1.0);

    expect_ok(interp, "tp = quantum_ket_tensor_product([1;0], [0;1])");
    EXPECT_EQ(interp.state().matrices.at("tp").rows(), 4u);

    expect_ok(interp, "cypha_nig_mean(0, 1, 0, 1)");

    expect_ok(interp, "tp2 = theta1_prime(0.1, 0.2)");
    EXPECT_TRUE(std::isfinite(interp.state().scalars.at("tp2")));
    expect_contains(interp, "help", "cfd_run_advection_3d");
}

TEST(IntegrationFrameworks,  CfdAdvection3d) {
    Interpreter interp;

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)");
    expect_ok(interp, "uf = cfd_run_advection_3d(g3, u0, 0.2, 0, 0, 0.01, 0.001)");
    ASSERT_GT(interp.state().matrices.count("uf"), 0u);
}
