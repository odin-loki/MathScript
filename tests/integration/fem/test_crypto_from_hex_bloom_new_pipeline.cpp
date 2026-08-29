
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/frameworks/izaac/izaac.hpp"
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

TEST(IntegrationFem,  CryptoBloomTopoCompress) {
    Interpreter interp;
    ms::izaac::clear_session();

    expect_contains(interp, "help", "crypto_from_hex");
    expect_contains(interp, "help", "topo_simplicial_counts");
    expect_contains(interp, "help", "bloom_bit_count");

    expect_ok(interp, "b = crypto_from_hex(\"deadbeef\")");
    EXPECT_EQ(interp.state().matrices.at("b").rows(), 4u);

    expect_ok(interp, "izaac seed 42");
    expect_ok(interp, "bloom_new(bf, 100, 0.01)");
    expect_ok(interp, "bloom_insert(bf, \"wave277\")");
    expect_ok(interp, "bloom_bit_count(bf)");
    expect_ok(interp, "bloom_hash_count(bf)");

    expect_ok(interp, "D = [0, 1; 1, 0]");
    expect_ok(interp, "vr = topo_vietoris_rips(D, 1.0)");
    expect_ok(interp, "cnt = topo_simplicial_counts(vr)");
    expect_ok(interp, "dim = topo_simplicial_dimension(vr)");
    ASSERT_GT(interp.state().scalars.count("dim"), 0u);

    expect_ok(interp, "orig = [1, 2, 3, 4, 5]");
    expect_ok(interp, "enc = ans_encode_vec(orig)");
    expect_ok(interp, "dec = ans_decode_vec(orig, enc)");
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 5u);

    expect_ok(interp, "henc = huffman_encode_vec(orig)");
    expect_ok(interp, "hdec = huffman_decode_vec(orig, henc)");
    EXPECT_EQ(interp.state().matrices.at("hdec").rows(), 5u);
}

TEST(IntegrationFem,  FemCfdQuantum3d) {
    Interpreter interp;

    expect_ok(interp, "m3 = fem_mesh3d_box(0, 0, 0, 1, 1, 1, 2, 2, 2)");
    expect_ok(interp, "K3 = fem_stiffness_3d(m3)");
    expect_ok(interp, "f3 = fem_load_3d(m3, 1)");
    expect_ok(interp, "sys3 = fem_apply_dirichlet(K3, f3, [0], [0])");
    EXPECT_GT(interp.state().matrices.at("sys3").rows(), 0u);

    expect_ok(interp, "g3 = cfd_grid3d(0, 1, 0, 1, 0, 1, 4, 4, 4)");
    expect_ok(interp, "u0 = cfd_square_pulse_3d(g3, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2)");
    expect_ok(interp, "mass3 = cfd_integrated_mass_3d(g3, u0)");
    ASSERT_GT(interp.state().scalars.count("mass3"), 0u);

    expect_ok(interp, "H = [0, 1; 1, 0]");
    expect_ok(interp, "psi0 = [1; 0]");
    expect_ok(interp, "psif = quantum_schrodinger_final(H, psi0, 0, 0.1, 10)");
    EXPECT_EQ(interp.state().matrices.at("psif").rows(), 2u);
}
