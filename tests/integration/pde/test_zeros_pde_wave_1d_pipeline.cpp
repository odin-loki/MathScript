
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

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

TEST(IntegrationPde,  PdeSparseControl) {
    Interpreter interp;

    expect_contains(interp, "help", "pde_wave_1d");
    expect_contains(interp, "help", "sparse_spmv");

    expect_ok(interp, "u0 = zeros(5, 1)");
    expect_ok(interp, "v0 = zeros(5, 1)");
    expect_ok(interp, "w1 = pde_wave_1d(u0, v0, 1, 0.1, 0.1, 10)");
    EXPECT_EQ(interp.state().matrices.at("w1").rows(), 5u);

    expect_ok(interp, "u2 = zeros(5, 5)");
    expect_ok(interp, "h2 = pde_heat_2d(u2, 0.1, 0.1, 0.1, 0.01, 8)");
    EXPECT_EQ(interp.state().matrices.at("h2").rows(), 5u);

    expect_ok(interp, "f = ones(5, 1)");
    expect_ok(interp, "p1 = pde_poisson_1d(f, 0.1, 0, 0)");
    EXPECT_EQ(interp.state().matrices.at("p1").rows(), 5u);

    expect_ok(interp, "adv = pde_advection_1d([0; 1; 0], 1, 0.1, 0.01, 10)");
    EXPECT_EQ(interp.state().matrices.at("adv").rows(), 3u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "y = sparse_spmv(A, [1; 2; 3])");
    expect_ok(interp, "D = sparse_to_dense(A)");
    EXPECT_GT(interp.state().matrices.at("y").rows(), 0u);
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 3u);

    expect_ok(interp, "V = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(V, 0)");
    expect_ok(interp, "RD = wavelet_decompress_vec(WC)");
    EXPECT_EQ(interp.state().matrices.at("RD").rows(), 8u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    expect_ok(interp, "Pp = control_kalman_predict_cov([0], [1], [1], [0.05])");
    EXPECT_GT(interp.state().matrices.at("Ob").rows(), 0u);
}

TEST(IntegrationPde,  OdeSignalSpecial) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 10)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "sr = control_step_response([1], [1, 1], 5, 50)");
    EXPECT_GT(interp.state().matrices.at("sr").rows(), 0u);

    expect_ok(interp, "A = [1; 2; 3]");
    expect_ok(interp, "K = [1; 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    EXPECT_GT(interp.state().matrices.at("C").rows(), 0u);

    expect_ok(interp, "sh = struve_h(1, 0.5)");
    expect_ok(interp, "kb = kelvin_ber(0, 0.5)");
    expect_ok(interp, "js = jacobi_sn(0.5, 0.5)");
    EXPECT_TRUE(interp.state().scalars.count("sh") > 0);
    EXPECT_TRUE(interp.state().scalars.count("kb") > 0);
    EXPECT_TRUE(interp.state().scalars.count("js") > 0);
}
