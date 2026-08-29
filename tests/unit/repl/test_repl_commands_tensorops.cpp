#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, tensorops_and_finance_bs) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_norm(T)");

    expect_ok(interp, "n = tensorops_norm([3; 4])");
    EXPECT_NEAR(interp.state().scalars.at("n"), 5.0, 1e-9);
    expect_contains(interp, "tensorops_norm([3, 4])", "5");

    expect_ok(interp, "c = finance_bs_call(100, 100, 1, 0.05, 0.2)");
    EXPECT_GT(interp.state().scalars.at("c"), 0.0);
    expect_contains(interp, "finance_bs_call(100, 100, 1, 0.05, 0.2)", "\n");
}

TEST(ReplCommandsTest, tensorops_matmul) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_matmul(A,B)");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = [5, 6; 7, 8]");
    expect_ok(interp, "C = tensorops_matmul(A, B)");
    const auto& C = interp.state().matrices.at("C");
    EXPECT_NEAR(C(0, 0), 19.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 22.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 43.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 50.0, 1e-9);
    expect_contains(interp, "tensorops_matmul([1, 2; 3, 4], [5, 6; 7, 8])", "19");
}

TEST(ReplCommandsTest, tensorops_einsum) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_einsum(A,B)");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "B = [5, 6; 7, 8]");
    expect_ok(interp, "C = tensorops_einsum(A, B)");
    const auto& C = interp.state().matrices.at("C");
    EXPECT_NEAR(C(0, 0), 19.0, 1e-9);
    EXPECT_NEAR(C(0, 1), 22.0, 1e-9);
    EXPECT_NEAR(C(1, 0), 43.0, 1e-9);
    EXPECT_NEAR(C(1, 1), 50.0, 1e-9);
    expect_contains(interp, "tensorops_einsum([1, 2; 3, 4], [5, 6; 7, 8])", "19");
}

TEST(ReplCommandsTest, tensorops_inner) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_inner(A,B)");

    expect_ok(interp, "V1 = [1; 2; 3]");
    expect_ok(interp, "V2 = [4; 5; 6]");
    expect_ok(interp, "ti = tensorops_inner(V1, V2)");
    EXPECT_NEAR(interp.state().scalars.at("ti"), 32.0, 1e-9);

    expect_contains(interp, "tensorops_inner([1; 2; 3], [4; 5; 6])", "32");
}

TEST(ReplCommandsTest, tensorops_nmf_tt) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_decompose_nmf(nmf, V, rank)");
    expect_contains(interp, "help", "tensorops_decompose_tt(tt, T, [n1,n2,n3], eps)");
    expect_contains(interp, "help", "tensorops_reconstruct_nmf(nmf)");
    expect_contains(interp, "help", "tensorops_reconstruct_tt(tt)");

    const auto nmf_created = interp.execute(
        "tensorops_decompose_nmf(nmf1, [4, 7, 7; 7, 6, 11; 7, 11, 12], 2)");
    ASSERT_TRUE(nmf_created.has_value());
    EXPECT_NE(nmf_created->find("NMFDecomposition"), std::string::npos);
    EXPECT_NE(nmf_created->find("final_error="), std::string::npos);

    const auto nmf_reconstructed = interp.execute("tensorops_reconstruct_nmf(nmf1)");
    ASSERT_TRUE(nmf_reconstructed.has_value());
    EXPECT_NE(nmf_reconstructed->find("["), std::string::npos);
    EXPECT_FALSE(nmf_reconstructed->empty());

    const auto tt_created = interp.execute(
        "tensorops_decompose_tt(tt1, [15, 18, 20, 24; 30, 36, 40, 48], [2, 2, 2], 1e-8)");
    ASSERT_TRUE(tt_created.has_value());
    EXPECT_NE(tt_created->find("TTDecomposition"), std::string::npos);

    const auto tt_reconstructed = interp.execute("tensorops_reconstruct_tt(tt1)");
    ASSERT_TRUE(tt_reconstructed.has_value());
    EXPECT_NE(tt_reconstructed->find("15"), std::string::npos);
    EXPECT_NE(tt_reconstructed->find("48"), std::string::npos);

    const auto listed = interp.execute("session_objects()");
    ASSERT_TRUE(listed.has_value());
    EXPECT_NE(listed->find("nmf1 nmf"), std::string::npos);
    EXPECT_NE(listed->find("tt1 tt"), std::string::npos);

    expect_error_contains(
        interp, "tensorops_decompose_tt(tt_bad, [1, 2; 3, 4], [2, 2], 1e-6)", "tensorops_decompose_tt");
    expect_error_contains(
        interp, "tensorops_reconstruct_nmf(cp1)", "tensorops_reconstruct_nmf");
}

TEST(ReplCommandsTest, tensorops_decompose_cp) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_decompose_cp");

    expect_contains(interp, "tensorops_decompose_cp(cp1, [1, 2; 3, 4], 1)", "CPDecomposition");
    expect_contains(interp, "tensorops_reconstruct_cp(cp1)", "[");

    expect_error_contains(interp, "tensorops_decompose_cp(cp1, [1, 2; 3, 4], 1)",
                         "already exists");
    expect_error_contains(interp, "tensorops_decompose_cp(cp_bad, [1, 2; 3, 4])",
                         "tensorops_decompose_cp");
    expect_error_contains(interp, "tensorops_reconstruct_cp(missing)",
                         "tensorops_reconstruct_cp");
}

TEST(ReplCommandsTest, tensorops_decompose_tucker) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_decompose_tucker");

    expect_contains(interp, "tensorops_decompose_tucker(tk1, [1, 2, 3; 4, 5, 6], [1, 2])",
                    "TuckerDecomposition");
    expect_contains(interp, "tensorops_reconstruct_tucker(tk1)", "[");

    expect_error_contains(interp, "tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [1, 1])",
                         "already exists");
    expect_error_contains(interp, "tensorops_decompose_tucker(tk_bad, [1, 2; 3, 4], [1])",
                         "dimensionality");
    expect_error_contains(interp, "tensorops_reconstruct_tucker(missing)",
                         "tensorops_reconstruct_tucker");
}

TEST(ReplCommandsTest, tensorops_decompose_hosvd) {
    Interpreter interp;
    expect_contains(interp, "help", "tensorops_decompose_hosvd");

    expect_contains(interp, "tensorops_decompose_hosvd(hs1, [1, 2, 3; 4, 5, 6], [1, 2])",
                    "TuckerDecomposition");
    expect_contains(interp, "tensorops_reconstruct_tucker(hs1)", "[");

    expect_error_contains(interp, "tensorops_decompose_hosvd(hs1, [1, 2; 3, 4], [1, 1])",
                         "already exists");
    expect_error_contains(interp, "tensorops_decompose_hosvd(hs_bad, [1, 2; 3, 4])",
                         "tensorops_decompose_hosvd");
    expect_error_contains(interp, "tensorops_decompose_hosvd(hs_rank, [1, 2; 3, 4], [3, 1])",
                         "rank exceeds");
}

TEST(ReplCommandsTest, matmul_tensorops) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_2) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_3) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_4) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_5) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_6) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_7) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_8) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_9) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_10) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_11) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_12) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_13) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_14) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_15) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_16) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_17) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_18) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_19) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_20) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_21) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_22) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_23) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_24) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_25) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_26) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_27) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_28) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_29) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_30) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_31) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, matmul_tensorops_32) {
    Interpreter interp;

    expect_ok(interp, "M1 = [1, 2; 3, 4]");
    expect_ok(interp, "M2 = [5, 6; 7, 8]");
    expect_ok(interp, "C = matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("C")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "Tm = tensorops_matmul(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("Tm"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 0), 19.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(0, 1), 22.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 0), 43.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("Tm")(1, 1), 50.0, 1e-8);

    expect_ok(interp, "E = tensorops_einsum(M1, M2)");
    ASSERT_GT(interp.state().matrices.count("E"), 0u);
    ASSERT_GT(interp.state().matrices.at("E").rows(), 0u);
}

TEST(ReplCommandsTest, tensorops_norm_noassign) {
    Interpreter interp;
    expect_contains(interp, "tensorops_norm([3; 4])", "5");
    expect_error_contains(interp, "tensorops_norm(no_such_matrix)", "unknown matrix");
}
