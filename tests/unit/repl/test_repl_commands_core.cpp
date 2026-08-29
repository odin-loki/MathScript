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

TEST(ReplCommandsTest, sparse_spmv) {
    Interpreter interp;
    expect_contains(interp, "help", "sparse_from_coo(rows,cols,row_idx,col_idx,values)");
    expect_contains(interp, "help", "sparse_spmv(A_packed,x)");
    expect_contains(interp, "help", "sparse_to_dense(A_packed)");
    expect_contains(interp, "help", "sparse_add(A_packed,B_packed)");
    expect_contains(interp, "help", "Packed layout: (nnz+1)x3");

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    ASSERT_GT(interp.state().matrices.count("A"), 0u);
    const auto& packed = interp.state().matrices.at("A");
    EXPECT_EQ(packed.rows(), 4u);
    EXPECT_EQ(packed.cols(), 3u);
    EXPECT_NEAR(packed(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(packed(0, 1), 3.0, 1e-9);
    EXPECT_NEAR(packed(0, 2), 3.0, 1e-9);

    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("y")(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("y")(2, 0), 4.0, 1e-9);

    expect_ok(interp, "D = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 0), 2.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(2, 2), 4.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("D")(0, 1), 0.0, 1e-9);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "C = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    expect_ok(interp, "Dc = sparse_to_dense(C)");
    EXPECT_NEAR(interp.state().matrices.at("Dc")(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Dc")(1, 2), 1.0, 1e-9);
    EXPECT_NEAR(interp.state().matrices.at("Dc")(0, 0), 2.0, 1e-9);
}

TEST(ReplCommandsTest, sparse_spmv_dense_add) {
    Interpreter interp;

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-9);

    expect_ok(interp, "D = sparse_to_dense(A)");
    EXPECT_NEAR(interp.state().matrices.at("D")(1, 1), 3.0, 1e-9);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_2) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_3) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_4) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_5) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_6) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_7) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_8) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_9) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_10) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_11) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_12) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_13) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_14) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_15) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_16) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_17) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_18) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_19) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_20) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_21) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_22) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_23) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_24) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_25) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_26) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_27) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_28) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_29) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_30) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_31) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}

TEST(ReplCommandsTest, decompress_sparse_32) {
    Interpreter interp;

    expect_ok(interp, "D = [10; 250; 128; 64; 32; 16; 8; 4]");
    expect_ok(interp, "WC = wavelet_compress_vec(D, 0)");
    expect_ok(interp, "WR = wavelet_decompress_vec(WC)");
    ASSERT_GT(interp.state().matrices.count("WR"), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    expect_ok(interp, "x = [1; 1; 1]");
    expect_ok(interp, "y = sparse_spmv(A, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("y")(0, 0), 2.0, 1e-8);

    expect_ok(interp, "Dens = sparse_to_dense(A)");
    ASSERT_GT(interp.state().matrices.count("Dens"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Dens")(1, 1), 3.0, 1e-8);

    expect_ok(interp, "ri2 = [0; 1]");
    expect_ok(interp, "ci2 = [1; 2]");
    expect_ok(interp, "vv2 = [1; 1]");
    expect_ok(interp, "B = sparse_from_coo(3, 3, ri2, ci2, vv2)");
    expect_ok(interp, "S = sparse_add(A, B)");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
}
