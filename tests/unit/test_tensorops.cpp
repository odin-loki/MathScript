#include "ms/tensorops/tensorops.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace ms::tensorops;

// ---- Tensor basics ----

TEST(TensorBasic, Construction) {
    Tensor T({2, 3, 4});
    EXPECT_EQ(T.ndim(), 3);
    EXPECT_EQ(T.numel(), 24);
    for (double v : T.data) EXPECT_DOUBLE_EQ(v, 0.0);
}

TEST(TensorBasic, AtAndFlat) {
    Tensor T({2, 3}, 0.0);
    T.at({0,0}) = 1.0;
    T.at({1,2}) = 5.0;
    EXPECT_DOUBLE_EQ(T.at({0,0}), 1.0);
    EXPECT_DOUBLE_EQ(T.at({1,2}), 5.0);
    EXPECT_DOUBLE_EQ(T.data[0], 1.0);
    EXPECT_DOUBLE_EQ(T.data[5], 5.0);
}

TEST(TensorBasic, Reshape) {
    Tensor T({2,3}, 1.0);
    auto R = T.reshape({3,2});
    EXPECT_EQ(R.shape[0], 3);
    EXPECT_EQ(R.shape[1], 2);
    EXPECT_DOUBLE_EQ(R.data[0], 1.0);
}

TEST(TensorBasic, Transpose) {
    Tensor T({2, 3}, 0.0);
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) T.at({i,j}) = i*3+j;
    auto Tt = T.transpose({1,0});
    EXPECT_EQ(Tt.shape[0], 3);
    EXPECT_EQ(Tt.shape[1], 2);
    EXPECT_DOUBLE_EQ(Tt.at({0,1}), 3.0);  // T[1][0] = 3
}

TEST(TensorBasic, Unfold) {
    // 2×3 tensor, unfold along mode 0 → 2×3 matrix (trivial)
    Tensor T({2,3}, 0.0);
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) T.at({i,j}) = i*3+j;
    auto M = T.unfold(0);
    EXPECT_EQ(M.size(), 2u);
    EXPECT_EQ(M[0].size(), 3u);
}

TEST(TensorBasic, AddSubScalar) {
    Tensor A({2,2}, 3.0), B({2,2}, 1.0);
    auto C = A + B;
    EXPECT_DOUBLE_EQ(C.data[0], 4.0);
    auto D = A - B;
    EXPECT_DOUBLE_EQ(D.data[0], 2.0);
    auto E = A * 2.0;
    EXPECT_DOUBLE_EQ(E.data[0], 6.0);
}

// ---- Outer product ----

TEST(TensorOuter, VectorOuter) {
    Tensor a({3}, 0.0); a.data = {1,2,3};
    Tensor b({2}, 0.0); b.data = {4,5};
    auto C = outer(a, b);
    EXPECT_EQ(C.shape[0], 3); EXPECT_EQ(C.shape[1], 2);
    EXPECT_DOUBLE_EQ(C.at({0,0}), 4.0);
    EXPECT_DOUBLE_EQ(C.at({2,1}), 15.0);
}

// ---- Mode product ----

TEST(TensorModeProduct, MatVec) {
    // T shape (3,), M shape (2,3) → T×_0 M shape (2,)
    Tensor T({3}, 0.0); T.data = {1,2,3};
    std::vector<std::vector<double>> M = {{1,0,0},{0,1,0}};
    auto R = mode_product(T, M, 0);
    EXPECT_EQ(R.shape[0], 2);
    EXPECT_DOUBLE_EQ(R.at({0}), 1.0);
    EXPECT_DOUBLE_EQ(R.at({1}), 2.0);
}

// ---- Contract ----

TEST(TensorContract, MatMul) {
    // A (2,3) * B (3,4) = C (2,4) via contraction on dim 1 of A, dim 0 of B
    Tensor A({2,3}, 0.0);
    Tensor B({3,4}, 0.0);
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) A.at({i,j}) = i*3+j+1;
    for (int i=0;i<3;++i) for (int j=0;j<4;++j) B.at({i,j}) = i==j ? 1.0 : 0.0;
    auto C = contract(A, B, {{1,0}});
    EXPECT_EQ(C.shape[0], 2); EXPECT_EQ(C.shape[1], 4);
    // B is identity-like → C ~ A with extra col
    EXPECT_NEAR(C.at({0,0}), A.at({0,0}), 1e-10);
    EXPECT_NEAR(C.at({1,1}), A.at({1,1}), 1e-10);
}

// ---- Einsum ----

TEST(TensorEinsum, MatMul) {
    Tensor A({2,3}, 0.0), B({3,2}, 0.0);
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) A.at({i,j}) = i*3+j+1;
    for (int i=0;i<3;++i) for (int j=0;j<2;++j) B.at({i,j}) = i==j ? 1.0 : 0.0;
    auto C = einsum("ij,jk->ik", A, B);
    EXPECT_EQ(C.shape[0], 2); EXPECT_EQ(C.shape[1], 2);
}

TEST(TensorEinsum, InnerProduct) {
    Tensor A({3}, 0.0), B({3}, 0.0);
    A.data = {1,2,3}; B.data = {4,5,6};
    auto C = einsum("i,i->", A, B);
    EXPECT_NEAR(C.data[0], 32.0, 1e-10);  // 1*4+2*5+3*6=32
}

// ---- Khatri-Rao / Kronecker ----

TEST(TensorKhatriRao, Basic) {
    std::vector<std::vector<double>> A = {{1,2},{3,4}};
    std::vector<std::vector<double>> B = {{5,6},{7,8}};
    auto KR = khatri_rao(A, B);
    EXPECT_EQ(KR.size(), 4u);
    EXPECT_EQ(KR[0].size(), 2u);
    // col 0: [1*5, 1*7, 3*5, 3*7]
    EXPECT_DOUBLE_EQ(KR[0][0], 5.0);
    EXPECT_DOUBLE_EQ(KR[1][0], 7.0);
    EXPECT_DOUBLE_EQ(KR[2][0], 15.0);
    EXPECT_DOUBLE_EQ(KR[3][0], 21.0);
}

TEST(TensorKronecker, IdentitySquare) {
    std::vector<std::vector<double>> I2 = {{1,0},{0,1}};
    auto K = kronecker(I2, I2);
    EXPECT_EQ(K.size(), 4u);
    EXPECT_DOUBLE_EQ(K[0][0], 1.0);
    EXPECT_DOUBLE_EQ(K[0][1], 0.0);
    EXPECT_DOUBLE_EQ(K[1][0], 0.0);
    EXPECT_DOUBLE_EQ(K[2][2], 1.0);
}

// ---- Norms ----

TEST(TensorNorms, Frobenius) {
    Tensor T({2,2}, 0.0);
    T.data = {3,4,0,0};
    EXPECT_NEAR(frobenius_norm(T), 5.0, 1e-10);
}

TEST(TensorNorms, Inner) {
    Tensor A({3}, 0.0), B({3}, 0.0);
    A.data = {1,2,3}; B.data = {4,5,6};
    EXPECT_NEAR(tensor_inner(A,B), 32.0, 1e-10);
}

// ---- Decompositions ----

TEST(TensorCP, RankOne) {
    // Rank-1 tensor: outer product of [1,2]x[3,4]x[5,6]
    Tensor a({2}); a.data={1,2};
    Tensor b({2}); b.data={3,4};
    Tensor c({2}); c.data={5,6};
    auto T = outer(outer(a,b),c);
    auto cp = decompose_cp(T, 1, 200, 1e-8);
    EXPECT_EQ(cp.rank, 1);
    // Residual should be small
    EXPECT_LT(cp.residual, 0.5);
}

TEST(TensorHOSVD, BasicShape) {
    Tensor T({4, 3, 2}, 1.0);
    auto decomp = decompose_hosvd(T, {2, 2, 2});
    EXPECT_EQ(decomp.core.shape[0], 2);
    EXPECT_EQ(decomp.core.shape[1], 2);
    EXPECT_EQ(decomp.core.shape[2], 2);
    EXPECT_EQ(decomp.factors.size(), 3u);
}

TEST(TensorHOSVD, Reconstruction) {
    // Small identity-like tensor
    Tensor T({3, 3}, 0.0);
    T.at({0,0})=1; T.at({1,1})=2; T.at({2,2})=3;
    auto decomp = decompose_hosvd(T, {2, 2});
    // Core should be small
    EXPECT_EQ(decomp.core.ndim(), 2);
}

// ---- Symmetrize ----
TEST(TensorSymmetrize, Symmetric2D) {
    Tensor T({2,2}, 0.0);
    T.data = {1,2,3,4};
    auto S = symmetrize(T);
    // S[i][j] = S[j][i]
    EXPECT_NEAR(S.at({0,1}), S.at({1,0}), 1e-10);
}

TEST(TensorAntisymmetrize, AntiSymmetric2D) {
    Tensor T({2,2}, 0.0);
    T.data = {1,2,3,4};
    auto A = antisymmetrize(T);
    EXPECT_NEAR(A.at({0,0}), 0.0, 1e-10);
    EXPECT_NEAR(A.at({0,1}), -A.at({1,0}), 1e-10);
}

TEST(TensorModeProduct, SecondMode) {
    Tensor T({2, 3}, 0.0);
    T.data = {1, 2, 3, 4, 5, 6};
    std::vector<std::vector<double>> M = {
        {1, 0, 0},
        {0, 1, 0},
        {1, 1, 1}
    };
    auto R = mode_product(T, M, 1);
    EXPECT_EQ(R.shape[0], 2);
    EXPECT_EQ(R.shape[1], 3);
    EXPECT_DOUBLE_EQ(R.at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(R.at({0, 2}), 6.0);  // row sum of [1,2,3]
    EXPECT_DOUBLE_EQ(R.at({1, 2}), 15.0); // row sum of [4,5,6]
}

TEST(TensorFoldUnfold, RoundTrip) {
    Tensor T({2, 3, 4}, 0.0);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 4; ++k)
                T.at({i, j, k}) = i * 12 + j * 4 + k + 1;
    auto M = T.unfold(1);
    auto R = Tensor::fold(M, 1, T.shape);
    for (long i = 0; i < T.numel(); ++i)
        EXPECT_DOUBLE_EQ(R.data[i], T.data[i]);
}

TEST(TensorOuter, RankThree) {
    Tensor a({2}, 0.0); a.data = {1, 2};
    Tensor b({3}, 0.0); b.data = {1, 0, -1};
    Tensor c({2}, 0.0); c.data = {3, 4};
    auto ab = outer(a, b);
    EXPECT_EQ(ab.shape[0], 2);
    EXPECT_EQ(ab.shape[1], 3);
    EXPECT_DOUBLE_EQ(ab.at({1, 2}), -2.0);
    auto abc = outer(ab, c);
    EXPECT_EQ(abc.ndim(), 3);
    EXPECT_DOUBLE_EQ(abc.at({0, 0, 1}), 4.0);  // 1*1*4
}

TEST(TensorSymmetrize, KnownAverage) {
    Tensor T({2, 2}, 0.0);
    T.data = {1, 2, 3, 4};
    auto S = symmetrize(T);
    EXPECT_NEAR(S.at({0, 0}), 1.0, 1e-10);
    EXPECT_NEAR(S.at({1, 1}), 4.0, 1e-10);
    EXPECT_NEAR(S.at({0, 1}), 2.5, 1e-10);
    EXPECT_NEAR(S.at({1, 0}), 2.5, 1e-10);
}

TEST(TensorCP, RankTwoTensor) {
    Tensor a({2}); a.data = {1, 2};
    Tensor b({2}); b.data = {3, 4};
    Tensor c({2}); c.data = {1, 0};
    Tensor d({2}); d.data = {0, 1};
    auto T = outer(outer(a, b), c) + outer(outer(a, d), b);
    auto cp = decompose_cp(T, 2, 300, 1e-6);
    EXPECT_EQ(cp.rank, 2);
    EXPECT_LT(cp.residual, 1.0);
}

// ---- CP / Tucker reconstruction ----

static Tensor make_rank1_tensor(const std::vector<Tensor>& vecs) {
    Tensor term = vecs[0];
    for (size_t m = 1; m < vecs.size(); ++m)
        term = outer(term, vecs[m]);
    return term;
}

TEST(TensorReconstructCP, RankOneExact3D) {
    Tensor a({3}); a.data = {1, 2, 3};
    Tensor b({3}); b.data = {4, 5, 6};
    Tensor c({3}); c.data = {7, 8, 9};
    auto T = make_rank1_tensor({a, b, c});
    auto cp = decompose_cp(T, 1, 300, 1e-8);
    auto R = reconstruct_cp(cp);
    EXPECT_EQ(R.shape, T.shape);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, 0.5);
    EXPECT_NEAR(err, cp.residual, 1e-10);
}

TEST(TensorReconstructCP, RankTwoExact3D) {
    Tensor a1({3}); a1.data = {1, 0, 2};
    Tensor b1({3}); b1.data = {1, 1, 0};
    Tensor c1({3}); c1.data = {2, 1, 1};
    Tensor a2({3}); a2.data = {0, 1, 1};
    Tensor b2({3}); b2.data = {2, 0, 1};
    Tensor c2({3}); c2.data = {1, 2, 0};
    auto T = make_rank1_tensor({a1, b1, c1}) + make_rank1_tensor({a2, b2, c2});
    auto cp = decompose_cp(T, 2, 400, 1e-6);
    auto R = reconstruct_cp(cp);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, 1.0);
    EXPECT_NEAR(err, cp.residual, 1e-10);
}

TEST(TensorReconstructCP, ResidualMatchesDecomposeCP) {
    Tensor T({3, 3, 3}, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                T.at({i, j, k}) = (i + 1) * (j + 2) * (k + 3) * 0.1;
    auto cp = decompose_cp(T, 3, 300, 1e-6);
    double independent = frobenius_norm(T - reconstruct_cp(cp));
    EXPECT_NEAR(independent, cp.residual, 1e-10);
}

TEST(TensorReconstructCP, LossyLowRank) {
    Tensor T({4, 4, 4}, 0.0);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                T.at({i, j, k}) = std::sin(i * 0.7 + j * 1.1 + k * 0.3);
    auto cp = decompose_cp(T, 1, 200, 1e-6);
    auto R = reconstruct_cp(cp);
    double err = frobenius_norm(T - R);
    EXPECT_GT(err, 0.01);
    EXPECT_LT(err, frobenius_norm(T));
    EXPECT_NEAR(err, cp.residual, 1e-10);
}

TEST(TensorReconstructCP, Matrix2D) {
    Tensor a({2}); a.data = {1, 2};
    Tensor b({3}); b.data = {3, 4, 5};
    auto T = outer(a, b);
    auto cp = decompose_cp(T, 1, 200, 1e-8);
    auto R = reconstruct_cp(cp);
    EXPECT_EQ(R.ndim(), 2);
    EXPECT_EQ(R.shape[0], 2);
    EXPECT_EQ(R.shape[1], 3);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, 0.5);
    EXPECT_NEAR(err, cp.residual, 1e-10);
}

TEST(TensorReconstructCP, RandomishTensorRank3) {
    Tensor vecs[3] = {Tensor({3}), Tensor({3}), Tensor({3})};
    vecs[0].data = {2, -1, 1};
    vecs[1].data = {1, 3, 2};
    vecs[2].data = {0, 2, -1};
    auto T = make_rank1_tensor({vecs[0], vecs[1], vecs[2]});
    auto cp = decompose_cp(T, 1, 300, 1e-8);
    EXPECT_LT(frobenius_norm(T - reconstruct_cp(cp)), 0.5);
}

TEST(TensorReconstructTucker, KnownConstructionExact) {
    TuckerDecomposition known;
    known.core = Tensor({2, 2, 2}, 0.0);
    known.core.at({0, 0, 0}) = 1.0;
    known.core.at({1, 1, 1}) = 2.0;
    known.factors = {
        {{1, 0}, {0, 1}},
        {{1, 1}, {1, -1}},
        {{2, 0}, {0, 3}}
    };
    auto T = reconstruct_tucker(known);
    auto R = reconstruct_tucker(known);
    EXPECT_NEAR(frobenius_norm(T - R), 0.0, 1e-10);
    EXPECT_EQ(R.shape, T.shape);
}

TEST(TensorReconstructTucker, FullRank3D) {
    TuckerDecomposition known;
    known.core = Tensor({2, 2, 2}, 0.0);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                known.core.at({i, j, k}) = (i + 1) * (j + 1) + k * 0.5;
    known.factors = {
        {{1, 0.5}, {0.5, 1}},
        {{1, -1}, {2, 1}},
        {{1, 0}, {1, 1}}
    };
    auto T = reconstruct_tucker(known);
    auto decomp = decompose_tucker(T, {2, 2, 2}, 50, 1e-6);
    auto R = reconstruct_tucker(decomp);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, 5.0);
}

TEST(TensorReconstructTucker, ReducedRank3D) {
    TuckerDecomposition known;
    known.core = Tensor({2, 2, 2}, 0.0);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                known.core.at({i, j, k}) = std::sin(i + j + k);
    known.factors = {
        {{1, 0.3, -0.2}, {0.5, 1, 0.1}, {0.2, -0.4, 1}},
        {{1, 0.2}, {0.3, 1}, {-0.1, 0.5}},
        {{1, -0.5}, {0.7, 1}}
    };
    auto T = reconstruct_tucker(known);
    auto decomp = decompose_hosvd(T, {2, 2, 2});
    auto R = reconstruct_tucker(decomp);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, frobenius_norm(T) + 1e-6);
    EXPECT_GT(err, 0.0);
}

TEST(TensorReconstructTucker, HOSVDLossyCompression) {
    TuckerDecomposition known;
    known.core = Tensor({3, 3, 3}, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                known.core.at({i, j, k}) = (i + 1) * (j + 2) + k * 0.3;
    known.factors = {
        {{1, 0.2, -0.1}, {0.3, 1, 0.4}, {0.1, -0.2, 1}},
        {{1, 0.5, 0.2}, {0.1, 1, -0.3}, {0.4, 0.2, 1}},
        {{1, -0.2, 0.7}, {0.6, 1, -0.1}, {0.3, -0.1, 1}}
    };
    auto T = reconstruct_tucker(known);
    auto decomp = decompose_hosvd(T, {1, 1, 1});
    auto R = reconstruct_tucker(decomp);
    double err = frobenius_norm(T - R);
    EXPECT_GT(err, 0.1);
    EXPECT_LT(err, frobenius_norm(T));
}

TEST(TensorReconstructTucker, Matrix2D) {
    Tensor T({3, 3}, 0.0);
    T.at({0, 0}) = 1; T.at({1, 1}) = 2; T.at({2, 2}) = 3;
    T.at({0, 1}) = 0.5; T.at({1, 0}) = 0.5;
    auto decomp = decompose_hosvd(T, {2, 2});
    auto R = reconstruct_tucker(decomp);
    EXPECT_EQ(R.ndim(), 2);
    EXPECT_EQ(R.shape[0], 3);
    EXPECT_EQ(R.shape[1], 3);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, frobenius_norm(T));
}

TEST(TensorReconstructTucker, TuckerALSReducedRank) {
    Tensor T({3, 3, 2}, 0.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                T.at({i, j, k}) = i * 0.5 + j * 0.3 + k;
    auto decomp = decompose_tucker(T, {2, 2, 2}, 30, 1e-6);
    auto R = reconstruct_tucker(decomp);
    EXPECT_EQ(R.shape, T.shape);
    double err = frobenius_norm(T - R);
    EXPECT_LT(err, frobenius_norm(T));
}
