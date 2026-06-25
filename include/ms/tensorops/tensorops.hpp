#pragma once
#include <functional>
#include <string>
#include <vector>

namespace ms {
namespace tensorops {

// A dense tensor stored in row-major order (first index fastest = C order)
class Tensor {
public:
    std::vector<int>    shape;
    std::vector<double> data;

    Tensor() = default;
    Tensor(std::vector<int> shape, double fill = 0.0);
    Tensor(std::vector<int> shape, std::vector<double> data);

    int  ndim()    const { return static_cast<int>(shape.size()); }
    long numel()   const;
    double& at(const std::vector<int>& idx);
    double  at(const std::vector<int>& idx) const;
    // Flat index from multi-index
    long flat(const std::vector<int>& idx) const;

    Tensor reshape(std::vector<int> new_shape) const;
    Tensor transpose(std::vector<int> perm) const;
    // Mode-n unfolding (matricisation): returns 2D matrix
    std::vector<std::vector<double>> unfold(int mode) const;
    // Fold a matrix back into a tensor along a given mode
    static Tensor fold(const std::vector<std::vector<double>>& mat,
                       int mode, const std::vector<int>& shape);

    // Elementwise ops
    Tensor operator+(const Tensor& o) const;
    Tensor operator-(const Tensor& o) const;
    Tensor operator*(double s) const;
};

// ========================== Contractions ==========================

// Einstein summation: contract(A, B, {pairs of (dim_A, dim_B)})
Tensor contract(const Tensor& A, const Tensor& B,
                const std::vector<std::pair<int,int>>& contractions);

// Outer product
Tensor outer(const Tensor& A, const Tensor& B);

// Mode-n product T ×_n M where M is a matrix (rows=new_dim, cols=mode_dim)
Tensor mode_product(const Tensor& T, const std::vector<std::vector<double>>& M, int mode);

// ========================== Einsum ==========================
// Simplified numpy-style einsum for 2 tensors only
// e.g. "ij,jk->ik"  (matrix multiply)
//      "ij,ij->"    (Frobenius inner product)
Tensor einsum(const std::string& subscripts, const Tensor& A, const Tensor& B);

// ========================== Khatri-Rao product ==========================
// Column-wise Kronecker: A ⊙ B, A and B must have same number of columns
std::vector<std::vector<double>>
khatri_rao(const std::vector<std::vector<double>>& A,
           const std::vector<std::vector<double>>& B);

// Kronecker product of two matrices
std::vector<std::vector<double>>
kronecker(const std::vector<std::vector<double>>& A,
          const std::vector<std::vector<double>>& B);

// ========================== Symmetrisation ==========================
// Symmetrize tensor over all index permutations
Tensor symmetrize(const Tensor& T);
// Anti-symmetrize tensor
Tensor antisymmetrize(const Tensor& T);

// ========================== Decompositions ==========================

// CP/CANDECOMP-PARAFAC decomposition via ALS
// Returns factor matrices F[mode] of shape (shape[mode], rank)
struct CPDecomposition {
    int rank;
    std::vector<std::vector<std::vector<double>>> factors;  // [mode][dim][component]
    std::vector<double> weights;  // lambda
    double residual;
};
CPDecomposition decompose_cp(const Tensor& T, int rank,
                              int max_iter = 200, double tol = 1e-6);

// Tucker decomposition via HOSVD
struct TuckerDecomposition {
    Tensor core;
    std::vector<std::vector<std::vector<double>>> factors;  // [mode][dim][component]
};
TuckerDecomposition decompose_hosvd(const Tensor& T, const std::vector<int>& ranks);
TuckerDecomposition decompose_tucker(const Tensor& T, const std::vector<int>& ranks,
                                      int max_iter = 50, double tol = 1e-6);

// ========================== Norms ==========================
double frobenius_norm(const Tensor& T);
double tensor_inner(const Tensor& A, const Tensor& B);

} // namespace tensorops
} // namespace ms
