#include "ms/tensorops/tensorops.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <sstream>

namespace ms {
namespace tensorops {

// ========================== Tensor basics ==========================

static long numel_shape(const std::vector<int>& shape) {
    long n = 1;
    for (int s : shape) n *= s;
    return n;
}

Tensor::Tensor(std::vector<int> shape, double fill)
    : shape(std::move(shape)), data(numel_shape(this->shape), fill) {}

Tensor::Tensor(std::vector<int> s, std::vector<double> d)
    : shape(std::move(s)), data(std::move(d)) {}

long Tensor::numel() const { return numel_shape(shape); }

long Tensor::flat(const std::vector<int>& idx) const {
    long f = 0, stride = 1;
    for (int i = ndim()-1; i >= 0; --i) { f += idx[i]*stride; stride *= shape[i]; }
    return f;
}

double& Tensor::at(const std::vector<int>& idx) { return data[flat(idx)]; }
double  Tensor::at(const std::vector<int>& idx) const { return data[flat(idx)]; }

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    return Tensor(std::move(new_shape), data);
}

Tensor Tensor::transpose(std::vector<int> perm) const {
    int N = ndim();
    std::vector<int> new_shape(N);
    for (int i=0;i<N;++i) new_shape[i] = shape[perm[i]];
    Tensor out(new_shape);
    // Iterate over all multi-indices
    std::vector<int> idx(N, 0);
    long total = numel();
    for (long f=0; f<total; ++f) {
        // Compute multi-index from flat f
        std::vector<int> midx(N);
        long tmp = f;
        for (int i=N-1; i>=0; --i) { midx[i] = tmp%shape[i]; tmp/=shape[i]; }
        // Permuted index
        std::vector<int> pidx(N);
        for (int i=0;i<N;++i) pidx[i] = midx[perm[i]];
        out.at(pidx) = data[f];
        (void)idx;
    }
    return out;
}

// Mode-n unfolding: rows = mode dimension, cols = product of others
std::vector<std::vector<double>> Tensor::unfold(int mode) const {
    int N = ndim();
    int rows = shape[mode];
    long cols = numel() / rows;
    std::vector<std::vector<double>> mat(rows, std::vector<double>(cols, 0.0));
    long total = numel();
    for (long f=0; f<total; ++f) {
        std::vector<int> midx(N);
        long tmp = f;
        for (int i=N-1; i>=0; --i) { midx[i] = tmp%shape[i]; tmp/=shape[i]; }
        int row = midx[mode];
        // Col index: all other indices in fixed order
        long col = 0, stride = 1;
        for (int i=N-1; i>=0; --i) {
            if (i == mode) continue;
            col += midx[i] * stride;
            stride *= shape[i];
        }
        mat[row][col] = data[f];
    }
    return mat;
}

Tensor Tensor::fold(const std::vector<std::vector<double>>& mat,
                    int mode, const std::vector<int>& shape) {
    Tensor T(shape, 0.0);
    int N = static_cast<int>(shape.size());
    long total = T.numel();
    for (long f=0; f<total; ++f) {
        std::vector<int> midx(N);
        long tmp = f;
        for (int i=N-1; i>=0; --i) { midx[i] = tmp%shape[i]; tmp/=shape[i]; }
        int row = midx[mode];
        long col = 0, stride = 1;
        for (int i=N-1; i>=0; --i) {
            if (i == mode) continue;
            col += midx[i] * stride;
            stride *= shape[i];
        }
        T.data[f] = mat[row][col];
    }
    return T;
}

Tensor Tensor::operator+(const Tensor& o) const {
    Tensor res(shape); for (size_t i=0;i<data.size();++i) res.data[i]=data[i]+o.data[i];
    return res;
}
Tensor Tensor::operator-(const Tensor& o) const {
    Tensor res(shape); for (size_t i=0;i<data.size();++i) res.data[i]=data[i]-o.data[i];
    return res;
}
Tensor Tensor::operator*(double s) const {
    Tensor res(shape); for (size_t i=0;i<data.size();++i) res.data[i]=data[i]*s;
    return res;
}

// ========================== Contractions ==========================

Tensor outer(const Tensor& A, const Tensor& B) {
    std::vector<int> shape;
    for (int s : A.shape) shape.push_back(s);
    for (int s : B.shape) shape.push_back(s);
    Tensor C(shape);
    long nA = A.numel(), nB = B.numel();
    for (long i=0; i<nA; ++i)
        for (long j=0; j<nB; ++j)
            C.data[i*nB+j] = A.data[i] * B.data[j];
    return C;
}

Tensor contract(const Tensor& A, const Tensor& B,
                const std::vector<std::pair<int,int>>& contractions) {
    // Build result shape
    int NA = A.ndim(), NB = B.ndim();
    std::vector<bool> skipA(NA, false), skipB(NB, false);
    for (auto& p : contractions) { skipA[p.first]=true; skipB[p.second]=true; }
    std::vector<int> rshape;
    for (int i=0;i<NA;++i) if (!skipA[i]) rshape.push_back(A.shape[i]);
    for (int i=0;i<NB;++i) if (!skipB[i]) rshape.push_back(B.shape[i]);

    Tensor C(rshape, 0.0);
    // Iterate over free and contracted indices
    // This is a general but slow implementation
    long totalA = A.numel();
    for (long fa=0; fa<totalA; ++fa) {
        std::vector<int> ia(NA);
        long tmp = fa;
        for (int k=NA-1; k>=0; --k) { ia[k]=tmp%A.shape[k]; tmp/=A.shape[k]; }
        // Iterate over free B indices and contracted indices
        long totalB = B.numel();
        for (long fb=0; fb<totalB; ++fb) {
            std::vector<int> ib(NB);
            tmp = fb;
            for (int k=NB-1; k>=0; --k) { ib[k]=tmp%B.shape[k]; tmp/=B.shape[k]; }
            // Check contracted indices match
            bool ok = true;
            for (auto& p : contractions)
                if (ia[p.first] != ib[p.second]) { ok=false; break; }
            if (!ok) continue;
            // Build result index
            std::vector<int> ic;
            for (int k=0;k<NA;++k) if (!skipA[k]) ic.push_back(ia[k]);
            for (int k=0;k<NB;++k) if (!skipB[k]) ic.push_back(ib[k]);
            C.at(ic) += A.data[fa] * B.data[fb];
        }
    }
    return C;
}

Tensor mode_product(const Tensor& T, const std::vector<std::vector<double>>& M, int mode) {
    int new_dim = static_cast<int>(M.size());
    int old_dim = T.shape[mode];
    std::vector<int> new_shape = T.shape;
    new_shape[mode] = new_dim;
    Tensor out(new_shape, 0.0);
    long total = out.numel();
    int N = T.ndim();
    for (long f=0; f<total; ++f) {
        std::vector<int> midx(N);
        long tmp = f;
        for (int k=N-1; k>=0; --k) { midx[k]=tmp%new_shape[k]; tmp/=new_shape[k]; }
        int j = midx[mode];
        for (int i=0; i<old_dim; ++i) {
            std::vector<int> tidx = midx;
            tidx[mode] = i;
            out.data[f] += M[j][i] * T.at(tidx);
        }
    }
    return out;
}

// ========================== Einsum ==========================

Tensor einsum(const std::string& subscripts, const Tensor& A, const Tensor& B) {
    // Parse "ij,jk->ik" style
    auto arrow = subscripts.find("->");
    auto comma = subscripts.find(',');
    if (arrow == std::string::npos || comma == std::string::npos)
        return Tensor({1}, 0.0);  // invalid subscripts
    std::string sA = subscripts.substr(0, comma);
    std::string sB = subscripts.substr(comma+1, arrow-comma-1);
    std::string sC = subscripts.substr(arrow+2);

    // Build index universe
    std::string all_idx;
    for (char c : sA) if (all_idx.find(c)==std::string::npos) all_idx+=c;
    for (char c : sB) if (all_idx.find(c)==std::string::npos) all_idx+=c;

    // Map each char to a size
    std::map<char, int> idx_size;
    for (int i=0;i<(int)sA.size();++i) idx_size[sA[i]] = A.shape[i];
    for (int i=0;i<(int)sB.size();++i) idx_size[sB[i]] = B.shape[i];

    // Output shape
    std::vector<int> oshape;
    if (sC.empty()) { oshape = {1}; }  // scalar
    else for (char c : sC) oshape.push_back(idx_size[c]);

    Tensor C(oshape, 0.0);

    // Iterate over all combinations of all_idx
    int M = static_cast<int>(all_idx.size());
    std::vector<int> sizes(M);
    for (int i=0;i<M;++i) sizes[i] = idx_size[all_idx[i]];
    long total = 1;
    for (int s : sizes) total *= s;

    for (long f=0; f<total; ++f) {
        std::map<char, int> val;
        long tmp = f;
        for (int k=M-1; k>=0; --k) { val[all_idx[k]] = tmp%sizes[k]; tmp/=sizes[k]; }
        // Get A index
        std::vector<int> ia, ib;
        for (char c : sA) ia.push_back(val[c]);
        for (char c : sB) ib.push_back(val[c]);
        // Get C index
        if (sC.empty()) {
            C.data[0] += A.at(ia) * B.at(ib);
        } else {
            std::vector<int> ic;
            for (char c : sC) ic.push_back(val[c]);
            C.at(ic) += A.at(ia) * B.at(ib);
        }
    }
    return C;
}

// ========================== Khatri-Rao / Kronecker ==========================

std::vector<std::vector<double>>
khatri_rao(const std::vector<std::vector<double>>& A,
           const std::vector<std::vector<double>>& B) {
    int rA = static_cast<int>(A.size()), rB = static_cast<int>(B.size());
    int cols = static_cast<int>(A[0].size());
    if ((int)B[0].size() != cols) return {};  // column mismatch
    int rows = rA * rB;
    std::vector<std::vector<double>> res(rows, std::vector<double>(cols));
    for (int c=0; c<cols; ++c)
        for (int i=0; i<rA; ++i)
            for (int j=0; j<rB; ++j)
                res[i*rB+j][c] = A[i][c] * B[j][c];
    return res;
}

std::vector<std::vector<double>>
kronecker(const std::vector<std::vector<double>>& A,
          const std::vector<std::vector<double>>& B) {
    int rA=A.size(), cA=A[0].size(), rB=B.size(), cB=B[0].size();
    std::vector<std::vector<double>> res(rA*rB, std::vector<double>(cA*cB));
    for (int ia=0;ia<rA;++ia)
        for (int ja=0;ja<cA;++ja)
            for (int ib=0;ib<rB;++ib)
                for (int jb=0;jb<cB;++jb)
                    res[ia*rB+ib][ja*cB+jb] = A[ia][ja]*B[ib][jb];
    return res;
}

// ========================== Symmetrisation ==========================

Tensor symmetrize(const Tensor& T) {
    int N = T.ndim();
    int n = T.shape[0];
    // Assume all dims equal
    Tensor out(T.shape, 0.0);
    // Generate all permutations
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    int count = 0;
    do {
        Tensor Tp = T.transpose(perm);
        for (size_t i=0;i<out.data.size();++i) out.data[i] += Tp.data[i];
        ++count;
    } while (std::next_permutation(perm.begin(), perm.end()));
    for (auto& v : out.data) v /= count;
    (void)n;
    return out;
}

Tensor antisymmetrize(const Tensor& T) {
    int N = T.ndim();
    Tensor out(T.shape, 0.0);
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);
    do {
        // Compute sign of permutation
        int sign = 1;
        std::vector<int> p2 = perm;
        for (int i=0; i<N; ++i)
            for (int j=i+1; j<N; ++j)
                if (p2[i] > p2[j]) sign = -sign;
        Tensor Tp = T.transpose(perm);
        for (size_t i=0;i<out.data.size();++i) out.data[i] += sign * Tp.data[i];
    } while (std::next_permutation(perm.begin(), perm.end()));
    long fac = 1; for (int i=1;i<=N;++i) fac*=i;
    for (auto& v : out.data) v /= fac;
    return out;
}

// ========================== Norms ==========================

double frobenius_norm(const Tensor& T) {
    double s=0; for (double v : T.data) s+=v*v; return std::sqrt(s);
}

double tensor_inner(const Tensor& A, const Tensor& B) {
    double s=0; for (size_t i=0;i<A.data.size();++i) s+=A.data[i]*B.data[i];
    return s;
}

// ========================== CP Decomposition (ALS) ==========================

// Matrix multiply: A(m×k) * B(k×n) → C(m×n)
static std::vector<std::vector<double>>
mat_mul(const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) {
    int m=A.size(), k=A[0].size(), n=B[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(n,0));
    for (int i=0;i<m;++i) for (int l=0;l<k;++l) for (int j=0;j<n;++j)
        C[i][j]+=A[i][l]*B[l][j];
    return C;
}

// Transpose matrix
static std::vector<std::vector<double>>
mat_T(const std::vector<std::vector<double>>& A) {
    int m=A.size(), n=A[0].size();
    std::vector<std::vector<double>> B(n, std::vector<double>(m));
    for (int i=0;i<m;++i) for (int j=0;j<n;++j) B[j][i]=A[i][j];
    return B;
}

// Solve m×m system via Gauss
static std::vector<std::vector<double>>
mat_solve_system(std::vector<std::vector<double>> A,
                 std::vector<std::vector<double>> B) {
    int n=A.size(), nb=B[0].size();
    for (int col=0;col<n;++col) {
        int pivot=col;
        for (int row=col+1;row<n;++row)
            if (std::abs(A[row][col])>std::abs(A[pivot][col])) pivot=row;
        std::swap(A[col],A[pivot]); std::swap(B[col],B[pivot]);
        double sc=A[col][col];
        if (std::abs(sc)<1e-14) continue;
        for (int j=col;j<n;++j) A[col][j]/=sc;
        for (int j=0;j<nb;++j) B[col][j]/=sc;
        for (int row=0;row<n;++row) if (row!=col) {
            double f=A[row][col];
            for (int j=col;j<n;++j) A[row][j]-=f*A[col][j];
            for (int j=0;j<nb;++j) B[row][j]-=f*B[col][j];
        }
    }
    return B;
}

// Element-wise product of two matrices (same size)
static std::vector<std::vector<double>>
mat_hadamard(const std::vector<std::vector<double>>& A,
             const std::vector<std::vector<double>>& B) {
    int m=A.size(), n=A[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(n));
    for (int i=0;i<m;++i) for (int j=0;j<n;++j) C[i][j]=A[i][j]*B[i][j];
    return C;
}

// Pseudo-inverse via normal equations
static std::vector<std::vector<double>>
mat_pinv(const std::vector<std::vector<double>>& A) {
    // (A^T A)^{-1} A^T
    auto At = mat_T(A);
    auto AtA = mat_mul(At, A);
    // Identity
    int n=AtA.size();
    std::vector<std::vector<double>> I(n, std::vector<double>(n,0));
    for (int i=0;i<n;++i) I[i][i]=1.0;
    auto inv = mat_solve_system(AtA, I);
    return mat_mul(inv, At);
}

CPDecomposition decompose_cp(const Tensor& T, int rank,
                              int max_iter, double tol) {
    int N = T.ndim();
    std::vector<int> shapes = T.shape;
    // Initialise factor matrices with random-ish values (deterministic: column-stacked identity)
    std::vector<std::vector<std::vector<double>>> factors(N);
    for (int mode=0; mode<N; ++mode) {
        int dim = shapes[mode];
        factors[mode].assign(dim, std::vector<double>(rank, 0.0));
        for (int i=0; i<dim; ++i) factors[mode][i][i % rank] = 1.0;
    }
    std::vector<double> weights(rank, 1.0);
    double prev_res = 1e18;
    for (int iter=0; iter<max_iter; ++iter) {
        for (int mode=0; mode<N; ++mode) {
            // Compute Khatri-Rao product of all factors except mode
            std::vector<std::vector<double>> KR;
            {
                // Start with factors[0] (skip mode) or factors[1] etc.
                int start = (mode == 0) ? 1 : 0;
                KR = factors[start];
                for (int m=start+1; m<N; ++m) {
                    if (m == mode) continue;
                    KR = khatri_rao(KR, factors[m]);
                }
            }
            // Unfolded T
            auto Tm = T.unfold(mode);
            // Update: factor[mode] = Tm * KR * (KR^T KR)^{-1}
            // KR^T KR (Hadamard product of Gram matrices)
            auto KRt = mat_T(KR);
            auto G = mat_mul(KRt, KR);
            // Solve factor[mode]^T from: factor[mode] * G = Tm * KR
            // => factor[mode] = (Tm * KR) * G^{-1}
            auto TmKR = mat_mul(Tm, KR);
            // G is rank×rank; solve G X = (TmKR)^T => X^T = factor[mode]
            auto G_inv_rhs = mat_solve_system(G, mat_T(TmKR));
            factors[mode] = mat_T(G_inv_rhs);
            // Normalise columns
            for (int r=0; r<rank; ++r) {
                double norm=0;
                for (int i=0;i<shapes[mode];++i) norm+=factors[mode][i][r]*factors[mode][i][r];
                norm=std::sqrt(norm);
                if (norm > 1e-12) {
                    weights[r] = norm;
                    for (int i=0;i<shapes[mode];++i) factors[mode][i][r]/=norm;
                }
            }
        }
        // Compute residual
        // Reconstruct tensor via rank-1 terms
        Tensor Trecon(T.shape, 0.0);
        for (int r=0; r<rank; ++r) {
            // Outer product of all factor columns
            Tensor term({shapes[0]});
            for (int i=0;i<shapes[0];++i) term.data[i] = factors[0][i][r];
            for (int mode=1; mode<N; ++mode) {
                Tensor fvec({shapes[mode]});
                for (int i=0;i<shapes[mode];++i) fvec.data[i] = factors[mode][i][r];
                term = outer(term, fvec);
            }
            for (size_t i=0;i<Trecon.data.size();++i) Trecon.data[i] += weights[r]*term.data[i];
        }
        double res=0;
        for (size_t i=0;i<T.data.size();++i) {
            double diff = T.data[i]-Trecon.data[i];
            res+=diff*diff;
        }
        res=std::sqrt(res);
        if (std::abs(prev_res - res) < tol) break;
        prev_res = res;
    }
    return {rank, factors, weights, prev_res};
}

// ========================== HOSVD ==========================

// Truncated SVD of a matrix → returns first 'rank' left singular vectors
static std::vector<std::vector<double>>
truncated_svd_left(const std::vector<std::vector<double>>& M, int rank) {
    // Power iteration for each singular vector
    int m=M.size(), n=M[0].size();
    auto Mt = mat_T(M);
    auto MtM = mat_mul(Mt, M);
    std::vector<std::vector<double>> U(m, std::vector<double>(rank, 0.0));
    // Compute leading 'rank' right singular vectors via power method on M^T M
    std::vector<std::vector<double>> V(n, std::vector<double>(rank, 0.0));
    std::vector<double> deflated_v(n, 0);
    for (int r=0; r<rank; ++r) {
        // Initial vector
        std::vector<double> v(n, 0);
        v[r % n] = 1.0;
        for (int it=0; it<100; ++it) {
            // v = MtM * v
            std::vector<double> nv(n, 0);
            for (int i=0;i<n;++i) for (int j=0;j<n;++j) nv[i]+=MtM[i][j]*v[j];
            // Deflate previous components
            for (int prev=0; prev<r; ++prev) {
                double dot=0; for (int i=0;i<n;++i) dot+=V[i][prev]*nv[i];
                for (int i=0;i<n;++i) nv[i]-=dot*V[i][prev];
            }
            double norm=0; for (double x:nv) norm+=x*x; norm=std::sqrt(norm);
            if (norm<1e-14) break;
            for (int i=0;i<n;++i) v[i]=nv[i]/norm;
        }
        for (int i=0;i<n;++i) V[i][r]=v[i];
        // u = M v / ||M v||
        std::vector<double> u(m, 0);
        for (int i=0;i<m;++i) for (int j=0;j<n;++j) u[i]+=M[i][j]*v[j];
        double norm=0; for (double x:u) norm+=x*x; norm=std::sqrt(norm);
        if (norm>1e-14) for (int i=0;i<m;++i) U[i][r]=u[i]/norm;
    }
    return U;
}

TuckerDecomposition decompose_hosvd(const Tensor& T, const std::vector<int>& ranks) {
    int N = T.ndim();
    std::vector<std::vector<std::vector<double>>> factors(N);
    Tensor core = T;
    for (int mode=0; mode<N; ++mode) {
        auto Tm = core.unfold(mode);
        int r = ranks[mode];
        factors[mode] = truncated_svd_left(Tm, r);
        // Update core: core = core ×_mode U^T
        auto Ut = mat_T(factors[mode]);
        core = mode_product(core, Ut, mode);
    }
    return {core, factors};
}

TuckerDecomposition decompose_tucker(const Tensor& T, const std::vector<int>& ranks,
                                      int max_iter, double tol) {
    // HOOI: Iterative Tucker
    auto decomp = decompose_hosvd(T, ranks);
    int N = T.ndim();
    double prev_res = 1e18;
    for (int iter=0; iter<max_iter; ++iter) {
        for (int mode=0; mode<N; ++mode) {
            // Compute Y_mode = T ×_{all n≠mode} U_n^T
            Tensor Y = T;
            for (int m=0; m<N; ++m) {
                if (m == mode) continue;
                auto Ut = mat_T(decomp.factors[m]);
                Y = mode_product(Y, Ut, m);
            }
            // Update U_mode = leading rank singular vectors of Y_(mode)
            auto Ym = Y.unfold(mode);
            decomp.factors[mode] = truncated_svd_left(Ym, ranks[mode]);
        }
        // Compute core
        Tensor core = T;
        for (int m=0; m<N; ++m) {
            auto Ut = mat_T(decomp.factors[m]);
            core = mode_product(core, Ut, m);
        }
        decomp.core = core;
        // Residual
        Tensor Trecon = core;
        for (int m=0; m<N; ++m) Trecon = mode_product(Trecon, decomp.factors[m], m);
        double res=0;
        for (size_t i=0;i<T.data.size();++i) {
            double diff = T.data[i]-Trecon.data[i]; res+=diff*diff;
        }
        res=std::sqrt(res);
        if (std::abs(prev_res - res) < tol) break;
        prev_res=res;
    }
    return decomp;
}

} // namespace tensorops
} // namespace ms
