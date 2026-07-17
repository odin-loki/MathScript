#include "ms/tensorops/tensorops.hpp"
#include "ms/error/error_types.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

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

namespace {

std::vector<long> row_major_strides(const std::vector<int>& shape) {
    const int nd = static_cast<int>(shape.size());
    std::vector<long> strides(nd, 1);
    long stride = 1;
    for (int i = nd - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

long linear_index(const std::vector<int>& idx, const std::vector<long>& strides) {
    long flat = 0;
    for (size_t i = 0; i < idx.size(); ++i)
        flat += static_cast<long>(idx[i]) * strides[i];
    return flat;
}

bool odometer_advance(std::vector<int>& idx, const std::vector<int>& sizes) {
    for (int d = static_cast<int>(idx.size()) - 1; d >= 0; --d) {
        ++idx[d];
        if (idx[d] < sizes[d]) return true;
        idx[d] = 0;
    }
    return false;
}

Tensor contract_matmul_2d(const Tensor& A, const Tensor& B, int m, int k, int n) {
    Tensor C({m, n}, 0.0);
    const double* Ap = A.data.data();
    const double* Bp = B.data.data();
    double* Cp = C.data.data();
    for (int i = 0; i < m; ++i) {
        const double* Ai = Ap + static_cast<long>(i) * k;
        double* Ci = Cp + static_cast<long>(i) * n;
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l)
                sum += Ai[l] * Bp[static_cast<long>(l) * n + j];
            Ci[j] = sum;
        }
    }
    return C;
}

Tensor contract_vec_inner(const Tensor& A, const Tensor& B) {
    const int n = A.shape[0];
    const double* Ap = A.data.data();
    const double* Bp = B.data.data();
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += Ap[i] * Bp[i];
    return Tensor({}, sum);
}

// Iterate contracted indices in the inner loop; free indices in the outer loop.
Tensor contract_general(const Tensor& A, const Tensor& B,
                        const std::vector<std::pair<int, int>>& contractions,
                        const std::vector<bool>& skipA, const std::vector<bool>& skipB,
                        const std::vector<int>& rshape) {
    const int NA = A.ndim();
    const int NB = B.ndim();
    const auto stA = row_major_strides(A.shape);
    const auto stB = row_major_strides(B.shape);
    const auto stC = row_major_strides(rshape);

    std::vector<int> freeA_axes, freeB_axes, conA_axes, conB_axes;
    std::vector<int> freeA_sizes, freeB_sizes, con_sizes;
    freeA_axes.reserve(NA);
    freeB_axes.reserve(NB);
    conA_axes.reserve(contractions.size());
    conB_axes.reserve(contractions.size());
    con_sizes.reserve(contractions.size());

    for (int i = 0; i < NA; ++i) {
        if (skipA[i]) continue;
        freeA_axes.push_back(i);
        freeA_sizes.push_back(A.shape[i]);
    }
    for (int i = 0; i < NB; ++i) {
        if (skipB[i]) continue;
        freeB_axes.push_back(i);
        freeB_sizes.push_back(B.shape[i]);
    }
    for (const auto& [a_ax, b_ax] : contractions) {
        conA_axes.push_back(a_ax);
        conB_axes.push_back(b_ax);
        con_sizes.push_back(A.shape[a_ax]);
    }

    Tensor C(rshape, 0.0);
    std::vector<int> idxA(NA, 0), idxB(NB, 0);
    std::vector<int> free_all;
    free_all.reserve(freeA_sizes.size() + freeB_sizes.size());
    free_all.insert(free_all.end(), freeA_sizes.begin(), freeA_sizes.end());
    free_all.insert(free_all.end(), freeB_sizes.begin(), freeB_sizes.end());
    std::vector<int> free_idx(free_all.size(), 0);
    std::vector<int> con(con_sizes.size(), 0);

    do {
        for (size_t i = 0; i < freeA_axes.size(); ++i)
            idxA[freeA_axes[i]] = free_idx[i];
        for (size_t i = 0; i < freeB_axes.size(); ++i)
            idxB[freeB_axes[i]] = free_idx[freeA_axes.size() + i];

        double acc = 0.0;
        do {
            for (size_t i = 0; i < conA_axes.size(); ++i) {
                idxA[conA_axes[i]] = con[i];
                idxB[conB_axes[i]] = con[i];
            }
            acc += A.data[static_cast<size_t>(linear_index(idxA, stA))] *
                   B.data[static_cast<size_t>(linear_index(idxB, stB))];
        } while (odometer_advance(con, con_sizes));

        long c_flat = 0;
        for (size_t i = 0; i < free_idx.size(); ++i)
            c_flat += static_cast<long>(free_idx[i]) * stC[static_cast<int>(i)];
        C.data[static_cast<size_t>(c_flat)] = acc;
    } while (odometer_advance(free_idx, free_all));

    return C;
}

} // namespace

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
    if (contractions.empty())
        return outer(A, B);

    const int NA = A.ndim(), NB = B.ndim();
    std::vector<bool> skipA(NA, false), skipB(NB, false);
    for (const auto& p : contractions) {
        skipA[p.first] = true;
        skipB[p.second] = true;
    }
    std::vector<int> rshape;
    rshape.reserve(NA + NB);
    for (int i = 0; i < NA; ++i) if (!skipA[i]) rshape.push_back(A.shape[i]);
    for (int i = 0; i < NB; ++i) if (!skipB[i]) rshape.push_back(B.shape[i]);

    // 2D matrix multiply: contract A's last axis with B's first axis.
    if (NA == 2 && NB == 2 && contractions.size() == 1) {
        const auto [a_ax, b_ax] = contractions[0];
        if (a_ax == 1 && b_ax == 0 && A.shape[1] == B.shape[0])
            return contract_matmul_2d(A, B, A.shape[0], A.shape[1], B.shape[1]);
    }

    // Vector inner product.
    if (NA == 1 && NB == 1 && contractions.size() == 1 &&
        contractions[0].first == 0 && contractions[0].second == 0 &&
        A.shape[0] == B.shape[0]) {
        return contract_vec_inner(A, B);
    }

    return contract_general(A, B, contractions, skipA, skipB, rshape);
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

    std::vector<std::pair<int,int>> contractions;
    for (int i = 0; i < static_cast<int>(sA.size()); ++i) {
        const char c = sA[static_cast<size_t>(i)];
        if (sC.find(c) != std::string::npos) continue;
        const auto posB = sB.find(c);
        if (posB == std::string::npos) continue;
        contractions.emplace_back(i, static_cast<int>(posB));
    }

    return contract(A, B, contractions);
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

static Tensor reconstruct_cp_from_factors(
    const std::vector<std::vector<std::vector<double>>>& factors,
    const std::vector<double>& weights,
    int rank) {
    int N = static_cast<int>(factors.size());
    std::vector<int> shapes(N);
    for (int mode = 0; mode < N; ++mode)
        shapes[mode] = static_cast<int>(factors[mode].size());
    Tensor Trecon(shapes, 0.0);
    for (int r = 0; r < rank; ++r) {
        Tensor term({shapes[0]});
        for (int i = 0; i < shapes[0]; ++i) term.data[i] = factors[0][i][r];
        for (int mode = 1; mode < N; ++mode) {
            Tensor fvec({shapes[mode]});
            for (int i = 0; i < shapes[mode]; ++i) fvec.data[i] = factors[mode][i][r];
            term = outer(term, fvec);
        }
        for (size_t i = 0; i < Trecon.data.size(); ++i)
            Trecon.data[i] += weights[r] * term.data[i];
    }
    return Trecon;
}

Tensor reconstruct_cp(const CPDecomposition& cp) {
    return reconstruct_cp_from_factors(cp.factors, cp.weights, cp.rank);
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
    double residual = prev_res;
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
        Tensor Trecon = reconstruct_cp_from_factors(factors, weights, rank);
        double res = frobenius_norm(T - Trecon);
        residual = res;
        if (std::abs(prev_res - res) < tol) break;
        prev_res = res;
    }
    return {rank, factors, weights, residual};
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

Tensor reconstruct_tucker(const TuckerDecomposition& tucker) {
    Tensor result = tucker.core;
    int N = static_cast<int>(tucker.factors.size());
    for (int m = 0; m < N; ++m)
        result = mode_product(result, tucker.factors[m], m);
    return result;
}

// ========================== Non-negative Matrix Factorization (NMF) ==========================

namespace {

constexpr double kNmfEps = 1e-10;

static double mat_frobenius_norm(const std::vector<std::vector<double>>& A) {
    double s = 0.0;
    for (const auto& row : A)
        for (double v : row) s += v * v;
    return std::sqrt(s);
}

static std::vector<std::vector<double>>
mat_sub(const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B) {
    int m = static_cast<int>(A.size()), n = static_cast<int>(A[0].size());
    std::vector<std::vector<double>> C(m, std::vector<double>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) C[i][j] = A[i][j] - B[i][j];
    return C;
}

static std::vector<std::vector<double>>
mat_element_divide(const std::vector<std::vector<double>>& num,
                   const std::vector<std::vector<double>>& den,
                   double eps) {
    int m = static_cast<int>(num.size()), n = static_cast<int>(num[0].size());
    std::vector<std::vector<double>> C(m, std::vector<double>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) C[i][j] = num[i][j] / (den[i][j] + eps);
    return C;
}

// Deterministic seeded non-negative initialization in (0.1, 1.1].
static std::vector<std::vector<double>>
random_nonneg_matrix(int rows, int cols, unsigned seed) {
    std::vector<std::vector<double>> M(rows, std::vector<double>(cols));
    unsigned state = seed;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            state = state * 1664525u + 1013904223u;
            M[i][j] = static_cast<double>(state % 10000u) / 10000.0 + 0.1;
        }
    }
    return M;
}

static bool matrix_has_negative(const std::vector<std::vector<double>>& M) {
    for (const auto& row : M)
        for (double v : row)
            if (v < 0.0) return true;
    return false;
}

} // namespace

std::vector<std::vector<double>> reconstruct_nmf(const NMFDecomposition& nmf) {
    return mat_mul(nmf.W, nmf.H);
}

Result<NMFDecomposition> decompose_nmf(const std::vector<std::vector<double>>& matrix, int rank,
                                       int max_iter, double tol) {
    if (matrix.empty() || matrix[0].empty()) {
        return std::unexpected(DomainError{"decompose_nmf", "matrix must not be empty"});
    }
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    for (int i = 1; i < m; ++i) {
        if (static_cast<int>(matrix[i].size()) != n) {
            return std::unexpected(DomainError{"decompose_nmf", "matrix must be rectangular"});
        }
    }
    if (rank <= 0) {
        return std::unexpected(DomainError{"decompose_nmf", "rank must be positive"});
    }
    if (max_iter <= 0) {
        return std::unexpected(DomainError{"decompose_nmf", "max_iter must be positive"});
    }
    if (matrix_has_negative(matrix)) {
        return std::unexpected(
            DomainError{"decompose_nmf", "matrix entries must be non-negative"});
    }

    // Lee-Seung: alternate multiplicative updates on W (m×k) and H (k×n).
    auto W = random_nonneg_matrix(m, rank, 42u);
    auto H = random_nonneg_matrix(rank, n, 137u);

    std::vector<double> error_history;
    error_history.reserve(static_cast<size_t>(max_iter));

    double error = mat_frobenius_norm(mat_sub(matrix, mat_mul(W, H)));
    error_history.push_back(error);

    for (int iter = 1; iter <= max_iter; ++iter) {
        // H <- H .* (W^T V) ./ (W^T W H + eps)
        auto Wt = mat_T(W);
        auto WtV = mat_mul(Wt, matrix);
        auto WtW = mat_mul(Wt, W);
        auto WtWH = mat_mul(WtW, H);
        H = mat_hadamard(H, mat_element_divide(WtV, WtWH, kNmfEps));

        // W <- W .* (V H^T) ./ (W H H^T + eps)
        auto Ht = mat_T(H);
        auto VHt = mat_mul(matrix, Ht);
        auto WH = mat_mul(W, H);
        auto WHHt = mat_mul(WH, Ht);
        W = mat_hadamard(W, mat_element_divide(VHt, WHHt, kNmfEps));

        error = mat_frobenius_norm(mat_sub(matrix, mat_mul(W, H)));
        error_history.push_back(error);

        if (error_history.size() >= 2u) {
            double prev = error_history[error_history.size() - 2u];
            if (std::abs(prev - error) < tol) {
                return NMFDecomposition{rank, std::move(W), std::move(H), iter,
                                        error, std::move(error_history)};
            }
        }
    }

    return NMFDecomposition{rank, std::move(W), std::move(H), max_iter,
                            error, std::move(error_history)};
}

// ========================== Tensor-Train (TT) Decomposition ==========================

namespace {

// Thin SVD of an (rows x cols) matrix: M = U * diag(S) * V^T, S sorted descending,
// r = min(rows, cols) components.
struct ThinSVD {
    std::vector<std::vector<double>> U;  // rows x r
    std::vector<double> S;               // size r, sorted descending
    std::vector<std::vector<double>> V;  // cols x r
};

// One-sided Jacobi SVD of an (rows x cols) matrix with rows >= cols, producing all `cols`
// components. Unlike eigenvalue-decomposition-of-Gram-matrix approaches, one-sided Jacobi
// only ever needs pairwise column dot products, so it stays robust when several singular
// values are (near-)zero or repeated -- exactly the case TT-SVD hits constantly, since
// truncation is only meaningful once a matrix genuinely has a rank-deficient tail.
void jacobi_svd_tall(int rows, int cols, const std::vector<std::vector<double>>& M,
                     std::vector<std::vector<double>>& U_out,
                     std::vector<double>& S_out,
                     std::vector<std::vector<double>>& V_out) {
    // Wcol[j] holds column j of the working matrix (length rows); Vcol[j] holds column j
    // of the accumulated rotation product (length cols).
    std::vector<std::vector<double>> Wcol(cols, std::vector<double>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) Wcol[j][i] = M[i][j];

    std::vector<std::vector<double>> Vcol(cols, std::vector<double>(cols, 0.0));
    for (int j = 0; j < cols; ++j) Vcol[j][j] = 1.0;

    constexpr int kMaxSweeps = 60;
    constexpr double kConvergedThreshold = 1e-28;
    for (int sweep = 0; sweep < kMaxSweeps && cols > 1; ++sweep) {
        double off_norm_sq = 0.0;
        for (int p = 0; p < cols - 1; ++p) {
            for (int q = p + 1; q < cols; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < rows; ++i) {
                    alpha += Wcol[p][i] * Wcol[p][i];
                    beta += Wcol[q][i] * Wcol[q][i];
                    gamma += Wcol[p][i] * Wcol[q][i];
                }
                off_norm_sq += gamma * gamma;
                if (gamma == 0.0) continue;
                // Standard symmetric Jacobi rotation angle that zeroes out column pair
                // (p, q)'s cross term: cot(2*theta) = (beta - alpha) / (2*gamma).
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0.0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = c * t;
                for (int i = 0; i < rows; ++i) {
                    double wp = Wcol[p][i], wq = Wcol[q][i];
                    Wcol[p][i] = c * wp - s * wq;
                    Wcol[q][i] = s * wp + c * wq;
                }
                for (int i = 0; i < cols; ++i) {
                    double vp = Vcol[p][i], vq = Vcol[q][i];
                    Vcol[p][i] = c * vp - s * vq;
                    Vcol[q][i] = s * vp + c * vq;
                }
            }
        }
        if (off_norm_sq < kConvergedThreshold) break;
    }

    std::vector<double> S(cols);
    std::vector<std::vector<double>> U(rows, std::vector<double>(cols, 0.0));
    for (int j = 0; j < cols; ++j) {
        double norm_sq = 0.0;
        for (int i = 0; i < rows; ++i) norm_sq += Wcol[j][i] * Wcol[j][i];
        double norm = std::sqrt(norm_sq);
        S[j] = norm;
        if (norm > 0.0)
            for (int i = 0; i < rows; ++i) U[i][j] = Wcol[j][i] / norm;
    }
    std::vector<std::vector<double>> V(cols, std::vector<double>(cols));
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j < cols; ++j) V[i][j] = Vcol[j][i];

    std::vector<int> order(cols);
    for (int j = 0; j < cols; ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return S[a] > S[b]; });

    S_out.resize(cols);
    U_out.assign(rows, std::vector<double>(cols));
    V_out.assign(cols, std::vector<double>(cols));
    for (int j = 0; j < cols; ++j) {
        int src = order[j];
        S_out[j] = S[src];
        for (int i = 0; i < rows; ++i) U_out[i][j] = U[i][src];
        for (int i = 0; i < cols; ++i) V_out[i][j] = V[i][src];
    }
}

ThinSVD compute_thin_svd(const Tensor& M2d) {
    int rows = M2d.shape[0];
    int cols = M2d.shape[1];
    std::vector<std::vector<double>> M(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) M[i][j] = M2d.data[static_cast<size_t>(i) * cols + j];

    ThinSVD out;
    if (rows >= cols) {
        jacobi_svd_tall(rows, cols, M, out.U, out.S, out.V);
    } else {
        // M is wide: SVD(M^T) is a tall problem; M = (M^T)^T = V2 * S * U2^T, so U/V swap.
        std::vector<std::vector<double>> Mt(cols, std::vector<double>(rows));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) Mt[j][i] = M[i][j];
        std::vector<std::vector<double>> U2, V2;
        jacobi_svd_tall(cols, rows, Mt, U2, out.S, V2);
        out.U = std::move(V2);
        out.V = std::move(U2);
    }
    return out;
}

// Smallest truncation rank r_k (1 <= r_k <= S.size()) such that the sum of squared
// discarded singular values (indices r_k..end, S sorted descending) is <= delta^2.
int truncation_rank(const std::vector<double>& S, double delta) {
    int r_full = static_cast<int>(S.size());
    double budget_sq = delta * delta;
    // discarded[i] = sum_{j=i..r_full-1} S[j]^2, i.e. the squared reconstruction error
    // incurred by keeping only the first i singular values.
    std::vector<double> discarded(r_full + 1, 0.0);
    for (int i = r_full - 1; i >= 0; --i) discarded[i] = discarded[i + 1] + S[i] * S[i];
    for (int r_k = 1; r_k <= r_full; ++r_k)
        if (discarded[r_k] <= budget_sq) return r_k;
    return r_full;
}

// Prepend a leading dimension of size 1 to a shape.
std::vector<int> prepend_one(const std::vector<int>& shape) {
    std::vector<int> out;
    out.reserve(shape.size() + 1);
    out.push_back(1);
    out.insert(out.end(), shape.begin(), shape.end());
    return out;
}

} // namespace

Result<TTDecomposition> decompose_tt(const Tensor& T, double eps) {
    if (eps < 0.0) {
        return std::unexpected(DomainError{"decompose_tt", "eps must be non-negative"});
    }
    if (T.numel() == 0) {
        return std::unexpected(DomainError{"decompose_tt", "tensor must not be empty"});
    }
    int d = T.ndim();
    if (d < 3) {
        return std::unexpected(
            DomainError{"decompose_tt", "TT decomposition requires tensor order >= 3"});
    }

    double norm_T = frobenius_norm(T);
    // Oseledets' TT-SVD error bound: distribute the total relative error budget `eps`
    // equally (in the Frobenius-norm sense) across the d-1 sequential SVD truncations, so
    // the accumulated error stays within eps * ||T|| overall.
    double delta = (eps / std::sqrt(static_cast<double>(d - 1))) * norm_T;

    std::vector<Tensor> cores;
    std::vector<int> ranks;
    ranks.push_back(1);

    // remaining always has shape (r_prev, n_k, n_{k+1}, ..., n_{d-1}) at the start of
    // iteration k; start it as (1, n_0, ..., n_{d-1}) so the loop body is uniform.
    Tensor remaining = T.reshape(prepend_one(T.shape));

    for (int k = 0; k < d - 1; ++k) {
        int r_prev = remaining.shape[0];
        int n_k = remaining.shape[1];
        long rows = static_cast<long>(r_prev) * n_k;
        long cols = remaining.numel() / rows;

        Tensor M2d = remaining.reshape({static_cast<int>(rows), static_cast<int>(cols)});
        ThinSVD sv = compute_thin_svd(M2d);
        int r_k = truncation_rank(sv.S, delta);

        // Core k: reshape U's first r_k columns into shape (r_prev, n_k, r_k). Row index of
        // U is the merged (r_prev, n_k) index in the same row-major order used to build M2d.
        Tensor core({r_prev, n_k, r_k}, 0.0);
        for (long row = 0; row < rows; ++row)
            for (int l = 0; l < r_k; ++l)
                core.data[row * r_k + l] = sv.U[static_cast<size_t>(row)][static_cast<size_t>(l)];
        cores.push_back(std::move(core));
        ranks.push_back(r_k);

        // Next remaining = S_trunc * V_trunc^T, shape (r_k, cols), then reshaped to
        // (r_k, n_{k+1}, ..., n_{d-1}) using the tensor's known original dimensions.
        std::vector<double> next_data(static_cast<size_t>(r_k) * static_cast<size_t>(cols));
        for (int l = 0; l < r_k; ++l)
            for (long col = 0; col < cols; ++col)
                next_data[static_cast<size_t>(l) * static_cast<size_t>(cols) + static_cast<size_t>(col)] =
                    sv.S[static_cast<size_t>(l)] * sv.V[static_cast<size_t>(col)][static_cast<size_t>(l)];

        std::vector<int> next_shape;
        next_shape.push_back(r_k);
        for (int m = k + 1; m < d; ++m) next_shape.push_back(T.shape[m]);
        remaining = Tensor(next_shape, std::move(next_data));
    }

    // Final remaining has shape (r_{d-2}, n_{d-1}); it becomes the last core.
    int r_last_prev = remaining.shape[0];
    int n_last = remaining.shape[1];
    Tensor last_core = remaining.reshape({r_last_prev, n_last, 1});
    cores.push_back(std::move(last_core));
    ranks.push_back(1);

    return TTDecomposition{std::move(cores), std::move(ranks), eps};
}

Tensor reconstruct_tt(const TTDecomposition& tt) {
    int d = static_cast<int>(tt.cores.size());
    if (d == 0) return Tensor();

    Tensor result = tt.cores[0];  // shape (1, n_0, r_1)
    for (int k = 1; k < d; ++k) {
        int last_dim = result.ndim() - 1;
        result = contract(result, tt.cores[k], {{last_dim, 0}});
    }
    // result now has shape (1, n_0, n_1, ..., n_{d-1}, 1); drop the boundary rank-1 dims.
    std::vector<int> final_shape(result.shape.begin() + 1, result.shape.end() - 1);
    return result.reshape(final_shape);
}

} // namespace tensorops
} // namespace ms
