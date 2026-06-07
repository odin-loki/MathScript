#include "ms/cpu/blas.hpp"
#include "ms/cpu/lapack.hpp"

#include <vector>

namespace ms::cpu::lapack {

namespace {

void apply_reflector_left_rows(
    int m,
    int k,
    int nrhs,
    const double* A,
    int lda,
    double tau,
    double* B,
    int ldb) {
    if (tau == 0.0 || nrhs <= 0) {
        return;
    }

    for (int rhs = 0; rhs < nrhs; ++rhs) {
        double dot = B[static_cast<std::size_t>(k) + static_cast<std::size_t>(rhs) * static_cast<std::size_t>(ldb)];
        for (int r = k + 1; r < m; ++r) {
            dot += A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(r)] *
                   B[static_cast<std::size_t>(r) + static_cast<std::size_t>(rhs) * static_cast<std::size_t>(ldb)];
        }

        B[static_cast<std::size_t>(k) + static_cast<std::size_t>(rhs) * static_cast<std::size_t>(ldb)] -= tau * dot;
        for (int r = k + 1; r < m; ++r) {
            B[static_cast<std::size_t>(r) + static_cast<std::size_t>(rhs) * static_cast<std::size_t>(ldb)] -=
                tau * dot *
                A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(r)];
        }
    }
}

} // namespace

void dormqr(
    char side,
    char trans,
    int m,
    int n,
    int k,
    const double* A,
    int lda,
    const double* tau,
    double* C,
    int ldc) {
    if (m <= 0 || n <= 0 || k <= 0 || A == nullptr || tau == nullptr || C == nullptr) {
        return;
    }
    if (side != 'L' && side != 'l') {
        return;
    }
    if (trans != 'T' && trans != 't' && trans != 'N' && trans != 'n') {
        return;
    }

    const int apply_count = (k < m) ? k : m;
    if (trans == 'T' || trans == 't') {
        for (int i = 0; i < apply_count; ++i) {
            apply_reflector_left_rows(m, i, n, A, lda, tau[i], C, ldc);
        }
        return;
    }

    for (int i = apply_count - 1; i >= 0; --i) {
        apply_reflector_left_rows(m, i, n, A, lda, tau[i], C, ldc);
    }
}

int dgels(int m, int n, int nrhs, double* A, int lda, double* B, int ldb) {
    if (m <= 0 || n <= 0 || nrhs <= 0) {
        return 0;
    }
    if (A == nullptr || B == nullptr) {
        return 1;
    }
    if (m < n) {
        return 2;
    }

    std::vector<double> tau(static_cast<std::size_t>(n));
    const int info = dgeqrf(m, n, A, lda, tau.data());
    if (info != 0) {
        return info;
    }

    dormqr('L', 'T', m, nrhs, n, A, lda, tau.data(), B, ldb);

    blas::dtrsm('L', 'U', 'N', 'N', n, nrhs, 1.0, A, lda, B, ldb);
    return 0;
}

} // namespace ms::cpu::lapack
