#include "ms/cpu/blas.hpp"
#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ms::cpu::lapack {

namespace {

int bidiag_singular_values_steqr(int k, const double* D, const double* E, double* S) {
    if (k <= 0) {
        return 0;
    }

    std::vector<double> diag(static_cast<std::size_t>(k));
    std::vector<double> off(static_cast<std::size_t>((std::max)(0, k - 1)));
    for (int i = 0; i < k; ++i) {
        diag[static_cast<std::size_t>(i)] =
            D[static_cast<std::size_t>(i)] * D[static_cast<std::size_t>(i)] +
            (i > 0 ? E[static_cast<std::size_t>(i - 1)] * E[static_cast<std::size_t>(i - 1)] : 0.0);
    }
    for (int i = 0; i < k - 1; ++i) {
        off[static_cast<std::size_t>(i)] = D[static_cast<std::size_t>(i)] * E[static_cast<std::size_t>(i)];
    }

    dsteqr(k, diag.data(), off.data(), nullptr, 0);

    std::vector<int> order(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        order[static_cast<std::size_t>(i)] = i;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return diag[static_cast<std::size_t>(a)] > diag[static_cast<std::size_t>(b)];
    });

    for (int j = 0; j < k; ++j) {
        const int idx = order[static_cast<std::size_t>(j)];
        S[j] = std::sqrt((std::max)(0.0, diag[static_cast<std::size_t>(idx)]));
    }

    return 0;
}

void vbt_rows_to_vb_cols(int n, const double* vbt_rows, int ldvbt, double* vb, int ldvb) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            vb[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldvb)] =
                vbt_rows[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldvbt) + static_cast<std::size_t>(i)];
        }
    }
}

void vbt_rows_to_vb_cols_rect(int m, const double* vbt_rows, int ldvbt, double* vb, int ldvb) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            vb[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldvb)] =
                vbt_rows[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldvbt) + static_cast<std::size_t>(i)];
        }
    }
}

int dgesvd_tall(int m, int n, const double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt) {
    std::vector<double> work(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            work[static_cast<std::size_t>(i + j * lda)] = A[static_cast<std::size_t>(i + j * lda)];
        }
    }

    std::vector<double> D(static_cast<std::size_t>(n));
    std::vector<double> E(static_cast<std::size_t>((std::max)(0, n - 1)));
    std::vector<double> tauq(static_cast<std::size_t>(n));
    std::vector<double> taup(static_cast<std::size_t>(n));
    if (dgebrd(m, n, work.data(), lda, D.data(), E.data(), tauq.data(), taup.data()) != 0) {
        return 1;
    }

    std::vector<double> bidiag(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            bidiag[static_cast<std::size_t>(i + j * lda)] = work[static_cast<std::size_t>(i + j * lda)];
        }
    }

    std::vector<double> qfac(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    dorgbr('Q', m, n, n, bidiag.data(), lda, tauq.data(), qfac.data(), m);

    std::vector<double> pt(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    dorgbr('P', n, n, n, bidiag.data(), lda, taup.data(), pt.data(), n);

    std::vector<double> ub(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    std::vector<double> vbt(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    std::vector<double> d = D;
    std::vector<double> e = E;
    if (dbdsqr('U', n, d.data(), e.data(), ub.data(), n, vbt.data(), n) != 0) {
        return 1;
    }

    blas::dgemm('N', 'N', m, n, n, 1.0, qfac.data(), m, ub.data(), n, 0.0, U, ldu);

    std::vector<double> vb(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    vbt_rows_to_vb_cols(n, vbt.data(), n, vb.data(), n);

    std::vector<double> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    blas::dgemm('T', 'N', n, n, n, 1.0, pt.data(), n, vb.data(), n, 0.0, v.data(), n);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            VT[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldvt)] =
                v[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)];
        }
    }

    for (int i = 0; i < n; ++i) {
        S[i] = d[static_cast<std::size_t>(i)];
    }
    return 0;
}

int dgesvd_wide(int m, int n, const double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt) {
    std::vector<double> at(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            at[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)] =
                A[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda)];
        }
    }

    std::vector<double> ut(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
    std::vector<double> vtt(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    if (dgesvd_tall(n, m, at.data(), n, S, ut.data(), n, vtt.data(), m) != 0) {
        std::vector<double> work(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
        std::vector<double> D(static_cast<std::size_t>(m));
        std::vector<double> E(static_cast<std::size_t>((std::max)(0, m - 1)));
        std::vector<double> tauq(static_cast<std::size_t>(m));
        std::vector<double> taup(static_cast<std::size_t>(m));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                work[static_cast<std::size_t>(i + j * lda)] = A[static_cast<std::size_t>(i + j * lda)];
            }
        }
        if (dgebrd(m, n, work.data(), lda, D.data(), E.data(), tauq.data(), taup.data()) != 0) {
            return 1;
        }
        (void)bidiag_singular_values_steqr(m, D.data(), E.data(), S);
        return 1;
    }

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            U[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldu)] =
                vtt[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) + static_cast<std::size_t>(i)];
        }
    }

    for (int q = 0; q < n; ++q) {
        for (int k = 0; k < m; ++k) {
            VT[static_cast<std::size_t>(k) + static_cast<std::size_t>(q) * static_cast<std::size_t>(ldvt)] =
                ut[static_cast<std::size_t>(q) + static_cast<std::size_t>(k) * static_cast<std::size_t>(n)];
        }
    }

    return 0;
}

} // namespace

int dgesvd(int m, int n, double* A, int lda, double* S, double* U, int ldu, double* VT, int ldvt) {
    if (m <= 0 || n <= 0) {
        return 0;
    }
    if (A == nullptr || S == nullptr || U == nullptr || VT == nullptr) {
        return 1;
    }

    if (m >= n) {
        return dgesvd_tall(m, n, A, lda, S, U, ldu, VT, ldvt);
    }
    return dgesvd_wide(m, n, A, lda, S, U, ldu, VT, ldvt);
}

} // namespace ms::cpu::lapack
