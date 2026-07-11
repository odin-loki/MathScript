#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <vector>

namespace ms::cpu::lapack {

namespace {

void apply_reflector_left_out(
    double* C,
    int ldc,
    int m,
    int n,
    int r0,
    int c0,
    int c1,
    const double* A,
    int lda,
    int v_r,
    int v_c,
    int v_inc,
    double tau) {
    const int cols = c1 - c0;
    if (tau == 0.0 || cols == 0) {
        return;
    }

    int lastv = m - r0;
    while (lastv > 1) {
        const double vi = (v_inc == 1) ? A[(v_r + lastv - 1) + v_c * lda] : A[v_r + (v_c + lastv - 1) * lda];
        if (vi != 0.0) {
            break;
        }
        --lastv;
    }

    int lastc = 0;
    for (int j = c0; j < c1; ++j) {
        for (int i = r0; i < r0 + lastv; ++i) {
            if (C[i + j * ldc] != 0.0) {
                lastc = j - c0 + 1;
            }
        }
    }
    if (lastc == 0) {
        return;
    }

    if (lastv == 1) {
        for (int j = c0; j < c0 + lastc; ++j) {
            C[r0 + j * ldc] *= 1.0 - tau;
        }
        return;
    }

    std::vector<double> work(static_cast<std::size_t>(lastc));
    for (int jj = 0; jj < lastc; ++jj) {
        const int j = c0 + jj;
        double sum = C[r0 + j * ldc];
        for (int ii = 1; ii < lastv; ++ii) {
            const int i = r0 + ii;
            const double vi = (v_inc == 1) ? A[(v_r + ii) + v_c * lda] : A[v_r + (v_c + ii) * lda];
            sum += C[i + j * ldc] * vi;
        }
        work[static_cast<std::size_t>(jj)] = sum;
    }

    for (int jj = 0; jj < lastc; ++jj) {
        const int j = c0 + jj;
        C[r0 + j * ldc] -= tau * work[static_cast<std::size_t>(jj)];
    }

    for (int ii = 1; ii < lastv; ++ii) {
        const int i = r0 + ii;
        const double vi = (v_inc == 1) ? A[(v_r + ii) + v_c * lda] : A[v_r + (v_c + ii) * lda];
        for (int jj = 0; jj < lastc; ++jj) {
            const int j = c0 + jj;
            C[i + j * ldc] -= tau * vi * work[static_cast<std::size_t>(jj)];
        }
    }
}

void apply_reflector_right_out(
    double* C,
    int ldc,
    int m,
    int n,
    int r0,
    int r1,
    int c0,
    int c1,
    const double* A,
    int lda,
    int v_r,
    int v_c,
    int v_inc,
    double tau) {
    const int rows = r1 - r0;
    const int cols = c1 - c0;
    if (tau == 0.0 || rows == 0) {
        return;
    }

    int lastv = cols;
    while (lastv > 1) {
        const double vj = (v_inc == 1) ? A[(v_r + lastv - 1) + v_c * lda] : A[v_r + (v_c + lastv - 1) * lda];
        if (vj != 0.0) {
            break;
        }
        --lastv;
    }

    int lastc = 0;
    for (int i = r0; i < r1; ++i) {
        for (int j = c0; j < c0 + lastv; ++j) {
            if (C[i + j * ldc] != 0.0) {
                lastc = i - r0 + 1;
            }
        }
    }
    if (lastc == 0) {
        return;
    }

    if (lastv == 1) {
        for (int i = r0; i < r0 + lastc; ++i) {
            C[i + c0 * ldc] *= 1.0 - tau;
        }
        return;
    }

    std::vector<double> work(static_cast<std::size_t>(lastc));
    for (int ii = 0; ii < lastc; ++ii) {
        const int i = r0 + ii;
        double sum = C[i + c0 * ldc];
        for (int jj = 1; jj < lastv; ++jj) {
            const int j = c0 + jj;
            const double vj = (v_inc == 1) ? A[(v_r + jj) + v_c * lda] : A[v_r + (v_c + jj) * lda];
            sum += C[i + j * ldc] * vj;
        }
        work[static_cast<std::size_t>(ii)] = sum;
    }

    for (int ii = 0; ii < lastc; ++ii) {
        const int i = r0 + ii;
        C[i + c0 * ldc] -= tau * work[static_cast<std::size_t>(ii)];
    }

    for (int jj = 1; jj < lastv; ++jj) {
        const int j = c0 + jj;
        const double vj = (v_inc == 1) ? A[(v_r + jj) + v_c * lda] : A[v_r + (v_c + jj) * lda];
        for (int ii = 0; ii < lastc; ++ii) {
            const int i = r0 + ii;
            C[i + j * ldc] -= tau * vj * work[static_cast<std::size_t>(ii)];
        }
    }
}

void apply_reflector_right_sub(
    double* A,
    int lda,
    int r0,
    int r1,
    int c0,
    int c1,
    int v_r,
    int v_c,
    int v_inc,
    double tau) {
    const int m = r1 - r0;
    const int n = c1 - c0;
    if (tau == 0.0 || m == 0) {
        return;
    }

    int lastv = n;
    while (lastv > 1) {
        const double vj = (v_inc == 1) ? A[(v_r + lastv - 1) + v_c * lda] : A[v_r + (v_c + lastv - 1) * lda];
        if (vj != 0.0) {
            break;
        }
        --lastv;
    }

    int lastc = 0;
    for (int i = r0; i < r1; ++i) {
        for (int j = c0; j < c0 + lastv; ++j) {
            if (A[i + j * lda] != 0.0) {
                lastc = i - r0 + 1;
            }
        }
    }
    if (lastc == 0) {
        return;
    }

    if (lastv == 1) {
        for (int i = r0; i < r0 + lastc; ++i) {
            A[i + c0 * lda] *= 1.0 - tau;
        }
        return;
    }

    std::vector<double> work(static_cast<std::size_t>(lastc));
    for (int ii = 0; ii < lastc; ++ii) {
        const int i = r0 + ii;
        double sum = A[i + c0 * lda];
        for (int jj = 1; jj < lastv; ++jj) {
            const int j = c0 + jj;
            const double vj = (v_inc == 1) ? A[(v_r + jj) + v_c * lda] : A[v_r + (v_c + jj) * lda];
            sum += A[i + j * lda] * vj;
        }
        work[static_cast<std::size_t>(ii)] = sum;
    }

    for (int ii = 0; ii < lastc; ++ii) {
        const int i = r0 + ii;
        A[i + c0 * lda] -= tau * work[static_cast<std::size_t>(ii)];
    }

    for (int jj = 1; jj < lastv; ++jj) {
        const int j = c0 + jj;
        const double vj = (v_inc == 1) ? A[(v_r + jj) + v_c * lda] : A[v_r + (v_c + jj) * lda];
        for (int ii = 0; ii < lastc; ++ii) {
            const int i = r0 + ii;
            A[i + j * lda] -= tau * work[static_cast<std::size_t>(ii)] * vj;
        }
    }
}

void dorgl2(int m, int n, int k, double* A, int lda, const double* tau) {
    if (m <= 0 || k <= 0) {
        return;
    }
    const int count = (std::min)(k, m);
    for (int i = count - 1; i >= 0; --i) {
        if (i < n - 1) {
            if (i < m - 1) {
                apply_reflector_right_sub(A, lda, i + 1, m, i, n, i, i, lda, tau[i]);
            }
            for (int j = i + 1; j < n; ++j) {
                A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda)] *= -tau[i];
            }
        }
        A[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(lda)] = 1.0 - tau[i];
        for (int j = 0; j < i; ++j) {
            A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda)] = 0.0;
        }
    }
}

void apply_q_right(int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool trans) {
    if (k <= 0) {
        return;
    }
    const int count = (std::min)(k, m);
    if (trans) {
        for (int i = count - 1; i >= 0; --i) {
            apply_reflector_right_out(C, ldc, m, n, 0, m, i, n, A, lda, i, i, 1, tau[i]);
        }
    } else {
        for (int i = 0; i < count; ++i) {
            apply_reflector_right_out(C, ldc, m, n, 0, m, i, n, A, lda, i, i, 1, tau[i]);
        }
    }
}

void transpose_square(int n, double* C, int ldc) {
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double tmp = C[i + j * ldc];
            C[i + j * ldc] = C[j + i * ldc];
            C[j + i * ldc] = tmp;
        }
    }
}

void apply_p_right_tall(int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool apply_pt) {
    const int count = (std::min)(k, n) - 1;
    if (count <= 0) {
        return;
    }
    if (apply_pt) {
        for (int i = 0; i < count; ++i) {
            apply_reflector_right_out(C, ldc, n, n, 0, n, i + 1, n, A, lda, i, i + 1, lda, tau[i]);
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            apply_reflector_right_out(C, ldc, n, n, 0, n, i + 1, n, A, lda, i, i + 1, lda, tau[i]);
        }
    }
}

void apply_p_left_tall(int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool apply_pt) {
    if (apply_pt) {
        transpose_square(n, C, ldc);
        apply_p_right_tall(n, k, A, lda, tau, C, ldc, true);
        transpose_square(n, C, ldc);
    } else {
        transpose_square(n, C, ldc);
        apply_p_right_tall(n, k, A, lda, tau, C, ldc, false);
        transpose_square(n, C, ldc);
    }
}

void apply_p_right_wide(int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool apply_pt) {
    const int count = (std::max)(0, (std::min)(k, m) - 1);
    if (count <= 0) {
        return;
    }
    if (apply_pt) {
        for (int i = 0; i < count; ++i) {
            apply_reflector_right_out(C, ldc, m, n, 0, m, i, n, A, lda, i, i, lda, tau[i]);
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            apply_reflector_right_out(C, ldc, m, n, 0, m, i, n, A, lda, i, i, lda, tau[i]);
        }
    }
}

// Used by dorgbr wide-P generation; dormbr P-left requires m >= n so this is not wired there.
void apply_p_left_wide(int m, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool apply_p) {
    if (apply_p) {
        transpose_square(m, C, ldc);
        apply_p_right_wide(m, m, k, A, lda, tau, C, ldc, true);
        transpose_square(m, C, ldc);
    } else {
        transpose_square(m, C, ldc);
        apply_p_right_wide(m, m, k, A, lda, tau, C, ldc, false);
        transpose_square(m, C, ldc);
    }
}

void dorgbr_p_wide(int m, int n, int k, const double* A, int lda, const double* tau, double* Q, int ldq) {
    if (m <= 0) {
        return;
    }

    std::vector<double> work(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            work[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)] =
                A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(lda)];
        }
    }

    dorgl2(m, n, (std::min)(k, m), work.data(), m, tau);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Q[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq)] =
                work[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)];
        }
    }
}

bool is_trans(char trans) {
    return trans == 'T' || trans == 't';
}

bool is_left(char side) {
    return side == 'L' || side == 'l';
}

} // namespace

void dorgbr(
    char vect,
    int m,
    int n,
    int k,
    const double* A,
    int lda,
    const double* tau,
    double* Q,
    int ldq) {
    if (m <= 0 || n <= 0 || k <= 0 || A == nullptr || tau == nullptr || Q == nullptr) {
        return;
    }

    if (vect == 'Q' || vect == 'q') {
        dorgqr(m, n, k, A, lda, tau, Q, ldq);
        return;
    }

    if (vect != 'P' && vect != 'p') {
        return;
    }

    if (m >= n) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(i)] =
                    (i == j) ? 1.0 : 0.0;
            }
        }
        // apply_p_right_tall accumulates the elementary reflectors P(0), P(1), ...,
        // P(count-1) (each individually symmetric) onto the identity from the
        // right. Applying them in ascending order (apply_pt=true) reproduces the
        // forward product P = P(0)*P(1)*...*P(count-1) used by dgebd2 to build the
        // bidiagonal form of A, not its transpose. dgesvd_tall needs P**T here (as
        // documented above) to back-transform the bidiagonal-space right singular
        // vectors into the original basis, so the reflectors must be applied in
        // descending order instead, which is what apply_pt=false does.
        //
        // Passing true here previously produced P instead of P**T whenever the
        // reflector product doesn't happen to be self-transpose (i.e. whenever
        // count = min(k, n) - 1 >= 2, so P is a product of 2+ non-commuting
        // Householder reflectors). That silently corrupts every right singular
        // vector beyond the ones spanned by the first reflector for ANY m>=n input
        // with n >= 3 taking this code path — not only rank-deficient matrices,
        // though a rank-deficient probe with several trailing near-zero singular
        // values beyond the leading one(s) is what exposed it, because the error
        // shows up as soon as more than one reflector's worth of columns are read.
        apply_p_right_tall(n, k, A, lda, tau, Q, ldq, false);
    } else {
        dorgbr_p_wide(m, n, k, A, lda, tau, Q, ldq);
    }
}

int dormbr(
    char vect,
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
        return 0;
    }

    const bool apply_q = (vect == 'Q' || vect == 'q');
    const bool apply_p = (vect == 'P' || vect == 'p');
    if (!apply_q && !apply_p) {
        return 1;
    }

    const bool left = is_left(side);
    const bool tr = is_trans(trans);

    if (apply_q) {
        if (left) {
            dormqr(side, trans, m, n, k, A, lda, tau, C, ldc);
        } else {
            apply_q_right(m, n, k, A, lda, tau, C, ldc, tr);
        }
        return 0;
    }

    if (left) {
        // LAPACK requires M >= N when applying P from the left. apply_p_left_tall assumes
        // an N-by-N leading block of C; apply_p_left_wide is for square m-by-m C only and
        // is not used here because wide gebrd P-left with m < n is invalid.
        if (m < n) {
            return 1;
        }
        // P is n-by-n; apply to the leading n-by-n block of C (rows n..m-1 unchanged).
        apply_p_left_tall(n, k, A, lda, tau, C, ldc, tr);
    } else {
        if (m == n) {
            apply_p_right_tall(n, k, A, lda, tau, C, ldc, tr);
        } else {
            // Tall (m>n): C is n-by-m; wide (m<n): C is m-by-n. Both use wide-P reflector layout.
            apply_p_right_wide((std::min)(m, n), (std::max)(m, n), k, A, lda, tau, C, ldc, tr);
        }
    }
    return 0;
}

} // namespace ms::cpu::lapack
