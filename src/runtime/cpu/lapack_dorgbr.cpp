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

void apply_q_left(int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool trans) {
    if (k <= 0) {
        return;
    }
    const int count = (std::min)(k, m);
    if (trans) {
        for (int i = 0; i < count; ++i) {
            apply_reflector_left_out(C, ldc, m, n, i, i, n, A, lda, i, i, 1, tau[i]);
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            apply_reflector_left_out(C, ldc, m, n, i, i, n, A, lda, i, i, 1, tau[i]);
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
            apply_reflector_left_out(C, ldc, n, m, i, i, m, A, lda, i, i, 1, tau[i]);
        }
    } else {
        for (int i = 0; i < count; ++i) {
            apply_reflector_left_out(C, ldc, n, m, i, i, m, A, lda, i, i, 1, tau[i]);
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
        apply_p_right_tall(n, k, A, lda, tau, C, ldc, false);
        transpose_square(n, C, ldc);
    } else {
        transpose_square(n, C, ldc);
        apply_p_right_tall(n, k, A, lda, tau, C, ldc, true);
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

void apply_p_left_wide(int m, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool apply_pt) {
    if (apply_pt) {
        transpose_square(m, C, ldc);
        apply_p_right_wide(m, m, k, A, lda, tau, C, ldc, false);
        transpose_square(m, C, ldc);
    } else {
        transpose_square(m, C, ldc);
        apply_p_right_wide(m, m, k, A, lda, tau, C, ldc, true);
        transpose_square(m, C, ldc);
    }
}

void apply_q_left_wide(int m, int k, const double* A, int lda, const double* tau, double* C, int ldc, bool trans) {
    const int count = (std::min)(k, m) - 1;
    if (count <= 0) {
        return;
    }
    if (trans) {
        for (int i = 0; i < count; ++i) {
            apply_reflector_left_out(C, ldc, m, m, i + 1, i, m, A, lda, i + 1, i, 1, tau[i]);
        }
    } else {
        for (int i = count - 1; i >= 0; --i) {
            apply_reflector_left_out(C, ldc, m, m, i + 1, i, m, A, lda, i + 1, i, 1, tau[i]);
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

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(ldq) + static_cast<std::size_t>(i)] =
                (i == j) ? 1.0 : 0.0;
        }
    }

    if (m >= n) {
        apply_p_right_tall(n, k, A, lda, tau, Q, ldq, true);
    } else {
        apply_p_right_wide(m, n, k, A, lda, tau, Q, ldq, true);
    }
}

void dormbr(
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
        return;
    }

    const bool apply_q = (vect == 'Q' || vect == 'q');
    const bool apply_p = (vect == 'P' || vect == 'p');
    if (!apply_q && !apply_p) {
        return;
    }

    const bool left = is_left(side);
    const bool tr = is_trans(trans);

    if (apply_q) {
        if (left) {
            dormqr(side, trans, m, n, k, A, lda, tau, C, ldc);
        } else {
            apply_q_right(m, n, k, A, lda, tau, C, ldc, tr);
        }
        return;
    }

    if (left) {
        if (m > n) {
            apply_p_right_wide(m, n, k, A, lda, tau, C, ldc, tr);
        } else {
            apply_p_left_tall(n, k, A, lda, tau, C, ldc, tr);
        }
    } else {
        if (m > n) {
            apply_p_right_wide(k, m, k, A, lda, tau, C, ldc, tr);
        } else {
            apply_p_right_tall(n, k, A, lda, tau, C, ldc, tr);
        }
    }
}

} // namespace ms::cpu::lapack
