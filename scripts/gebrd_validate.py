"""Validate Golub-Kahan bidiagonalization against NumPy SVD."""
import math
import numpy as np


def dlarfg(n, alpha, x, incx):
    if n <= 1:
        return alpha, x, 0.0
    if incx == 1:
        xn = math.hypot(*x[: n - 1]) if n > 1 else 0.0
    else:
        xn = math.sqrt(sum(x[i * incx] ** 2 for i in range(n - 1)))
    if xn == 0.0:
        return alpha, x, 0.0
    beta = -math.copysign(math.hypot(alpha, xn), alpha)
    tau = (beta - alpha) / beta
    denom = alpha - beta
    for i in range(n - 1):
        x[i * incx] /= denom
    return beta, x, tau


def dlarf1f_left(m, n, v, incv, tau, C, ldc, work):
    if tau == 0.0 or n == 0:
        return
    lastv = m
    if incv > 0:
        idx = (lastv - 1) * incv
    else:
        idx = 0
    while lastv > 1 and v[idx] == 0.0:
        lastv -= 1
        idx -= incv
    lastc = 0
    for j in range(n):
        for i in range(lastv):
            if C[i + j * ldc] != 0.0:
                lastc = j + 1
    if lastc == 0:
        return
    if lastv == 1:
        for j in range(lastc):
            C[0 + j * ldc] *= 1.0 - tau
        return
    for j in range(lastc):
        s = C[0 + j * ldc]
        for i in range(1, lastv):
            s += C[i + j * ldc] * v[(i - 1 + 1) * incv if incv > 0 else (i - 1) * incv + incv]
        work[j] = s
    # fix v indexing: v points to v(2) in Fortran, v[0] is v(2)
    for j in range(lastc):
        s = C[0 + j * ldc]
        for i in range(1, lastv):
            vi = v[(i - 1) * incv] if incv > 0 else v[(i - 1) * incv]
            s += C[i + j * ldc] * vi
        work[j] = s
    for j in range(lastc):
        C[0 + j * ldc] -= tau * work[j]
    for i in range(1, lastv):
        vi = v[(i - 1) * incv]
        for j in range(lastc):
            C[i + j * ldc] -= tau * vi * work[j]


def dlarf1f_left_clean(m, n, v0_row, v0_col, v_incv, tau, C, ldc):
    """v implicit at (v0_row,v0_col), tail v(2:lastv) with stride v_incv."""
    if tau == 0.0 or n == 0:
        return
    lastv = m
    if v_incv == 1:
        while lastv > 1 and C[(v0_row + lastv - 1) + v0_col * ldc] == 0.0:
            lastv -= 1
    else:
        while lastv > 1 and C[v0_row + (v0_col + lastv - 1) * ldc] == 0.0:
            lastv -= 1
    lastc = 0
    for j in range(n):
        for i in range(lastv):
            if C[i + j * ldc] != 0.0:
                lastc = j + 1
    if lastc == 0:
        return
    if lastv == 1:
        for j in range(lastc):
            C[0 + j * ldc] *= 1.0 - tau
        return
    work = [0.0] * lastc
    for j in range(lastc):
        s = C[0 + j * ldc]
        for i in range(1, lastv):
            if v_incv == 1:
                vi = C[(v0_row + i) + v0_col * ldc]
            else:
                vi = C[v0_row + (v0_col + i) * ldc]
            s += C[i + j * ldc] * vi
        work[j] = s
    for j in range(lastc):
        C[0 + j * ldc] -= tau * work[j]
    for i in range(1, lastv):
        if v_incv == 1:
            vi = C[(v0_row + i) + v0_col * ldc]
        else:
            vi = C[v0_row + (v0_col + i) * ldc]
        for j in range(lastc):
            C[i + j * ldc] -= tau * vi * work[j]


def dlarf1f_right(m, n, v0_row, v0_col, v_incv, tau, C, ldc):
    if tau == 0.0 or m == 0:
        return
    lastv = n
    if v_incv == 1:
        while lastv > 1 and C[v0_row + (v0_col + lastv - 1) * ldc] == 0.0:
            lastv -= 1
    else:
        while lastv > 1 and C[(v0_row + lastv - 1) + v0_col * ldc] == 0.0:
            lastv -= 1
    lastc = 0
    for i in range(m):
        for j in range(lastv):
            if C[i + j * ldc] != 0.0:
                lastc = i + 1
    if lastc == 0:
        return
    if lastv == 1:
        for i in range(lastc):
            C[i + 0 * ldc] *= 1.0 - tau
        return
    work = [0.0] * lastc
    for i in range(lastc):
        s = C[i + 0 * ldc]
        for j in range(1, lastv):
            if v_incv == 1:
                vj = C[v0_row + (v0_col + j) * ldc]
            else:
                vj = C[(v0_row + j) + v0_col * ldc]
            s += C[i + j * ldc] * vj
        work[i] = s
    for i in range(lastc):
        C[i + 0 * ldc] -= tau * work[i]
    for j in range(1, lastv):
        if v_incv == 1:
            vj = C[v0_row + (v0_col + j) * ldc]
        else:
            vj = C[(v0_row + j) + v0_col * ldc]
        for i in range(lastc):
            C[i + j * ldc] -= tau * work[i] * vj


def dgebd2(m, n, A, lda):
    """Column-major A, returns D, E, tauq, taup."""
    D = [0.0] * min(m, n)
    E = [0.0] * (min(m, n) - 1)
    tauq = [0.0] * min(m, n)
    taup = [0.0] * min(m, n)
    if m >= n:
        for i in range(n):
            # left reflector on column i
            nn = m - i
            alpha = A[i + i * lda]
            x = [A[(i + k) + i * lda] for k in range(1, nn)]
            beta, x, tau = dlarfg(nn, alpha, x, 1)
            A[i + i * lda] = beta
            for k in range(1, nn):
                A[(i + k) + i * lda] = x[k - 1]
            tauq[i] = tau
            D[i] = A[i + i * lda]
            if i < n - 1:
                dlarf1f_left_clean(m - i, n - i - 1, i, i, 1, tau, A[i + (i + 1) * lda :], lda)
                # patch: operate on submatrix in place
                sub = A  # full matrix; offset handled below
                _ = sub
                dlarf1f_left_clean(m - i, n - i - 1, i, i, 1, tau, A, lda)
                # zero first row of submatrix offset - fix by passing pointer offset
            if i < n - 1:
                # re-run left apply with correct submatrix base
                pass
    return D, E, tauq, taup


def dgebd2_fixed(m, n, A, lda):
    A = list(A)
    k = min(m, n)
    D = [0.0] * k
    E = [0.0] * (k - 1)
    tauq = [0.0] * k
    taup = [0.0] * k
    if m >= n:
        for i in range(n):
            nn = m - i
            alpha = A[i + i * lda]
            x = [0.0] * (nn - 1)
            for t in range(nn - 1):
                x[t] = A[(i + 1 + t) + i * lda]
            beta, x, tau = dlarfg(nn, alpha, x, 1)
            A[i + i * lda] = beta
            for t in range(nn - 1):
                A[(i + 1 + t) + i * lda] = x[t]
            tauq[i] = tau
            D[i] = A[i + i * lda]
            if i < n - 1:
                # apply left to A[i:m, i+1:n]
                apply_left_sub(A, lda, i, m, i + 1, n, i, i, 1, tau)
                nn2 = n - i - 1
                alpha = A[i + (i + 1) * lda]
                x = [0.0] * (nn2 - 1)
                for t in range(nn2 - 1):
                    x[t] = A[i + (i + 2 + t) * lda]
                beta, x, tau = dlarfg(nn2, alpha, x, 1)
                A[i + (i + 1) * lda] = beta
                for t in range(nn2 - 1):
                    A[i + (i + 2 + t) * lda] = x[t]
                taup[i] = tau
                E[i] = A[i + (i + 1) * lda]
                apply_right_sub(A, lda, i + 1, m, i + 1, n, i, i + 1, lda, tau)
            else:
                taup[i] = 0.0
    else:
        for i in range(m):
            nn = n - i
            alpha = A[i + i * lda]
            x = [0.0] * (nn - 1)
            for t in range(nn - 1):
                x[t] = A[i + (i + 1 + t) * lda]
            beta, x, tau = dlarfg(nn, alpha, x, 1)
            A[i + i * lda] = beta
            for t in range(nn - 1):
                A[i + (i + 1 + t) * lda] = x[t]
            taup[i] = tau
            D[i] = A[i + i * lda]
            if i < m - 1:
                apply_right_sub(A, lda, i + 1, m, i, n, i, i, lda, tau)
                nn2 = m - i - 1
                alpha = A[(i + 1) + i * lda]
                x = [0.0] * (nn2 - 1)
                for t in range(nn2 - 1):
                    x[t] = A[(i + 2 + t) + i * lda]
                beta, x, tau = dlarfg(nn2, alpha, x, 1)
                A[(i + 1) + i * lda] = beta
                for t in range(nn2 - 1):
                    A[(i + 2 + t) + i * lda] = x[t]
                tauq[i] = tau
                E[i] = A[(i + 1) + i * lda]
                apply_left_sub(A, lda, i + 1, m, i + 1, n, i + 1, i, 1, tau)
            else:
                tauq[i] = 0.0
    return D, E, tauq, taup, A


def apply_left_sub(A, lda, r0, r1, c0, c1, v_r, v_c, v_inc, tau):
    m = r1 - r0
    n = c1 - c0
    if tau == 0.0 or n == 0:
        return
    lastv = m
    while lastv > 1:
        if v_inc == 1:
            vi = A[(v_r + lastv - 1) + v_c * lda]
        else:
            vi = A[v_r + (v_c + lastv - 1) * lda]
        if vi != 0.0:
            break
        lastv -= 1
    lastc = 0
    for j in range(c0, c1):
        for i in range(r0, r0 + lastv):
            if A[i + j * lda] != 0.0:
                lastc = j - c0 + 1
    if lastc == 0:
        return
    if lastv == 1:
        for j in range(c0, c0 + lastc):
            A[r0 + j * lda] *= 1.0 - tau
        return
    work = [0.0] * lastc
    for jj in range(lastc):
        j = c0 + jj
        s = A[r0 + j * lda]
        for ii in range(1, lastv):
            i = r0 + ii
            if v_inc == 1:
                vi = A[(v_r + ii) + v_c * lda]
            else:
                vi = A[v_r + (v_c + ii) * lda]
            s += A[i + j * lda] * vi
        work[jj] = s
    for jj in range(lastc):
        j = c0 + jj
        A[r0 + j * lda] -= tau * work[jj]
    for ii in range(1, lastv):
        i = r0 + ii
        if v_inc == 1:
            vi = A[(v_r + ii) + v_c * lda]
        else:
            vi = A[v_r + (v_c + ii) * lda]
        for jj in range(lastc):
            j = c0 + jj
            A[i + j * lda] -= tau * vi * work[jj]


def apply_right_sub(A, lda, r0, r1, c0, c1, v_r, v_c, v_inc, tau):
    m = r1 - r0
    n = c1 - c0
    if tau == 0.0 or m == 0:
        return
    lastv = n
    while lastv > 1:
        if v_inc == 1:
            vj = A[(v_r + lastv - 1) + v_c * lda]
        else:
            vj = A[v_r + (v_c + lastv - 1) * lda]
        if vj != 0.0:
            break
        lastv -= 1
    lastc = 0
    for i in range(r0, r1):
        for j in range(c0, c0 + lastv):
            if A[i + j * lda] != 0.0:
                lastc = i - r0 + 1
    if lastc == 0:
        return
    if lastv == 1:
        for i in range(r0, r0 + lastc):
            A[i + c0 * lda] *= 1.0 - tau
        return
    work = [0.0] * lastc
    for ii in range(lastc):
        i = r0 + ii
        s = A[i + c0 * lda]
        for jj in range(1, lastv):
            j = c0 + jj
            if v_inc == 1:
                vj = A[(v_r + jj) + v_c * lda]
            else:
                vj = A[v_r + (v_c + jj) * lda]
            s += A[i + j * lda] * vj
        work[ii] = s
    for ii in range(lastc):
        i = r0 + ii
        A[i + c0 * lda] -= tau * work[ii]
    for jj in range(1, lastv):
        j = c0 + jj
        if v_inc == 1:
            vj = A[(v_r + jj) + v_c * lda]
        else:
            vj = A[v_r + (v_c + jj) * lda]
        for ii in range(lastc):
            i = r0 + ii
            A[i + j * lda] -= tau * work[ii] * vj


def bidiag_svals(m, n, D, E):
    k = len(D)
    if m >= n:
        # upper bidiagonal B: diag D, superdiag E
        BtB_d = [0.0] * k
        BtB_e = [0.0] * (k - 1)
        for i in range(k):
            BtB_d[i] = D[i] ** 2 + (E[i - 1] ** 2 if i > 0 else 0.0)
        for i in range(k - 1):
            BtB_e[i] = D[i] * E[i]
    else:
        BtB_d = [0.0] * k
        BtB_e = [0.0] * (k - 1)
        for i in range(k):
            BtB_d[i] = D[i] ** 2 + (E[i - 1] ** 2 if i > 0 else 0.0)
        for i in range(k - 1):
            BtB_e[i] = D[i] * E[i]
    w = np.linalg.eigvalsh(np.diag(BtB_d) + np.diag(BtB_e, 1) + np.diag(BtB_e, -1))
    return np.sort(np.sqrt(np.maximum(w, 0)))[::-1]


def to_col(A):
    return np.ravel(A, order="F").tolist()


def test_case(m, n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((m, n))
    ref = np.linalg.svd(M, compute_uv=False)
    A = to_col(M.copy())
    lda = m
    D, E, _, _, _ = dgebd2_fixed(m, n, A, lda)
    got = bidiag_svals(m, n, D, E)
    err = np.max(np.abs(got - ref))
    print(f"{m}x{n} seed={seed} max_err={err:.2e} D={D} E={E}")
    assert err < 1e-10, err


if __name__ == "__main__":
    for m, n, s in [(2, 2, 0), (3, 2, 1), (4, 3, 2), (5, 3, 3), (8, 5, 4), (2, 3, 5), (3, 4, 6)]:
        test_case(m, n, s)
    print("all ok")
