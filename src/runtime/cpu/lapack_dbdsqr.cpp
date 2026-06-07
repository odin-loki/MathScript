#include "ms/cpu/lapack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ms::cpu::lapack {

namespace {

constexpr double kMeigth = -0.125;
constexpr int kMaxItr = 6;

void dlartg(double f, double g, double& cs, double& sn, double& r) {
    if (g == 0.0) {
        cs = 1.0;
        sn = 0.0;
        r = f;
        return;
    }
    if (f == 0.0) {
        cs = 0.0;
        sn = 1.0;
        r = g;
        return;
    }
    if (std::abs(g) > std::abs(f)) {
        const double tt = f / g;
        sn = 1.0 / std::hypot(1.0, tt);
        cs = sn * tt;
        r = g / std::copysign(sn, g);
    } else {
        const double tt = g / f;
        cs = 1.0 / std::hypot(1.0, tt);
        sn = cs * tt;
        r = f / std::copysign(cs, f);
    }
}

void dlas2(double f, double g, double h, double& ssmin, double& ssmax) {
    const double fa = std::abs(f);
    const double ga = std::abs(g);
    const double ha = std::abs(h);
    const double fhmn = std::fmin(fa, ha);
    const double fhmx = std::fmax(fa, ha);

    if (fhmn == 0.0) {
        ssmin = 0.0;
        ssmax = (fhmx == 0.0) ? ga : std::fmax(fhmx, ga) * std::hypot(1.0, std::fmin(fhmx, ga) / std::fmax(fhmx, ga));
        return;
    }

    if (ga < fhmx) {
        const double as = 1.0 + fhmn / fhmx;
        const double at = (fhmx - fhmn) / fhmx;
        const double au = std::pow(ga / fhmx, 2);
        const double c = 2.0 / (std::sqrt(as * as + au) + std::sqrt(at * at + au));
        ssmin = fhmn * c;
        ssmax = fhmx / c;
    } else {
        const double au = fhmx / ga;
        if (au == 0.0) {
            ssmin = (fhmn * fhmx) / ga;
            ssmax = ga;
        } else {
            const double as = 1.0 + fhmn / fhmx;
            const double at = (fhmx - fhmn) / fhmx;
            const double c = 1.0 / (std::sqrt(1.0 + std::pow(as * au, 2)) + std::sqrt(1.0 + std::pow(at * au, 2)));
            ssmin = 2.0 * (fhmn * c) * au;
            ssmax = ga / (2.0 * c);
        }
    }
}

void dlasv2(
    double f,
    double g,
    double h,
    double& ssmin,
    double& ssmax,
    double& snr,
    double& csr,
    double& snl,
    double& csl) {
    const double eps = std::numeric_limits<double>::epsilon();
    double ft = f;
    double fa = std::abs(ft);
    double ht = h;
    double ha = std::abs(ht);
    int pmax = 1;
    bool swap = ha > fa;
    if (swap) {
        pmax = 3;
        std::swap(ft, ht);
        std::swap(fa, ha);
    }

    const double gt = g;
    const double ga = std::abs(gt);
    if (ga == 0.0) {
        ssmin = ha;
        ssmax = fa;
        csl = 1.0;
        csr = 1.0;
        snl = 0.0;
        snr = 0.0;
        return;
    }

    bool gasmal = true;
    if (ga > fa) {
        pmax = 2;
        if (fa / ga < eps) {
            gasmal = false;
            ssmax = ga;
            if (ha > 1.0) {
                ssmin = fa / (ga / ha);
            } else {
                ssmin = (fa / ga) * ha;
            }
            csl = 1.0;
            snl = ht / gt;
            csr = 1.0;
            snr = ft / gt;
        }
    }

    double clt = 0.0;
    double crt = 0.0;
    double slt = 0.0;
    double srt = 0.0;

    if (gasmal) {
        const double d = fa - ha;
        const double l = (d == fa) ? 1.0 : d / fa;
        const double m = gt / ft;
        const double t = 2.0 - l;
        const double mm = m * m;
        const double tt = t * t;
        const double s = std::sqrt(tt + mm);
        const double r = (l == 0.0) ? std::abs(m) : std::sqrt(l * l + mm);
        const double a = 0.5 * (s + r);
        ssmin = ha / a;
        ssmax = fa * a;
        double tval = 0.0;
        if (mm == 0.0) {
            if (l == 0.0) {
                tval = std::copysign(2.0, ft) * std::copysign(1.0, gt);
            } else {
                tval = gt / std::copysign(d, ft) + m / t;
            }
        } else {
            tval = (m / (s + t) + m / (r + l)) * (1.0 + a);
        }
        const double lval = std::sqrt(tval * tval + 4.0);
        crt = 2.0 / lval;
        srt = tval / lval;
        clt = (crt + srt * m) / a;
        slt = (ht / ft) * srt / a;
    }

    if (swap) {
        csl = srt;
        snl = crt;
        csr = slt;
        snr = clt;
    } else {
        csl = clt;
        snl = slt;
        csr = crt;
        snr = srt;
    }

    double tsign = 1.0;
    if (pmax == 1) {
        tsign = std::copysign(1.0, csr) * std::copysign(1.0, csl) * std::copysign(1.0, f);
    } else if (pmax == 2) {
        tsign = std::copysign(1.0, snr) * std::copysign(1.0, csl) * std::copysign(1.0, g);
    } else {
        tsign = std::copysign(1.0, snr) * std::copysign(1.0, snl) * std::copysign(1.0, h);
    }
    ssmax = std::copysign(ssmax, tsign);
    ssmin = std::copysign(ssmin, tsign * std::copysign(1.0, f) * std::copysign(1.0, h));
}

void drot_rows(int row1, int row2, double cs, double sn, double* VT, int ldvt, int ncols) {
    for (int j = 0; j < ncols; ++j) {
        const double x = VT[row1 * ldvt + j];
        const double y = VT[row2 * ldvt + j];
        VT[row1 * ldvt + j] = cs * x + sn * y;
        VT[row2 * ldvt + j] = -sn * x + cs * y;
    }
}

void drot_cols(int col1, int col2, double cs, double sn, double* U, int ldu, int nrows) {
    for (int r = 0; r < nrows; ++r) {
        const double x = U[col1 * ldu + r];
        const double y = U[col2 * ldu + r];
        U[col1 * ldu + r] = cs * x + sn * y;
        U[col2 * ldu + r] = -sn * x + cs * y;
    }
}

void dlasr_left_v_forward(int nrows, int ncols, int row_off, const double* c, const double* s, double* A, int lda) {
    if (nrows <= 1) {
        return;
    }
    for (int j = 0; j < nrows - 1; ++j) {
        const double cs = c[j];
        const double sn = s[j];
        if (cs == 1.0 && sn == 0.0) {
            continue;
        }
        const int r0 = row_off + j;
        const int r1 = row_off + j + 1;
        for (int col = 0; col < ncols; ++col) {
            const double temp = A[r1 * lda + col];
            A[r1 * lda + col] = cs * temp - sn * A[r0 * lda + col];
            A[r0 * lda + col] = sn * temp + cs * A[r0 * lda + col];
        }
    }
}

void dlasr_left_v_backward(int nrows, int ncols, int row_off, const double* c, const double* s, double* A, int lda) {
    if (nrows <= 1) {
        return;
    }
    for (int j = nrows - 2; j >= 0; --j) {
        const double cs = c[j];
        const double sn = s[j];
        if (cs == 1.0 && sn == 0.0) {
            continue;
        }
        const int r0 = row_off + j;
        const int r1 = row_off + j + 1;
        for (int col = 0; col < ncols; ++col) {
            const double temp = A[r1 * lda + col];
            A[r1 * lda + col] = cs * temp - sn * A[r0 * lda + col];
            A[r0 * lda + col] = sn * temp + cs * A[r0 * lda + col];
        }
    }
}

void dlasr_right_v_forward(int nrows, int ncols, int col_off, const double* c, const double* s, double* A, int lda) {
    if (ncols <= 1) {
        return;
    }
    for (int j = 0; j < ncols - 1; ++j) {
        const double cs = c[j];
        const double sn = s[j];
        if (cs == 1.0 && sn == 0.0) {
            continue;
        }
        const int c0 = col_off + j;
        const int c1 = col_off + j + 1;
        for (int r = 0; r < nrows; ++r) {
            const double temp = A[r + c1 * lda];
            A[r + c1 * lda] = cs * temp - sn * A[r + c0 * lda];
            A[r + c0 * lda] = sn * temp + cs * A[r + c0 * lda];
        }
    }
}

void dlasr_right_v_backward(int nrows, int ncols, int col_off, const double* c, const double* s, double* A, int lda) {
    if (ncols <= 1) {
        return;
    }
    for (int j = ncols - 2; j >= 0; --j) {
        const double cs = c[j];
        const double sn = s[j];
        if (cs == 1.0 && sn == 0.0) {
            continue;
        }
        const int c0 = col_off + j;
        const int c1 = col_off + j + 1;
        for (int r = 0; r < nrows; ++r) {
            const double temp = A[r + c1 * lda];
            A[r + c1 * lda] = cs * temp - sn * A[r + c0 * lda];
            A[r + c0 * lda] = sn * temp + cs * A[r + c0 * lda];
        }
    }
}

void init_identity_rows(int n, double* VT, int ldvt) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            VT[i * ldvt + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void init_identity_cols(int nrows, int ncols, double* U, int ldu) {
    for (int j = 0; j < ncols; ++j) {
        for (int r = 0; r < nrows; ++r) {
            U[j * ldu + r] = (r == j) ? 1.0 : 0.0;
        }
    }
}

void recompute_vt_from_u(
    int n,
    const double* d_orig,
    const double* e_orig,
    char uplo,
    const double* d_s,
    const double* U,
    int ldu,
    int u_rows,
    double* VT,
    int ldvt) {
    if (VT == nullptr || U == nullptr || d_orig == nullptr) {
        return;
    }

    for (int k = 0; k < n; ++k) {
        const double inv_sk = (d_s[k] != 0.0) ? (1.0 / d_s[k]) : 0.0;
        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int i = 0; i < n; ++i) {
                double bij = 0.0;
                if (uplo == 'U' || uplo == 'u') {
                    if (i == j) {
                        bij = d_orig[i];
                    } else if (j == i + 1) {
                        bij = e_orig[i];
                    }
                } else {
                    if (i == j) {
                        bij = d_orig[i];
                    } else if (i == j + 1) {
                        bij = e_orig[j];
                    }
                }
                acc += U[k * ldu + i] * bij;
            }
            VT[k * ldvt + j] = inv_sk * acc;
        }
    }
}

void sort_singular_values_decreasing(int n, double* d, double* U, int ldu, int u_rows, double* VT, int ldvt) {
    for (int i = 0; i < n - 1; ++i) {
        int isub = 0;
        double smin = d[0];
        for (int j = 1; j < n - i; ++j) {
            if (d[j] <= smin) {
                isub = j;
                smin = d[j];
            }
        }
        const int target = n - 1 - i;
        if (isub == target) {
            continue;
        }

        d[isub] = d[target];
        d[target] = smin;

        if (VT != nullptr) {
            for (int j = 0; j < n; ++j) {
                std::swap(VT[isub * ldvt + j], VT[target * ldvt + j]);
            }
        }
        if (U != nullptr) {
            for (int r = 0; r < u_rows; ++r) {
                std::swap(U[isub * ldu + r], U[target * ldu + r]);
            }
        }
    }
}

int dbdsqr_upper(
    int n,
    double* d,
    double* e,
    double* U,
    int ldu,
    double* VT,
    int ldvt,
    bool init_vectors = true,
    int nru = 0) {
    if (n <= 0) {
        return 0;
    }
    if (n == 1) {
        if (d[0] < 0.0) {
            d[0] = -d[0];
            if (VT != nullptr) {
                VT[0] = -VT[0];
            }
        }
        return 0;
    }

    const int nm1 = (std::max)(0, n - 1);
    const int nm12 = nm1 + nm1;
    const int nm13 = nm12 + nm1;

    const int u_rows = (nru > 0) ? nru : n;

    const bool rotate = (U != nullptr) || (VT != nullptr);
    std::vector<double> work;
    if (rotate) {
        work.assign(static_cast<std::size_t>(4 * nm1 + static_cast<std::size_t>(n)), 0.0);
        if (init_vectors) {
            if (U != nullptr) {
                init_identity_cols(n, n, U, ldu);
            }
            if (VT != nullptr) {
                init_identity_rows(n, VT, ldvt);
            }
        }
    }

    const double eps = std::numeric_limits<double>::epsilon();
    const double unfl = std::numeric_limits<double>::min();
    const double tolmul = std::fmax(10.0, std::fmin(100.0, std::pow(eps, kMeigth)));
    const double tol = tolmul * eps;

    double sminoa = std::abs(d[0]);
    if (sminoa != 0.0) {
        double mu = sminoa;
        for (int i = 1; i < n; ++i) {
            mu = std::abs(d[i]) * (mu / (mu + std::abs(e[i - 1])));
            sminoa = std::fmin(sminoa, mu);
            if (sminoa == 0.0) {
                break;
            }
        }
    }
    sminoa /= std::sqrt(static_cast<double>(n));
    const double thresh = std::fmax(tol * sminoa, static_cast<double>(kMaxItr * n * n) * unfl);

    const int maxitdivn = kMaxItr * n;
    int iterdivn = 0;
    int iter = -1;
    int oldll = -1;
    int oldm = -1;
    int idir = 0;
    int m = n;

    auto apply_block_updates = [&](int ll, int mblock, const double* cosr, const double* sinr, const double* cosl, const double* sinl, bool backward) {
        if (!rotate) {
            return;
        }
        const int nblock = mblock - ll;
        if (nblock <= 1) {
            return;
        }
        if (backward) {
            for (int j = nblock - 2; j >= 0; --j) {
                if (VT != nullptr) {
                    drot_rows(ll + j, ll + j + 1, cosl[j], sinl[j], VT, ldvt, n);
                }
                if (U != nullptr) {
                    drot_cols(ll + j, ll + j + 1, cosr[j], sinr[j], U, ldu, u_rows);
                }
            }
        } else {
            for (int j = 0; j < nblock - 1; ++j) {
                if (VT != nullptr) {
                    drot_rows(ll + j, ll + j + 1, cosr[j], sinr[j], VT, ldvt, n);
                }
                if (U != nullptr) {
                    drot_cols(ll + j, ll + j + 1, cosl[j], sinl[j], U, ldu, u_rows);
                }
            }
        }
    };

    for (;;) {
        if (m <= 1) {
            break;
        }

        if (iter >= n) {
            iter -= n;
            ++iterdivn;
            if (iterdivn >= maxitdivn) {
                return 1;
            }
        }

        if (std::abs(d[m - 1]) <= thresh) {
            d[m - 1] = 0.0;
        }

        int split_e = -1;
        for (int lll = 1; lll <= m - 1; ++lll) {
            const int row = m - lll;
            if (std::abs(d[row - 1]) <= thresh) {
                d[row - 1] = 0.0;
            }
            if (std::abs(e[row - 1]) <= thresh) {
                split_e = row - 1;
                e[split_e] = 0.0;
                break;
            }
        }

        if (split_e == m - 2) {
            --m;
            continue;
        }

        const int ll = (split_e >= 0) ? (split_e + 1) : 0;

        if (ll == m - 2) {
            double sigmn = 0.0;
            double sigmx = 0.0;
            double sinr = 0.0;
            double cosr = 0.0;
            double sinl = 0.0;
            double cosl = 0.0;
            dlasv2(d[m - 2], e[m - 2], d[m - 1], sigmn, sigmx, sinr, cosr, sinl, cosl);
            d[m - 2] = sigmx;
            e[m - 2] = 0.0;
            d[m - 1] = sigmn;
            if (rotate) {
                if (VT != nullptr) {
                    drot_rows(m - 2, m - 1, cosr, sinr, VT, ldvt, n);
                }
                if (U != nullptr) {
                    drot_cols(m - 2, m - 1, cosl, sinl, U, ldu, u_rows);
                }
            }
            m -= 2;
            continue;
        }

        if (ll > oldm || m < oldll) {
            idir = (std::abs(d[ll]) >= std::abs(d[m - 1])) ? 1 : 2;
        }

        if (idir == 1) {
            if (std::abs(e[m - 2]) <= tol * std::abs(d[m - 1])) {
                e[m - 2] = 0.0;
                continue;
            }
        } else if (std::abs(e[ll]) <= tol * std::abs(d[ll])) {
            e[ll] = 0.0;
            continue;
        }

        oldll = ll;
        oldm = m;

        double shift = 0.0;
        double r = 0.0;
        if (idir == 1) {
            dlas2(d[m - 2], e[m - 2], d[m - 1], shift, r);
        } else {
            dlas2(d[ll], e[ll], d[ll + 1], shift, r);
        }
        const double sll = (idir == 1) ? std::abs(d[ll]) : std::abs(d[m - 1]);
        if (sll > 0.0 && std::pow(shift / sll, 2) < eps) {
            shift = 0.0;
        }

        iter += m - ll;

        if (shift == 0.0) {
            if (idir == 1) {
                double cs = 1.0;
                double oldcs = 1.0;
                double oldsn = 0.0;
                for (int i = ll; i <= m - 2; ++i) {
                    double sn = 0.0;
                    dlartg(d[i] * cs, e[i], cs, sn, r);
                    if (i > ll) {
                        e[i - 1] = oldsn * r;
                    }
                    dlartg(oldcs * r, d[i + 1] * sn, oldcs, oldsn, d[i]);
                    if (rotate) {
                        work[static_cast<std::size_t>(i - ll)] = cs;
                        work[static_cast<std::size_t>(i - ll + nm1)] = sn;
                        work[static_cast<std::size_t>(i - ll + nm12)] = oldcs;
                        work[static_cast<std::size_t>(i - ll + nm13)] = oldsn;
                    }
                }
                const double h = d[m - 1] * cs;
                d[m - 1] = h * oldcs;
                e[m - 2] = h * oldsn;
                if (rotate) {
                    apply_block_updates(ll, m, work.data(), work.data() + nm1, work.data() + nm12, work.data() + nm13, false);
                }
            } else {
                double cs = 1.0;
                double oldcs = 1.0;
                double oldsn = 0.0;
                for (int i = m - 1; i >= ll + 1; --i) {
                    double sn = 0.0;
                    const double super = (i > 0) ? e[i - 1] : 0.0;
                    dlartg(d[i] * cs, super, cs, sn, r);
                    if (i < m - 1) {
                        e[i] = oldsn * r;
                    }
                    dlartg(oldcs * r, d[i - 1] * sn, oldcs, oldsn, d[i]);
                    if (rotate) {
                        work[static_cast<std::size_t>(i - ll - 1)] = cs;
                        work[static_cast<std::size_t>(i - ll - 1 + nm1)] = -sn;
                        work[static_cast<std::size_t>(i - ll - 1 + nm12)] = oldcs;
                        work[static_cast<std::size_t>(i - ll - 1 + nm13)] = -oldsn;
                    }
                }
                const double h = d[ll] * cs;
                d[ll] = h * oldcs;
                e[ll] = h * oldsn;
                if (rotate) {
                    apply_block_updates(ll, m, work.data(), work.data() + nm1, work.data() + nm12, work.data() + nm13, true);
                }
            }
        } else if (idir == 1) {
            double f = (std::abs(d[ll]) - shift) * (std::copysign(1.0, d[ll]) + shift / d[ll]);
            double g = e[ll];
            for (int i = ll; i <= m - 2; ++i) {
                double cosr = 0.0;
                double sinr = 0.0;
                double cosl = 0.0;
                double sinl = 0.0;
                dlartg(f, g, cosr, sinr, r);
                if (i > ll) {
                    e[i - 1] = r;
                }
                f = cosr * d[i] + sinr * e[i];
                e[i] = cosr * e[i] - sinr * d[i];
                g = sinr * d[i + 1];
                d[i + 1] = cosr * d[i + 1];
                dlartg(f, g, cosl, sinl, r);
                d[i] = r;
                f = cosl * e[i] + sinl * d[i + 1];
                d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                if (i < m - 2) {
                    g = sinl * e[i + 1];
                    e[i + 1] = cosl * e[i + 1];
                }
                if (rotate) {
                    work[static_cast<std::size_t>(i - ll)] = cosr;
                    work[static_cast<std::size_t>(i - ll + nm1)] = sinr;
                    work[static_cast<std::size_t>(i - ll + nm12)] = cosl;
                    work[static_cast<std::size_t>(i - ll + nm13)] = sinl;
                }
            }
            e[m - 2] = f;
            if (rotate) {
                apply_block_updates(ll, m, work.data(), work.data() + nm1, work.data() + nm12, work.data() + nm13, false);
            }
        } else {
            double f = (std::abs(d[m - 1]) - shift) * (std::copysign(1.0, d[m - 1]) + shift / d[m - 1]);
            double g = e[m - 2];
            for (int i = m - 1; i >= ll + 1; --i) {
                double cosr = 0.0;
                double sinr = 0.0;
                double cosl = 0.0;
                double sinl = 0.0;
                dlartg(f, g, cosr, sinr, r);
                if (i < m - 1) {
                    e[i] = r;
                }
                f = cosr * d[i] + sinr * ((i > 0) ? e[i - 1] : 0.0);
                if (i > 0) {
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                }
                g = sinr * d[i - 1];
                d[i - 1] = cosr * d[i - 1];
                dlartg(f, g, cosl, sinl, r);
                d[i] = r;
                f = cosl * ((i > 0) ? e[i - 1] : 0.0) + sinl * d[i - 1];
                d[i - 1] = cosl * d[i - 1] - sinl * ((i > 0) ? e[i - 1] : 0.0);
                if (i > ll) {
                    g = sinl * e[i - 2];
                    e[i - 2] = cosl * e[i - 2];
                }
                if (rotate) {
                    work[static_cast<std::size_t>(i - ll - 1)] = cosr;
                    work[static_cast<std::size_t>(i - ll - 1 + nm1)] = -sinr;
                    work[static_cast<std::size_t>(i - ll - 1 + nm12)] = cosl;
                    work[static_cast<std::size_t>(i - ll - 1 + nm13)] = -sinl;
                }
            }
            e[ll] = f;
            if (rotate) {
                apply_block_updates(ll, m, work.data(), work.data() + nm1, work.data() + nm12, work.data() + nm13, true);
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        if (d[i] < 0.0) {
            d[i] = -d[i];
            if (VT != nullptr) {
                for (int j = 0; j < n; ++j) {
                    VT[i * ldvt + j] = -VT[i * ldvt + j];
                }
            }
        } else if (d[i] == 0.0) {
            d[i] = 0.0;
        }
    }

    sort_singular_values_decreasing(n, d, U, ldu, u_rows, VT, ldvt);
    for (int i = 0; i < n - 1; ++i) {
        e[i] = 0.0;
    }
    return 0;
}

} // namespace

int dbdsqr(
    char uplo,
    int n,
    double* d,
    double* e,
    double* U,
    int ldu,
    double* VT,
    int ldvt,
    int nru,
    bool init_vectors) {
    if (n <= 0) {
        return 0;
    }
    if (d == nullptr || (n > 1 && e == nullptr)) {
        return 1;
    }

    const int u_rows = (nru > 0) ? nru : n;
    std::vector<double> d_orig(static_cast<std::size_t>(n));
    std::vector<double> e_orig(static_cast<std::size_t>((std::max)(0, n - 1)));
    for (int i = 0; i < n; ++i) {
        d_orig[static_cast<std::size_t>(i)] = d[i];
    }
    for (int i = 0; i < n - 1; ++i) {
        e_orig[static_cast<std::size_t>(i)] = e[i];
    }
    const char bidiag_uplo = (uplo == 'L' || uplo == 'l') ? 'L' : 'U';

    int status = 0;
    if (uplo == 'L' || uplo == 'l') {
        if (init_vectors) {
            if (U != nullptr) {
                init_identity_cols(n, n, U, ldu);
            }
            if (VT != nullptr) {
                init_identity_rows(n, VT, ldvt);
            }
        }

        std::vector<double> conv_c;
        std::vector<double> conv_s;
        if (U != nullptr) {
            conv_c.resize(static_cast<std::size_t>(n - 1));
            conv_s.resize(static_cast<std::size_t>(n - 1));
        }
        for (int i = 0; i < n - 1; ++i) {
            double cs = 0.0;
            double sn = 0.0;
            double r = 0.0;
            dlartg(d[i], e[i], cs, sn, r);
            d[i] = r;
            e[i] = sn * d[i + 1];
            d[i + 1] = cs * d[i + 1];
            if (U != nullptr) {
                conv_c[static_cast<std::size_t>(i)] = cs;
                conv_s[static_cast<std::size_t>(i)] = sn;
            }
        }
        if (U != nullptr) {
            dlasr_right_v_forward(n, n, 0, conv_c.data(), conv_s.data(), U, ldu);
        }
        status = dbdsqr_upper(n, d, e, U, ldu, VT, ldvt, init_vectors, nru);
    } else if (uplo == 'U' || uplo == 'u') {
        status = dbdsqr_upper(n, d, e, U, ldu, VT, ldvt, init_vectors, nru);
    } else {
        return 1;
    }

    if (status == 0 && U != nullptr && VT != nullptr) {
        recompute_vt_from_u(n, d_orig.data(), e_orig.data(), bidiag_uplo, d, U, ldu, u_rows, VT, ldvt);
    }
    return status;
}

} // namespace ms::cpu::lapack
