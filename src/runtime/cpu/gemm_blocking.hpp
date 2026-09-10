// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

// Packing and cache blocking shared by every dgemm micro-kernel.
//
// The AVX-512 kernel this replaces had a 4x8 micro-kernel and nothing above it,
// and it read B through eight strided scalar loads per vector on every iteration
// of the inner loop -- a gather written as `_mm512_set_pd(B[(j0+7)*ldb + p], ...)`.
// Both problems are structural rather than local: a micro-kernel with no blocking
// above it stops scaling as soon as the operands leave cache, and a gather in the
// innermost loop pays its cost k times per tile instead of once.
//
// The fix is the Goto/BLIS decomposition, which exists precisely because these two
// problems are the whole difficulty of writing a fast gemm:
//
//   for each column block of C (NC)         B block streams from L3
//     for each depth block (KC)             pack B once into contiguous panels
//       for each row block of C (MC)        pack A once, resident in L2
//         for each NR column panel
//           for each MR row panel           micro-kernel, all loads contiguous
//
// Everything here is scalar C++ so it can be compiled at baseline ISA and included
// from a translation unit built with -mavx2 or -mavx512f without dragging vector
// instructions into code that runs before the ISA check.
//
// Matrices are column-major throughout: A[p*lda + i], B[j*ldb + p], C[j*ldc + i].

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace ms::cpu::blas::detail {

/// 64-byte aligned scratch, sized once per gemm call.
///
/// The panels are read with aligned vector loads in the micro-kernels, and an
/// unaligned buffer would turn every one of those into a fault or a slow path
/// depending on the instruction chosen.
///
/// The library is built with -fno-exceptions, so a failed allocation cannot be
/// thrown and must not be ignored. `get()` returns nullptr and the caller falls
/// back to an unpacked loop: a slower correct answer beats both a null dereference
/// and an abort inside a numerical routine.
class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t doubles) : size_(doubles) {
        if (doubles == 0) {
            return;
        }
        const std::size_t bytes = ((doubles * sizeof(double)) + 63U) & ~std::size_t{63};
#if defined(_MSC_VER)
        data_ = static_cast<double*>(_aligned_malloc(bytes, 64));
#else
        data_ = static_cast<double*>(std::aligned_alloc(64, bytes));
#endif
    }
    ~AlignedBuffer() {
        if (data_ == nullptr) {
            return;
        }
#if defined(_MSC_VER)
        _aligned_free(data_);
#else
        std::free(data_);
#endif
    }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    double* get() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    double* data_ = nullptr;
    std::size_t size_ = 0;
};

/// Cache blocking sizes.
///
/// Chosen so the packed A block fits L2 and the packed B block fits L3, which is
/// what the decomposition is for. They are deliberately parameters rather than
/// constants: the right values are a property of the machine, and the honest
/// position is that these are reasonable defaults measured on nothing in
/// particular, not tuned numbers.
struct BlockSizes {
    int mc;
    int kc;
    int nc;
};

constexpr int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

/// Pack a block of A into row panels of MR.
///
/// A is column-major, so the MR values a panel needs for one depth index are
/// already adjacent in memory and this is a run of copies rather than a
/// transpose. Rows past `m` are zero-filled: the micro-kernel computes the full
/// MR lanes regardless and the driver discards the surplus, but leaving them as
/// whatever the allocator returned would let a stray infinity or NaN pattern out
/// of an unused lane on some future kernel that reduces across them.
template <int MR>
void pack_a(const double* __restrict A, int lda, int ic, int mc, int pc, int kc,
            int m, double* __restrict Apack) {
    const std::size_t lda_u = static_cast<std::size_t>(lda);
    double* dst = Apack;
    for (int ip = 0; ip < mc; ip += MR) {
        const int rows = (mc - ip < MR) ? (mc - ip) : MR;
        const int global_i = ic + ip;
        const int valid = (global_i + rows > m) ? (m - global_i) : rows;
        for (int p = 0; p < kc; ++p) {
            const double* src = A + static_cast<std::size_t>(pc + p) * lda_u +
                                static_cast<std::size_t>(global_i);
            int i = 0;
            for (; i < valid; ++i) {
                dst[i] = src[i];
            }
            for (; i < MR; ++i) {
                dst[i] = 0.0;
            }
            dst += MR;
        }
    }
}

/// Pack a block of B into column panels of NR.
///
/// This is the gather, moved. B is column-major, so collecting NR consecutive
/// columns at one depth index is a strided read -- exactly what `load_b_panel`
/// used to do inside the innermost loop. Done here it happens once per (panel,
/// depth) instead of once per micro-kernel iteration, and the inner loop reads
/// the result sequentially.
template <int NR>
void pack_b(const double* __restrict B, int ldb, int jc, int nc, int pc, int kc,
            int n, double* __restrict Bpack) {
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    double* dst = Bpack;
    for (int jp = 0; jp < nc; jp += NR) {
        const int cols = (nc - jp < NR) ? (nc - jp) : NR;
        const int global_j = jc + jp;
        const int valid = (global_j + cols > n) ? (n - global_j) : cols;
        for (int p = 0; p < kc; ++p) {
            const std::size_t row = static_cast<std::size_t>(pc + p);
            int j = 0;
            for (; j < valid; ++j) {
                dst[j] = B[static_cast<std::size_t>(global_j + j) * ldb_u + row];
            }
            for (; j < NR; ++j) {
                dst[j] = 0.0;
            }
            dst += NR;
        }
    }
}

/// C += alpha * A * B, with no packing and no blocking.
///
/// The fallback when the packing buffers cannot be allocated. Same result, same
/// accumulation order as a plain rank-1 update; only the speed differs.
inline void gemm_unpacked(int m, int n, int k, double alpha,
                          const double* __restrict A, int lda,
                          const double* __restrict B, int ldb,
                          double* __restrict C, int ldc) {
    const std::size_t lda_u = static_cast<std::size_t>(lda);
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    for (int j = 0; j < n; ++j) {
        double* c_col = C + static_cast<std::size_t>(j) * ldc_u;
        for (int p = 0; p < k; ++p) {
            const double bpj = B[static_cast<std::size_t>(j) * ldb_u +
                                 static_cast<std::size_t>(p)];
            if (bpj == 0.0) {
                continue;
            }
            const double scale = alpha * bpj;
            const double* a_col = A + static_cast<std::size_t>(p) * lda_u;
            for (int i = 0; i < m; ++i) {
                c_col[i] += scale * a_col[i];
            }
        }
    }
}

/// The blocked loop nest.
///
/// `micro` computes one MR x NR tile and accumulates alpha * (Apanel * Bpanel)
/// into the C tile it is handed:
///
///     micro(kc, alpha, Apanel, Bpanel, Cptr, ldc)
///
/// C is only ever scaled by beta before this runs, so the micro-kernel always
/// accumulates and never has to know about beta.
///
/// Edge tiles are computed at full MR x NR into a local buffer and then folded
/// into C, so the kernel itself has no remainder handling and the ragged edges of
/// the matrix take the same arithmetic path as the interior. That costs a copy on
/// the boundary tiles and removes an entire class of off-by-one from the kernels.
template <int MR, int NR, typename Kernel>
void gemm_blocked(int m, int n, int k, double alpha,
                  const double* __restrict A, int lda,
                  const double* __restrict B, int ldb,
                  double* __restrict C, int ldc,
                  const BlockSizes& blk, Kernel&& micro) {
    const int mc_max = round_up(blk.mc, MR);
    const int nc_max = round_up(blk.nc, NR);

    AlignedBuffer apack(static_cast<std::size_t>(mc_max) * static_cast<std::size_t>(blk.kc));
    AlignedBuffer bpack(static_cast<std::size_t>(nc_max) * static_cast<std::size_t>(blk.kc));
    if (apack.get() == nullptr || bpack.get() == nullptr) {
        gemm_unpacked(m, n, k, alpha, A, lda, B, ldb, C, ldc);
        return;
    }

    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    alignas(64) double tile[static_cast<std::size_t>(MR) * static_cast<std::size_t>(NR)];

    for (int jc = 0; jc < n; jc += blk.nc) {
        const int nc = (n - jc < blk.nc) ? (n - jc) : blk.nc;
        const int nc_pad = round_up(nc, NR);

        for (int pc = 0; pc < k; pc += blk.kc) {
            const int kc = (k - pc < blk.kc) ? (k - pc) : blk.kc;
            pack_b<NR>(B, ldb, jc, nc_pad, pc, kc, n, bpack.get());

            for (int ic = 0; ic < m; ic += blk.mc) {
                const int mc = (m - ic < blk.mc) ? (m - ic) : blk.mc;
                const int mc_pad = round_up(mc, MR);
                pack_a<MR>(A, lda, ic, mc_pad, pc, kc, m, apack.get());

                for (int jp = 0; jp < nc_pad; jp += NR) {
                    const double* bpanel =
                        bpack.get() + static_cast<std::size_t>(jp / NR) *
                                          static_cast<std::size_t>(kc) *
                                          static_cast<std::size_t>(NR);
                    const int nr_eff = (nc - jp < NR) ? (nc - jp) : NR;
                    if (nr_eff <= 0) {
                        continue;
                    }

                    for (int ip = 0; ip < mc_pad; ip += MR) {
                        const double* apanel =
                            apack.get() + static_cast<std::size_t>(ip / MR) *
                                              static_cast<std::size_t>(kc) *
                                              static_cast<std::size_t>(MR);
                        const int mr_eff = (mc - ip < MR) ? (mc - ip) : MR;
                        if (mr_eff <= 0) {
                            continue;
                        }

                        double* cptr = C +
                                       static_cast<std::size_t>(jc + jp) * ldc_u +
                                       static_cast<std::size_t>(ic + ip);

                        if (mr_eff == MR && nr_eff == NR) {
                            micro(kc, alpha, apanel, bpanel, cptr, ldc);
                        } else {
                            std::memset(tile, 0, sizeof(tile));
                            micro(kc, alpha, apanel, bpanel, tile, MR);
                            for (int j = 0; j < nr_eff; ++j) {
                                double* dst = cptr + static_cast<std::size_t>(j) * ldc_u;
                                const double* src =
                                    tile + static_cast<std::size_t>(j) *
                                               static_cast<std::size_t>(MR);
                                for (int i = 0; i < mr_eff; ++i) {
                                    dst[i] += src[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Scale C by beta before any accumulation.
inline void scale_c(int m, int n, double beta, double* __restrict C, int ldc) {
    if (beta == 1.0) {
        return;
    }
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    for (int j = 0; j < n; ++j) {
        double* col = C + static_cast<std::size_t>(j) * ldc_u;
        if (beta == 0.0) {
            for (int i = 0; i < m; ++i) {
                col[i] = 0.0;
            }
        } else {
            for (int i = 0; i < m; ++i) {
                col[i] *= beta;
            }
        }
    }
}

} // namespace ms::cpu::blas::detail
