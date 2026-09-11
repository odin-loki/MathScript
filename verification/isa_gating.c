/* The ISA gate, extracted so a model checker can enumerate every input.
 *
 * The property: no wide path is ever reported without the OS state it needs.
 * CBMC explores all 2^k assignments of the CPUID and XCR0 bits rather than the
 * one combination this host happens to have. */
#include <assert.h>
#include <stdint.h>
#include "assume.h"

/* Declared so CBMC gives them the right type; an undeclared nondet_uint64_t is
 * assumed to return int and the harness then reports its own conversion. */
int nondet_int(void);
uint64_t nondet_uint64_t(void);

typedef struct { int sse2, sse41, avx, avx2, fma, avx512f; } feats;

#define XCR0_YMM 0x6ULL
#define XCR0_ZMM 0xE6ULL

static feats detect(int cpu_sse2, int cpu_sse41, int cpu_avx, int cpu_fma,
                    int cpu_avx2, int cpu_avx512, int osxsave, uint64_t xcr0) {
    feats f = {0, 0, 0, 0, 0, 0};
    f.sse2 = cpu_sse2;
    f.sse41 = cpu_sse41;
    int os_ymm = osxsave && ((xcr0 & XCR0_YMM) == XCR0_YMM);
    int os_zmm = osxsave && ((xcr0 & XCR0_ZMM) == XCR0_ZMM);
    f.avx = cpu_avx && os_ymm;
    f.fma = cpu_fma && os_ymm;
    f.avx2 = cpu_avx2 && os_ymm;
    f.avx512f = cpu_avx512 && os_zmm;
    return f;
}

int main(void) {
    int cpu_sse2 = nondet_int(), cpu_sse41 = nondet_int(), cpu_avx = nondet_int();
    int cpu_fma = nondet_int(), cpu_avx2 = nondet_int(), cpu_avx512 = nondet_int();
    int osxsave = nondet_int();
    uint64_t xcr0 = nondet_uint64_t();

    MS_ASSUME(cpu_sse2 == 0 || cpu_sse2 == 1);
    MS_ASSUME(cpu_sse41 == 0 || cpu_sse41 == 1);
    MS_ASSUME(cpu_avx == 0 || cpu_avx == 1);
    MS_ASSUME(cpu_fma == 0 || cpu_fma == 1);
    MS_ASSUME(cpu_avx2 == 0 || cpu_avx2 == 1);
    MS_ASSUME(cpu_avx512 == 0 || cpu_avx512 == 1);
    MS_ASSUME(osxsave == 0 || osxsave == 1);

    feats f = detect(cpu_sse2, cpu_sse41, cpu_avx, cpu_fma, cpu_avx2, cpu_avx512,
                     osxsave, xcr0);

    /* 1. Nothing wide is reported unless the OS saves the registers it uses. */
    if (f.avx)     assert(osxsave && (xcr0 & XCR0_YMM) == XCR0_YMM);
    if (f.avx2)    assert(osxsave && (xcr0 & XCR0_YMM) == XCR0_YMM);
    if (f.fma)     assert(osxsave && (xcr0 & XCR0_YMM) == XCR0_YMM);
    if (f.avx512f) assert(osxsave && (xcr0 & XCR0_ZMM) == XCR0_ZMM);

    /* 2. AVX-512 implies the YMM state too: 0xE6 has 0x6 as a subset. This is the
     *    property that keeps the reported set internally consistent. */
    if (f.avx512f) assert((xcr0 & XCR0_YMM) == XCR0_YMM);

    /* 3. Nothing is invented: a path is only reported if the CPU has it. */
    if (f.avx)     assert(cpu_avx);
    if (f.avx2)    assert(cpu_avx2);
    if (f.fma)     assert(cpu_fma);
    if (f.avx512f) assert(cpu_avx512);

    /* 4. The reported set is never wider than the OS agreed to. */
    if (!osxsave) {
        assert(!f.avx && !f.avx2 && !f.fma && !f.avx512f);
    }
    return 0;
}
