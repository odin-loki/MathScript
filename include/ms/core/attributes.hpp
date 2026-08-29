#pragma once

/// Portable forced-inline for hot BLAS/LAPACK kernels.
/// MSVC: __forceinline. GCC/Clang: inline always_inline.
#if defined(_MSC_VER)
#define MS_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define MS_FORCEINLINE inline __attribute__((always_inline))
#else
#define MS_FORCEINLINE inline
#endif
