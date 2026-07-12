# MathScript — Master Build Plan
**Version:** 0.1 Draft  
**Author:** Odin  
**Status:** Active Development Specification  

This document is the complete, authoritative specification for the MathScript
HPC Computer Algebra System and IDE. It covers every architectural decision,
every module, every function, and every build consideration from first
principles to shipping.

**Related documents:** This file is the original design specification and phased delivery roadmap.
`CHANGELOG.md` is the detailed implementation history (wave-by-wave).
`README.md` is the project overview — consult each for architecture, what shipped, or onboarding respectively.

---

## Table of Contents

1. Foundational Decisions & System Identity
2. Complete Capability Inventory (1,934 functions, 31 modules)
3. Repository Structure & Build System
4. Architecture & Core Design
5. Memory System
6. Type System & Core Types
7. Plugin & Enforcement
8. SIMD Layer
9. CUDA Layer
10. Distributed Processing
11. Qt6 IDE
12. Frameworks Integration
13. Security Model
14. Testing Strategy
15. Phased Delivery Plan
16. Math Runtime Implementation Guide

---

# MathScript — Master Build Plan
## Part 1: Foundational Decisions & System Identity

---

## 1.1 Name & Identity

**Name:** MathScript

**Binary names:**

| Binary | Purpose |
|---|---|
| `mathscript` | Main IDE executable (GUI) |
| `mathscriptc` | Headless compiler / batch runner |
| `mathscript-repl` | Standalone REPL, no GUI |
| `mathscript-server` | Headless distributed compute node |
| `libmathscript.dll` / `.so` | Runtime shared library |
| `libmathscript.a` | Static link variant |
| `mathscript-plugin.dll` / `.so` | Clang enforcement plugin |

**C++ namespace:** `ms::` — all public API. Internal: `ms::internal::`. Frameworks: `ms::gria::`, `ms::izaac::`, `ms::cypha::`, `ms::cellai::`, `ms::axiom::`.

**Include prefix:** `#include <ms/...>`

**CMake namespace:** `MathScript::`

---

## 1.2 Language Standard

**Host C++ standard: C++23** — mandatory, no fallback.

| Feature | Where used |
|---|---|
| `std::expected<T,E>` | Entire error handling system |
| `std::mdspan` | Tensor view types, non-owning matrix slices |
| `std::flat_map`, `std::flat_set` | Symbol table, dispatch registry |
| Deducing `this` | CRTP expression templates |
| `std::stacktrace` | Error diagnostics |
| `std::print` / `std::format` | All diagnostic and output formatting |
| Explicit `this` in lambdas | Recursive lambdas in evaluator |
| `std::ranges` v2 | Algorithm composition |

**CUDA C++ standard: C++20** — nvcc trails by one standard. CUDA translation units are isolated behind a compilation firewall.

**Minimum compiler versions:**

| Compiler | Minimum | Role |
|---|---|---|
| Clang | 17.0 | Primary — everything |
| MSVC | 19.38 (VS 2022 17.8) | Windows host for CUDA |
| nvcc | CUDA 12.3 | CUDA compilation only |
| GCC | 13.2 | CI cross-validation only |

---

## 1.3 Platform Targets

### Tier 1 — Fully Supported

| Platform | Architecture |
|---|---|
| Windows 10 22H2+ | x86_64 |
| Windows 11 | x86_64 |
| Ubuntu 22.04 LTS | x86_64 |
| Ubuntu 24.04 LTS | x86_64 |

### Tier 2 — Supported, No Binary Releases

| Platform | Architecture | Notes |
|---|---|---|
| Windows Server 2022 | x86_64 | Headless server mode |
| RHEL 9 / Rocky 9 | x86_64 | HPC cluster deployment |
| Debian 12 | x86_64 | Server deployment |

### Tier 3 — Future

| Platform | Architecture |
|---|---|
| Linux aarch64 | Grace Hopper, AWS Graviton |
| macOS 14+ | arm64 (if demanded) |

### GPU Compute Targets

| Architecture | CUDA SM | Notes |
|---|---|---|
| Ampere (RTX 3090, A100) | sm_80 / sm_86 | Primary development |
| Ada Lovelace (RTX 4090) | sm_89 | Primary development |
| Hopper (H100) | sm_90 | HPC deployment |
| Volta (V100) | sm_70 | Legacy support |

No AMD/ROCm in initial scope. Dispatch layer designed with a HIP shim reserved for future phase.

---

## 1.4 Build System

**CMake 3.28+ / Ninja 1.11+** — no alternatives.

MSBuild explicitly rejected. Ninja is faster, platform-consistent, required for large CUDA builds.

**Windows:** Clang-cl for all non-CUDA code (Clang front-end, MSVC ABI). MSVC toolchain for CUDA host compilation. Windows SDK 10.0.22621.0+.

**Linux:** Clang 17+ from LLVM apt repository. lld as linker in all builds.

**Build configurations:**

| Config | Sanitizers | Optimisation | Assertions | Plugin |
|---|---|---|---|---|
| Debug | ASan + UBSan | O0 | On | On |
| RelWithDebInfo | UBSan | O2 | On | On |
| Release | UBSan (min) | O3 + LTO | Off | On |
| Shipping | None | O3 + LTO + PGO | Off | On |

**The plugin is unconditionally on in all configurations. It cannot be disabled by any build flag. This is enforced in CMake.**

**Toolchain files:**

```
cmake/toolchains/
├── windows-clang-cl.cmake
├── windows-cuda.cmake
├── linux-clang-x64.cmake
├── linux-clang-cuda.cmake
└── linux-clang-aarch64.cmake
```

---

## 1.5 Licensing

**Proprietary — All Rights Reserved.**

Copyright held by Odin. CLA required for all external contributions. GPL and LGPL dependencies explicitly rejected — even LGPL creates distribution obligations incompatible with proprietary commercial use.

**Dependency licence audit:**

| Dependency | Licence | Status |
|---|---|---|
| LLVM/Clang | Apache 2.0 + LLVM exception | ✅ Approved |
| Qt6 | Commercial | ✅ Requires commercial licence |
| SymEngine | MIT | ✅ Transitional |
| mimalloc | MIT | ✅ |
| oneTBB | Apache 2.0 | ✅ |
| Google Highway | Apache 2.0 | ✅ |
| CUDA Toolkit | NVIDIA EULA | ✅ |
| NCCL | BSD-3 | ✅ |
| OpenMPI | BSD-3 | ✅ |
| hwloc | BSD-2 | ✅ |
| GoogleTest | BSD-3 | ✅ Test only |
| OpenBLAS | BSD-3 | ✅ Transitional |
| FFTW3 | GPL-2 | ❌ REJECTED — own FFT |

**Note on Qt:** Qt6 Widgets requires a Qt Commercial licence for proprietary distribution. This is a real cost that must be budgeted before distribution.

---

## 1.6 Library Philosophy

Every third-party library is classified:

**Class A — Transitional:** Gets functionality working early. Will be replaced by own implementations. Examples: OpenBLAS, SymEngine.

**Class B — Infrastructure:** Toolchain or platform infrastructure not worth replacing. Examples: LLVM/Clang, CUDA Toolkit, MPI runtime, Qt6.

**Class C — Trivial:** Small, single-purpose, audited. Not worth replacing. Examples: mimalloc, GoogleTest, hwloc.

The goal: by Shipping phase, the math and algorithm core is entirely own code. Class B infrastructure is the only external dependency remaining.

---

## 1.7 C++ Enforcement Profile

The MathScript C++ Profile is a named, versioned document enforced by a Clang plugin at compile time and inside the REPL. It is not optional.

### Prohibited

**Memory:**
- `new` / `delete` — use `ms::make_unique`, `ms::make_shared`
- Raw owning pointers as return types or struct members
- `malloc`, `free`, `calloc`, `realloc`, `alloca`
- Variable-length arrays
- `reinterpret_cast` — only inside `[[ms::unsafe("reason")]]`

**Error handling:**
- `throw` expressions — exceptions disabled at compiler level
- `catch` clauses
- Discarding `[[nodiscard]]` `std::expected` return values

**Type safety:**
- C-style casts `(T)x`
- `const_cast`
- `reinterpret_cast` outside unsafe blocks
- Implicit narrowing conversions
- Mixed signed/unsigned arithmetic
- Uninitialized variable use

**Concurrency:**
- `std::thread` — use `std::jthread` only
- `mutex.lock()` / `mutex.unlock()` — use `std::scoped_lock`
- `volatile` for synchronisation — use `std::atomic`
- `thread.detach()`

**Other:**
- RTTI disabled — `-fno-rtti` — use `std::variant` + `std::visit`
- Exceptions disabled — `-fno-exceptions`
- Raw pointer arithmetic — all indexing is bounds-checked; unchecked requires `[[ms::unsafe]]`

### Required

- Functions returning nullable values use `std::optional<T>`
- Functions that can fail return `std::expected<T, ms::Error>`
- All unsafe operations annotated `[[ms::unsafe("justification")]]`
- Narrowing uses `ms::narrow<T>()` (checked) or `ms::truncate<T>()` (explicit)

### The Escape Hatch

```cpp
[[ms::unsafe("CUDA interop: reinterpret device pointer as float*")]]
float* ptr = reinterpret_cast<float*>(device_buf);
```

The plugin requires a non-empty justification string. All unsafe sites are collected into an audit report on every build.

---

## 1.8 Error Handling System

One mechanism. `std::expected<T, ms::Error>` everywhere. No exceptions, no error codes, no nullable returns without `std::optional`.

```cpp
namespace ms {

struct DimensionMismatch  { size_t got_r, got_c, exp_r, exp_c; };
struct SingularMatrix     { double condition_number; };
struct DeviceError        { int code; std::string_view msg; };
struct AllocationFailure  { size_t requested_bytes; };
struct ConvergenceFail    { size_t iterations; double residual; };
struct DomainError        { std::string_view function; std::string_view reason; };
struct DistributedError   { int rank; int mpi_code; };
struct SymbolicError      { std::string_view reason; };
struct IOError            { std::string_view path; std::string_view reason; };
struct ParseError         { size_t line, col; std::string_view msg; };
struct OverflowError      { std::string_view op; };
struct PluginViolation    { std::string_view rule; std::string_view location; };

using Error = std::variant<
    DimensionMismatch, SingularMatrix, DeviceError,
    AllocationFailure, ConvergenceFail, DomainError,
    DistributedError, SymbolicError, IOError,
    ParseError, OverflowError, PluginViolation
>;

template<typename T>
using Result = std::expected<T, Error>;

} // namespace ms
```

**Call patterns:**

```cpp
// Monadic pipeline (preferred for multi-step operations)
auto x = ms::invert(A)
    .and_then([](auto& Ai){ return ms::matmul(Ai, b); })
    .or_else([](auto& e)  { return ms::log_error(e);  });

// Explicit check
auto r = ms::solve(A, b);
if (!r) { handle(r.error()); return; }
auto& x = r.value();

// Unsafe assume — requires [[ms::unsafe]] annotation
[[ms::unsafe("A verified nonsingular by precondition check at line 47")]]
auto& x = ms::solve(A, b).assume_value();
```

---

## 1.9 Version Scheme

**Semantic versioning: MAJOR.MINOR.PATCH**

Version 0.x.x during all development phases. Version 1.0.0 is the first shipping release.

Version generated into a header at build time:

```cpp
// Generated: include/ms/version.hpp
namespace ms {
    constexpr int VERSION_MAJOR = 0;
    constexpr int VERSION_MINOR = 3;
    constexpr int VERSION_PATCH = 0;
    constexpr std::string_view VERSION_STRING = "0.3.0";
    constexpr std::string_view BUILD_COMMIT   = "a1b2c3d4";
    constexpr std::string_view BUILD_DATE     = "2025-04-11";
}
```

---

## 1.10 Resolved Decisions Summary

| Decision | Resolution |
|---|---|
| Name | MathScript |
| C++ standard | C++23 (CUDA: C++20) |
| Primary compiler | Clang 17+ |
| Windows compiler | Clang-cl (MSVC ABI) |
| Build system | CMake 3.28+ / Ninja |
| Platforms | Windows 10/11, Ubuntu 22.04/24.04 |
| GPU | NVIDIA CUDA sm_70+ |
| Namespace | `ms::` |
| Error handling | `std::expected` exclusively |
| Exceptions | Disabled globally |
| RTTI | Disabled globally |
| Licensing | Proprietary |
| Library policy | Transitional → own implementations |
| Qt licence | Commercial required |
| FFTW | Rejected (GPL) — own FFT |
| Repository | Monorepo, no git submodules |
| Unsafe escape | `[[ms::unsafe("reason")]]` |
| Version scheme | Semantic versioning |


# MathScript — Master Build Plan
## Part 2: Complete Capability Inventory

All functions MathScript will implement. Every domain. Own code throughout.
Reference sources: Octave-Forge, SciPy/NumPy, Mathematica, Maple, MATLAB,
Julia, R, GSL, NAG, IMSL, Boost.Math, LAPACK, ROOT, QuantLib, OpenFOAM, DLMF.

---

## Module Index

| Module | Namespace | Function Count |
|---|---|---|
| Numeric Types | `ms::core` | 45 |
| Arbitrary Precision | `ms::bignum` | 52 |
| Linear Algebra | `ms::linalg` | 98 |
| FFT & Spectral | `ms::fft` | 38 |
| Signal Processing | `ms::signal` | 94 |
| Statistics | `ms::stats` | 112 |
| Probability | `ms::prob` | 74 |
| Optimisation | `ms::optim` | 58 |
| ODE / DAE Solvers | `ms::ode` | 28 |
| PDE Solvers | `ms::pde` | 36 |
| Polynomial Arithmetic | `ms::poly` | 48 |
| Symbolic CAS | `ms::sym` | 86 |
| Special Functions | `ms::special` | 214 |
| Number Theory | `ms::numthy` | 68 |
| Combinatorics | `ms::combo` | 44 |
| Graph Theory | `ms::graph` | 76 |
| Computational Geometry | `ms::geo` | 64 |
| Image Processing | `ms::image` | 108 |
| Machine Learning | `ms::ml` | 92 |
| Cryptography | `ms::crypto` | 46 |
| Compression | `ms::compress` | 28 |
| Financial Mathematics | `ms::finance` | 62 |
| Quantum Mechanics | `ms::quantum` | 44 |
| Finite Element Method | `ms::fem` | 38 |
| Fluid Dynamics | `ms::cfd` | 34 |
| Control Theory | `ms::control` | 56 |
| Information Theory | `ms::info` | 32 |
| Complex Analysis | `ms::cplx` | 38 |
| Differential Geometry | `ms::diffgeo` | 42 |
| Tensor Analysis | `ms::tensor` | 36 |
| Topology | `ms::topo` | 24 |
| **Total** | | **~1,934** |

---

## 2.1 ms::core — Numeric Types

### Scalar Types

```
int8, int16, int32, int64, int128
uint8, uint16, uint32, uint64, uint128
float16, float32, float64, float80, float128
complex32, complex64, complex128
rational<T>          — exact rational arithmetic
dual<T>              — dual numbers for automatic differentiation
hypercomplex<T>      — quaternions
octonion<T>
fixed<N,F>           — fixed-point arithmetic
```

### Type Operations

```
narrow<T>(x)         — checked narrowing cast (error on loss)
truncate<T>(x)       — explicit truncating cast
widen<T>(x)          — widening cast (always safe)
bitcast<T>(x)        — type-punning inside [[ms::unsafe]]
is_nan(x)
is_inf(x)
is_finite(x)
is_normal(x)
signbit(x)
nextafter(x, y)
ulp(x)               — unit in last place
eps(T)               — machine epsilon for type T
huge(T)              — maximum finite value
tiny(T)              — minimum positive normal value
```

### Checked Arithmetic

```
checked_add(a, b)    — returns Result<T, OverflowError>
checked_sub(a, b)
checked_mul(a, b)
checked_div(a, b)
checked_mod(a, b)
checked_neg(a)
checked_abs(a)
checked_pow(a, n)
saturating_add(a, b) — clamps instead of overflowing
saturating_sub(a, b)
saturating_mul(a, b)
wrapping_add(a, b)   — explicit modular arithmetic
wrapping_sub(a, b)
wrapping_mul(a, b)
```

---

## 2.2 ms::bignum — Arbitrary Precision

Own implementation throughout. No GMP/MPFR dependency.

### Integer Arithmetic (BigInt)

```
BigInt(string, base)
BigInt(int64)
add(a, b)
sub(a, b)
mul(a, b)             — Karatsuba for medium n, Schönhage-Strassen for large
div(a, b)             — returns {quotient, remainder}
mod(a, b)
pow(a, n)             — fast exponentiation
pow_mod(a, e, m)      — modular exponentiation
gcd(a, b)             — Lehmer / Binary GCD
lcm(a, b)
extended_gcd(a, b)    — Bezout coefficients
abs(a)
neg(a)
cmp(a, b)
sign(a)
bit_length(a)
is_zero(a)
is_one(a)
is_even(a)
is_odd(a)
to_string(a, base)
from_string(s, base)
```

### Rational Arithmetic

```
Rational(BigInt num, BigInt den)
add, sub, mul, div
reduce(r)             — normalise to lowest terms
floor(r)
ceil(r)
round(r)
to_float<T>(r)
```

### Arbitrary Precision Float (APFloat)

```
APFloat(precision_bits)
add, sub, mul, div, sqrt
exp, log, sin, cos, tan
asin, acos, atan, atan2
sinh, cosh, tanh
pow
abs, neg
round, floor, ceil, trunc
to_string(precision)
pi(precision_bits)    — Chudnovsky algorithm
e(precision_bits)
```

### Arbitrary Precision Complex

```
APComplex(APFloat re, APFloat im)
add, sub, mul, div
abs, arg, conj
exp, log, sqrt
sin, cos, tan
pow
```

---

## 2.3 ms::linalg — Linear Algebra

### Construction

```
zeros(m, n)
ones(m, n)
eye(n)
rand(m, n)
randn(m, n)
linspace(a, b, n)
logspace(a, b, n)
diag(v)              — vector to diagonal matrix
diag(A)              — matrix to diagonal vector
tril(A, k)
triu(A, k)
repmat(A, m, n)
reshape(A, m, n)
flatten(A)
horzcat(A, B)
vertcat(A, B)
kron(A, B)           — Kronecker product
```

### Basic Operations

```
transpose(A)
ctranspose(A)        — conjugate transpose
trace(A)
det(A)
rank(A)
norm(A, p)           — p = 1, 2, Inf, Fro
cond(A, p)
null(A)
orth(A)
rref(A)              — reduced row echelon form
```

### Decompositions

```
lu(A)                — LU with partial pivoting
lu_full(A)           — full pivoting
qr(A)                — thin and full variants
qr_pivoted(A)
chol(A)              — Cholesky (symmetric positive definite)
chol_incomplete(A)   — incomplete Cholesky
ldl(A)               — LDL^T decomposition
svd(A)               — singular value decomposition
svd_thin(A)
svd_trunc(A, k)      — truncated SVD, rank-k approximation
schur(A)             — Schur decomposition
hess(A)              — upper Hessenberg form
bidiag(A)            — bidiagonalisation
tridiag(A)           — tridiagonalisation (symmetric)
```

### Eigenvalue Problems

```
eig(A)               — eigenvalues and eigenvectors
eig_sym(A)           — symmetric/Hermitian (DSYEV equivalent)
eig_gen(A, B)        — generalised eigenvalue problem
eig_partial(A, k)    — k largest/smallest eigenvalues
eigs(A, k)           — sparse eigenvalues (Krylov-Schur)
svds(A, k)           — sparse SVD
```

### Linear Solvers

```
solve(A, b)          — general square system
solve_sym(A, b)      — symmetric positive definite
solve_tri(A, b)      — triangular system
solve_banded(A, b)   — banded matrix
solve_sparse(A, b)   — sparse direct solver
lsq(A, b)            — least squares (full rank)
lsq_constrained(A, b, C, d)
lsq_nonneg(A, b)     — nonnegative least squares
pinv(A)              — Moore-Penrose pseudoinverse
```

### Matrix Functions

```
expm(A)              — matrix exponential
logm(A)              — matrix logarithm
sqrtm(A)             — matrix square root
funm(A, f)           — general matrix function
signm(A)             — matrix sign function
cosm(A)
sinm(A)
```

### Iterative Solvers

```
cg(A, b)             — conjugate gradient
bicg(A, b)           — biconjugate gradient
bicgstab(A, b)
gmres(A, b)
minres(A, b)
qmr(A, b)
lsqr(A, b)
lsmr(A, b)
tfqmr(A, b)
```

### Preconditioners

```
precond_ilu(A)       — incomplete LU
precond_ic(A)        — incomplete Cholesky
precond_diag(A)      — Jacobi / diagonal scaling
precond_ssor(A)
precond_amg(A)       — algebraic multigrid
```

---

## 2.4 ms::fft — FFT & Spectral

Own implementation: Cooley-Tukey, Bluestein (non-power-of-2), Rader (prime lengths), split-radix.

```
fft(x)               — 1D complex FFT
ifft(x)
rfft(x)              — real input FFT (output N/2+1 complex)
irfft(x, n)
fft2(A)              — 2D FFT
ifft2(A)
fftn(A)              — N-dimensional FFT
ifftn(A)
fftshift(x)          — shift zero-frequency to centre
ifftshift(x)
fftfreq(n, d)        — frequency axis
rfftfreq(n, d)
dct(x, type)         — discrete cosine transform types I–IV
idct(x, type)
dst(x, type)         — discrete sine transform types I–IV
idst(x, type)
dht(x)               — discrete Hartley transform
czt(x, m, w, a)      — chirp Z-transform
goertzel(x, f, fs)   — single-frequency DFT
nufft(x, t, f)       — non-uniform FFT
nudft(x, t, f)       — non-uniform DFT direct
stft(x, window, noverlap, nfft)   — short-time Fourier transform
istft(X, window, noverlap, nfft)
wigner(x)            — Wigner-Ville distribution
cwt(x, wavelet, scales)           — continuous wavelet transform
icwt(W, wavelet, scales)
dwt(x, wavelet, level)            — discrete wavelet transform
idwt(c, wavelet)
wavedec(x, wavelet, n)            — multilevel DWT
waverec(c, wavelet)
```

---

## 2.5 ms::signal — Signal Processing

### Filtering

```
filter(b, a, x)      — IIR/FIR filter (direct form II transposed)
filtfilt(b, a, x)    — zero-phase forward-backward filter
sosfilt(sos, x)      — second-order sections filter
lfilter(b, a, x)     — causal filter with initial conditions
medfilt(x, n)        — median filter
wiener(x, n)         — Wiener filter
savgol(x, window, poly)           — Savitzky-Golay filter
kalman(A, C, Q, R, x0)            — Kalman filter
```

### Filter Design

```
butter(n, Wn, type)  — Butterworth
cheby1(n, rp, Wn)    — Chebyshev type I
cheby2(n, rs, Wn)    — Chebyshev type II
ellip(n, rp, rs, Wn) — elliptic (Cauer)
bessel(n, Wn)        — Bessel/Thomson
firwin(n, cutoff)    — windowed FIR
firwin2(n, freq, gain)
remez(n, bands, desired)          — Parks-McClellan optimal FIR
firls(n, bands, desired)          — least-squares FIR
kaiserord(ripple, width)          — Kaiser window order estimation
cheb1ord, cheb2ord, ellipord, buttord
freqz(b, a, n, fs)   — frequency response
freqs(b, a, w)       — analogue frequency response
grpdelay(b, a)       — group delay
phasez(b, a)
impz(b, a)           — impulse response
stepz(b, a)          — step response
zplane(b, a)         — pole-zero plot data
sos2tf, tf2sos, tf2zp, zp2tf, tf2ss, ss2tf
```

### Resampling

```
resample(x, p, q)    — rational resampling
decimate(x, q)
interpolate(x, p)
upsample(x, n)
downsample(x, n)
resample_poly(x, p, q, window)
```

### Spectral Analysis

```
periodogram(x, window, nfft, fs)
welch(x, window, noverlap, nfft, fs)
bartlett(x, nfft, fs)
blackman_tukey(x, lags, fs)
music(x, p, n, fs)   — MUSIC algorithm
esprit(x, p, n, fs)  — ESPRIT algorithm
pburg(x, p, fs)      — Burg AR spectrum
pyulear(x, p, fs)    — Yule-Walker AR spectrum
cohere(x, y, window, noverlap, nfft, fs)   — coherence
tfestimate(x, y, window, noverlap, nfft)   — transfer function estimate
mscohere(x, y)       — magnitude squared coherence
cpsd(x, y)           — cross power spectral density
```

### Time-Frequency

```
spectrogram(x, window, noverlap, nfft, fs)
hilbert(x)           — analytic signal
envelope(x)
instantaneous_freq(x, fs)
instantaneous_phase(x)
emd(x)               — empirical mode decomposition
hht(x, fs)           — Hilbert-Huang transform
vmd(x, alpha, tau, K)             — variational mode decomposition
```

### Correlation & Convolution

```
conv(a, b)           — polynomial/linear convolution (FFT-based for large n)
conv2(A, B)          — 2D convolution
convn(A, B)          — N-D convolution
deconv(y, b)
xcorr(x, y, maxlag)  — cross-correlation
xcov(x, y, maxlag)   — cross-covariance
autocorr(x, maxlag)
```

### Modulation

```
ammod(x, fc, fs)     — AM modulation
amdemod(y, fc, fs)
fmmod(x, fc, fs, dev)
fmdemod(y, fc, fs, dev)
pmmod(x, fc, fs, dev)
pmdemod(y, fc, fs, dev)
qammod(x, M)         — QAM
qamdemod(y, M)
pskmod(x, M)
pskdemod(y, M)
```

---

## 2.6 ms::stats — Statistics

### Descriptive

```
mean(x, dim)
median(x, dim)
mode(x)
var(x, flag, dim)
std(x, flag, dim)
skewness(x)
kurtosis(x)
moment(x, n)
quantile(x, p)
iqr(x)
range(x)
mad(x)               — median absolute deviation
trimmed_mean(x, p)
winsorized_mean(x, p)
geometric_mean(x)
harmonic_mean(x)
rms(x)
```

### Correlation

```
corr(x, y, type)     — Pearson, Spearman, Kendall
cov(X)
partial_corr(X)
xcorr_matrix(X)
concordance(x, y)
distance_corr(x, y)
```

### Hypothesis Testing

```
ttest(x, mu)         — one-sample t-test
ttest2(x, y)         — two-sample t-test
ttest_paired(x, y)
ztest(x, mu, sigma)
ftest(x, y)
chi2test(O, E)       — chi-squared goodness of fit
chi2ind(X)           — chi-squared test of independence
kstest(x, cdf)       — Kolmogorov-Smirnov one-sample
ks2test(x, y)        — two-sample KS
lillietest(x)        — Lilliefors normality
jbtest(x)            — Jarque-Bera normality
shapiro_wilk(x)
anderson_darling(x)
bartlett_test(groups)
levene_test(groups)
fligner_test(groups)
ranksum(x, y)        — Wilcoxon rank-sum
signtest(x, mu)
signrank(x, y)       — Wilcoxon signed-rank
kruskal_wallis(groups)
friedman(X)
anova1(groups)
anova2(X)
anovan(X, groups)
manova(X, group)
```

### Linear Models

```
fitlm(X, y)          — ordinary least squares
fitglm(X, y, dist)   — generalised linear model
fitlme(X, y, Z)      — linear mixed-effects
fitnlm(X, y, f)      — nonlinear least squares
fitgam(X, y)         — generalised additive model
stepwise(X, y)       — stepwise regression
ridge(X, y, lambda)
lasso(X, y, lambda)
elasticnet(X, y, alpha, lambda)
robustfit(X, y)      — robust regression (M-estimators)
```

### Multivariate

```
pca(X)               — principal component analysis
ica(X, n)            — independent component analysis
fa(X, n)             — factor analysis
mds(D, dim)          — multidimensional scaling
tsne(X, dim)         — t-SNE
umap(X, dim)         — UMAP
lda(X, y)            — linear discriminant analysis
qda(X, y)            — quadratic discriminant analysis
canonical_corr(X, Y)
hotelling_t2(X, mu)
```

### Resampling & Nonparametric

```
bootstrap(x, stat, B)
jackknife(x, stat)
permtest(x, y, stat, B)
crossval(model, X, y, k)
monte_carlo(f, n)
importance_sampling(f, g, n)
```

### Time Series

```
acf(x, lags)         — autocorrelation function
pacf(x, lags)        — partial autocorrelation
arfit(x, p)          — AR model fitting
mafit(x, q)          — MA model fitting
armafit(x, p, q)
arimafit(x, p, d, q)
sarima(x, order, seasonal)
varfit(X, p)         — vector AR
garchfit(x, p, q)    — GARCH
arch_test(x)
ljung_box(x, lags)
dickey_fuller(x)
kpss_test(x)
forecast(model, h)
```

---

## 2.7 ms::prob — Probability Distributions

For each distribution: `pdf`, `cdf`, `icdf` (quantile), `rng`, `fit`, `mean`, `var`, `std`, `skewness`, `kurtosis`, `entropy`, `mgf`, `cf` (characteristic function).

### Continuous Distributions

```
Normal(mu, sigma)
TruncatedNormal(mu, sigma, a, b)
HalfNormal(sigma)
FoldedNormal(mu, sigma)
LogNormal(mu, sigma)
Cauchy(x0, gamma)
StudentT(nu)
Laplace(mu, b)
Logistic(mu, s)
Uniform(a, b)
Exponential(lambda)
Gamma(alpha, beta)
InverseGamma(alpha, beta)
Beta(alpha, beta)
BetaPrime(alpha, beta)
Chi2(k)
InverseChi2(k)
ScaledInverseChi2(nu, tau2)
F(d1, d2)
Weibull(lambda, k)
InverseWeibull(lambda, k)
ExtremeValue(mu, sigma)
Gumbel(mu, beta)
Frechet(alpha, s, m)
GeneralisedExtremeValue(xi, mu, sigma)
GeneralisedPareto(xi, mu, sigma)
Rayleigh(sigma)
Rice(nu, sigma)
Maxwell(a)
Nakagami(m, omega)
Pareto(alpha, xm)
LogLogistic(alpha, beta)
LogGamma(alpha, beta)
Kumaraswamy(a, b)
Arcsine(a, b)
PowerFunction(a, b)
Triangular(a, b, c)
PERT(a, b, c)
Irwin_Hall(n)
Bates(n)
Trapezoidal(a, b, c, d)
VonMises(mu, kappa)
WrappedCauchy(mu, gamma)
WrappedNormal(mu, sigma)
NormalInverseGaussian(mu, alpha, beta, delta)
GeneralisedHyperbolic(lambda, alpha, beta, delta, mu)
VarianceGamma(mu, alpha, beta, lambda)
NIG(mu, alpha, beta, delta)
SkewNormal(xi, omega, alpha)
StableDistribution(alpha, beta, c, delta)
Levy(mu, c)
```

### Discrete Distributions

```
Bernoulli(p)
Binomial(n, p)
NegativeBinomial(r, p)
Geometric(p)
Poisson(lambda)
Hypergeometric(N, K, n)
BetaBinomial(n, alpha, beta)
BetaNegBinomial(r, alpha, beta)
ZipfMandelbrot(n, q, s)
Zipf(s, n)
DiscreteUniform(a, b)
Categorical(p)
```

### Multivariate Distributions

```
MultivariateNormal(mu, Sigma)
MultivariateTStudent(nu, mu, Sigma)
DirichletMultinomial(alpha, n)
Dirichlet(alpha)
Multinomial(n, p)
Wishart(nu, V)
InverseWishart(nu, Psi)
MatrixNormal(M, U, V)
```

### Copulas

```
GaussianCopula(R)
TCopula(nu, R)
ClaytonCopula(theta)
GumbelCopula(theta)
FrankCopula(theta)
JoeCopula(theta)
BB1Copula(theta, delta)
```

### Kernel Density Estimation

```
kde(x, bandwidth, kernel)
kde2d(x, y, bandwidth)
kden(X, bandwidth)
```

---

## 2.8 ms::optim — Optimisation

### Unconstrained

```
bfgs(f, x0)
lbfgs(f, x0, m)
cg_fr(f, x0)         — Fletcher-Reeves CG
cg_pr(f, x0)         — Polak-Ribiere CG
newton(f, x0)        — Newton's method with Hessian
gauss_newton(f, x0)  — for least squares
levenberg_marquardt(f, x0)
nelder_mead(f, x0)   — derivative-free
pattern_search(f, x0)
trust_region(f, x0)
trust_region_exact(f, x0)
dogleg(f, x0)
adam(f, x0)          — Adam optimiser
rmsprop(f, x0)
sgd(f, x0)
coordinate_descent(f, x0)
```

### Constrained

```
sqp(f, x0, ceq, cineq)           — sequential quadratic programming
ipopt_interior(f, x0, bounds, constraints)
auglag(f, x0, constraints)       — augmented Lagrangian
penalty(f, x0, constraints)
active_set(f, x0, constraints)
```

### Linear Programming

```
lp(c, A, b, Aeq, beq, bounds)   — simplex + interior point
milp(c, A, b, intcon)            — mixed-integer LP
glp(c, A, b)                     — general LP
revised_simplex(c, A, b)
dual_simplex(c, A, b)
```

### Quadratic Programming

```
qp(H, c, A, b, Aeq, beq)
socp(c, A, b, cones)             — second-order cone
sdp(C, A, b)                     — semidefinite programming
```

### Global Optimisation

```
basin_hopping(f, x0)
differential_evolution(f, bounds)
simulated_annealing(f, x0)
genetic_algorithm(f, bounds)
pso(f, bounds)                   — particle swarm
cma_es(f, x0, sigma)             — CMA-ES
direct(f, bounds)                — DIRECT algorithm
bayesian_opt(f, bounds)          — Bayesian optimisation
grid_search(f, grid)
random_search(f, bounds, n)
```

### Nonlinear Equations

```
fsolve(f, x0)        — general nonlinear system
fzero(f, a, b)       — scalar root finding
bisection(f, a, b)
brentq(f, a, b)      — Brent's method
illinois(f, a, b)
muller(f, x0, x1, x2)
newton_raphson(f, df, x0)
secant(f, x0, x1)
halley(f, df, d2f, x0)
fixed_point(g, x0)
```

### Multi-Objective

```
nsga2(f, bounds, n_obj)
nsga3(f, bounds, n_obj)
moead(f, bounds, n_obj)
pareto_front(solutions)
```

---

## 2.9 ms::ode — ODE / DAE Solvers

### Explicit Methods

```
euler(f, t, y0)
midpoint(f, t, y0)
rk4(f, t, y0)
rk45(f, tspan, y0)   — Dormand-Prince adaptive
rk23(f, tspan, y0)   — Bogacki-Shampine
rk89(f, tspan, y0)   — high-order Dormand-Prince
rkf45(f, tspan, y0)  — Fehlberg
cashkarp(f, tspan, y0)
dop853(f, tspan, y0) — Hairer's DOP853
vern7(f, tspan, y0)  — Verner 7th order
vern9(f, tspan, y0)
```

### Implicit / Stiff Methods

```
ode15s(f, tspan, y0) — BDF 1–5 (stiff)
ode23s(f, tspan, y0) — modified Rosenbrock
ode23t(f, tspan, y0) — trapezoidal rule
ode23tb(f, tspan, y0)
radau(f, tspan, y0)  — Radau IIA
gauss_legendre(f, tspan, y0)
lobatto(f, tspan, y0)— IIIC
sdirk(f, tspan, y0)  — singly diagonal implicit RK
```

### DAE Solvers

```
dae_solve(f, g, tspan, y0, z0)   — index-1 DAE
ida(f, g, tspan, y0, yp0)        — SUNDIALS IDA style
```

### Delay Differential Equations

```
dde23(f, lags, history, tspan)
dde_solver(f, lags, history, tspan)
```

### Boundary Value Problems

```
bvp4c(f, bc, init)
bvp5c(f, bc, init)
shooting(f, bc, t, y0)
collocation(f, bc, mesh, order)
```

### Event Detection

```
ode_event(f, event, tspan, y0)   — stop/record/restart on events
```

---

## 2.10 ms::pde — PDE Solvers

### Finite Difference

```
fd_poisson(f, domain, bc)
fd_heat(f, domain, bc, tspan)
fd_wave(f, domain, bc, tspan)
fd_advection(f, domain, bc, tspan)
fd_diffusion_reaction(f, g, domain, bc, tspan)
crank_nicolson(f, domain, bc, tspan)
adi(f, domain, bc, tspan)        — alternating direction implicit
upwind(f, domain, bc, tspan)
lax_wendroff(f, domain, bc, tspan)
```

### Spectral Methods

```
spectral_poisson(f, domain, bc, n)
chebyshev_collocation(L, f, bc)
fourier_galerkin(f, domain, n)
```

### Finite Volume

```
fv_euler(domain, ic, bc, tspan)  — Euler equations
fv_navier_stokes(domain, ic, bc, tspan, nu)
godunov(f, domain, ic, bc, tspan)
muscl(f, domain, ic, bc, tspan)
weno5(f, domain, ic, bc, tspan)
```

### Mesh Generation

```
mesh1d(a, b, n)
mesh2d_rect(domain, nx, ny)
mesh2d_tri(boundary, h)          — Delaunay triangulation
mesh3d_tet(boundary, h)
refine_mesh(mesh, criterion)
```

### Interpolation on Meshes

```
interp_mesh(mesh, values, points)
```

---

## 2.11 ms::poly — Polynomial Arithmetic

```
Poly(coeffs)         — dense polynomial
SparsePoly(terms)    — sparse polynomial
MultiPoly(coeffs, vars)          — multivariate

add(p, q)
sub(p, q)
mul(p, q)            — FFT-based O(n log n) for large n
div(p, q)            — returns {quotient, remainder}
mod(p, q)
gcd(p, q)            — half-GCD algorithm O(n log^2 n)
lcm(p, q)
extended_gcd(p, q)   — Bezout for polynomials
pow(p, n)
compose(p, q)        — p(q(x))
deriv(p, n)          — nth derivative
integ(p, c)          — indefinite integral
eval(p, x)           — Horner's method
eval_multi(p, xs)    — multipoint evaluation (subproduct tree)
interp(xs, ys)       — polynomial interpolation
interp_newton(xs, ys)
interp_lagrange(xs, ys)
interp_hermite(xs, ys, dys)
roots(p)             — eigenvalue method
roots_companion(p)
companion(p)         — companion matrix
sylvester(p, q)      — Sylvester matrix
resultant(p, q)
discriminant(p)
sturm(p)             — Sturm chain (root counting)
sturm_count(p, a, b) — roots in interval
factor(p)            — factorisation over Q, R, C, Fp
factor_zz(p)         — over integers
squarefree(p)        — square-free decomposition
monic(p)             — normalise leading coefficient
reverse(p)
shift(p, a)          — p(x - a)
scale(p, a)          — p(a*x)
cheb_expand(f, n)    — Chebyshev expansion
cheb_eval(coeffs, x)
legendre_expand(f, n)
bernstein(n, i, x)   — Bernstein basis polynomial
spline_basis(knots, degree)
```

---

## 2.12 ms::sym — Symbolic CAS

### Symbolic Types

```
Sym(expr_string)     — symbolic variable / expression
SymMatrix(...)       — symbolic matrix
SymPoly(...)         — symbolic polynomial
SymSet(...)          — symbolic set
```

### Calculus

```
diff(f, x, n)        — nth derivative
pdiff(f, x)          — partial derivative
grad(f, vars)        — gradient vector
hessian(f, vars)     — Hessian matrix
jacobian(F, vars)    — Jacobian matrix
div(F, vars)         — divergence
curl(F, vars)        — curl (3D)
laplacian(f, vars)   — scalar Laplacian
biharmonic(f, vars)
directional_deriv(f, v, vars)
lie_deriv(f, v, vars)
integrate(f, x)      — indefinite integral
integrate(f, x, a, b)             — definite integral
integrate_multi(f, vars, limits)  — multiple integral
line_integral(f, curve, t)
surface_integral(f, surface, params)
```

### Algebraic Manipulation

```
simplify(expr)
simplify_full(expr)
expand(expr)
factor(expr)
collect(expr, x)
cancel(expr)
apart(expr, x)       — partial fractions
together(expr)
numer(expr)
denom(expr)
coeff(expr, x, n)    — coefficient of x^n
leading_coeff(expr, x)
total_degree(expr)
```

### Series & Limits

```
series(f, x, x0, n) — Taylor series
asymptotic(f, x, inf)
limit(f, x, x0)
limit(f, x, x0, dir)             — one-sided limit
residue(f, x, x0)                — complex residue
order(f, x, x0)     — leading order term
```

### Solving

```
solve(eq, x)         — algebraic equation(s)
solve_system(eqs, vars)
dsolve(ode, f, x)    — ODE symbolic solution
dsolve_system(odes, funcs, x)
pdesolve(pde, f, vars)
isolate(eq, x)       — rearrange for x
```

### Transforms

```
laplace(f, t, s)
ilaplace(F, s, t)
fourier(f, t, omega)
ifourier(F, omega, t)
ztransform(f, n, z)
iztransform(F, z, n)
mellin(f, t, s)
imellin(F, s, t)
hankel(f, r, k)
```

### Boolean & Set Algebra

```
sym_and, sym_or, sym_not, sym_xor
sym_implies, sym_iff
sym_forall, sym_exists
sym_union, sym_intersect, sym_diff, sym_complement
sym_subset, sym_superset
sym_in, sym_notin
```

### Code Generation

```
codegen_cpp(expr)    — generate C++ expression
codegen_cuda(expr)   — generate CUDA kernel expression
cse(exprs)           — common subexpression elimination
```

---

## 2.13 ms::special — Special Functions (Full DLMF Catalogue)

### Error Functions

```
erf(x)
erfc(x)
erfi(x)              — imaginary error function
erfcx(x)             — scaled complementary error function
erfinv(x)
erfcinv(x)
dawson(x)            — Dawson function F(x)
fresnel_S(x)
fresnel_C(x)
voigt(x, sigma, gamma)           — Voigt profile
pseudo_voigt(x, sigma, gamma, eta)
```

### Gamma & Related

```
gamma(z)             — complex gamma function
lgamma(x)            — log-gamma
rgamma(x)            — reciprocal gamma
gamma_inc(a, x)      — lower incomplete gamma
gamma_inc_upper(a, x)
gamma_inc_reg(a, x)  — regularised
gamma_inc_reg_upper(a, x)
gamma_ratio(a, b)
digamma(z)           — psi function
trigamma(z)          — psi^(1)(z)
polygamma(n, z)      — psi^(n)(z)
beta(a, b)
lbeta(a, b)
beta_inc(x, a, b)    — incomplete beta
beta_inc_reg(x, a, b)
pochhammer(a, n)     — rising factorial
falling_factorial(a, n)
bernoulli(n)         — Bernoulli number
bernoulli_poly(n, x) — Bernoulli polynomial
euler_num(n)
euler_poly(n, x)
```

### Bessel Functions

```
besselj(nu, x)       — J_nu(x) — first kind
bessely(nu, x)       — Y_nu(x) — second kind (Neumann)
besseli(nu, x)       — I_nu(x) — modified first kind
besselk(nu, x)       — K_nu(x) — modified second kind
besselh(nu, kind, x) — Hankel functions H^(1), H^(2)
spherical_jn(n, x)   — spherical Bessel j_n
spherical_yn(n, x)   — spherical Bessel y_n
spherical_in(n, x)   — spherical modified i_n
spherical_kn(n, x)   — spherical modified k_n
riccati_jn(n, x)     — Riccati-Bessel S_n
riccati_yn(n, x)     — Riccati-Bessel C_n
bessel_zero_jnu(nu, n)           — nth zero of J_nu
bessel_zero_ynu(nu, n)           — nth zero of Y_nu
```

### Airy Functions

```
airy_ai(x)           — Ai(x)
airy_bi(x)           — Bi(x)
airy_ai_deriv(x)     — Ai'(x)
airy_bi_deriv(x)     — Bi'(x)
airy_ai_zero(n)      — nth zero of Ai
airy_bi_zero(n)      — nth zero of Bi
```

### Struve & Related

```
struve_H(nu, x)      — H_nu(x)
struve_L(nu, x)      — L_nu(x) modified
anger_J(nu, x)       — J_nu(x) Anger function
weber_E(nu, x)       — E_nu(x) Weber function
```

### Kelvin Functions

```
kelvin_ber(nu, x)
kelvin_bei(nu, x)
kelvin_ker(nu, x)
kelvin_kei(nu, x)
```

### Parabolic Cylinder

```
pcf_U(a, x)          — U(a, x)
pcf_V(a, x)          — V(a, x)
pcf_W(a, x)          — W(a, x)
```

### Coulomb Wave Functions

```
coulomb_F(l, eta, x) — regular Coulomb wave function
coulomb_G(l, eta, x) — irregular
coulomb_CL(l, eta)   — Gamow factor
```

### Hypergeometric Functions

```
hyp1f1(a, b, x)      — Kummer's confluent 1F1
hyp2f1(a, b, c, x)   — Gauss hypergeometric 2F1
hyp0f1(b, x)         — 0F1
hyp1f2(a, b1, b2, x)
hyp2f0(a1, a2, b, x)
hyp3f0(a1, a2, a3, x)
hyperpfq(a_list, b_list, x)      — generalised pFq
whittaker_M(kappa, mu, x)
whittaker_W(kappa, mu, x)
tricomi_U(a, b, x)   — Tricomi's U
kummer_M(a, b, x)    — alias for 1F1
```

### Meijer G & Fox H

```
meijer_G(m, n, p, q, a_list, b_list, x)
fox_H(m, n, p, q, a_list, alpha_list, b_list, beta_list, x)
```

### Orthogonal Polynomials

```
legendre_P(n, x)     — Legendre P_n
legendre_Q(n, x)     — Legendre Q_n
legendre_Pm(n, m, x) — associated P_n^m
legendre_Qm(n, m, x) — associated Q_n^m
sph_harm(l, m, theta, phi)       — spherical harmonics Y_l^m
hermite_H(n, x)      — probabilist's Hermite He_n
hermite_He(n, x)     — physicist's Hermite H_n
laguerre_L(n, x)     — Laguerre L_n
laguerre_La(n, a, x) — generalised Laguerre L_n^a
chebyshev_T(n, x)    — Chebyshev T_n first kind
chebyshev_U(n, x)    — Chebyshev U_n second kind
chebyshev_V(n, x)    — Chebyshev V_n
chebyshev_W(n, x)    — Chebyshev W_n
gegenbauer_C(n, a, x)— Gegenbauer C_n^a
jacobi_P(n, a, b, x) — Jacobi P_n^(a,b)
zernike_R(n, m, r)   — Zernike radial
zernike_Z(n, m, r, theta)        — Zernike full
```

### Elliptic Integrals

```
ellipK(k)            — complete elliptic integral K(k)
ellipE(k)            — complete elliptic integral E(k)
ellipPi(n, k)        — complete elliptic integral Pi(n,k)
ellipF(phi, k)       — incomplete elliptic F(phi,k)
ellipE(phi, k)       — incomplete elliptic E(phi,k)
ellipPi(n, phi, k)   — incomplete elliptic Pi
ellipD(k)            — D(k) = (K-E)/k^2
ellipJ(u, k)         — Jacobi sn, cn, dn, am simultaneously
```

### Elliptic Functions

```
jacobi_sn(u, k)
jacobi_cn(u, k)
jacobi_dn(u, k)
jacobi_am(u, k)      — amplitude
jacobi_sc(u, k)
jacobi_sd(u, k)
jacobi_nd(u, k)
jacobi_nc(u, k)
jacobi_dc(u, k)
jacobi_cs(u, k)
jacobi_ns(u, k)
jacobi_ds(u, k)
jacobi_cd(u, k)
weierstrass_P(z, g2, g3)         — Weierstrass ℘ function
weierstrass_Pprime(z, g2, g3)
weierstrass_zeta(z, g2, g3)
weierstrass_sigma(z, g2, g3)
```

### Theta Functions

```
theta1(z, q)
theta2(z, q)
theta3(z, q)
theta4(z, q)
theta1_prime(z, q)
jacobi_theta(n, z, tau)          — nome parametrisation
```

### Zeta & L-Functions

```
zeta(s)              — Riemann zeta
zeta_hurwitz(s, a)   — Hurwitz zeta
lerch_phi(z, s, a)   — Lerch transcendent
eta_dirichlet(s)     — Dirichlet eta
beta_dirichlet(s)    — Dirichlet beta
lambda_dirichlet(s)
polylog(n, z)        — polylogarithm Li_n(z)
clausen(x)           — Clausen function Cl_2(x)
legendre_chi(n, z)   — Legendre chi function
lerch_zeta(x, a, s)
stieltjes_gamma(n)   — Stieltjes constants
```

### Exponential & Logarithmic Integrals

```
Ei(x)                — exponential integral Ei(x)
E1(x)                — E_1(x)
En(n, x)             — E_n(x)
Li(x)                — logarithmic integral Li(x)
li(x)                — li(x) = Li(x) — Li(2)
Si(x)                — sine integral
Ci(x)                — cosine integral
Shi(x)               — hyperbolic sine integral
Chi(x)               — hyperbolic cosine integral
```

### Lambert & Wright

```
lambertW(k, z)       — Lambert W, branch k
wright_W(z)          — Wright omega function
wright_phi(a, b, z)  — Wright function
```

### Mittag-Leffler

```
mittag_leffler(alpha, z)         — E_alpha(z)
mittag_leffler(alpha, beta, z)   — E_{alpha,beta}(z)
```

### Mathieu Functions

```
mathieu_a(n, q)      — characteristic values a_n
mathieu_b(n, q)      — characteristic values b_n
mathieu_ce(n, q, x)  — even Mathieu function
mathieu_se(n, q, x)  — odd Mathieu function
mathieu_Mc(n, q, x)  — modified first kind
mathieu_Ms(n, q, x)  — modified second kind
```

### Spheroidal Wave Functions

```
spheroidal_S1(n, m, c, x)        — prolate/oblate first kind
spheroidal_S2(n, m, c, x)        — second kind
spheroidal_lambda(n, m, c)       — eigenvalues
```

### Painlevé Transcendents

```
painleve1(x, init)   — P-I transcendent (numerical)
painleve2(x, init, alpha)        — P-II
painleve3(x, init, params)       — P-III
painleve4(x, init, params)       — P-IV
painleve5(x, init, params)       — P-V
painleve6(x, init, params)       — P-VI
```

### Heun Functions

```
heun_G(a, q, alpha, beta, gamma, delta, z)   — general Heun
heun_C(q, alpha, beta, gamma, delta, z)      — confluent Heun
heun_D(q, alpha, gamma, delta, z)            — doubly confluent
heun_B(q, alpha, beta, delta, z)             — biconfluent
heun_T(q, alpha, beta, gamma, z)             — triconfluent
```

### Combinatorial Special Functions

```
binomial(n, k)       — exact via bignum
multinomial(n, ks)
stirling1(n, k)      — unsigned Stirling numbers of first kind
stirling2(n, k)      — Stirling numbers of second kind
bell_num(n)          — Bell numbers
catalan_num(n)       — Catalan numbers
motzkin_num(n)
fubini_num(n)        — ordered Bell / Fubini numbers
```

---

## 2.14 ms::numthy — Number Theory

```
isprime(n)           — Miller-Rabin + BPSW
nextprime(n)
prevprime(n)
primes(n)            — sieve of Eratosthenes / Atkin
prime_pi(n)          — prime counting function
prime_nth(n)         — nth prime
factor(n)            — integer factorisation (trial division + Pollard rho + SIQS)
factor_small(n)      — trial division only
factor_pollard(n)    — Pollard rho
factor_ecm(n)        — elliptic curve method
factor_qs(n)         — quadratic sieve
divisors(n)
num_divisors(n)       — tau function
sum_divisors(n)       — sigma function
euler_phi(n)         — Euler totient
jordan_totient(k, n)
mobius(n)            — Möbius function
liouville(n)         — Liouville lambda
von_mangoldt(n)
legendre_symbol(a, p)
jacobi_symbol(a, n)
kronecker_symbol(a, n)
gcd(a, b)
lcm(a, b)
extended_gcd(a, b)
mod_inv(a, m)        — modular inverse
crt(remainders, moduli)          — Chinese remainder theorem
mod_pow(base, exp, mod)
discrete_log(g, h, p)            — baby-step giant-step
is_primitive_root(g, p)
primitive_root(p)
quadratic_residues(p)
tonelli_shanks(n, p)             — modular square root
cornacchia(d, p)     — representation as sum of squares
pell_solve(D)        — Pell equation x^2 - Dy^2 = 1
continued_fraction(x, n)
convergents(cf)
farey(n)             — Farey sequence
stern_brocot(n)
partition(n)         — number of partitions
partition_list(n)    — actual partitions
```

---

## 2.15 ms::combo — Combinatorics

```
factorial(n)
double_factorial(n)
subfactorial(n)      — derangements
binomial(n, k)
multinomial(n, ks)
permutations(n, k)
combinations(n, k)
combinations_with_rep(n, k)
permutations_with_rep(n, ks)
next_permutation(v)
prev_permutation(v)
next_combination(v, n)
rank_permutation(v)
unrank_permutation(n, r)
rank_combination(v, n)
unrank_combination(n, k, r)
all_permutations(n)
all_subsets(n)
all_compositions(n)
all_partitions(n)
set_partitions(n)
integer_partitions(n)
restricted_partitions(n, k)
necklaces(n, k)
bracelets(n, k)
lyndon_words(n, k)
derangements(n)
involutions(n)
ballot_sequences(n)
catalan_paths(n)
dyck_paths(n)
motzkin_paths(n)
latin_squares(n)
```

---

## 2.16 ms::graph — Graph Theory

### Construction

```
Graph(n_vertices, edges)         — adjacency list
DiGraph(n_vertices, edges)       — directed
WeightedGraph(n, edges, weights)
from_matrix(A)
from_edge_list(edges)
add_vertex(G)
add_edge(G, u, v)
remove_vertex(G, v)
remove_edge(G, u, v)
subgraph(G, vertices)
complement(G)
line_graph(G)
```

### Traversal

```
bfs(G, source)
dfs(G, source)
bfs_tree(G, source)
dfs_tree(G, source)
topological_sort(G)
lexicographic_bfs(G)
```

### Shortest Paths

```
dijkstra(G, source)
bellman_ford(G, source)
floyd_warshall(G)
johnson(G)
astar(G, source, target, h)
bidirectional_dijkstra(G, s, t)
all_pairs_shortest(G)
```

### Connectivity

```
is_connected(G)
is_strongly_connected(G)
connected_components(G)
strongly_connected_components(G) — Tarjan / Kosaraju
weakly_connected(G)
biconnected_components(G)
articulation_points(G)
bridges(G)
```

### Spanning Trees

```
mst_prim(G)          — Prim's MST
mst_kruskal(G)       — Kruskal's MST
mst_boruvka(G)       — Borůvka's MST
steiner_tree(G, terminals)
arborescence(G, root)            — Edmonds' algorithm
```

### Flow

```
max_flow(G, s, t)    — Dinic's algorithm
min_cut(G, s, t)
min_cost_flow(G, s, t, demands)
circulation(G, demands, capacities)
push_relabel(G, s, t)
```

### Matching

```
bipartite_match(G)   — Hopcroft-Karp
max_matching(G)      — Blossom algorithm (Edmonds)
min_weight_match(G)
perfect_matching(G)
```

### Colouring

```
chromatic_number(G)
greedy_colour(G)
dsatur_colour(G)
edge_colour(G)
list_colour(G, lists)
```

### Planarity & Embedding

```
is_planar(G)         — Boyer-Myrvold
planar_embedding(G)
kuratowski_subgraph(G)
genus(G)
```

### Spectral Graph Theory

```
adjacency_matrix(G)
laplacian(G)
normalised_laplacian(G)
signless_laplacian(G)
spectrum(G)
spectral_radius(G)
algebraic_connectivity(G)        — Fiedler value
cheeger_constant(G)
```

### Clustering & Communities

```
pagerank(G, alpha)
hits(G)
betweenness_centrality(G)
closeness_centrality(G)
eigenvector_centrality(G)
katz_centrality(G, alpha)
louvain(G)           — community detection
girvan_newman(G)
label_propagation(G)
spectral_cluster(G, k)
```

### Miscellaneous

```
is_bipartite(G)
is_tree(G)
is_dag(G)
diameter(G)
radius(G)
centre(G)
girth(G)
clique_number(G)
independence_number(G)
vertex_cover(G)
dominating_set(G)
hamiltonian_path(G)
euler_circuit(G)
graph_isomorphism(G, H)          — VF2 algorithm
subgraph_isomorphism(G, H)
```

---

## 2.17 ms::geo — Computational Geometry

### 2D Primitives

```
Point2D, Line2D, Segment2D, Ray2D
Triangle2D, Polygon2D, Circle2D, Ellipse2D
AABB2D, OBB2D
```

### 3D Primitives

```
Point3D, Line3D, Segment3D, Ray3D, Plane3D
Triangle3D, Tetrahedron3D, Sphere3D, AABB3D, OBB3D
Cylinder3D, Cone3D, Frustum3D
```

### Convex Hull

```
convex_hull_2d(points)           — Graham scan / Quickhull
convex_hull_3d(points)           — 3D Quickhull
convex_hull_nd(points)
upper_hull(points)
lower_hull(points)
```

### Triangulation

```
delaunay_2d(points)
constrained_delaunay(points, edges)
voronoi(points)
power_diagram(points, weights)
ruppert(boundary, h)             — quality Delaunay mesh
```

### Range Searching

```
KDTree(points)
KDTree::nearest(q, k)
KDTree::range(q, r)
BVH(triangles)
BVH::ray_cast(ray)
BVH::nearest(q)
rtree_2d(rects)
rtree_3d(boxes)
quadtree(points, bounds)
octree(points, bounds)
```

### Intersection

```
intersect(a, b)      — any primitive pair
intersect_ray_tri(ray, tri)
intersect_ray_sphere(ray, sphere)
intersect_ray_aabb(ray, aabb)
intersect_seg_seg(s1, s2)
intersect_poly_poly(p1, p2)
overlap(a, b)
distance(a, b)       — minimum distance between primitives
```

### Boolean Operations

```
poly_union(A, B)
poly_intersect(A, B)
poly_diff(A, B)
poly_xor(A, B)
minkowski_sum(A, B)
```

### Curves & Surfaces

```
bezier_eval(control_pts, t)
bezier_deriv(control_pts, t)
bezier_subdivide(control_pts, t)
bspline_eval(control_pts, knots, degree, t)
nurbs_eval(control_pts, weights, knots, degree, t)
catmull_rom(control_pts, t)
hermite_curve(p0, m0, p1, m1, t)
bezier_surface(ctrl_net, u, v)
```

### Measurements

```
area(polygon)
signed_area(polygon)
perimeter(polygon)
volume(mesh)
surface_area(mesh)
centroid(polygon)
moment_of_inertia(polygon, axis)
```

---

## 2.18 ms::image — Image Processing

### I/O & Conversion

```
imread(path)
imwrite(path, img)
imshow(img)
rgb2gray(img)
gray2rgb(img)
rgb2hsv, hsv2rgb
rgb2lab, lab2rgb
rgb2xyz, xyz2rgb
rgb2ycbcr, ycbcr2rgb
im2double, im2uint8, im2uint16
imresize(img, scale, method)
imcrop(img, rect)
imrotate(img, angle)
imflip(img, dir)
impad(img, pad, method)
```

### Filtering

```
imfilter(img, kernel)
imgaussfilt(img, sigma)
imgaussfilt3(vol, sigma)
medfilt2(img, size)
wiener2(img, size)
bilateral(img, sigma_s, sigma_r)
nlmeans(img, h, patch, search)   — non-local means
guided_filter(img, guide, r, eps)
anisotropic_diffusion(img, n, kappa, gamma)
```

### Edge & Feature Detection

```
edge(img, method)    — Sobel, Prewitt, Canny, LoG, Roberts
canny(img, low, high, sigma)
sobel(img)
prewitt(img)
roberts(img)
laplacian_of_gaussian(img, sigma)
scharr(img)
gradient(img)
hough_lines(img)
hough_circles(img)
harris(img, k)       — Harris corner detector
shi_tomasi(img, n)
fast(img, threshold)
sift(img)            — SIFT keypoints + descriptors
surf(img)
orb(img)
brief(img, keypoints)
```

### Morphology

```
imdilate(img, se)
imerode(img, se)
imopen(img, se)
imclose(img, se)
imtophat(img, se)
imbothat(img, se)
imgradient_morph(img, se)
strel(shape, params) — structuring element
bwmorph(bw, op)
```

### Segmentation

```
im_threshold(img, method)        — Otsu, adaptive
kmeans_seg(img, k)
watershed(img)
active_contour(img, init, alpha, beta, gamma)
graph_cut(img, lambda)
slic(img, n, compactness)        — superpixels
felzenszwalb(img, scale, sigma, min_size)
```

### Transforms

```
radon(img, theta)
iradon(R, theta)
hough(img)
log_polar(img)
polar(img)
dft_image(img)
```

### Colour & Histogram

```
imhist(img, nbins)
histeq(img)
adapthisteq(img)
imadjust(img, in_range, out_range)
imsharpen(img, radius, amount)
```

### 3D Volume

```
volshow(vol)
volfilter(vol, kernel)
vol_threshold(vol, t)
marchingcubes(vol, iso)          — isosurface extraction
volinterp(vol, points)
```

---

## 2.19 ms::ml — Machine Learning

### Supervised

```
LinearRegression(fit, predict, score)
RidgeRegression(alpha)
LassoRegression(alpha)
ElasticNet(alpha, l1_ratio)
LogisticRegression(C, solver)
LinearSVC(C)
SVC(C, kernel, gamma)
SVR(C, kernel, epsilon)
DecisionTree(max_depth, criterion)
RandomForest(n_estimators, max_depth)
GradientBoosting(n_estimators, lr, max_depth)
XGBoost style implementation
AdaBoost(n_estimators)
BaggingClassifier(base, n)
NaiveBayes(var_smoothing)
KNN(n_neighbors, metric)
GaussianProcess(kernel, noise)
```

### Unsupervised

```
KMeans(k, init, n_init)
KMedoids(k)
DBSCAN(eps, min_samples)
HDBSCAN(min_cluster_size)
AgglomerativeClustering(n, linkage)
GaussianMixture(n_components, cov_type)
IsolationForest(n_estimators, contamination)
OneClassSVM(nu, kernel)
LocalOutlierFactor(n_neighbors)
```

### Dimensionality Reduction

```
PCA(n_components)
KernelPCA(n, kernel)
SparsePCA(n, alpha)
IncrementalPCA(n, batch)
TruncatedSVD(n)
FastICA(n, algorithm)
NMF(n, init, solver)
FactorAnalysis(n)
tSNE(n, perplexity, lr)
UMAP(n, n_neighbors, min_dist)
Isomap(n, n_neighbors)
LLE(n, n_neighbors)
LTSA(n, n_neighbors)
SpectralEmbedding(n)
```

### Neural Networks

```
Layer::Dense(units, activation)
Layer::Conv1D(filters, kernel_size, stride)
Layer::Conv2D(filters, kernel_size, stride)
Layer::Conv3D(filters, kernel_size, stride)
Layer::DepthwiseConv2D(kernel_size)
Layer::SeparableConv2D(filters, kernel_size)
Layer::TransposedConv2D(filters, kernel_size)
Layer::MaxPool1D/2D/3D(pool_size)
Layer::AvgPool1D/2D/3D(pool_size)
Layer::GlobalAvgPool1D/2D
Layer::BatchNorm(eps, momentum)
Layer::LayerNorm(eps)
Layer::GroupNorm(n_groups, eps)
Layer::Dropout(rate)
Layer::Embedding(vocab, dim)
Layer::LSTM(units, return_seq)
Layer::GRU(units, return_seq)
Layer::RNN(units, cell)
Layer::MultiHeadAttention(heads, key_dim)
Layer::Flatten
Layer::Reshape(shape)
Layer::Residual(sublayer)
```

### Autodiff

```
Tape()               — reverse-mode AD
grad(f, x)           — gradient of scalar function
jacobian(F, x)       — Jacobian matrix
hessian(f, x)        — Hessian via double reverse
jvp(f, x, v)         — forward-mode Jacobian-vector product
vjp(f, x, v)         — reverse-mode vector-Jacobian product
```

### Optimisers (Neural)

```
SGD(lr, momentum, nesterov)
Adam(lr, beta1, beta2, eps)
AdamW(lr, weight_decay)
RMSProp(lr, rho, eps)
Adagrad(lr, eps)
Adadelta(rho, eps)
LAMB(lr, beta1, beta2)
LARS(lr, momentum, eta)
```

### Loss Functions

```
mse_loss
mae_loss
huber_loss(delta)
binary_crossentropy
categorical_crossentropy
sparse_categorical_crossentropy
kl_divergence
hinge_loss
focal_loss(alpha, gamma)
contrastive_loss(margin)
triplet_loss(margin)
cosine_similarity_loss
```

---

## 2.20 ms::crypto — Cryptography (Izaac integration)

```
sha256(data)
sha512(data)
sha3_256(data)
sha3_512(data)
blake2b(data, key, size)
blake3(data, key)
hmac(key, data, hash_fn)
aes128_ecb, aes128_cbc, aes128_gcm
aes256_ecb, aes256_cbc, aes256_gcm
chacha20(key, nonce, counter, data)
chacha20_poly1305(key, nonce, aad, data)
poly1305(key, data)
curve25519_keygen()
curve25519_dh(priv, pub)
ed25519_keygen()
ed25519_sign(key, msg)
ed25519_verify(pub, msg, sig)
x25519(scalar, point)
ecdsa_p256_keygen()
ecdsa_p256_sign(key, hash)
ecdsa_p256_verify(pub, hash, sig)
rsa_keygen(bits)
rsa_encrypt(pub, msg)
rsa_decrypt(priv, ct)
rsa_sign(priv, hash)
rsa_verify(pub, hash, sig)
pbkdf2(password, salt, iter, dklen, prf)
hkdf(ikm, salt, info, len, hash)
argon2id(password, salt, t, m, p)
csprng(n)            — Izaac VRF-based CSPRNG
vrf_prove(key, msg)  — Izaac VRF
vrf_verify(pub, msg, proof)
```

---

## 2.21 ms::compress — Compression

```
deflate(data, level)
inflate(data)
lz4_compress(data, level)
lz4_decompress(data)
zstd_compress(data, level)
zstd_decompress(data)
lzma_compress(data, preset)
lzma_decompress(data)
brotli_compress(data, quality)
brotli_decompress(data)
rle_encode(data)
rle_decode(data)
huffman_encode(data)
huffman_decode(data, tree)
arithmetic_encode(data, model)
arithmetic_decode(data, model)
bwt(data)            — Burrows-Wheeler transform
ibwt(data, idx)
mtf_encode(data)
mtf_decode(data)
ans_encode(data, freq)           — asymmetric numeral systems
ans_decode(data, freq)
wavelet_compress(data, threshold, wavelet)
```

---

## 2.22 ms::finance — Financial Mathematics

```
black_scholes(S, K, r, sigma, T, type)   — call/put price
black_scholes_greeks(S, K, r, sigma, T)  — delta, gamma, theta, vega, rho
black76(F, K, r, sigma, T)
bachelier(F, K, r, sigma, T)
heston(S, K, r, v0, kappa, theta, sigma, rho, T)
sabr(F, K, T, alpha, beta, rho, nu)      — SABR model
binomial_tree(S, K, r, sigma, T, n, type)
trinomial_tree(S, K, r, sigma, T, n, type)
monte_carlo_option(S, K, r, sigma, T, n, type)
american_option(S, K, r, sigma, T, type, n)
barrier_option(S, K, B, r, sigma, T, type)
asian_option(S, K, r, sigma, T, n, type)
lookback_option(S, r, sigma, T, type)
digital_option(S, K, r, sigma, T, type)
bond_price(face, coupon, r, T, freq)
bond_ytm(price, face, coupon, T, freq)
bond_duration(face, coupon, r, T, freq)
bond_convexity(face, coupon, r, T, freq)
bond_dv01(face, coupon, r, T, freq)
zero_rate(discount_factors, times)
forward_rate(r1, t1, r2, t2)
swap_rate(fixed_schedule, float_schedule, discount)
pv(cash_flows, times, r)
npv(cash_flows, times, r)
irr(cash_flows, times)
xirr(cash_flows, dates)
var(returns, alpha)  — Value at Risk
cvar(returns, alpha) — Conditional VaR
expected_shortfall(returns, alpha)
portfolio_var(weights, cov_matrix)
sharpe_ratio(returns, rf)
sortino_ratio(returns, rf)
max_drawdown(returns)
information_ratio(returns, benchmark)
treynor_ratio(returns, rf, beta)
capm(rf, beta, rm)
fama_french(returns, factors)
black_litterman(Sigma, P, Q, Omega, pi, tau)
markowitz(mu, Sigma, target_return)
efficient_frontier(mu, Sigma, n_points)
```

---

## 2.23 ms::quantum — Quantum Mechanics

```
ket(state_vector)    — ket |psi>
bra(state_vector)    — bra <psi|
inner(bra, ket)      — <phi|psi>
outer(ket, bra)      — |psi><phi| density matrix
tensor_product(A, B) — Kronecker product for operators
commutator(A, B)     — [A,B] = AB - BA
anticommutator(A, B) — {A,B} = AB + BA
expectation(psi, A)  — <psi|A|psi>
uncertainty(psi, A, B)
pauli_x, pauli_y, pauli_z
pauli_plus, pauli_minus
hadamard_gate
phase_gate(theta)
cnot_gate
toffoli_gate
swap_gate
qft_gate(n)          — quantum Fourier transform
rotation_x(theta)
rotation_y(theta)
rotation_z(theta)
density_matrix(psi)
partial_trace(rho, subsystem)
fidelity(rho, sigma)
trace_distance(rho, sigma)
von_neumann_entropy(rho)
entanglement_entropy(psi, partition)
bell_states()
ghz_state(n)
w_state(n)
schrodinger(H, psi0, tspan)      — time evolution
time_evolution_operator(H, t)
eigenspectrum(H)
ground_state(H)
variational_eigensolver(H, ansatz, params)
wigner_function(rho, x, p)
husimi_Q(rho, alpha)
coherent_state(alpha, n_max)
fock_state(n, n_max)
```

---

## 2.24 ms::fem — Finite Element Method

```
mesh1d(a, b, n, element_type)
mesh2d_tri(boundary, h, element_type)
mesh2d_quad(domain, nx, ny)
mesh3d_tet(boundary, h)
mesh3d_hex(domain, nx, ny, nz)
lagrange_basis(degree, ref_element)
hermite_basis(degree, ref_element)
serendipity_basis(degree, ref_element)
gauss_quadrature(n, dim)
gauss_lobatto(n)
assemble_stiffness(mesh, basis, coeff)
assemble_mass(mesh, basis)
assemble_load(mesh, basis, f)
apply_dirichlet(K, f, boundary, g)
apply_neumann(f, mesh, basis, boundary, g)
solve_fem(K, f)
interpolate_fem(solution, mesh, points)
error_L2(solution, exact, mesh)
error_H1(solution, exact, mesh)
adaptive_refine(mesh, error_estimator)
zienkiewicz_zhu_estimator(solution, mesh)
```

---

## 2.25 ms::cfd — Fluid Dynamics

```
navier_stokes_2d(domain, ic, bc, nu, tspan)
navier_stokes_3d(domain, ic, bc, nu, tspan)
euler_equations(domain, ic, bc, tspan)
stokes_flow(domain, f, bc, nu)
darcy_flow(domain, K, f, bc)
lid_driven_cavity(n, Re)
channel_flow(n, Re)
boundary_layer(U_inf, nu, x)
blasius_profile(eta)
von_karman_integral(theta0, Re, x)
reynolds_number(rho, U, L, mu)
mach_number(U, a)
prandtl_number(mu, cp, k)
normal_shock(M1, gamma)
oblique_shock(M1, theta, gamma)
prandtl_meyer(M, gamma)
```

---

## 2.26 ms::control — Control Theory

```
tf(num, den)         — transfer function
ss(A, B, C, D)       — state space
zpk(z, p, k)         — zero-pole-gain
series(sys1, sys2)
parallel(sys1, sys2)
feedback(sys, sens)
poles(sys)
zeros(sys)
dcgain(sys)
bode(sys, w)
nyquist(sys, w)
nichols(sys, w)
root_locus(sys, k)
margin(sys)          — gain and phase margins
step_response(sys, t)
impulse_response(sys, t)
ramp_response(sys, t)
lsim(sys, u, t, x0)  — linear simulation
c2d(sys, Ts, method) — continuous to discrete
d2c(sys, method)     — discrete to continuous
pidtune(sys, type)   — PID autotuning
rlocus_design(sys, spec)
place(A, B, poles)   — pole placement
lqr(A, B, Q, R)      — linear quadratic regulator
lqe(A, C, Q, R)      — Kalman filter / linear quadratic estimator
lqg(A, B, C, Q_proc, Q_meas) — LQG controller
h2syn(P, nmeas, ncont)        — H2 synthesis
hinfsyn(P, nmeas, ncont)      — H-inf synthesis
mixsyn(P, W1, W2, W3)         — mixed sensitivity
obsv(A, C)           — observability matrix
ctrb(A, B)           — controllability matrix
is_observable(A, C)
is_controllable(A, B)
gram(sys, type)       — controllability/observability Gramian
balreal(sys)         — balanced realisation
minreal(sys)         — minimal realisation
norm_h2(sys)
norm_hinf(sys)
covar(sys, Q)        — covariance
lyap(A, Q)           — Lyapunov equation
dlyap(A, Q)          — discrete Lyapunov
riccati(A, B, Q, R)  — algebraic Riccati
dare(A, B, Q, R)     — discrete algebraic Riccati
```

---

## 2.27 ms::info — Information Theory

```
entropy(p)           — Shannon entropy H(X)
entropy_joint(p_xy)  — H(X,Y)
entropy_cond(p_xy)   — H(X|Y)
mutual_info(p_xy)    — I(X;Y)
mutual_info_norm(p_xy)
kl_divergence(p, q)  — D_KL(P||Q)
js_divergence(p, q)  — Jensen-Shannon
renyi_entropy(p, alpha)
tsallis_entropy(p, q)
cross_entropy(p, q)
channel_capacity(W)  — discrete memoryless channel
channel_capacity_awgn(snr)
rate_distortion(D)
blahut_arimoto(W, eps)           — Blahut-Arimoto algorithm
huffman_code(symbols, probs)
arithmetic_code(symbols, probs)
lempel_ziv_complexity(x)
kolmogorov_complexity_approx(x)  — via compression ratio
minkowski_functional(p, A, B)
fisher_info(theta, f)            — Fisher information
```

---

## 2.28 ms::cplx — Complex Analysis

```
residue(f, z0)
residues(f, poles)
winding_number(gamma, z0)
argument_principle(f, gamma)
contour_integral(f, gamma)
cauchy_integral(f, z0, gamma)
conformal_map(f, domain)
joukowski(z, c)      — Joukowski transform
schwarz_christoffel(vertices, angles)
riemann_map(domain, z0)          — numerical Riemann mapping
blaschke_product(zeros, n)
poisson_kernel(theta, r)
green_function_disk(z, z0)
harmonic_conjugate(u, domain)
analytic_continuation(f, z0, path)
mobius(a, b, c, d)   — Möbius transformation
inversion(z, center, r)
cross_ratio(z1, z2, z3, z4)
hyperbolic_distance(z1, z2)      — Poincaré disk
```

---

## 2.29 ms::diffgeo — Differential Geometry

```
manifold(dim, charts)
tangent_vector(M, p, components)
cotangent_vector(M, p, components)
tensor_field(M, type, components)
metric_tensor(M, components)
christoffel_symbols(g, coords)   — Γ^k_{ij}
riemann_tensor(g, coords)        — R^l_{ijk}
ricci_tensor(g, coords)
ricci_scalar(g, coords)
einstein_tensor(g, coords)
weyl_tensor(g, coords)
geodesic(M, p, v, t)             — geodesic equation
exp_map(M, p, v)                 — exponential map
log_map(M, p, q)                 — logarithmic map
parallel_transport(M, gamma, v)
covariant_deriv(T, v)
lie_bracket(X, Y)
lie_derivative(T, X)
exterior_deriv(omega)
hodge_star(omega, g)
wedge(omega, eta)
pullback(f, omega)
pushforward(f, X)
de_rham_cohomology(M)
gauss_curvature(surface)         — K
mean_curvature(surface)          — H
principal_curvatures(surface)    — k1, k2
second_fundamental_form(surface)
gauss_bonnet(surface)
```

---

## 2.30 ms::tensor — Tensor Analysis

```
Tensor(shape)
Tensor(shape, data)
rank(T)
shape(T)
reshape(T, new_shape)
transpose(T, perm)
contract(T, U, idx_T, idx_U)     — tensor contraction
outer(T, U)          — outer product
inner(T, U)          — full contraction
einsum(subscripts, T, U)         — Einstein summation
trace(T, axis1, axis2)
diag(T, axis1, axis2)
symmetrise(T)
antisymmetrise(T)
symmetric_part(T)
antisymmetric_part(T)
levi_civita(n)       — Levi-Civita symbol
kronecker_delta(n)
metric_raise(T, g_inv, idx)
metric_lower(T, g, idx)
decompose_cp(T, rank)            — CANDECOMP/PARAFAC
decompose_tucker(T, ranks)       — Tucker decomposition
decompose_tt(T, eps)             — tensor train
decompose_hosvd(T)               — higher-order SVD
norm_frobenius(T)
norm_nuclear(T)
unfold(T, mode)
fold(M, mode, shape)
khatri_rao(A, B)
hadamard(A, B)
mode_product(T, A, mode)         — n-mode product
```

---

## 2.31 ms::topo — Topology

```
SimplicialComplex(vertices, simplices)
CubicalComplex(bitmap)
VietorisRips(points, epsilon)
Cech(points, epsilon)
AlphaComplex(points)
persistence_diagram(complex)     — persistent homology
betti_numbers(complex)           — β_0, β_1, β_2, ...
euler_characteristic(complex)
fundamental_group(complex)       — π_1 (presentation)
homology_groups(complex)         — H_k with Z coefficients
cohomology_groups(complex)
wasserstein_dist(dgm1, dgm2)     — between persistence diagrams
bottleneck_dist(dgm1, dgm2)
mapper(X, f, cover, cluster)     — Mapper algorithm (TDA)
```

---

## Reference Sources by Domain

| Domain | Primary References |
|---|---|
| Linear algebra | LAPACK, BLAS, Eigen, MATLAB, NAG |
| FFT | FFTW manual, Numerical Recipes, Oppenheim |
| Signal | MATLAB Signal Processing Toolbox, Octave-Forge signal |
| Statistics | R base + packages, MATLAB Stats Toolbox, NIST |
| Probability | Distributions.jl, SciPy stats, MATLAB |
| Optimisation | NLopt, MATLAB Optimization Toolbox, Nocedal & Wright |
| ODE/DAE | Hairer & Wanner, SUNDIALS, MATLAB ode suite |
| PDE | FEniCS, deal.ii, Numerical Recipes |
| Polynomial | Cohen "Computer Algebra", PARI/GP |
| Symbolic CAS | Mathematica, Maple, SymPy, Maxima |
| Special functions | DLMF (NIST), BOOST Math, GSL, Abramowitz & Stegun |
| Number theory | PARI/GP, SageMath, Cohen "Number Theory" |
| Combinatorics | Knuth TAOCP, SageMath |
| Graph | NetworkX, igraph, Boost.Graph |
| Geometry | CGAL, Shamos & Preparata |
| Image | OpenCV, ITK, MATLAB Image Toolbox, scikit-image |
| ML | scikit-learn, PyTorch, JAX |
| Crypto | NIST FIPS, RFC standards, Bernstein et al. |
| Finance | QuantLib, Wilmott, Hull |
| Quantum | Qiskit, Cirq, QuTiP |
| FEM | FEniCS, deal.ii, Zienkiewicz & Taylor |
| CFD | OpenFOAM function set, Anderson |
| Control | MATLAB Control Toolbox, Skogestad & Postlethwaite |
| Info theory | Cover & Thomas, Elements of Information Theory |
| Complex analysis | Ahlfors, Needham |
| Differential geometry | do Carmo, Lee, Misner Thorne Wheeler |
| Tensor analysis | Kolda & Bader, Cichocki |
| Topology | Edelsbrunner & Harer, Gudhi |

---

*MathScript Master Build Plan — Part 2 of N*
*Next: Part 3 — Repository Structure, Build System & Toolchain*

---

## Part 3: Repository Structure & Build System

---

## 3.1 Repository Layout

Single monorepo. No git submodules. All vendored dependencies copied into `vendor/` at a pinned commit and committed to the repository.

```
mathscript/
│
├── CMakeLists.txt
├── CMakePresets.json
│
├── cmake/
│   ├── toolchains/
│   │   ├── windows-clang-cl.cmake
│   │   ├── windows-cuda.cmake
│   │   ├── linux-clang-x64.cmake
│   │   ├── linux-clang-cuda.cmake
│   │   └── linux-clang-aarch64.cmake
│   ├── modules/
│   │   ├── FindCUDAToolkit.cmake
│   │   ├── FindMPI.cmake
│   │   ├── FindhwlocMS.cmake
│   │   ├── FindNCCL.cmake
│   │   └── FindSLATE.cmake
│   ├── options.cmake
│   ├── platform.cmake
│   ├── version.cmake
│   └── compiler_flags.cmake
│
├── include/
│   └── ms/
│       ├── version.hpp               (generated)
│       ├── ms.hpp                    (umbrella header)
│       ├── core/
│       │   ├── types.hpp
│       │   ├── matrix.hpp
│       │   ├── tensor.hpp
│       │   ├── sparse.hpp
│       │   ├── vector.hpp
│       │   ├── sym.hpp
│       │   ├── sym_matrix.hpp
│       │   ├── index.hpp
│       │   ├── slice.hpp
│       │   └── scalar.hpp
│       ├── memory/
│       │   ├── aligned_allocator.hpp
│       │   ├── pinned_allocator.hpp
│       │   ├── numa_allocator.hpp
│       │   ├── pool_allocator.hpp
│       │   ├── arena.hpp
│       │   └── policy.hpp
│       ├── runtime/
│       │   ├── dispatch.hpp
│       │   ├── exec_policy.hpp
│       │   ├── topology.hpp
│       │   └── thread_pool.hpp
│       ├── simd/
│       │   ├── isa.hpp
│       │   ├── portable.hpp
│       │   └── intrinsics.hpp
│       ├── cuda/
│       │   ├── device.hpp
│       │   ├── stream_pool.hpp
│       │   ├── device_buffer.hpp
│       │   ├── pinned_buffer.hpp
│       │   ├── multi_gpu.hpp
│       │   └── kernels.hpp
│       ├── distributed/
│       │   ├── mpi_context.hpp
│       │   ├── dist_matrix.hpp
│       │   ├── collective.hpp
│       │   ├── gpu_aware_mpi.hpp
│       │   └── topology.hpp
│       ├── linalg/
│       ├── fft/
│       ├── signal/
│       ├── stats/
│       ├── prob/
│       ├── optim/
│       ├── ode/
│       ├── pde/
│       ├── poly/
│       ├── sym/
│       ├── special/
│       ├── numthy/
│       ├── combo/
│       ├── graph/
│       ├── geo/
│       ├── image/
│       ├── ml/
│       ├── crypto/
│       ├── compress/
│       ├── finance/
│       ├── quantum/
│       ├── fem/
│       ├── cfd/
│       ├── control/
│       ├── info/
│       ├── cplx/
│       ├── diffgeo/
│       ├── tensor/
│       ├── topo/
│       └── frameworks/
│           ├── gria/
│           ├── izaac/
│           ├── cyphadif/
│           ├── cellai/
│           └── axiom/
│
├── src/
│   ├── core/
│   │   ├── matrix.cpp
│   │   ├── tensor.cpp
│   │   ├── sparse.cpp
│   │   └── bignum/
│   │       ├── bigint.cpp
│   │       ├── apfloat.cpp
│   │       └── rational.cpp
│   ├── memory/
│   │   ├── numa_allocator.cpp
│   │   ├── pool_allocator.cpp
│   │   └── arena.cpp
│   ├── runtime/
│   │   ├── topology.cpp
│   │   ├── thread_pool.cpp
│   │   ├── dispatch.cpp
│   │   ├── cpu/
│   │   │   ├── blas_impl.cpp
│   │   │   ├── lapack_impl.cpp
│   │   │   ├── fft_impl.cpp
│   │   │   └── simd/
│   │   │       ├── avx512_gemm.cpp
│   │   │       ├── avx2_elementwise.cpp
│   │   │       └── neon_elementwise.cpp
│   │   └── cuda/
│   │       ├── device.cu
│   │       ├── stream_pool.cu
│   │       ├── kernels/
│   │       │   ├── elementwise.cu
│   │       │   ├── reduction.cu
│   │       │   ├── trig.cu
│   │       │   ├── complex_ops.cu
│   │       │   ├── transpose.cu
│   │       │   ├── conv2d.cu
│   │       │   ├── fill.cu
│   │       │   ├── gather_scatter.cu
│   │       │   └── bitonic_sort.cu
│   │       ├── cublas_wrap.cu
│   │       ├── cusolver_wrap.cu
│   │       ├── cufft_wrap.cu
│   │       └── cusparse_wrap.cu
│   ├── linalg/
│   ├── fft/
│   ├── signal/
│   ├── stats/
│   ├── prob/
│   ├── optim/
│   ├── ode/
│   ├── pde/
│   ├── poly/
│   ├── sym/
│   ├── special/
│   ├── numthy/
│   ├── combo/
│   ├── graph/
│   ├── geo/
│   ├── image/
│   ├── ml/
│   ├── crypto/
│   ├── compress/
│   ├── finance/
│   ├── quantum/
│   ├── fem/
│   ├── cfd/
│   ├── control/
│   ├── info/
│   ├── cplx/
│   ├── diffgeo/
│   ├── tensor/
│   ├── topo/
│   ├── distributed/
│   │   ├── mpi_context.cpp
│   │   ├── dist_matrix.cpp
│   │   └── gpu_aware_mpi.cpp
│   ├── interp/
│   │   ├── repl_engine.cpp
│   │   ├── jit_backend.cpp
│   │   ├── preamble.cpp
│   │   ├── session.cpp
│   │   └── reflect.cpp
│   ├── plugin/
│   │   ├── MsPlugin.cpp
│   │   ├── diagnostics.hpp
│   │   └── rules/
│   │       ├── NoRawNew.cpp
│   │       ├── NoMalloc.cpp
│   │       ├── NoCStyleCast.cpp
│   │       ├── NoConstCast.cpp
│   │       ├── NoUnsafeReinterpret.cpp
│   │       ├── NoThrow.cpp
│   │       ├── NoCatch.cpp
│   │       ├── UnusedExpected.cpp
│   │       ├── NoUninit.cpp
│   │       ├── NarrowingCheck.cpp
│   │       ├── NoRawThread.cpp
│   │       ├── NoRawMutexLock.cpp
│   │       ├── NoVolatileSync.cpp
│   │       ├── NoDetach.cpp
│   │       ├── NoSignedUnsignedMix.cpp
│   │       ├── NoRawPtrArithmetic.cpp
│   │       └── UnsafeAudit.cpp
│   └── gui/
│       ├── MainWindow.cpp
│       ├── Editor/
│       │   ├── CodeEditor.cpp
│       │   ├── Highlighter.cpp
│       │   ├── LSPClient.cpp
│       │   ├── CompletionModel.cpp
│       │   └── DiagnosticMargin.cpp
│       ├── Console/
│       │   ├── ReplWidget.cpp
│       │   ├── InputBar.cpp
│       │   ├── RichOutput.cpp
│       │   └── OutputRouter.cpp
│       ├── Workspace/
│       │   ├── VariableModel.cpp
│       │   ├── VariableDelegate.cpp
│       │   └── TypeRenderer.cpp
│       ├── Plotter/
│       │   ├── PlotManager.cpp
│       │   ├── Plot2D.cpp
│       │   ├── Plot3D.cpp
│       │   └── PlotBridge.cpp
│       ├── Debugger/
│       │   ├── BreakpointManager.cpp
│       │   ├── CallStackModel.cpp
│       │   ├── WatchModel.cpp
│       │   └── DebugController.cpp
│       ├── FileBrowser/
│       │   └── FileWidget.cpp
│       ├── PackageManager/
│       │   └── PackageWidget.cpp
│       └── SystemMonitor/
│           └── HardwareBar.cpp
│
├── frameworks/
│   ├── gria/
│   │   ├── include/ms/frameworks/gria/
│   │   └── src/
│   ├── izaac/
│   ├── cyphadif/
│   ├── cellai/
│   └── axiom/
│
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/
│   │   ├── core/
│   │   ├── linalg/
│   │   ├── fft/
│   │   └── ...
│   ├── integration/
│   ├── compliance/
│   │   ├── no_raw_new_fail.cpp
│   │   ├── no_raw_new_ok.cpp
│   │   ├── no_cstyle_cast_fail.cpp
│   │   └── ...
│   ├── numerical/
│   ├── performance/
│   ├── distributed/
│   └── fuzz/
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── frameworks/
│   └── build/
│
├── scripts/
│   ├── setup/
│   │   ├── setup_windows.ps1
│   │   └── setup_linux.sh
│   ├── build.sh
│   ├── build.ps1
│   ├── run_tests.sh
│   ├── run_tests.ps1
│   └── unsafe_report.sh
│
└── vendor/
    ├── mimalloc/
    ├── oneTBB/
    ├── highway/
    ├── symengine/
    └── googletest/
```

---

## 3.2 Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.28)

project(MathScript
    VERSION 0.1.0
    DESCRIPTION "MathScript HPC CAS Environment"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Pull in platform detection and compiler flags
include(cmake/platform.cmake)
include(cmake/compiler_flags.cmake)
include(cmake/options.cmake)
include(cmake/version.cmake)

# Find all dependencies
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(Qt6 REQUIRED COMPONENTS
    Core Widgets WebEngineWidgets OpenGL Charts)
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(TBB REQUIRED)

# Vendored dependencies
add_subdirectory(vendor/mimalloc)
add_subdirectory(vendor/oneTBB)
add_subdirectory(vendor/highway)
add_subdirectory(vendor/symengine)

# Core targets
add_subdirectory(src/plugin)
add_subdirectory(src/runtime)
add_subdirectory(src/core)
add_subdirectory(src/interp)
add_subdirectory(frameworks)

# Math library modules
add_subdirectory(src/linalg)
add_subdirectory(src/fft)
# ... all 31 modules

# GUI
if(MS_BUILD_GUI)
    add_subdirectory(src/gui)
endif()

# Main executables
add_subdirectory(src/exe)

# Tests
if(MS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(vendor/googletest)
    add_subdirectory(tests)
endif()
```

---

## 3.3 CMakePresets.json

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "linux-debug",
      "displayName": "Linux Debug",
      "toolchainFile": "cmake/toolchains/linux-clang-x64.cmake",
      "binaryDir": "${sourceDir}/build/linux-debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "MS_BUILD_TESTS": "ON",
        "MS_BUILD_GUI": "ON",
        "MS_ENABLE_CUDA": "ON"
      }
    },
    {
      "name": "linux-release",
      "displayName": "Linux Release",
      "toolchainFile": "cmake/toolchains/linux-clang-cuda.cmake",
      "binaryDir": "${sourceDir}/build/linux-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "MS_BUILD_TESTS": "OFF",
        "MS_BUILD_GUI": "ON",
        "MS_ENABLE_CUDA": "ON"
      }
    },
    {
      "name": "windows-debug",
      "displayName": "Windows Debug",
      "toolchainFile": "cmake/toolchains/windows-clang-cl.cmake",
      "binaryDir": "${sourceDir}/build/windows-debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "MS_BUILD_TESTS": "ON",
        "MS_BUILD_GUI": "ON",
        "MS_ENABLE_CUDA": "ON"
      }
    },
    {
      "name": "windows-release",
      "displayName": "Windows Release",
      "toolchainFile": "cmake/toolchains/windows-cuda.cmake",
      "binaryDir": "${sourceDir}/build/windows-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "MS_BUILD_TESTS": "OFF",
        "MS_ENABLE_CUDA": "ON"
      }
    },
    {
      "name": "headless-server",
      "displayName": "Headless Server (No GUI)",
      "toolchainFile": "cmake/toolchains/linux-clang-cuda.cmake",
      "binaryDir": "${sourceDir}/build/server",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "MS_BUILD_GUI": "OFF",
        "MS_ENABLE_CUDA": "ON",
        "MS_ENABLE_MPI": "ON"
      }
    }
  ]
}
```

---

## 3.4 compiler_flags.cmake

```cmake
# Flags applied to all targets unconditionally

set(MS_CXX_FLAGS
    -Wall -Wextra -Wpedantic -Werror
    -Wconversion
    -Wshadow
    -Wnull-dereference
    -Wdouble-promotion
    -Wimplicit-fallthrough
    -Wno-unused-parameter   # Too noisy during development — remove before 1.0
)

set(MS_CXX_FLAGS_DEBUG
    -O0 -g3
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
)

set(MS_CXX_FLAGS_RELWITHDEBINFO
    -O2 -g
    -fsanitize=undefined
)

set(MS_CXX_FLAGS_RELEASE
    -O3
    -flto=thin
    -fsanitize=undefined
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=3
)

set(MS_CXX_FLAGS_SHIPPING
    -O3
    -flto=full
    -fprofile-use
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=3
)

# Applied globally — these are not negotiable
add_compile_options(
    -fno-exceptions
    -fno-rtti
    -fno-strict-aliasing
)

# Linker flags for Release/Shipping on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options(
        -Wl,-z,relro,-z,now
        -pie
        -fuse-ld=lld
    )
endif()

# Plugin — loaded for all non-plugin targets
if(NOT MS_BUILDING_PLUGIN)
    add_compile_options(-fplugin=${MS_PLUGIN_PATH})
endif()
```

---

## 3.5 options.cmake

```cmake
option(MS_BUILD_GUI         "Build Qt6 IDE"              ON)
option(MS_BUILD_TESTS       "Build test suite"           ON)
option(MS_ENABLE_CUDA       "Enable CUDA GPU backend"    ON)
option(MS_ENABLE_MPI        "Enable distributed MPI"     OFF)
option(MS_ENABLE_NCCL       "Enable multi-GPU NCCL"      ON)
option(MS_ENABLE_AVX512     "Enable AVX-512 kernels"     ON)
option(MS_ENABLE_ASAN       "Force ASan regardless of build type" OFF)
option(MS_ENABLE_COVERAGE   "Enable coverage instrumentation"     OFF)
option(MS_VERBOSE_DISPATCH  "Log dispatch decisions at runtime"   OFF)

# CUDA architecture targets
set(MS_CUDA_ARCHITECTURES "70;80;86;89;90"
    CACHE STRING "CUDA SM architectures to compile for")
set_property(CACHE MS_CUDA_ARCHITECTURES PROPERTY STRINGS
    "70" "80" "86" "89" "90" "70;80;86;89;90")
```

---

## 3.6 Dependency Management

**No package managers (no vcpkg, no Conan).** All dependencies are either:

1. Vendored in `vendor/` — committed at a specific git hash
2. Found on the system via CMake `find_package` (MPI, CUDA, Qt6)

**Vendored dependency policy:**

- Update by copying new source, recording the source URL and commit hash in `vendor/VERSIONS.txt`
- No auto-updates, no network access during build
- `vendor/VERSIONS.txt` format:

```
mimalloc    2.1.7    https://github.com/microsoft/mimalloc    commit:abc1234
oneTBB      2021.11  https://github.com/oneapi-src/oneTBB     commit:def5678
highway     1.0.7    https://github.com/google/highway        commit:ghi9012
symengine   0.12.0   https://github.com/symengine/symengine   commit:jkl3456
googletest  1.14.0   https://github.com/google/googletest     commit:mno7890
```

---

## 3.7 Version Generation

`cmake/version.cmake` reads `CMakeLists.txt` project version and the current git commit:

```cmake
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MS_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_BINARY_DIR}/include/ms/version.hpp
    @ONLY
)
```

---

## 3.8 Windows-Specific Setup

**Required tools (setup_windows.ps1 installs these):**

```powershell
winget install LLVM.LLVM              # Clang + Clang-cl
winget install Kitware.CMake
winget install Ninja-build.Ninja
winget install Microsoft.VisualStudio.2022.BuildTools  # MSVC runtime + headers
# Qt6 Commercial — manual install from Qt installer
# CUDA Toolkit — manual install from NVIDIA
```

**Windows-specific CMake toolchain (windows-clang-cl.cmake):**

```cmake
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)

# MSVC ABI — required for CUDA interop on Windows
set(CMAKE_CXX_FLAGS_INIT "/EHs- /GR- /W4")

# mimalloc Windows override
add_compile_definitions(MI_OVERRIDE=1)
```

**CUDA on Windows (windows-cuda.cmake):**

```cmake
set(CMAKE_CUDA_HOST_COMPILER cl)   # MSVC as CUDA host compiler
set(CMAKE_CUDA_FLAGS_INIT
    "--compiler-options /EHs- /GR-"
    "-Xcompiler /W4"
)
```


---

## Part 4: Architecture & Core Design

---

## 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Qt6 IDE Layer                            │
│  Editor(clangd) │ REPL Console │ Plotter │ Workspace │ Debugger │
└────────────────────────────┬────────────────────────────────────┘
                             │ In-process async queue (Qt signals)
┌────────────────────────────▼────────────────────────────────────┐
│                    Interpreter Core                             │
│       Clang-REPL ──► LLVM ORC JIT v2                           │
│       Preamble loader │ Session state │ Symbol reflection       │
│       Plugin enforcement (runs on every input)                  │
└──────┬──────────────────────┬──────────────────────────────────┘
       │                      │
┌──────▼──────┐    ┌──────────▼──────────────────────────────────┐
│  Symbolic   │    │          Math Runtime                        │
│  CAS Layer  │    │   Dispatch ─► CPU path / CUDA path           │
│ (ms::sym)   │    │   Load balancer │ Policy tags                │
└─────────────┘    └──────────┬──────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────▼────────┐ ┌────────▼───────┐ ┌────────▼───────────┐
│   CPU Backend    │ │   CUDA Backend │ │ Distributed Backend │
│ Own impls + SIMD │ │ Own kernels +  │ │ MPI + NCCL + SLATE  │
│ AVX2/AVX-512/    │ │ cuBLAS/cuFFT/  │ │ GPU-direct RDMA     │
│ NEON             │ │ cuSOLVER       │ │                     │
└──────────────────┘ └────────────────┘ └────────────────────┘
```

---

## 4.2 Interpreter Core Design

The interpreter is behind an interface. The backend is swappable.

```cpp
// src/interp/interpreter.hpp
namespace ms::interp {

struct ExprResult {
    std::string output;           // text representation
    std::string type_name;        // C++ type as string
    bool has_plot_data = false;
    PlotData plot_data;
};

class IInterpreter {
public:
    virtual Result<ExprResult>  execute(std::string_view code)      = 0;
    virtual Result<void>        load_library(std::filesystem::path) = 0;
    virtual Result<void>        add_include_path(std::filesystem::path) = 0;
    virtual Result<void>        add_preamble(std::string_view code) = 0;
    virtual SymbolTable         symbols() const                      = 0;
    virtual void                reset_session()                      = 0;
    virtual ~IInterpreter()     = default;
};

// Factory — returns Clang-REPL implementation, ORC JIT available later
std::unique_ptr<IInterpreter> create_interpreter(const InterpreterConfig&);

} // namespace ms::interp
```

**Clang-REPL embedding:**

```cpp
// src/interp/repl_engine.cpp
class ClangReplInterpreter : public IInterpreter {
    std::unique_ptr<clang::Interpreter> interp_;
    SessionState                        state_;

public:
    explicit ClangReplInterpreter(const InterpreterConfig& cfg) {
        clang::IncrementalCompilerBuilder builder;
        builder.SetCompilerArgs({
            "-std=c++23",
            "-fno-exceptions",
            "-fno-rtti",
            "-O1",                            // light optimisation in REPL
            "-fplugin=" + cfg.plugin_path,    // enforcement plugin always loaded
            "-I" + cfg.include_root,
        });
        auto compiler = llvm::cantFail(builder.CreateCpp());
        interp_ = llvm::cantFail(
            clang::Interpreter::create(std::move(compiler)));
        inject_preamble();
    }

private:
    void inject_preamble() {
        // Every session starts with the full MathScript library pre-included
        constexpr std::string_view preamble = R"(
            #include <ms/ms.hpp>
            using namespace ms;
            using namespace ms::linalg;
            using namespace ms::fft;
            using namespace ms::signal;
            using namespace ms::stats;
            using namespace ms::sym;
            using namespace ms::special;
            using namespace ms::gria;
            using namespace ms::izaac;
            using namespace ms::cypha;
            using namespace ms::cellai;
            using namespace ms::axiom;
        )";
        llvm::cantFail(interp_->Parse(preamble));
    }
};
```

---

## 4.3 Session State & Symbol Reflection

The workspace panel in the GUI needs to know what variables exist, their types, and their values. This is done by reflecting over the LLVM JIT symbol table after each execution.

```cpp
// src/interp/session.hpp
struct VariableInfo {
    std::string name;
    std::string type;             // demangled C++ type string
    std::string preview;          // short string representation
    size_t      size_bytes;
    bool        on_device;        // true if in GPU memory
};

class SessionState {
    std::vector<VariableInfo> variables_;

public:
    void update_from_jit(clang::Interpreter& interp);
    std::span<const VariableInfo> variables() const;
    std::optional<VariableInfo>   find(std::string_view name) const;
    void clear();
};
```

---

## 4.4 Dispatch Layer

The dispatch layer is the heart of the runtime. Every math operation passes through it. It decides: CPU or GPU, which SIMD width, which algorithm variant.

```cpp
// include/ms/runtime/dispatch.hpp
namespace ms::runtime {

enum class Backend { CPU, CUDA };

struct DispatchDecision {
    Backend  backend;
    int      cuda_device;          // -1 if CPU
    int      n_threads;            // 0 = all available
    ISA      simd_isa;
    bool     use_distributed;
};

// Dispatch oracle — called by every math function
DispatchDecision decide(
    size_t        n_elements,
    OpClass       op,
    ExecPolicy    user_policy,    // CPU{}, GPU{}, AUTO{}
    const SystemTopology& topo
);

} // namespace ms::runtime
```

Dispatch rules (in priority order):

1. If user specified `CPU{}` or `GPU{}` — honour it
2. If CUDA unavailable — CPU
3. If `n_elements < cpu_threshold[op]` — CPU (GPU launch overhead not worth it)
4. If GPU utilisation > 85% — CPU (avoid contention)
5. If PCIe transfer cost > compute savings — CPU
6. Otherwise — GPU

These thresholds are tuned per operation class during benchmarking and stored in a config table, not hardcoded.

---

## 4.5 Execution Policy Tags

User-facing policy types that feed into the dispatch layer:

```cpp
// include/ms/runtime/exec_policy.hpp
namespace ms {

struct CPU {
    int threads = -1;   // -1 = hardware_concurrency()
    ISA isa = ISA::AUTO;
};

struct GPU {
    int device = 0;
    cudaStream_t stream = 0;
};

struct AUTO {};

struct NUMA {
    int node = -1;      // -1 = nearest to calling thread
};

struct Distributed {
    MPIContext* ctx;
    int root_rank = 0;
};

using ExecPolicy = std::variant<CPU, GPU, AUTO, NUMA, Distributed>;

} // namespace ms
```

Usage in user code:

```cpp
auto R = ms::matmul(A, B);              // AUTO — dispatch decides
auto R = ms::matmul(A, B, GPU{1});      // explicit GPU device 1
auto R = ms::matmul(A, B, CPU{8});      // explicit 8-thread CPU
auto R = ms::fft(x, GPU{});             // GPU FFT
auto R = ms::solve(dA, b, Distributed{&mpi_ctx});
```

---

## 4.6 Load Balancer

The load balancer operates at the dispatch decision layer. It maintains a runtime state object updated by a background monitor thread:

```cpp
// src/runtime/load_balancer.hpp
class LoadBalancer {
    struct State {
        std::atomic<float> gpu_utilisation;     // 0.0 – 1.0
        std::atomic<float> gpu_memory_free_gb;
        std::atomic<int>   active_gpu_streams;
        std::atomic<float> cpu_utilisation;
        std::atomic<size_t> free_host_memory_gb;
    };

    State state_;
    std::jthread monitor_thread_;

public:
    LoadBalancer();

    // Called by dispatch layer
    Backend recommend(size_t n_elements, OpClass op) const;

    // Called by CUDA stream completion callbacks
    void notify_gpu_complete(size_t stream_id);

    // Background monitor — queries nvml + /proc/stat
    void monitor_loop(std::stop_token st);
};
```

The monitor thread queries NVML (NVIDIA Management Library) for GPU utilisation and `/proc/stat` on Linux or `GetSystemTimes` on Windows for CPU load. It updates the atomic state every 100ms. The dispatch layer reads this state lock-free.

---

## 4.7 Thread Model

**GUI thread:** Qt event loop only. Never blocks. Never calls into the interpreter directly.

**Interpreter thread:** Single `QThread` running the Clang-REPL session. Receives work via a thread-safe queue. Posts results back via `QMetaObject::invokeMethod` with `Qt::QueuedConnection`.

**TBB pool threads:** oneTBB work-stealing pool. Drives all CPU parallelism in math functions. NUMA-pinned — one TBB arena per NUMA node.

**CUDA stream threads:** CUDA operations are asynchronous. The CUDA driver has its own internal threads for the DMA engine and the execution engine. MathScript's CUDA backend uses a stream pool and posts completion callbacks that update the load balancer.

**MPI ranks:** In distributed mode, ranks 1..N run a headless interpreter session. The GUI only runs on rank 0.

**Communication between GUI thread and interpreter thread:**

```cpp
// Posted by interpreter thread, consumed by GUI thread
struct ReplOutputEvent {
    QString text;
    bool is_error;
};

struct PlotReadyEvent {
    PlotData data;
};

struct VariablesChangedEvent {
    std::vector<VariableInfo> variables;
};
```

---

## 4.8 Preamble System

The preamble is injected once at session start. It is configurable per session via the `InterpreterConfig`:

```cpp
struct InterpreterConfig {
    std::filesystem::path  plugin_path;
    std::filesystem::path  include_root;
    std::vector<std::string> extra_preamble;   // user-defined startup code
    std::string            session_name;
    bool                   headless = false;
};
```

The default preamble includes all MathScript namespaces. Users can add their own startup code via the Settings panel — equivalent to a `.mathscriptrc` file. This startup code goes through the same plugin enforcement as everything else.


---

## Part 5: Memory System

---

## 5.1 Memory Architecture Overview

Five memory classes, each with a dedicated allocator. The correct allocator is selected automatically based on context. Users never interact with allocators directly.

```
┌─────────────────────────────────────────────────────────┐
│ Class 1: General Heap — mimalloc                        │
│ Replaces malloc system-wide. Thread-local caches.       │
├─────────────────────────────────────────────────────────┤
│ Class 2: SIMD-Aligned — AlignedAllocator<T, 64>         │
│ 64-byte alignment for AVX-512. All Matrix/Tensor data.  │
├─────────────────────────────────────────────────────────┤
│ Class 3: CUDA Pinned — PinnedAllocator<T>               │
│ cudaMallocHost. DMA-capable. GPU-bound matrices.        │
├─────────────────────────────────────────────────────────┤
│ Class 4: NUMA-Local — NumaAllocator<T>                  │
│ libnuma. Allocated on nearest NUMA node to caller.      │
├─────────────────────────────────────────────────────────┤
│ Class 5: Scratch/Arena — ArenaAllocator                 │
│ std::pmr monotonic. Per-thread. Resets each expression. │
└─────────────────────────────────────────────────────────┘
```

---

## 5.2 Class 1 — mimalloc

Replaces the system allocator globally. No code changes — just link order.

```cmake
# CMakeLists.txt — must be first library linked
target_link_libraries(mathscript PRIVATE mimalloc)
```

On Windows: `mimalloc-override.lib` is linked first, intercepting all CRT allocation calls.
On Linux: `LD_PRELOAD` or link-order override.

mimalloc provides:
- Thread-local allocation caches — zero lock contention at scale
- Low fragmentation over long sessions
- Free-list randomisation — security hardening
- Heap isolation per thread
- Double-free detection

---

## 5.3 Class 2 — AlignedAllocator

```cpp
// include/ms/memory/aligned_allocator.hpp
namespace ms::memory {

template<typename T, size_t Align = 64>
class AlignedAllocator {
public:
    using value_type      = T;
    using size_type       = size_t;
    using difference_type = ptrdiff_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Align>; };

    [[nodiscard]] T* allocate(size_t n) {
        if (n == 0) return nullptr;
        const size_t bytes = n * sizeof(T);
        void* ptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes, Align);
#else
        ptr = std::aligned_alloc(Align, bytes);
#endif
        if (!ptr) throw std::bad_alloc{};
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Align>&) const noexcept {
        return true;
    }

    static constexpr size_t alignment = Align;
};

} // namespace ms::memory
```

All `Matrix<T>` and `Tensor<T>` use `AlignedAllocator<T, 64>` as their default allocator. This is baked into the type definition — users cannot accidentally use an unaligned allocator for these types.

---

## 5.4 Class 3 — PinnedAllocator

```cpp
// include/ms/memory/pinned_allocator.hpp
namespace ms::memory {

template<typename T>
class PinnedAllocator {
public:
    using value_type = T;

    [[nodiscard]] T* allocate(size_t n) {
        T* ptr = nullptr;
        const cudaError_t err = cudaMallocHost(
            reinterpret_cast<void**>(&ptr),
            n * sizeof(T)
        );
        if (err != cudaSuccess || !ptr) {
            // Cannot throw — use Result<> at higher level
            // Caller must check for null
            return nullptr;
        }
        return ptr;
    }

    void deallocate(T* ptr, size_t) noexcept {
        if (ptr) cudaFreeHost(ptr);
    }

    template<typename U>
    bool operator==(const PinnedAllocator<U>&) const noexcept { return true; }
};

template<typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;

} // namespace ms::memory
```

Used automatically when a `Matrix` is created with `GPU{}` policy or when the dispatch layer promotes a matrix to device use. The `DevicePair` class (see CUDA layer) wraps a `PinnedVector` for the host side.

---

## 5.5 Class 4 — NumaAllocator

```cpp
// include/ms/memory/numa_allocator.hpp
namespace ms::memory {

class NumaTopology {
public:
    static NumaTopology& instance();

    int  nearest_node() const;
    int  node_for_cpu(int cpu_id) const;
    int  num_nodes() const;
    bool available() const;   // false on systems without NUMA
    size_t node_memory_free(int node) const;

private:
    NumaTopology();
    // hwloc_topology_t on Linux, GetNumaHighestNodeNumber on Windows
};

template<typename T>
class NumaAllocator {
    int node_;
public:
    using value_type = T;

    explicit NumaAllocator(int node = -1)
        : node_(node == -1
            ? NumaTopology::instance().nearest_node()
            : node) {}

    [[nodiscard]] T* allocate(size_t n) {
        if (!NumaTopology::instance().available()) {
            // Fall back to aligned allocator on non-NUMA systems
            return AlignedAllocator<T, 64>{}.allocate(n);
        }
#if defined(_WIN32)
        void* ptr = VirtualAllocExNuma(GetCurrentProcess(),
            nullptr, n * sizeof(T),
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, node_);
#else
        void* ptr = numa_alloc_onnode(n * sizeof(T), node_);
#endif
        if (!ptr) throw std::bad_alloc{};
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t n) noexcept {
        if (!ptr) return;
#if defined(_WIN32)
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        numa_free(ptr, n * sizeof(T));
#endif
    }

    template<typename U>
    bool operator==(const NumaAllocator<U>& o) const noexcept {
        return node_ == o.node_;
    }

    int node() const { return node_; }
};

} // namespace ms::memory
```

---

## 5.6 Class 5 — Arena Allocator

Per-thread monotonic arena. Resets at the end of each top-level expression evaluation. Eliminates heap pressure from short-lived temporaries in expression templates and symbolic evaluation.

```cpp
// include/ms/memory/arena.hpp
namespace ms::memory {

class Arena {
    static constexpr size_t DEFAULT_CAPACITY = 64 * 1024 * 1024; // 64 MB

    std::pmr::monotonic_buffer_resource resource_;
    std::pmr::polymorphic_allocator<std::byte> alloc_;
    size_t capacity_;

public:
    explicit Arena(size_t capacity = DEFAULT_CAPACITY)
        : resource_(capacity, std::pmr::get_default_resource())
        , alloc_(&resource_)
        , capacity_(capacity) {}

    template<typename T>
    T* allocate(size_t n = 1) {
        return alloc_.allocate_object<T>(n);
    }

    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate<T>();
        std::construct_at(ptr, std::forward<Args>(args)...);
        return ptr;
    }

    void reset() noexcept {
        resource_.release();
    }

    size_t capacity() const { return capacity_; }
};

// Per-thread arena — zero contention, no locking
inline thread_local Arena scratch_arena;

// RAII scope guard — resets arena on destruction
struct [[nodiscard]] ScratchScope {
    ~ScratchScope() { scratch_arena.reset(); }
};

} // namespace ms::memory
```

The interpreter wraps every statement execution in a `ScratchScope`:

```cpp
Result<ExprResult> ClangReplInterpreter::execute(std::string_view code) {
    ms::memory::ScratchScope scope;   // arena resets when this returns
    return execute_impl(code);
}
```

---

## 5.7 Pool Allocator for Fixed-Size Objects

The symbolic CAS layer allocates millions of small expression tree nodes. A slab-based pool allocator eliminates fragmentation and overhead:

```cpp
// include/ms/memory/pool_allocator.hpp
namespace ms::memory {

template<size_t BlockSize, size_t BlocksPerSlab = 4096>
class PoolAllocator {
    static_assert(BlockSize >= sizeof(void*));

    struct Slab {
        alignas(64) std::byte data[BlockSize * BlocksPerSlab];
        Slab* next;
    };

    Slab*  slabs_    = nullptr;
    void*  freelist_ = nullptr;
    size_t allocated_= 0;
    std::mutex mtx_;              // pool is shared, needs a lock

public:
    void* allocate() {
        std::lock_guard lock(mtx_);
        if (!freelist_) grow();
        void* ptr = freelist_;
        freelist_ = *static_cast<void**>(freelist_);
        ++allocated_;
        return ptr;
    }

    void deallocate(void* ptr) noexcept {
        std::lock_guard lock(mtx_);
        *static_cast<void**>(ptr) = freelist_;
        freelist_ = ptr;
        --allocated_;
    }

    size_t allocated() const { return allocated_; }

private:
    void grow() {
        auto* slab = new Slab;
        slab->next = slabs_;
        slabs_ = slab;
        // Chain all blocks into the free list
        for (size_t i = 0; i < BlocksPerSlab; ++i) {
            void* block = slab->data + i * BlockSize;
            *static_cast<void**>(block) = freelist_;
            freelist_ = block;
        }
    }
};

// Pre-defined sizes for common symbolic node sizes
extern PoolAllocator<16>  pool16;
extern PoolAllocator<32>  pool32;
extern PoolAllocator<64>  pool64;
extern PoolAllocator<128> pool128;

} // namespace ms::memory
```

---

## 5.8 Allocator Policy Selection

```cpp
// include/ms/memory/policy.hpp
namespace ms::memory {

enum class MemoryKind {
    Default,    // AlignedAllocator<T, 64>
    Pinned,     // PinnedAllocator<T>      — GPU-bound
    NUMA,       // NumaAllocator<T>        — NUMA-local
    Scratch,    // From thread-local arena — short-lived
};

template<typename T, MemoryKind Kind = MemoryKind::Default>
using SelectAllocator = std::conditional_t<
    Kind == MemoryKind::Pinned,  PinnedAllocator<T>,
    std::conditional_t<
    Kind == MemoryKind::NUMA,    NumaAllocator<T>,
    std::conditional_t<
    Kind == MemoryKind::Scratch, std::pmr::polymorphic_allocator<T>,
                                 AlignedAllocator<T, 64>>>>;

} // namespace ms::memory
```

---

## 5.9 System Topology Detection

Hardware topology is detected once at startup using hwloc on Linux and GetSystemInfo/GetNumaHighestNodeNumber on Windows. The result drives allocator selection, thread pinning, and MPI rank assignment.

```cpp
// include/ms/runtime/topology.hpp
namespace ms::runtime {

struct CPUNode {
    int            numa_id;
    std::vector<int> logical_cpu_ids;
    std::vector<int> gpu_ids;          // GPUs attached to this NUMA node
    size_t         memory_bytes;
};

struct SystemTopology {
    std::vector<CPUNode>   numa_nodes;
    int                    total_logical_cpus;
    int                    total_gpus;
    bool                   has_infiniband;
    bool                   has_gpu_direct;

    static SystemTopology  detect();   // called once at startup

    // For a given MPI rank, which NUMA node + GPUs does it own?
    CPUNode& assignment_for_rank(int rank);

    // For dispatching: nearest GPU to calling thread's NUMA node
    int nearest_gpu() const;
};

// Global topology — populated at startup
extern const SystemTopology g_topology;

} // namespace ms::runtime
```

At startup sequence:

```cpp
// src/exe/main.cpp
int main(int argc, char** argv) {
    // 1. Detect hardware topology
    ms::runtime::g_topology = ms::runtime::SystemTopology::detect();

    // 2. Initialise mimalloc (already active via link order)

    // 3. Set up TBB arenas pinned to NUMA nodes
    ms::runtime::ThreadPool::initialise(ms::runtime::g_topology);

    // 4. Initialise CUDA devices
    ms::cuda::DeviceManager::initialise(ms::runtime::g_topology);

    // 5. Start GUI (or headless session)
    ...
}
```


---

## Part 6: Type System & Core Types

---

## 6.1 Design Principles

- Column-major storage by default — BLAS/LAPACK/cuBLAS compatibility
- Expression templates for lazy evaluation — zero temporaries in pipelines
- Owning vs non-owning distinction enforced by the type system
- Allocator as a template parameter — layout and memory class are compile-time properties
- No virtual dispatch — all polymorphism via `std::variant` + CRTP
- Bounds checking on by default — unchecked access requires `[[ms::unsafe]]`

---

## 6.2 Matrix Type

```cpp
// include/ms/core/matrix.hpp
namespace ms {

enum class StorageOrder { ColMajor, RowMajor };

template<
    typename Scalar,
    StorageOrder Order  = StorageOrder::ColMajor,
    template<typename> class Alloc = memory::AlignedAllocator
>
class Matrix : public expr::MatExpr<Matrix<Scalar, Order, Alloc>> {
public:
    using scalar_type     = Scalar;
    using allocator_type  = Alloc<Scalar>;
    using storage_type    = std::vector<Scalar, Alloc<Scalar>>;

    // Construction
    Matrix() = default;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, Scalar fill);
    Matrix(std::initializer_list<std::initializer_list<Scalar>> init);

    // From expression template — evaluates lazily
    template<typename Expr>
    explicit Matrix(const expr::MatExpr<Expr>& expr);

    // Shape
    size_t rows()  const noexcept { return rows_; }
    size_t cols()  const noexcept { return cols_; }
    size_t size()  const noexcept { return rows_ * cols_; }
    bool   empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    // Bounds-checked access (default)
    Result<Scalar&>       at(size_t r, size_t c);
    Result<const Scalar&> at(size_t r, size_t c) const;

    // Unchecked access — requires [[ms::unsafe]]
    [[nodiscard]] Scalar& operator()(size_t r, size_t c) noexcept;
    [[nodiscard]] const Scalar& operator()(size_t r, size_t c) const noexcept;

    // Raw data — for BLAS/CUDA interop inside [[ms::unsafe]]
    Scalar*       data()       noexcept { return storage_.data(); }
    const Scalar* data() const noexcept { return storage_.data(); }

    // Non-owning view — zero copy
    MatrixView<Scalar, Order> view() noexcept;
    MatrixView<const Scalar, Order> view() const noexcept;

    // Slice — non-owning sub-matrix
    MatrixView<Scalar, Order> slice(
        size_t row_start, size_t row_end,
        size_t col_start, size_t col_end);

    // Reshape — non-owning reinterpretation
    Result<MatrixView<Scalar, Order>> reshape(size_t new_rows, size_t new_cols);

    // Device management
    void         to_device(cudaStream_t s = 0);
    void         to_host(cudaStream_t s = 0);
    bool         on_device() const;
    Scalar*      device_data();

    // Print / display
    void         print(size_t max_rows = 10, size_t max_cols = 10) const;
    std::string  to_string() const;

private:
    size_t       rows_     = 0;
    size_t       cols_     = 0;
    storage_type storage_;
    // Optional device mirror — nullptr if not on GPU
    cuda::DeviceBuffer<Scalar> device_buf_;
    bool device_valid_  = false;
    bool host_valid_    = true;
};

// Common aliases
template<typename S> using ColMatrix = Matrix<S, StorageOrder::ColMajor>;
template<typename S> using RowMatrix = Matrix<S, StorageOrder::RowMajor>;
template<typename S> using PinnedMatrix = Matrix<S, StorageOrder::ColMajor,
                                                  memory::PinnedAllocator>;
template<typename S> using NumaMatrix   = Matrix<S, StorageOrder::ColMajor,
                                                  memory::NumaAllocator>;

} // namespace ms
```

---

## 6.3 MatrixView — Non-Owning

```cpp
// include/ms/core/matrix.hpp (continued)
namespace ms {

template<typename Scalar, StorageOrder Order = StorageOrder::ColMajor>
class MatrixView {
public:
    MatrixView(Scalar* data, size_t rows, size_t cols, size_t stride);

    size_t rows()   const noexcept;
    size_t cols()   const noexcept;
    size_t stride() const noexcept;

    Result<Scalar&>       at(size_t r, size_t c);
    Result<const Scalar&> at(size_t r, size_t c) const;

    [[nodiscard]] Scalar& operator()(size_t r, size_t c) noexcept;

    // Convert to owning Matrix (copies data)
    Matrix<Scalar, Order> copy() const;

private:
    Scalar* data_;
    size_t  rows_, cols_, stride_;
};

} // namespace ms
```

---

## 6.4 Tensor Type

```cpp
// include/ms/core/tensor.hpp
namespace ms {

template<
    typename Scalar,
    size_t Rank,
    template<typename> class Alloc = memory::AlignedAllocator
>
class Tensor {
public:
    using shape_type = std::array<size_t, Rank>;

    Tensor() = default;
    explicit Tensor(shape_type shape);
    Tensor(shape_type shape, Scalar fill);

    shape_type shape()  const noexcept { return shape_; }
    size_t     size()   const noexcept;
    size_t     rank()   const noexcept { return Rank; }

    // Multi-index bounds-checked access
    template<typename... Idx>
    Result<Scalar&> at(Idx... indices)
        requires (sizeof...(Idx) == Rank);

    // Unchecked
    template<typename... Idx>
    Scalar& operator()(Idx... indices) noexcept
        requires (sizeof...(Idx) == Rank);

    // Non-owning view via std::mdspan
    auto view() noexcept {
        return std::mdspan(storage_.data(), shape_);
    }

    Scalar*       data()       noexcept { return storage_.data(); }
    const Scalar* data() const noexcept { return storage_.data(); }

    // Reshape — returns TensorView (zero copy)
    template<size_t NewRank>
    Result<TensorView<Scalar, NewRank>> reshape(
        std::array<size_t, NewRank> new_shape);

private:
    shape_type   shape_{};
    std::vector<Scalar, Alloc<Scalar>> storage_;
};

} // namespace ms
```

---

## 6.5 Sparse Matrix

```cpp
// include/ms/core/sparse.hpp
namespace ms {

enum class SparseFormat { COO, CSR, CSC };

template<typename Scalar, typename Index = int32_t>
class Sparse {
public:
    // Construction
    static Sparse from_triplets(
        size_t rows, size_t cols,
        std::span<const Index> row_idx,
        std::span<const Index> col_idx,
        std::span<const Scalar> values);

    static Sparse identity(size_t n);
    static Sparse diag(std::span<const Scalar> d);

    // Shape
    size_t rows()  const noexcept;
    size_t cols()  const noexcept;
    size_t nnz()   const noexcept;    // number of non-zeros

    // Format conversion
    Sparse to_csr() const;
    Sparse to_csc() const;
    Sparse to_coo() const;

    // Element access — O(log nnz) for CSR
    Result<Scalar> get(size_t r, size_t c) const;
    Result<void>   set(size_t r, size_t c, Scalar v);

    // Convert to dense
    Matrix<Scalar> to_dense() const;

    // Arithmetic (returns Sparse)
    Sparse add(const Sparse& other) const;
    Sparse mul(const Sparse& other) const;    // sparse-sparse multiply
    Matrix<Scalar> mul_dense(const Matrix<Scalar>& B) const; // SpMM

    // Internal storage — accessible for CUDA/BLAS under [[ms::unsafe]]
    const std::vector<Index>&  row_ptr()  const;   // CSR row pointers
    const std::vector<Index>&  col_idx()  const;   // CSR column indices
    const std::vector<Scalar>& values()   const;

private:
    size_t rows_, cols_;
    SparseFormat format_ = SparseFormat::CSR;
    std::vector<Index>  row_ptr_;
    std::vector<Index>  col_idx_;
    std::vector<Scalar> values_;
};

} // namespace ms
```

---

## 6.6 Expression Templates

Lazy evaluation eliminates temporaries in expressions like `A * B + C * D`.

```cpp
// include/ms/core/expr.hpp
namespace ms::expr {

// Base CRTP class for all matrix expressions
template<typename Derived>
class MatExpr {
public:
    const Derived& self() const { return static_cast<const Derived&>(*this); }

    size_t rows() const { return self().rows_impl(); }
    size_t cols() const { return self().cols_impl(); }

    // Force evaluation to an owning Matrix
    template<typename S, StorageOrder O, template<typename> class A>
    operator Matrix<S, O, A>() const {
        Matrix<S, O, A> result(rows(), cols());
        for (size_t r = 0; r < rows(); ++r)
            for (size_t c = 0; c < cols(); ++c) {
                [[ms::unsafe("bounds guaranteed by expression shape")]]
                result(r, c) = self().eval_at(r, c);
            }
        return result;
    }
};

// Add expression
template<typename L, typename R>
class MatAdd : public MatExpr<MatAdd<L, R>> {
    const L& l_; const R& r_;
public:
    MatAdd(const MatExpr<L>& l, const MatExpr<R>& r) : l_(l.self()), r_(r.self()) {}
    size_t rows_impl() const { return l_.rows(); }
    size_t cols_impl() const { return l_.cols(); }
    auto   eval_at(size_t r, size_t c) const { return l_.eval_at(r, c) + r_.eval_at(r, c); }
};

// Scale expression
template<typename M, typename S>
class MatScale : public MatExpr<MatScale<M, S>> {
    const M& m_; S s_;
public:
    MatScale(const MatExpr<M>& m, S s) : m_(m.self()), s_(s) {}
    size_t rows_impl() const { return m_.rows(); }
    size_t cols_impl() const { return m_.cols(); }
    auto   eval_at(size_t r, size_t c) const { return m_.eval_at(r, c) * s_; }
};

// Operators
template<typename L, typename R>
auto operator+(const MatExpr<L>& l, const MatExpr<R>& r) {
    return MatAdd<L, R>(l, r);
}

template<typename M, typename S>
auto operator*(const MatExpr<M>& m, S s) {
    return MatScale<M, S>(m, s);
}

} // namespace ms::expr
```

---

## 6.7 Symbolic Types

```cpp
// include/ms/core/sym.hpp
namespace ms {

// Symbolic scalar — wraps a SymEngine expression
class Sym {
public:
    explicit Sym(std::string_view name);   // creates a symbol x
    explicit Sym(double value);            // wraps a numeric constant
    explicit Sym(const BigInt& value);
    explicit Sym(const Rational& value);

    // Arithmetic operators — return new Sym expressions
    Sym operator+(const Sym& other) const;
    Sym operator-(const Sym& other) const;
    Sym operator*(const Sym& other) const;
    Sym operator/(const Sym& other) const;
    Sym operator^(const Sym& exp) const;   // power
    Sym operator-() const;

    // Numeric evaluation
    Result<double>  eval_double(
        std::span<const std::pair<std::string, double>> subs) const;
    Result<APFloat> eval_precise(
        std::span<const std::pair<std::string, APFloat>> subs,
        size_t precision_bits) const;

    // Representation
    std::string to_string() const;
    std::string to_latex()  const;
    std::string to_cpp()    const;    // code generation

    // Internal — for CAS functions
    const void* expr_ptr() const;     // SymEngine::RCP<Basic>

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ms
```

---

## 6.8 Scalar with Units

Physical units as a compile-time or runtime annotation. Arithmetic between incompatible units is an error.

```cpp
// include/ms/core/scalar.hpp
namespace ms {

// Base SI units as compile-time tags
struct units {
    int m = 0;    // metres
    int kg = 0;   // kilograms
    int s = 0;    // seconds
    int A = 0;    // amperes
    int K = 0;    // kelvin
    int mol = 0;  // moles
    int cd = 0;   // candela

    constexpr bool operator==(const units&) const = default;
    constexpr units operator*(const units& o) const;
    constexpr units operator/(const units& o) const;
};

template<typename T, units U = {}>
class Scalar {
public:
    explicit Scalar(T value) : value_(value) {}

    T value() const { return value_; }
    static constexpr units unit() { return U; }

    // Compatible units — same exponent vector
    Scalar operator+(const Scalar& o) const
        requires (U == o.unit()) { return Scalar(value_ + o.value_); }

    // Multiplication — units multiply
    template<units V>
    Scalar<T, U * V> operator*(const Scalar<T, V>& o) const {
        return Scalar<T, U * V>(value_ * o.value());
    }

private:
    T value_;
};

// Pre-defined unit types
using Metres    = Scalar<double, units{.m=1}>;
using Kilograms = Scalar<double, units{.kg=1}>;
using Seconds   = Scalar<double, units{.s=1}>;
using Newtons   = Scalar<double, units{.m=1, .kg=1, .s=-2}>;
using Joules    = Scalar<double, units{.m=2, .kg=1, .s=-2}>;
using Watts     = Scalar<double, units{.m=2, .kg=1, .s=-3}>;
using Pascals   = Scalar<double, units{.m=-1, .kg=1, .s=-2}>;

} // namespace ms
```


---

## Part 7: Plugin & Enforcement

---

## 7.1 Plugin Architecture

The MathScript C++ Profile is enforced by a Clang AST plugin. It runs as part of every compilation — in the CMake build and in every Clang-REPL session. It cannot be disabled.

The plugin is compiled as a shared library (`mathscript-plugin.dll` / `.so`) and loaded via:
- Build system: `-fplugin=path/to/mathscript-plugin.so`
- REPL: `clang::Interpreter::LoadDynamicLibrary()` before first user input

**Plugin structure:**

```
src/plugin/
├── MsPlugin.cpp          # Plugin registration + ASTConsumer
├── diagnostics.hpp       # Custom diagnostic IDs and messages
├── unsafe_registry.hpp   # Collects [[ms::unsafe]] sites
└── rules/
    ├── NoRawNew.cpp
    ├── NoMalloc.cpp
    ├── NoOwningRawPtr.cpp
    ├── NoVLA.cpp
    ├── NoCStyleCast.cpp
    ├── NoConstCast.cpp
    ├── NoUnsafeReinterpret.cpp
    ├── NarrowingCheck.cpp
    ├── NoThrow.cpp
    ├── NoCatch.cpp
    ├── UnusedExpected.cpp
    ├── NoUninit.cpp
    ├── NoStoredSpan.cpp
    ├── NoRawThread.cpp
    ├── NoRawMutexLock.cpp
    ├── NoVolatileSync.cpp
    ├── NoDetach.cpp
    ├── NoSignedUnsignedMix.cpp
    ├── NoRawPtrArithmetic.cpp
    └── UnsafeAudit.cpp
```

---

## 7.2 Plugin Registration

```cpp
// src/plugin/MsPlugin.cpp
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "rules/NoRawNew.hpp"
#include "rules/NoCStyleCast.hpp"
// ... all rule headers

namespace ms::plugin {

class MsASTConsumer : public clang::ASTConsumer {
    clang::CompilerInstance& CI_;
    UnsafeRegistry           unsafe_registry_;

public:
    explicit MsASTConsumer(clang::CompilerInstance& CI)
        : CI_(CI) {}

    void HandleTranslationUnit(clang::ASTContext& ctx) override {
        // Run all rules on this translation unit
        rules::NoRawNew         {CI_, ctx}.run();
        rules::NoMalloc         {CI_, ctx}.run();
        rules::NoOwningRawPtr   {CI_, ctx}.run();
        rules::NoVLA            {CI_, ctx}.run();
        rules::NoCStyleCast     {CI_, ctx}.run();
        rules::NoConstCast      {CI_, ctx}.run();
        rules::NoUnsafeReinterpret{CI_, ctx}.run();
        rules::NarrowingCheck   {CI_, ctx}.run();
        rules::NoThrow          {CI_, ctx}.run();
        rules::NoCatch          {CI_, ctx}.run();
        rules::UnusedExpected   {CI_, ctx}.run();
        rules::NoRawThread      {CI_, ctx}.run();
        rules::NoRawMutexLock   {CI_, ctx}.run();
        rules::NoVolatileSync   {CI_, ctx}.run();
        rules::NoDetach         {CI_, ctx}.run();
        rules::NoSignedUnsignedMix{CI_, ctx}.run();
        rules::NoRawPtrArithmetic {CI_, ctx}.run();
        rules::UnsafeAudit      {CI_, ctx, unsafe_registry_}.run();
    }
};

class MsPlugin : public clang::PluginASTAction {
public:
    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance& CI,
                      llvm::StringRef) override {
        return std::make_unique<MsASTConsumer>(CI);
    }

    bool ParseArgs(const clang::CompilerInstance&,
                   const std::vector<std::string>&) override {
        return true;
    }

    clang::PluginASTAction::ActionType getActionType() override {
        return AddBeforeMainAction;   // runs before main compilation
    }
};

} // namespace ms::plugin

static clang::FrontendPluginRegistry::Add<ms::plugin::MsPlugin>
    X("ms-profile", "MathScript C++ Profile Enforcement");
```

---

## 7.3 Example Rule Implementation

```cpp
// src/plugin/rules/NoRawNew.hpp
namespace ms::plugin::rules {

class NoRawNew {
    clang::CompilerInstance& CI_;
    clang::ASTContext&       ctx_;

public:
    NoRawNew(clang::CompilerInstance& CI, clang::ASTContext& ctx)
        : CI_(CI), ctx_(ctx) {}

    void run() {
        using namespace clang::ast_matchers;
        auto matcher = cxxNewExpr().bind("new_expr");

        MatchFinder finder;
        finder.addMatcher(matcher, this);
        finder.matchAST(ctx_);
    }

    void onMatch(const clang::ast_matchers::MatchFinder::MatchResult& r) {
        const auto* expr = r.Nodes.getNodeAs<clang::CXXNewExpr>("new_expr");
        if (!expr) return;

        // Check if this new-expression is inside [[ms::unsafe]]
        if (is_inside_unsafe_block(expr, ctx_)) return;

        auto& diag = CI_.getDiagnostics();
        const auto id = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'new' is prohibited. "
            "Use ms::make_unique or ms::make_shared instead."
        );
        diag.Report(expr->getBeginLoc(), id);
    }
};

} // namespace ms::plugin::rules
```

---

## 7.4 The [[ms::unsafe]] Annotation

The `[[ms::unsafe("reason")]]` attribute is a custom Clang attribute. It marks a declaration or statement as exempt from profile rules and records the justification.

**Attribute definition (registered in the plugin):**

```cpp
// Parsed from [[ms::unsafe("reason string")]]
struct UnsafeAttr {
    std::string justification;
    clang::SourceLocation location;
    std::string filename;
    unsigned    line;
};
```

**Usage:**

```cpp
// On a statement
[[ms::unsafe("AVX-512 load: ptr is 64-byte aligned by construction")]]
__m512d v = _mm512_loadu_pd(ptr);

// On a function — entire function body is exempt
[[ms::unsafe("CUDA interop boundary: raw pointer required by cudaMemcpy")]]
void copy_to_device(float* dst, const float* src, size_t n) {
    cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
}

// On a block (via lambda)
auto result = [&]() [[ms::unsafe("legacy C API boundary")]] {
    return old_c_function(raw_ptr, size);
}();
```

---

## 7.5 Unsafe Audit Report

Every build generates an audit report. CI is configured to alert if new unsafe sites appear without corresponding review.

```
$ mathscriptc --unsafe-report ./src

MathScript Unsafe Surface Report
Generated: 2025-04-11 14:23:01
Build: 0.3.0 (commit abc1234)

Total unsafe sites: 14

src/runtime/cuda/cublas_wrap.cu:47
  [[ms::unsafe("cuBLAS requires raw device pointer")]]
  File: cublas_wrap.cu, Line: 47

src/runtime/cuda/memory.cu:112
  [[ms::unsafe("cudaMemcpy requires void* — reinterpret at CUDA boundary")]]
  File: memory.cu, Line: 112

src/runtime/cpu/avx512_gemm.cpp:89
  [[ms::unsafe("AVX-512 store: 64-byte alignment guaranteed by AlignedAllocator")]]
  File: avx512_gemm.cpp, Line: 89

... (11 more)

Categories:
  CUDA interop:        8 sites
  SIMD intrinsics:     4 sites
  Legacy C API:        2 sites

Diff from last build: +0 new, -0 removed
```

The report is generated by `UnsafeAudit.cpp` which writes to `build/unsafe_report.txt` and fails the build if the site count increases without a corresponding entry in `UNSAFE_REVIEW.md`.

---

## 7.6 Compliance Test Infrastructure

Every plugin rule has two test files:

```
tests/compliance/
├── no_raw_new/
│   ├── fail.cpp           # MUST fail to compile with the plugin
│   └── ok.cpp             # MUST compile cleanly
├── no_cstyle_cast/
│   ├── fail.cpp
│   └── ok.cpp
├── no_throw/
│   ├── fail.cpp
│   └── ok.cpp
...
```

CMake drives compile-failure tests:

```cmake
# tests/compliance/CMakeLists.txt
function(add_compliance_test rule)
    # Fail test — must not compile
    add_executable(compliance_${rule}_fail ${rule}/fail.cpp)
    set_target_properties(compliance_${rule}_fail PROPERTIES
        EXCLUDE_FROM_ALL TRUE)
    add_test(NAME compliance_${rule}_fail
        COMMAND ${CMAKE_COMMAND}
            --build ${CMAKE_BINARY_DIR}
            --target compliance_${rule}_fail
    )
    set_tests_properties(compliance_${rule}_fail PROPERTIES
        WILL_FAIL TRUE)

    # OK test — must compile
    add_executable(compliance_${rule}_ok ${rule}/ok.cpp)
    add_test(NAME compliance_${rule}_ok
        COMMAND compliance_${rule}_ok)
endfunction()

add_compliance_test(no_raw_new)
add_compliance_test(no_cstyle_cast)
add_compliance_test(no_throw)
# ... all rules
```

---

## 7.7 Compiler Flags

These flags apply to all translation units unconditionally. They complement the plugin.

```cmake
# Applied to all targets system-wide
target_compile_options(ms_global_flags INTERFACE
    # Exceptions and RTTI off
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fno-exceptions -fno-rtti>
    $<$<CXX_COMPILER_ID:MSVC>:/EHs- /GR->

    # Warnings as errors
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Werror
        -Wconversion -Wshadow -Wnull-dereference
        -Wdouble-promotion -Wimplicit-fallthrough
    >
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>

    # Security
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:
        -fstack-protector-strong
    >
    $<$<CXX_COMPILER_ID:MSVC>:/GS>
)

# Debug-only sanitizers
target_compile_options(ms_global_flags INTERFACE
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang,AppleClang>>:
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    >
)

# Release security hardening
target_compile_options(ms_global_flags INTERFACE
    $<$<AND:$<CONFIG:Release>,$<PLATFORM_ID:Linux>>:
        -D_FORTIFY_SOURCE=3
        -fsanitize=undefined
    >
    $<$<AND:$<CONFIG:Release>,$<PLATFORM_ID:Windows>>:
        /sdl /guard:cf
    >
)

# Release linker flags (Linux)
target_link_options(ms_global_flags INTERFACE
    $<$<AND:$<CONFIG:Release>,$<PLATFORM_ID:Linux>>:
        -Wl,-z,relro,-z,now -pie -fuse-ld=lld
    >
)
```


---

## Part 8: SIMD Layer

---

## 8.1 Three Tiers

**Tier 1 — Automatic:** Library internals use intrinsics. Users see nothing.

**Tier 2 — Portable SIMD API:** `ms::simd::` namespace. ISA-portable over AVX2/AVX-512/NEON/SVE. Uses Google Highway as the portability backend (transitional — will be replaced with own implementation).

**Tier 3 — Raw Intrinsics:** Full `<immintrin.h>` / `<arm_neon.h>` available directly in the REPL and user code. No wrapping. Inside `[[ms::unsafe]]` for any pointer arithmetic.

---

## 8.2 ISA Detection

```cpp
// include/ms/simd/isa.hpp
namespace ms::simd {

struct ISACapabilities {
    // x86
    bool sse2      = false;
    bool sse4_1    = false;
    bool sse4_2    = false;
    bool avx       = false;
    bool avx2      = false;
    bool avx512f   = false;
    bool avx512bw  = false;
    bool avx512dq  = false;
    bool avx512vl  = false;
    bool avx512vnni= false;
    bool amx       = false;    // Intel AMX for matrix ops
    bool f16c      = false;
    bool fma       = false;

    // ARM
    bool neon      = false;
    bool sve       = false;
    bool sve2      = false;

    // Derived
    int  max_vector_bits = 128;   // 128, 256, or 512

    static ISACapabilities detect();
};

// Detected once at startup — read-only thereafter
extern const ISACapabilities g_isa;

inline bool has_avx512()  { return g_isa.avx512f; }
inline bool has_avx2()    { return g_isa.avx2;    }
inline bool has_neon()    { return g_isa.neon;    }
inline int  vector_bits() { return g_isa.max_vector_bits; }

} // namespace ms::simd
```

Detection via CPUID on x86, `/proc/cpuinfo` / hwcap on Linux/ARM, `IsProcessorFeaturePresent` on Windows.

---

## 8.3 Portable SIMD API

```cpp
// include/ms/simd/portable.hpp
namespace ms::simd {

// Portable vector type — width is ISA-dependent
template<typename T>
struct Vec;   // specialised per ISA at compile time via Highway

// Load/store
template<typename T>
Vec<T> load(const T* ptr);

template<typename T>
Vec<T> load_unaligned(const T* ptr);

template<typename T>
void store(T* ptr, Vec<T> v);

template<typename T>
void store_unaligned(T* ptr, Vec<T> v);

// Arithmetic
template<typename T> Vec<T> add(Vec<T> a, Vec<T> b);
template<typename T> Vec<T> sub(Vec<T> a, Vec<T> b);
template<typename T> Vec<T> mul(Vec<T> a, Vec<T> b);
template<typename T> Vec<T> div(Vec<T> a, Vec<T> b);
template<typename T> Vec<T> fmadd(Vec<T> a, Vec<T> b, Vec<T> c);  // a*b+c
template<typename T> Vec<T> fnmadd(Vec<T> a, Vec<T> b, Vec<T> c); // -(a*b)+c

// Comparison
template<typename T> Vec<T> min(Vec<T> a, Vec<T> b);
template<typename T> Vec<T> max(Vec<T> a, Vec<T> b);
template<typename T> auto   eq(Vec<T> a, Vec<T> b);   // mask
template<typename T> auto   lt(Vec<T> a, Vec<T> b);

// Math
template<typename T> Vec<T> sqrt(Vec<T> a);
template<typename T> Vec<T> rsqrt(Vec<T> a);
template<typename T> Vec<T> rcp(Vec<T> a);
template<typename T> Vec<T> abs(Vec<T> a);
template<typename T> Vec<T> neg(Vec<T> a);

// Reductions
template<typename T> T sum(Vec<T> a);
template<typename T> T reduce_min(Vec<T> a);
template<typename T> T reduce_max(Vec<T> a);

// Broadcast
template<typename T> Vec<T> set1(T scalar);

// Lane count
template<typename T> constexpr size_t lanes();

// Runtime ISA dispatch — runs the best available implementation
template<typename F>
void dispatch(F&& kernel);

} // namespace ms::simd
```

---

## 8.4 Custom AVX-512 Kernels

Key hot-path kernels are written in raw intrinsics, not via the portable layer. They live in `src/runtime/cpu/simd/`:

### DGEMM micro-kernel (AVX-512)

The micro-kernel is the innermost loop of matrix multiplication. For a 8x24 register block using AVX-512 zmm registers:

```cpp
// src/runtime/cpu/simd/avx512_dgemm.cpp
[[ms::unsafe("AVX-512 register blocking: ptr arithmetic is loop-controlled and bounded")]]
void avx512_dgemm_8x24(
    size_t k,
    const double* __restrict__ A,  // 8 x k panel
    const double* __restrict__ B,  // k x 24 panel
    double* __restrict__ C,        // 8 x 24 accumulator
    size_t ldc)
{
    // 24 zmm accumulators (8 rows x 3 zmm per row = 24 accumulators)
    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd(), c02 = _mm512_setzero_pd();
    // ... declare all 24 accumulators

    for (size_t p = 0; p < k; ++p) {
        // Load 3 zmm from B (8 doubles each = 24 columns)
        __m512d b0 = _mm512_loadu_pd(B + p * 24 + 0);
        __m512d b1 = _mm512_loadu_pd(B + p * 24 + 8);
        __m512d b2 = _mm512_loadu_pd(B + p * 24 + 16);

        // Broadcast each of 8 A elements, FMA with B rows
        for (size_t i = 0; i < 8; ++i) {
            __m512d a = _mm512_set1_pd(A[p * 8 + i]);
            c[i][0] = _mm512_fmadd_pd(a, b0, c[i][0]);
            c[i][1] = _mm512_fmadd_pd(a, b1, c[i][1]);
            c[i][2] = _mm512_fmadd_pd(a, b2, c[i][2]);
        }
    }

    // Store accumulators to C
    for (size_t i = 0; i < 8; ++i) {
        _mm512_storeu_pd(C + i * ldc + 0,  c[i][0]);
        _mm512_storeu_pd(C + i * ldc + 8,  c[i][1]);
        _mm512_storeu_pd(C + i * ldc + 16, c[i][2]);
    }
}
```

### Elementwise ops (AVX-512)

```cpp
// Vectorised exp() using Cephes polynomial approximation
[[ms::unsafe("AVX-512: pointer arithmetic bounded by loop")]]
void avx512_exp_f64(const double* src, double* dst, size_t n) {
    constexpr size_t W = 8;  // doubles per zmm
    size_t i = 0;
    for (; i + W <= n; i += W) {
        __m512d x = _mm512_loadu_pd(src + i);
        __m512d r = avx512_exp_poly(x);
        _mm512_storeu_pd(dst + i, r);
    }
    // Scalar tail
    for (; i < n; ++i) dst[i] = std::exp(src[i]);
}
```

---

## 8.5 SIMD Dispatch at Runtime

For functions with multiple ISA implementations, dispatch selects at runtime once and caches the function pointer:

```cpp
// src/runtime/cpu/dispatch_table.hpp
namespace ms::runtime {

// Function pointer type for elementwise double operations
using ElemwiseFnD = void(*)(const double*, double*, size_t);

struct DispatchTable {
    ElemwiseFnD exp_f64;
    ElemwiseFnD log_f64;
    ElemwiseFnD sin_f64;
    ElemwiseFnD cos_f64;
    // ... all dispatched functions

    static DispatchTable build(const simd::ISACapabilities& isa) {
        DispatchTable t;
        if (isa.avx512f) {
            t.exp_f64 = simd::avx512::exp_f64;
            t.log_f64 = simd::avx512::log_f64;
        } else if (isa.avx2) {
            t.exp_f64 = simd::avx2::exp_f64;
            t.log_f64 = simd::avx2::log_f64;
        } else if (isa.neon) {
            t.exp_f64 = simd::neon::exp_f64;
            t.log_f64 = simd::neon::log_f64;
        } else {
            t.exp_f64 = scalar::exp_f64;
            t.log_f64 = scalar::log_f64;
        }
        return t;
    }
};

extern const DispatchTable g_dispatch;  // built at startup

} // namespace ms::runtime
```

---

## 8.6 Intrinsics Wrapper Typedefs

For user code that wants to use intrinsics portably without raw Intel type names:

```cpp
// include/ms/simd/intrinsics.hpp
namespace ms::simd {

#if defined(__AVX512F__)
    using f64x8  = __m512d;   // 8 × f64
    using f32x16 = __m512;    // 16 × f32
    using i64x8  = __m512i;
    using i32x16 = __m512i;
#elif defined(__AVX2__)
    using f64x4  = __m256d;   // 4 × f64
    using f32x8  = __m256;    // 8 × f32
    using i64x4  = __m256i;
    using i32x8  = __m256i;
#elif defined(__ARM_NEON)
    using f64x2  = float64x2_t;
    using f32x4  = float32x4_t;
    using i32x4  = int32x4_t;
#endif

// Unified load/store that selects the right intrinsic
template<typename VecT, typename ScalarT>
VecT vec_load(const ScalarT* ptr);

template<typename VecT, typename ScalarT>
void vec_store(ScalarT* ptr, VecT v);

} // namespace ms::simd
```


---

## Part 9: CUDA Layer

---

## 9.1 CUDA Architecture

```
ms::cuda::
├── DeviceManager          — device enumeration, selection, properties
├── StreamPool             — per-device pool of CUDA streams
├── DeviceBuffer<T>        — RAII device memory
├── PinnedBuffer<T>        — RAII pinned host memory
├── DevicePair<T>          — paired host + device with transfer management
├── MultiGPUContext        — NCCL multi-GPU collective communication
├── KernelConfig           — launch configuration utilities
└── kernels/               — own CUDA kernel implementations
```

---

## 9.2 Device Manager

```cpp
// include/ms/cuda/device.hpp
namespace ms::cuda {

struct DeviceProperties {
    int    device_id;
    std::string name;
    size_t total_memory_bytes;
    int    compute_capability_major;
    int    compute_capability_minor;
    int    sm_count;
    int    max_threads_per_sm;
    size_t l2_cache_bytes;
    bool   supports_unified_memory;
    bool   supports_gpu_direct;
    int    numa_node;               // nearest CPU NUMA node
};

class DeviceManager {
public:
    static void  initialise(const runtime::SystemTopology& topo);
    static int   device_count();
    static const DeviceProperties& properties(int device_id);
    static bool  cuda_available();

    // Select best device for a given operation + data size
    static int   select_device(OpClass op, size_t n_elements);

    // Activate device for calling thread
    static void  set_device(int device_id);
    static int   current_device();
};

} // namespace ms::cuda
```

---

## 9.3 Stream Pool

```cpp
// include/ms/cuda/stream_pool.hpp
namespace ms::cuda {

class StreamPool {
    static constexpr size_t DEFAULT_STREAMS = 8;

    std::vector<cudaStream_t> streams_;
    std::atomic<size_t>       next_stream_{0};
    int                       device_id_;

public:
    explicit StreamPool(int device_id, size_t n_streams = DEFAULT_STREAMS);
    ~StreamPool();

    // Round-robin stream acquisition — lock-free
    cudaStream_t acquire() noexcept {
        return streams_[next_stream_.fetch_add(1,
            std::memory_order_relaxed) % streams_.size()];
    }

    void sync_all();
    void sync(cudaStream_t s);

    int device_id() const { return device_id_; }
};

// One stream pool per device — initialised at startup
extern std::vector<StreamPool> g_stream_pools;

inline cudaStream_t get_stream(int device = 0) {
    return g_stream_pools[device].acquire();
}

} // namespace ms::cuda
```

---

## 9.4 Device Buffer

```cpp
// include/ms/cuda/device_buffer.hpp
namespace ms::cuda {

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count, int device = 0);
    ~DeviceBuffer();

    // Non-copyable, movable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;

    size_t  size()    const noexcept { return count_; }
    bool    empty()   const noexcept { return count_ == 0; }
    int     device()  const noexcept { return device_; }

    // Raw pointer — only accessible inside [[ms::unsafe]]
    T*       device_ptr()       noexcept { return ptr_; }
    const T* device_ptr() const noexcept { return ptr_; }

    // Async transfers
    Result<void> copy_from_host(const T* src, size_t n,
                                cudaStream_t s = 0) noexcept;
    Result<void> copy_to_host(T* dst, size_t n,
                              cudaStream_t s = 0) const noexcept;
    Result<void> copy_from_device(const DeviceBuffer& src,
                                  cudaStream_t s = 0) noexcept;

    // Zero-fill
    Result<void> zero(cudaStream_t s = 0) noexcept;

    // Resize — reallocates if needed
    Result<void> resize(size_t new_count);

private:
    T*     ptr_    = nullptr;
    size_t count_  = 0;
    int    device_ = 0;
};

} // namespace ms::cuda
```

---

## 9.5 DevicePair — Host + Device Mirror

```cpp
// include/ms/cuda/device_buffer.hpp (continued)
namespace ms::cuda {

template<typename T>
class DevicePair {
public:
    explicit DevicePair(size_t count, int device = 0)
        : host_(count), device_(count, device) {}

    memory::PinnedVector<T>& host()         { return host_; }
    DeviceBuffer<T>&         device_buf()   { return device_; }

    // Transfer — async, uses stream pool
    Result<void> to_device(cudaStream_t s = 0) {
        if (device_valid_) return {};   // skip if already current
        MS_TRY(device_.copy_from_host(host_.data(), host_.size(), s));
        device_valid_ = true;
        return {};
    }

    Result<void> to_host(cudaStream_t s = 0) {
        if (host_valid_) return {};
        MS_TRY(device_.copy_to_host(host_.data(), host_.size(), s));
        host_valid_ = true;
        return {};
    }

    void invalidate_device() noexcept { device_valid_ = false; }
    void invalidate_host()   noexcept { host_valid_   = false; }

    bool device_valid() const { return device_valid_; }
    bool host_valid()   const { return host_valid_;   }

private:
    memory::PinnedVector<T> host_;
    DeviceBuffer<T>         device_;
    bool                    device_valid_ = false;
    bool                    host_valid_   = true;
};

} // namespace ms::cuda
```

---

## 9.6 Kernel Launch Configuration

```cpp
// include/ms/cuda/kernel_config.hpp
namespace ms::cuda {

struct LaunchConfig {
    dim3   grid;
    dim3   block;
    size_t shared_bytes = 0;
    cudaStream_t stream = 0;
};

// 1D kernel — n_elements work items
inline LaunchConfig configure_1d(size_t n_elements, int device = 0,
                                  cudaStream_t s = 0) {
    constexpr int BLOCK_SIZE = 256;
    const size_t grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return { .grid = {static_cast<unsigned>(grid), 1, 1},
             .block = {BLOCK_SIZE, 1, 1},
             .stream = s };
}

// 2D kernel — rows × cols
inline LaunchConfig configure_2d(size_t rows, size_t cols,
                                  int device = 0, cudaStream_t s = 0) {
    constexpr int TX = 16, TY = 16;
    return {
        .grid  = { static_cast<unsigned>((cols + TX - 1) / TX),
                   static_cast<unsigned>((rows + TY - 1) / TY), 1 },
        .block = { TX, TY, 1 },
        .stream = s
    };
}

// Optimal block size via CUDA occupancy API
LaunchConfig configure_optimal(const void* kernel,
                                size_t n_elements,
                                size_t shared_bytes = 0,
                                int device = 0,
                                cudaStream_t s = 0);

} // namespace ms::cuda
```

---

## 9.7 Custom CUDA Kernels

All custom kernels are in `src/runtime/cuda/kernels/`. Each has a host-callable wrapper declared in `include/ms/cuda/kernels.hpp`.

### Elementwise Operations

```cuda
// src/runtime/cuda/kernels/elementwise.cu

__global__ void kernel_exp_f64(const double* __restrict__ src,
                                double* __restrict__ dst, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = exp(src[i]);
}

__global__ void kernel_log_f64(const double* __restrict__ src,
                                double* __restrict__ dst, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = log(src[i]);
}

__global__ void kernel_elementwise_add_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ c, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void kernel_scale_f64(const double* __restrict__ src,
                                  double* __restrict__ dst,
                                  double s, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * s;
}
```

### Reduction

```cuda
// src/runtime/cuda/kernels/reduction.cu
// Uses shared memory tree reduction

__global__ void kernel_sum_f64(const double* __restrict__ src,
                                double* __restrict__ out, size_t n) {
    extern __shared__ double sdata[];
    const size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + tid;

    double sum = 0.0;
    while (i < n) {
        sum += src[i];
        if (i + blockDim.x < n) sum += src[i + blockDim.x];
        i += gridDim.x * blockDim.x * 2;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
```

### 2D Convolution with Shared Memory Tiling

```cuda
// src/runtime/cuda/kernels/conv2d.cu
#define TILE_SIZE 16
#define KERNEL_RADIUS 1   // generalise at runtime

__global__ void kernel_conv2d_f32(
    const float* __restrict__ input,
    const float* __restrict__ kernel_w,
    float* __restrict__ output,
    int in_h, int in_w,
    int k_h, int k_w) {

    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    // Load tile with halo into shared memory
    // ... (standard shared-memory tiling pattern)

    __syncthreads();

    if (row < in_h && col < in_w) {
        float acc = 0.0f;
        for (int kr = 0; kr < k_h; ++kr)
            for (int kc = 0; kc < k_w; ++kc)
                acc += tile[ty + kr][tx + kc] * kernel_w[kr * k_w + kc];
        output[row * in_w + col] = acc;
    }
}
```

---

## 9.8 cuBLAS Wrapper

```cpp
// src/runtime/cuda/cublas_wrap.cu
namespace ms::cuda::blas {

cublasHandle_t& get_handle(int device = 0) {
    thread_local std::vector<cublasHandle_t> handles;
    if (handles.empty()) {
        handles.resize(DeviceManager::device_count());
        for (int d = 0; d < DeviceManager::device_count(); ++d) {
            cudaSetDevice(d);
            cublasCreate(&handles[d]);
            cublasSetMathMode(handles[d], CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }
    return handles[device];
}

Result<void> dgemm(
    bool trans_A, bool trans_B,
    int M, int N, int K,
    double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double beta,
    double* C, int ldc,
    int device, cudaStream_t stream)
{
    [[ms::unsafe("cuBLAS requires raw device pointers — validated by DeviceBuffer")]]
    cublasSetStream(get_handle(device), stream);

    const cublasStatus_t st = cublasDgemm(
        get_handle(device),
        trans_A ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_B ? CUBLAS_OP_T : CUBLAS_OP_N,
        M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);

    if (st != CUBLAS_STATUS_SUCCESS)
        return ms::unexpected(DeviceError{st, "cublasDgemm failed"});
    return {};
}

} // namespace ms::cuda::blas
```

---

## 9.9 Multi-GPU via NCCL

```cpp
// include/ms/cuda/multi_gpu.hpp
namespace ms::cuda {

class MultiGPUContext {
public:
    MultiGPUContext();   // auto-detects all GPUs
    ~MultiGPUContext();

    int device_count() const;

    // AllReduce across all GPUs
    template<typename T>
    Result<void> all_reduce(
        std::span<T*> device_buffers,
        size_t count,
        ReduceOp op = ReduceOp::Sum);

    // Broadcast from root to all GPUs
    template<typename T>
    Result<void> broadcast(
        std::span<T*> device_buffers,
        size_t count, int root = 0);

    // Scatter: split buffer[root] across GPUs
    template<typename T>
    Result<void> scatter(T* src, std::span<T*> dst,
                         size_t per_gpu_count, int root = 0);

    // Gather: collect from all GPUs to root
    template<typename T>
    Result<void> gather(std::span<T*> src, T* dst,
                        size_t per_gpu_count, int root = 0);

    void sync_all();

private:
    std::vector<ncclComm_t>    comms_;
    std::vector<cudaStream_t>  streams_;
};

} // namespace ms::cuda
```


---

## Part 10: Distributed Processing

---

## 10.1 Distribution Architecture

Three tiers of parallelism compose cleanly:

```
Tier 3: Multi-Node   — MPI + UCX transport + NCCL GPU collectives
Tier 2: Multi-GPU    — NCCL within a single machine
Tier 1: Multi-Core   — oneTBB work-stealing pool, NUMA-pinned
```

All three can be active simultaneously. The dispatch layer selects the appropriate tier based on data size, topology, and policy.

**Execution model: SPMD (Single Program Multiple Data)**

All ranks run the same session. Standard operations are local. Distributed operations are explicit via collective functions. No master/worker distinction. The GUI runs on rank 0 only; ranks 1..N run headless sessions.

---

## 10.2 Thread Pool (Tier 1)

```cpp
// include/ms/runtime/thread_pool.hpp
namespace ms::runtime {

class ThreadPool {
    std::vector<tbb::task_arena> arenas_;   // one per NUMA node
    tbb::flow::graph             flow_;

public:
    static void initialise(const SystemTopology& topo);
    static ThreadPool& instance();

    // Submit work to the arena nearest to the calling thread's NUMA node
    template<typename F>
    void submit(F&& f) {
        arenas_[nearest_arena()].execute(std::forward<F>(f));
    }

    // Parallel for — auto-partitioned
    template<typename F>
    void parallel_for(size_t n, F&& f) {
        arenas_[nearest_arena()].execute([&] {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t i = r.begin(); i < r.end(); ++i) f(i);
                }
            );
        });
    }

    // Parallel for with grain size hint
    template<typename F>
    void parallel_for(size_t n, size_t grain, F&& f);

    // DAG execution via TBB flow graph
    tbb::flow::graph& flow_graph() { return flow_; }

    // Wait for all submitted work to complete
    void wait();

private:
    int nearest_arena() const;
};

} // namespace ms::runtime
```

---

## 10.3 MPI Context (Tier 3)

```cpp
// include/ms/distributed/mpi_context.hpp
namespace ms::distributed {

class MPIContext {
public:
    MPIContext(int* argc, char*** argv) {
        int provided;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        comm_ = MPI_COMM_WORLD;
    }

    ~MPIContext() { MPI_Finalize(); }

    int  rank()       const { return rank_; }
    int  world_size() const { return world_size_; }
    bool is_root()    const { return rank_ == 0; }

    // Blocking collectives
    template<typename T>
    Result<void> broadcast(T* data, size_t count, int root = 0);

    template<typename T>
    Result<void> all_reduce(const T* send, T* recv, size_t count, ReduceOp op);

    template<typename T>
    Result<void> reduce(const T* send, T* recv, size_t count,
                        ReduceOp op, int root = 0);

    template<typename T>
    Result<void> scatter(const T* send, T* recv, size_t count, int root = 0);

    template<typename T>
    Result<void> gather(const T* send, T* recv, size_t count, int root = 0);

    template<typename T>
    Result<void> all_gather(const T* send, T* recv, size_t count);

    template<typename T>
    Result<void> all_to_all(const T* send, T* recv, size_t count);

    template<typename T>
    Result<void> scan(const T* send, T* recv, size_t count, ReduceOp op);

    // Non-blocking point-to-point
    template<typename T>
    Result<MPI_Request> isend(const T* data, size_t count, int dest, int tag = 0);

    template<typename T>
    Result<MPI_Request> irecv(T* data, size_t count, int src, int tag = 0);

    Result<void> wait(MPI_Request& req);
    Result<void> waitall(std::span<MPI_Request> reqs);

    // Barrier
    Result<void> barrier();

    // Sub-communicators
    Result<MPIContext> split(int colour, int key);

    // Topology query
    bool has_gpu_direct() const;

private:
    int      rank_, world_size_;
    MPI_Comm comm_;
};

} // namespace ms::distributed
```

---

## 10.4 Distributed Matrix

```cpp
// include/ms/distributed/dist_matrix.hpp
namespace ms::distributed {

enum class Distribution {
    Block,           // contiguous row blocks per rank
    Cyclic,          // cyclic 1D row distribution
    BlockCyclic,     // ScaLAPACK-style 2D block-cyclic
    Replicated       // full copy on all ranks
};

template<typename S, Distribution D = Distribution::Block>
class DistMatrix {
public:
    // Create distributed matrix from global shape
    static Result<DistMatrix> create(
        size_t global_rows, size_t global_cols,
        MPIContext& mpi);

    // Scatter from rank 0 to all ranks
    static Result<DistMatrix> scatter(
        const Matrix<S>& global, MPIContext& mpi, int root = 0);

    // Shape
    size_t global_rows() const;
    size_t global_cols() const;
    size_t local_rows()  const;    // rows owned by this rank
    size_t local_cols()  const;

    // Local portion
    Matrix<S>&       local()       { return local_; }
    const Matrix<S>& local() const { return local_; }

    // Row range this rank owns
    std::pair<size_t, size_t> row_range() const;

    // Gather to rank 0
    Result<std::optional<Matrix<S>>> gather(int root = 0) const;

    // Redistribute between layouts
    template<Distribution NewD>
    Result<DistMatrix<S, NewD>> redistribute() const;

    MPIContext& mpi() { return *mpi_; }

private:
    Matrix<S>   local_;
    MPIContext* mpi_;
    size_t      global_rows_, global_cols_;
    int         rank_, world_size_;
};

} // namespace ms::distributed
```

---

## 10.5 GPU-Direct MPI

```cpp
// include/ms/distributed/gpu_aware_mpi.hpp
namespace ms::distributed {

// Detect if the MPI build supports GPU-direct RDMA
bool has_gpu_direct_mpi();

// Send GPU buffer directly — bypasses CPU staging if GPU-direct available
template<typename T>
Result<void> gpu_direct_send(
    cuda::DeviceBuffer<T>& buf,
    int dest, MPIContext& mpi, int tag = 0)
{
    if (has_gpu_direct_mpi()) {
        [[ms::unsafe("GPU-direct: MPI sees device pointer directly")]]
        MPI_Send(buf.device_ptr(), static_cast<int>(buf.size()),
                 mpi_type<T>(), dest, tag, MPI_COMM_WORLD);
    } else {
        // Stage through pinned host memory
        cuda::PinnedBuffer<T> host_buf(buf.size());
        MS_TRY(buf.copy_to_host(host_buf.data(), buf.size()));
        cudaDeviceSynchronize();
        MPI_Send(host_buf.data(), static_cast<int>(buf.size()),
                 mpi_type<T>(), dest, tag, MPI_COMM_WORLD);
    }
    return {};
}

} // namespace ms::distributed
```

---

## 10.6 Distributed Linear Algebra

Distributed linear algebra (solve, SVD, eigenvalues on distributed matrices) dispatches to SLATE (the modern ScaLAPACK replacement). SLATE uses MPI + cuBLAS internally.

```cpp
// include/ms/distributed/dist_linalg.hpp
namespace ms::distributed {

// Distributed LU solve — uses SLATE internally
template<typename S>
Result<DistMatrix<S>> solve(
    DistMatrix<S>& A,
    DistMatrix<S>& b,
    MPIContext& mpi);

// Distributed SVD
template<typename S>
Result<std::tuple<DistMatrix<S>, Matrix<S>, DistMatrix<S>>>
svd(DistMatrix<S>& A, MPIContext& mpi);

// Distributed eigenvalues (symmetric)
template<typename S>
Result<std::tuple<DistMatrix<S>, Matrix<S>>>
eig_sym(DistMatrix<S>& A, MPIContext& mpi);

// Distributed matrix multiply
template<typename S>
Result<DistMatrix<S>> matmul(
    const DistMatrix<S>& A,
    const DistMatrix<S>& B,
    MPIContext& mpi);

} // namespace ms::distributed
```

---

## 10.7 NUMA-Aware Thread Pinning

At startup, threads are pinned to NUMA nodes and a TBB arena is created per node:

```cpp
// src/runtime/topology.cpp
void ThreadPool::initialise(const SystemTopology& topo) {
    arenas_.clear();
    for (const auto& node : topo.numa_nodes) {
        // Create a TBB arena with one slot per logical CPU on this node
        tbb::task_arena arena(
            static_cast<int>(node.logical_cpu_ids.size()),
            0  // reserved slots
        );

        // Pin the arena's threads to this NUMA node's CPUs
        arena.initialize();
        arenas_.push_back(std::move(arena));
    }
}
```

Matrix allocations default to the NUMA node of the thread that creates them:

```cpp
// The NumaAllocator default constructor queries the calling thread's NUMA node
template<typename T>
NumaAllocator<T>::NumaAllocator()
    : node_(NumaTopology::instance().nearest_node()) {}
```

This means a matrix created on a thread pinned to NUMA node 1 will have its data allocated on node 1's memory — minimising cross-node memory traffic.

---

## 10.8 MPI Rank to Hardware Assignment

```cpp
// src/distributed/topology.cpp
void assign_ranks_to_hardware(
    const runtime::SystemTopology& topo,
    const distributed::MPIContext& mpi)
{
    // Divide NUMA nodes evenly among ranks
    const int nodes_per_rank =
        std::max(1, static_cast<int>(topo.numa_nodes.size()) / mpi.world_size());

    const int my_start_node = mpi.rank() * nodes_per_rank;
    const int my_end_node   = std::min(
        my_start_node + nodes_per_rank,
        static_cast<int>(topo.numa_nodes.size()));

    // Pin this process to its NUMA nodes
    for (int n = my_start_node; n < my_end_node; ++n) {
        for (int cpu : topo.numa_nodes[n].logical_cpu_ids)
            cpu_set(cpu);   // hwloc / SetThreadAffinityMask
    }

    // Assign GPUs: each rank owns GPUs on its NUMA nodes
    for (int n = my_start_node; n < my_end_node; ++n)
        for (int gpu : topo.numa_nodes[n].gpu_ids)
            ms::cuda::DeviceManager::claim_device(gpu);
}
```


---

## Part 11: Qt6 IDE

---

## 11.1 Window Layout

```
┌──────────┬──────────────────────────────────┬───────────────┐
│  File    │           Editor                 │  Workspace    │
│ Browser  │   (clangd-powered C++23)          │  Browser      │
│          │                                  │               │
│          ├──────────────────────────────────│  Variable     │
│          │         REPL Console             │  Inspector    │
│          │   (input bar + rich output)      │               │
├──────────┴──────────────────────────────────┴───────────────┤
│              Plot / Visualisation Panel                      │
├──────────────────────────────────────────────────────────────┤
│  Status: GPU 43% │ CPU 12% │ RAM 4.2/32 GB │ Rank 0/4 │ ●  │
└──────────────────────────────────────────────────────────────┘
```

All panels are dockable and undockable via `QDockWidget`. The layout is saved and restored between sessions.

---

## 11.2 Component Structure

```
src/gui/
├── MainWindow.hpp/cpp
├── interp_thread.hpp/cpp          — QThread wrapping IInterpreter
├── Editor/
│   ├── CodeEditor.hpp/cpp         — QPlainTextEdit subclass
│   ├── Highlighter.hpp/cpp        — QSyntaxHighlighter for C++
│   ├── LSPClient.hpp/cpp          — clangd over stdio (Language Server Protocol)
│   ├── CompletionModel.hpp/cpp    — LSP completions → QCompleter
│   └── DiagnosticMargin.hpp/cpp   — Error/warning squiggles from plugin
├── Console/
│   ├── ReplWidget.hpp/cpp         — Full REPL panel
│   ├── InputBar.hpp/cpp           — Command input with history
│   ├── RichOutput.hpp/cpp         — Inline LaTeX (QWebEngineView) + inline plots
│   └── OutputRouter.hpp/cpp       — stdout / stderr / diagnostic stream mux
├── Workspace/
│   ├── VariableModel.hpp/cpp      — QAbstractItemModel over JIT symbol table
│   ├── VariableDelegate.hpp/cpp   — Type-aware rendering
│   └── TypeRenderer.hpp/cpp       — Small inline matrix heatmap previews
├── Plotter/
│   ├── PlotManager.hpp/cpp        — Manages multiple plot windows
│   ├── Plot2D.hpp/cpp             — QCustomPlot wrapper
│   ├── Plot3D.hpp/cpp             — OpenGL 3D surface / VTK
│   └── PlotBridge.hpp/cpp         — ms::plot() → Qt signal bridge
├── Debugger/
│   ├── BreakpointManager.hpp/cpp
│   ├── CallStackModel.hpp/cpp
│   ├── WatchModel.hpp/cpp
│   └── DebugController.hpp/cpp
├── FileBrowser/
│   └── FileWidget.hpp/cpp
├── PackageManager/
│   └── PackageWidget.hpp/cpp
└── SystemMonitor/
    └── HardwareBar.hpp/cpp
```

---

## 11.3 Thread Model

```cpp
// src/gui/interp_thread.hpp
class InterpThread : public QThread {
    Q_OBJECT

    std::unique_ptr<ms::interp::IInterpreter> interp_;
    std::queue<std::string>                   work_queue_;
    std::mutex                                queue_mutex_;
    std::condition_variable                   queue_cv_;
    bool                                      running_ = true;

public:
    explicit InterpThread(ms::interp::InterpreterConfig cfg,
                          QObject* parent = nullptr);

    // Called from GUI thread — posts work to interpreter thread
    void execute(std::string code);
    void interrupt();
    void reset_session();

protected:
    void run() override;

signals:
    // Emitted by interpreter thread — consumed by GUI thread
    void output_ready(QString text, bool is_error);
    void plot_ready(ms::gui::PlotData data);
    void variables_changed(QList<ms::gui::VariableInfo> vars);
    void execution_done(bool success, qint64 elapsed_ms);
    void diagnostic(QString message, QString severity);
};
```

Rule: **the GUI thread never calls into the interpreter directly.** All communication is via signals.

---

## 11.4 Editor & LSP Integration

The editor provides full C++23 editing with live diagnostics from both clangd (type errors, completion) and the MathScript plugin (profile violations).

```cpp
// src/gui/Editor/LSPClient.hpp
class LSPClient : public QObject {
    Q_OBJECT

    QProcess clangd_proc_;
    // JSON-RPC communication over clangd's stdio

public:
    explicit LSPClient(QObject* parent = nullptr);
    void start(const QString& project_root,
               const QStringList& extra_args);

    // Requests to clangd
    void request_completion(const QString& uri,
                            int line, int character);
    void request_hover(const QString& uri,
                       int line, int character);
    void request_signature_help(const QString& uri,
                                int line, int character);
    void did_open(const QString& uri, const QString& content);
    void did_change(const QString& uri, const QString& content);
    void did_save(const QString& uri);

signals:
    void completion_ready(QList<CompletionItem> items);
    void hover_ready(QString content);
    void diagnostics_ready(QString uri, QList<Diagnostic> diags);
    void signature_help_ready(SignatureHelp help);
};
```

clangd is launched as a child process with the MathScript include paths and the compile flags from the active CMake configuration. Diagnostics from both clangd and the plugin (which runs as part of clangd's semantic analysis) appear as red/yellow underlines in the editor gutter.

---

## 11.5 Plot Bridge

The `ms::plot()` family of functions posts plot data to the Qt event loop rather than blocking:

```cpp
// include/ms/gui/plot_bridge.hpp
namespace ms::gui {

// All plot data types — sent as signals to PlotManager
struct Line2DData   { std::vector<double> x, y; std::string label; };
struct Scatter2DData{ std::vector<double> x, y; std::string label; };
struct Surface3DData{ Matrix<double> X, Y, Z; };
struct HeatmapData  { Matrix<double> Z; std::string colormap; };
struct HistogramData{ std::vector<double> data; size_t bins; };
struct BarData      { std::vector<std::string> labels; std::vector<double> values; };
struct ImageData    { Matrix<uint8_t> img; bool is_rgb; };
struct SpyData      { Sparse<double> A; };  // sparsity pattern

using PlotData = std::variant<
    Line2DData, Scatter2DData, Surface3DData,
    HeatmapData, HistogramData, BarData,
    ImageData, SpyData
>;

// Function pointer set by GUI at startup — nullptr in headless mode
inline std::function<void(PlotData)> g_plot_sink;

// Called by user in REPL
inline void plot(std::vector<double> x, std::vector<double> y,
                 std::string label = "") {
    if (g_plot_sink)
        g_plot_sink(Line2DData{std::move(x), std::move(y), std::move(label)});
}

inline void surf(Matrix<double> X, Matrix<double> Y, Matrix<double> Z) {
    if (g_plot_sink)
        g_plot_sink(Surface3DData{std::move(X), std::move(Y), std::move(Z)});
}

inline void imshow(Matrix<uint8_t> img) {
    if (g_plot_sink)
        g_plot_sink(ImageData{std::move(img), false});
}

} // namespace ms::gui
```

At GUI startup:

```cpp
// src/gui/MainWindow.cpp
ms::gui::g_plot_sink = [this](ms::gui::PlotData data) {
    QMetaObject::invokeMethod(plot_manager_, "receive_plot",
        Qt::QueuedConnection,
        Q_ARG(ms::gui::PlotData, std::move(data)));
};
```

---

## 11.6 Workspace Variable Inspector

The workspace panel reflects the JIT symbol table after every execution. Types are rendered inline:

```cpp
// src/gui/Workspace/TypeRenderer.hpp
class TypeRenderer {
public:
    // Returns a small QPixmap preview for display in workspace panel
    static QPixmap render(const ms::interp::VariableInfo& info);

private:
    // Matrix → heatmap thumbnail (64×64 pixels)
    static QPixmap render_matrix(const std::string& name,
                                 size_t rows, size_t cols);
    // Scalar → formatted number string
    static QPixmap render_scalar(const std::string& name,
                                 const std::string& value);
    // Symbolic expression → rendered LaTeX
    static QPixmap render_sym(const std::string& latex);
    // Vector → sparkline
    static QPixmap render_vector(const std::string& name, size_t size);
};
```

Double-clicking a variable in the workspace opens a full variable viewer that shows the complete matrix/tensor with scrollable grid view and colour mapping.

---

## 11.7 Hardware Monitor Bar

The status bar shows live hardware utilisation. A background thread polls NVML and system APIs every 500ms:

```cpp
// src/gui/SystemMonitor/HardwareBar.hpp
class HardwareBar : public QStatusBar {
    Q_OBJECT

    QLabel* gpu_label_;
    QLabel* cpu_label_;
    QLabel* mem_label_;
    QLabel* rank_label_;
    QProgressBar* gpu_bar_;
    QTimer* timer_;

public:
    explicit HardwareBar(QWidget* parent = nullptr);

private slots:
    void update_stats();

private:
    float query_gpu_utilisation(int device = 0);  // NVML
    float query_cpu_utilisation();                 // /proc/stat or GetSystemTimes
    std::pair<size_t, size_t> query_memory();      // used / total
};
```

---

## 11.8 Session Save/Restore

Sessions can be saved to disk and restored. A session file records:

```json
{
  "version": "0.3.0",
  "session_name": "my_analysis",
  "preamble": "auto custom_fn = [](double x) { return x * x; };",
  "history": [
    { "input": "Matrix<double> A = randn(100, 100);" },
    { "input": "auto [L, U] = lu(A);" }
  ],
  "open_files": ["/home/odin/scripts/analysis.cpp"],
  "layout": "<Qt dock layout XML>"
}
```

Variables themselves are not saved (they live in the JIT — not serialisable in general). The history is replayed on restore to reconstruct the session state.

---

## 11.9 Settings

Settings are stored via `QSettings` in the platform-appropriate location (Registry on Windows, `~/.config/mathscript/` on Linux).

Key settings categories:
- Editor font, size, tab width, colour scheme
- REPL auto-completion on/off
- GPU auto-dispatch threshold overrides
- Default plot style and colormap
- MPI launch command for distributed sessions
- Custom startup code (appended to preamble)
- clangd path and extra flags
- Qt style (Fusion dark, system native)


---

## Part 12: Frameworks Integration

---

## 12.1 Framework Architecture

All five frameworks are first-class modules — auto-included in every session, fully integrated with the math runtime and dispatch layer. They are not plugins or add-ons. They are part of the MathScript core.

```
ms::gria::      — Graded Reversible-Irreversible Algebra
ms::izaac::     — Deterministic randomness, 12 applications
ms::cypha::     — CyphaDIF online ML, NIG uncertainty
ms::cellai::    — Cellular memory, Hebbian learning
ms::axiom::     — Genetic programming over algorithm space
```

Each framework lives in `frameworks/<name>/` with its own `include/` and `src/` subtree, built as a static library and linked into `libmathscript`.

---

## 12.2 GRIA — Graded Reversible-Irreversible Algebra

```cpp
// include/ms/frameworks/gria/gria.hpp
namespace ms::gria {

// Grand Unified Law: α = 1 − H(f(X)) / H(X)
// α ∈ [0,1]: 0 = fully reversible, 1 = fully irreversible
// α ≈ 0.5 = edge-of-chaos critical point

template<typename T>
double compute_alpha(
    std::span<const T> input,
    std::function<std::vector<T>(std::span<const T>)> transform);

// GRIA grade of a matrix transformation
double matrix_alpha(const Matrix<double>& X,
                    const Matrix<double>& fX);

// Edge-of-chaos test
bool is_critical(double alpha, double tolerance = 0.05);

// GF(2^n) operations
namespace gf2n {
    uint64_t mul(uint64_t a, uint64_t b, uint64_t poly);
    uint64_t pow(uint64_t a, uint64_t n, uint64_t poly);
    uint64_t inv(uint64_t a, uint64_t poly);
    std::vector<uint64_t> generate_field(int n);
}

// Cellular automaton analysis
namespace ca {
    std::vector<uint8_t> step(std::span<const uint8_t> state, uint8_t rule);
    double langton_lambda(uint8_t rule);
    double alpha_ca(uint8_t rule, size_t steps, size_t n);
}

// LFSR analysis
namespace lfsr {
    uint64_t step(uint64_t state, uint64_t poly);
    double   alpha_lfsr(uint64_t poly, size_t steps);
    bool     is_maximal(uint64_t poly, int n);
}

// Dispatch hint — high-α operations are candidates for lossy GPU paths
enum class ComputeClass { Reversible, Critical, Irreversible };
ComputeClass classify(double alpha);

} // namespace ms::gria
```

**Integration with dispatch layer:**

```cpp
// src/runtime/dispatch.cpp
DispatchDecision decide(size_t n, OpClass op, ExecPolicy policy,
                        const SystemTopology& topo) {
    // GRIA integration: high-α operations tolerate lossy GPU paths
    if (const auto* alpha = gria_hint_registry().get(op)) {
        if (gria::classify(*alpha) == gria::ComputeClass::Irreversible) {
            // Irreversible computation — GPU lossy path acceptable
            if (cuda::available() && n > 1000)
                return { .backend = Backend::CUDA };
        }
    }
    // Standard dispatch logic follows...
}
```

---

## 12.3 Izaac — Deterministic Randomness

Izaac is the cryptographic randomness framework. In MathScript, all stochastic operations in the math library use Izaac's VRF as the entropy source when an Izaac session is active — giving auditable, reproducible randomness.

```cpp
// include/ms/frameworks/izaac/izaac.hpp
namespace ms::izaac {

// Core VRF
struct VRFKey {
    std::array<uint8_t, 32> private_key;
    std::array<uint8_t, 32> public_key;
};

struct VRFProof {
    std::array<uint8_t, 80> proof;
    std::array<uint8_t, 64> output;
};

VRFKey   keygen();
VRFProof prove(const VRFKey& key, std::span<const uint8_t> msg);
bool     verify(const std::array<uint8_t, 32>& pub,
                std::span<const uint8_t> msg,
                const VRFProof& proof);

// CSPRNG seeded from VRF output
class CSPRNG {
    std::array<uint8_t, 32> state_;
public:
    explicit CSPRNG(const VRFProof& seed);
    explicit CSPRNG(std::array<uint8_t, 32> seed);

    void     fill(std::span<uint8_t> buf);
    uint64_t next_u64();
    double   next_f64();           // uniform [0, 1)
    double   next_normal();        // standard normal via Box-Muller
};

// Session RNG — the global RNG for MathScript math functions
// Set at session start; all ms::randn, ms::rand etc. use this
extern std::optional<CSPRNG> g_session_rng;

void seed_session(const VRFKey& key, std::span<const uint8_t> nonce);
void seed_session(uint64_t deterministic_seed);

// 12 Izaac applications
namespace crypto   { /* ... */ }
namespace vrf      { /* ... */ }
namespace consensus{ /* ... */ }
namespace mpc      { /* ... */ }
namespace compress { /* ... */ }
namespace bloom    { /* ... */ }
namespace mc       { /* ... */ }  // Monte Carlo
namespace ratelimit{ /* ... */ }
namespace diffpriv { /* ... */ }  // differential privacy
namespace fuzz     { /* ... */ }
namespace backtest { /* ... */ }
namespace military { /* ... */ }

} // namespace ms::izaac
```

---

## 12.4 CyphaDIF — Online ML with NIG Uncertainty

```cpp
// include/ms/frameworks/cyphadif/cypha.hpp
namespace ms::cypha {

// Normal-Inverse-Gaussian distribution parameters
struct NIGParams {
    double mu, alpha, beta, delta;
};

// CyphaDIF model configuration
struct DifConfig {
    size_t   input_dim;
    size_t   output_dim;
    size_t   n_experts      = 8;
    double   learning_rate  = 0.01;
    double   gh_protection  = 0.95;    // GH posterior coverage
    bool     online         = true;
    size_t   max_components = 100;
};

// Full CyphaDIF model
class DifModel {
public:
    explicit DifModel(const DifConfig& cfg);

    // Online update — call once per data point or batch
    Result<void> update(const Matrix<double>& X,
                        const Matrix<double>& y);

    // Predict — returns mean + uncertainty
    struct Prediction {
        Matrix<double> mean;
        Matrix<double> lower;   // credible interval lower
        Matrix<double> upper;   // credible interval upper
        double         nig_alpha, nig_beta;
    };

    Result<Prediction> predict(const Matrix<double>& X) const;

    // OOD detection — is this input out-of-distribution?
    Result<double> ood_score(const Matrix<double>& X) const;

    // GH posterior gate — protect against adversarial inputs
    bool gh_gate(const Matrix<double>& X) const;

    // Serialise / deserialise
    Result<void>        save(const std::filesystem::path& path) const;
    static Result<DifModel> load(const std::filesystem::path& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// NIG primitive operations
NIGParams nig_fit(const Matrix<double>& data);
double    nig_pdf(double x, const NIGParams& p);
double    nig_cdf(double x, const NIGParams& p);
Matrix<double> nig_sample(const NIGParams& p, size_t n,
                           izaac::CSPRNG& rng);

} // namespace ms::cypha
```

**CUDA dispatch:** CyphaDIF tensor operations go through the standard MathScript dispatch layer. The model's internal matrices are `ms::Matrix<double>` — they automatically use CUDA paths above the dispatch threshold.

---

## 12.5 CellAI — Cellular Memory

```cpp
// include/ms/frameworks/cellai/cellai.hpp
namespace ms::cellai {

// Hebbian learning rule — updates synaptic weights
Matrix<double> hebbian_update(
    const Matrix<double>& W,    // current weight matrix
    const Matrix<double>& x,    // input pattern
    const Matrix<double>& y,    // output pattern
    double learning_rate);

// Boltzmann machine energy
double energy(const Matrix<double>& W,
              const Matrix<double>& v,
              const Matrix<double>& h);

// Multi-scale SSM-equivalent memory kernel
// Implements CellAI's core memory architecture
class CellMemory {
public:
    explicit CellMemory(size_t input_dim, size_t memory_dim,
                        std::vector<double> time_scales);

    // Process one timestep — updates internal state
    Result<Matrix<double>> step(const Matrix<double>& input);

    // Retrieve from memory at a given time scale
    Result<Matrix<double>> recall(double time_scale) const;

    // Consolidate short-term to long-term
    Result<void> consolidate();

    // Reset all memory
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// CellAI ↔ CyphaDIF bridge
// CellAI memory output feeds into CyphaDIF as input features
Matrix<double> cell_to_cypha_features(
    const CellMemory& cell,
    const std::vector<double>& time_scales);

} // namespace ms::cellai
```

**Memory allocation:** CellAI's short-lived Hebbian update matrices use the arena allocator, keeping the weight update path allocation-free.

---

## 12.6 AXIOM — Genetic Programming

```cpp
// include/ms/frameworks/axiom/axiom.hpp
namespace ms::axiom {

// An algorithm in AXIOM is a 5-component symbolic object:
// (representation, evaluation, selection, mutation, fitness)
struct Algorithm {
    ms::sym::Sym representation;
    ms::sym::Sym evaluation;
    ms::sym::Sym selection;
    ms::sym::Sym mutation;
    double       fitness;   // GRIA α as universal fitness signal
};

// The primitive space — all MathScript functions are available
// AXIOM searches over compositions of these
struct PrimitiveRegistry {
    // Populated at startup from the full ms:: namespace
    std::vector<std::string> function_names;
    std::vector<ms::sym::Sym> function_symbols;

    static PrimitiveRegistry build_from_ms_namespace();
};

// AXIOM evolution configuration
struct EvolutionConfig {
    size_t   population_size   = 100;
    size_t   max_generations   = 1000;
    double   mutation_rate     = 0.1;
    double   crossover_rate    = 0.7;
    size_t   tournament_size   = 5;
    double   target_alpha      = 0.5;  // target GRIA α (edge-of-chaos)
    size_t   max_depth         = 10;   // maximum expression tree depth
    bool     use_cuda          = true;
    izaac::CSPRNG* rng         = nullptr;
};

// AXIOM runner
class Axiom {
public:
    explicit Axiom(const EvolutionConfig& cfg,
                   const PrimitiveRegistry& primitives);

    // Run evolution — returns best algorithm found
    Result<Algorithm> evolve(
        std::function<double(const Algorithm&)> objective,
        std::function<bool(const Algorithm&)>   termination);

    // Evaluate an algorithm on data
    Result<Matrix<double>> evaluate(
        const Algorithm& algo,
        const Matrix<double>& data);

    // Fitness function using GRIA α
    double gria_fitness(const Algorithm& algo,
                        const Matrix<double>& data);

    const std::vector<Algorithm>& population() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ms::axiom
```

---

## 12.7 Preamble Injection

All five frameworks are injected in the session preamble automatically:

```cpp
constexpr std::string_view FRAMEWORK_PREAMBLE = R"cpp(
    // GRIA
    using namespace ms::gria;
    auto alpha = [](auto& X, auto f) { return ms::gria::compute_alpha(X, f); };

    // Izaac — seed session RNG at startup
    ms::izaac::seed_session(ms::izaac::keygen(),
        std::array<uint8_t,8>{0,1,2,3,4,5,6,7});

    // CyphaDIF shorthand
    using DIF = ms::cypha::DifModel;
    using ms::cypha::nig_fit;
    using ms::cypha::nig_pdf;

    // CellAI shorthand
    using Cell = ms::cellai::CellMemory;

    // AXIOM shorthand
    using ms::axiom::Axiom;
    using ms::axiom::EvolutionConfig;
    const auto& primitives = ms::axiom::PrimitiveRegistry::build_from_ms_namespace();
)cpp";
```


---

## Part 13: Security Model

---

## 13.1 Security Layers

Security in MathScript operates at five independent layers. A failure in one layer does not compromise the others.

```
Layer 1: Compile-time   — Plugin enforcement, sanitizers, hardening flags
Layer 2: Runtime        — mimalloc hardening, bounds checking, UBSan
Layer 3: Cryptographic  — Izaac primitives, auditable RNG, HMAC
Layer 4: Distributed    — MPI auth, UCX encryption, GPU-direct policy
Layer 5: Audit          — Unsafe site registry, coverage, fuzz corpus
```

---

## 13.2 Compile-Time Security

**Plugin enforcement** (Part 7) is the primary compile-time control. It prevents the entire class of memory misuse, type confusion, and concurrency bugs by making them unrepresentable.

**Compiler flags:**

| Flag | Effect |
|---|---|
| `-fno-exceptions` | Eliminates exception unwinding paths |
| `-fno-rtti` | Eliminates type_info and dynamic_cast |
| `-fsanitize=address,undefined` | Catches buffer overflows, UB (Debug) |
| `-fsanitize=undefined` | Low-overhead UB detection (Release) |
| `-fstack-protector-strong` | Stack canaries against overflows |
| `-D_FORTIFY_SOURCE=3` | glibc hardened string/memory functions |
| `-ftrapv` | Signed integer overflow traps (Debug) |
| `-Werror` | All warnings are errors |
| `-fPIE` / `-pie` | Position-independent executable |
| `-Wl,-z,relro,-z,now` | Full RELRO — no writable GOT (Linux) |
| `/guard:cf` | Control Flow Guard (Windows) |
| `/sdl` | Additional SDL checks (Windows) |

---

## 13.3 Runtime Memory Security

**mimalloc hardening:**
- Free-list randomisation — prevents heap spray
- Guard pages between arenas — detects out-of-bounds
- Double-free detection — terminates on second free of same pointer
- Heap isolation per thread — cross-thread heap corruption harder

**Bounds checking:**
All `Matrix` and `Tensor` element access uses checked `at()` by default. The unchecked `operator()` requires `[[ms::unsafe]]`. The plugin enforces this — `operator()` calls outside unsafe blocks are flagged.

```cpp
// Default (safe) — returns Result<T&>
auto elem = A.at(r, c);

// Unchecked — inside [[ms::unsafe]] only
[[ms::unsafe("hot path: r,c validated by loop bounds")]]
double val = A(r, c);
```

**Arena zeroing:** The arena allocator zeroes released memory (via `resource_.release()` which reinitialises the buffer). No data from previous expressions leaks into the next.

**CUDA buffer zeroing:** All `DeviceBuffer` allocations are zero-initialised via `cudaMemset` on allocation. No uninitialised GPU memory.

---

## 13.4 Cryptographic Security

All cryptographic operations in MathScript use Izaac primitives exclusively. No direct use of OpenSSL or system crypto APIs.

**Random number generation policy:**

```cpp
// All stochastic functions in ms:: use this when set
// If not set, falls back to std::mt19937 with a hardware seed
extern std::optional<ms::izaac::CSPRNG> g_session_rng;
```

This means that with a VRF key, every random operation in a session is:
- Deterministic given the key and nonce
- Cryptographically secure
- Auditable — the VRF proof can be verified

**Key derivation for sessions:** Session keys are derived via Izaac's HKDF implementation, not exposed directly to user code.

---

## 13.5 Distributed Security

**MPI authentication:** OpenMPI 5.0 uses PMIx for process management. PMIx credentials authenticate MPI ranks at launch time. Connections from unauthorised processes are rejected.

**Transport encryption:** UCX 1.15+ supports TLS encryption over TCP. For cluster deployments, UCX is configured with `UCX_TLS=tcp,tls` to mandate encryption over wide-area links. InfiniBand traffic is encrypted at the RDMA layer via RoCE v2 with MACsec.

**GPU-direct policy:** GPU-direct RDMA (device memory directly to network) bypasses the CPU. This means kernel memory protection does not apply to GPU-direct transfers. MathScript's `gpu_direct_send()` only activates GPU-direct when the network is verified as private (InfiniBand or trusted RoCE). Over untrusted networks, staging through pinned host memory is mandatory.

---

## 13.6 Unsafe Surface Audit

The unsafe site registry (Part 7) is the audit mechanism. Rules:

1. Every unsafe site requires a justification string — no empty annotations
2. New unsafe sites require a review entry in `UNSAFE_REVIEW.md`
3. CI fails if unsafe count increases without a review entry
4. The audit report is archived with every release build

**UNSAFE_REVIEW.md format:**

```markdown
## Unsafe Site Review Log

### 2025-04-11 — avx512_dgemm.cpp:89
**Justification:** AVX-512 store requires raw pointer. 64-byte alignment
guaranteed by AlignedAllocator at construction. Bounds guaranteed by
blocked loop iteration.
**Reviewer:** Odin
**Status:** Approved

### 2025-04-11 — cuda/memory.cu:112
**Justification:** cudaMemcpy API requires void*. Pointer is a DeviceBuffer
device_ptr() which is always valid for the declared element count.
**Reviewer:** Odin
**Status:** Approved
```

---

## 13.7 Fuzzing

Key attack surfaces are covered by libFuzzer targets:

```
tests/fuzz/
├── fuzz_matrix_ops.cpp         — fuzz matrix arithmetic with arbitrary inputs
├── fuzz_sym_parser.cpp         — fuzz symbolic expression parser
├── fuzz_repl_input.cpp         — fuzz interpreter with arbitrary C++ fragments
├── fuzz_bignum.cpp             — fuzz bignum arithmetic
├── fuzz_poly_ops.cpp           — fuzz polynomial operations
├── fuzz_special_fns.cpp        — fuzz special functions with edge-case inputs
└── fuzz_mpi_message.cpp        — fuzz MPI message parsing
```

Fuzzing corpus seeds are checked into `tests/fuzz/corpus/`. CI runs a short fuzzing session (30s per target) on every merge. Discovered crashes are regression-tested permanently.

---

## 13.8 Supply Chain Security

**Dependency integrity:**

All vendored sources in `vendor/` have a corresponding SHA-256 hash in `vendor/CHECKSUMS.sha256`. A CMake pre-build step verifies these hashes before building.

```cmake
# cmake/verify_vendor.cmake
function(verify_vendor name expected_sha256)
    file(SHA256 ${CMAKE_SOURCE_DIR}/vendor/${name}/CMakeLists.txt actual)
    if(NOT actual STREQUAL expected_sha256)
        message(FATAL_ERROR "Vendor integrity check failed for ${name}")
    endif()
endfunction()

verify_vendor(mimalloc   "abc123...")
verify_vendor(highway    "def456...")
verify_vendor(symengine  "ghi789...")
```

**No network access during build:** CMake is configured with `CMAKE_DISABLE_FIND_PACKAGE_<X>` for any package that could trigger a network download. All deps are either in `vendor/` or installed system-wide by the setup script.


---

## Part 14: Testing Strategy

---

## 14.1 Test Categories

```
tests/
├── unit/          — per-function correctness
├── integration/   — cross-module workflows
├── compliance/    — plugin rule enforcement
├── numerical/     — accuracy regression vs reference values
├── performance/   — benchmark suite with regression detection
├── distributed/   — MPI multi-rank tests
└── fuzz/          — libFuzzer targets for attack surfaces
```

All tests run under ASan + UBSan in Debug configuration. All tests are run on both Linux and Windows in CI.

---

## 14.2 Unit Tests

Every function in every module has at least one unit test. Tests are parameterised over scalar types (`float`, `double`, `complex<float>`, `complex<double>`) where applicable.

**Test structure:**

```cpp
// tests/unit/linalg/test_lu.cpp
#include <gtest/gtest.h>
#include <ms/linalg.hpp>

template<typename T>
class LUTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double,
    std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LUTest, ScalarTypes);

TYPED_TEST(LUTest, basic_decomposition) {
    using S = TypeParam;
    auto A = ms::Matrix<S>{{1,2,3},{4,5,6},{7,8,10}};
    auto result = ms::linalg::lu(A);
    ASSERT_TRUE(result.has_value()) << result.error();
    auto [L, U, P] = result.value();

    // Verify PA = LU
    auto PA  = ms::linalg::matmul(P, A).value();
    auto LU  = ms::linalg::matmul(L, U).value();
    EXPECT_NEAR(ms::linalg::norm(
        ms::linalg::sub(PA, LU).value()).value(), 0.0, 1e-12);
}

TYPED_TEST(LUTest, singular_matrix_returns_error) {
    auto A = ms::Matrix<TypeParam>{{1,2},{2,4}};   // rank 1
    auto result = ms::linalg::lu(A);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<ms::SingularMatrix>(result.error()));
}
```

---

## 14.3 Numerical Accuracy Tests

Every mathematical function is tested against reference values. Reference values are generated by:
- Mathematica (for special functions, symbolic results)
- MPFR at 256-bit precision (for elementary functions)
- NIST reference implementations (for special functions)
- SciPy/NumPy (for signal processing, statistics)

```cpp
// tests/numerical/test_bessel.cpp
struct BesselJRef {
    double nu, x, expected, tolerance;
};

const std::vector<BesselJRef> BESSEL_J_REFERENCE = {
    { 0.0,   0.0,   1.0,              1e-15 },
    { 0.0,   1.0,   0.7651976866,     1e-10 },
    { 0.0,   2.4048, 0.0,             1e-6  },  // first zero
    { 1.0,   1.0,   0.4400505858,     1e-10 },
    { 0.5,   1.0,   0.6713967071,     1e-10 },
    { 10.0, 20.0,   0.1860405018,     1e-9  },
    // ...hundreds more reference points
};

TEST(SpecialFunctions, BesselJ_reference) {
    for (const auto& ref : BESSEL_J_REFERENCE) {
        auto result = ms::special::besselj(ref.nu, ref.x);
        ASSERT_TRUE(result.has_value());
        EXPECT_NEAR(result.value(), ref.expected, ref.tolerance)
            << "J_" << ref.nu << "(" << ref.x << ")";
    }
}
```

---

## 14.4 Compliance Tests

Each plugin rule has a compile-fail test and a compile-pass test:

```cpp
// tests/compliance/no_raw_new/fail.cpp
// This file MUST NOT compile when the plugin is active

int* p = new int(5);          // plugin fires: NoRawNew
```

```cpp
// tests/compliance/no_raw_new/ok.cpp
// This file MUST compile cleanly

auto p = ms::make_unique<int>(5);   // correct pattern
```

CMake drives these via `WILL_FAIL` test properties (see Part 7).

---

## 14.5 Performance Benchmarks

Google Benchmark suite. Every hot path has a benchmark. Benchmarks run on every CI merge and compare against a stored baseline. Regressions > 5% fail the build.

```cpp
// tests/performance/bench_matmul.cpp
#include <benchmark/benchmark.h>
#include <ms/linalg.hpp>

static void BM_MatmulCPU_1000x1000(benchmark::State& state) {
    auto A = ms::linalg::randn<double>(1000, 1000);
    auto B = ms::linalg::randn<double>(1000, 1000);
    for (auto _ : state) {
        auto C = ms::linalg::matmul(A, B, ms::CPU{});
        benchmark::DoNotOptimize(C);
    }
    state.SetBytesProcessed(
        state.iterations() * 1000LL * 1000LL * sizeof(double));
}
BENCHMARK(BM_MatmulCPU_1000x1000)->Unit(benchmark::kMillisecond);

static void BM_MatmulGPU_1000x1000(benchmark::State& state) {
    auto A = ms::linalg::randn<double>(1000, 1000, ms::GPU{});
    auto B = ms::linalg::randn<double>(1000, 1000, ms::GPU{});
    for (auto _ : state) {
        auto C = ms::linalg::matmul(A, B, ms::GPU{});
        ms::cuda::get_stream().sync();   // wait for GPU completion
        benchmark::DoNotOptimize(C);
    }
}
BENCHMARK(BM_MatmulGPU_1000x1000)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
```

**Benchmark categories:**

| Category | Examples |
|---|---|
| Dense linalg | matmul 100/500/1000/5000, lu, svd, eig |
| Sparse linalg | SpMM various sparsity patterns, sparse solve |
| FFT | Various sizes including non-power-of-2, 2D |
| Signal | filter, filtfilt, welch, stft |
| Special functions | besselj, gamma, hyp2f1 — scalar and vectorised |
| Memory | allocation, transfer H2D, D2H |
| Symbolic | diff, integrate, simplify large expressions |
| SIMD | elementwise ops scalar vs AVX2 vs AVX-512 |

---

## 14.6 Distributed Tests

MPI tests are run with 1, 2, and 4 ranks. They verify:
- Scatter / gather correctness
- Distributed solve matches single-rank result
- AllReduce produces correct sum
- DistMatrix redistribution preserves values

```cmake
# tests/distributed/CMakeLists.txt
function(add_mpi_test name source n_ranks)
    add_executable(mpi_${name} ${source})
    target_link_libraries(mpi_${name} MathScript::distributed GTest::gtest)

    add_test(NAME mpi_${name}_${n_ranks}ranks
        COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${n_ranks}
                $<TARGET_FILE:mpi_${name}>)
endfunction()

add_mpi_test(scatter_gather test_scatter_gather.cpp 4)
add_mpi_test(dist_solve      test_dist_solve.cpp     4)
add_mpi_test(all_reduce      test_all_reduce.cpp     4)
```

---

## 14.7 CI Pipeline

```
Push / PR →

  Stage 1: Build
    ├── Linux Debug   (Clang 17, ASan+UBSan)
    ├── Linux Release (Clang 17, O3+LTO)
    ├── Windows Debug (Clang-cl 17)
    └── Windows Release (Clang-cl 17)

  Stage 2: Compliance Tests (all platforms)
    └── All plugin rules — compile-fail and compile-pass

  Stage 3: Unit Tests (all platforms, ASan+UBSan)
    └── All 31 modules, all scalar type parameterisations

  Stage 4: Numerical Accuracy Tests
    └── All reference comparisons — tolerances are hard failures

  Stage 5: Performance Benchmarks (Linux Release)
    └── Compare to baseline — >5% regression fails build
    └── Save new baseline if all pass

  Stage 6: Distributed Tests (Linux, 4 MPI ranks)
    └── All MPI test cases

  Stage 7: Fuzz (Linux Debug, 30s per target)
    └── All fuzz targets

  Stage 8: Reports
    ├── Unsafe surface report (delta from last build)
    ├── Coverage report (lcov)
    └── Benchmark trend graph

All stages must pass. Any failure blocks merge.
```

---

## 14.8 Coverage Target

Target coverage: **85% line coverage** by Phase 7 (Hardening).

Coverage excludes:
- `[[ms::unsafe]]` blocks (separately audited)
- Generated code (`version.hpp`)
- CUDA kernel bodies (covered by numerical accuracy tests, not line coverage)
- Third-party vendored code

Coverage is measured with `llvm-cov` / `lcov` in the Debug build with `-fprofile-arcs -ftest-coverage`.


---

## Part 15: Phased Delivery Plan

---

## 15.1 Phase Overview

| Phase | Name | Duration | Deliverable |
|---|---|---|---|
| 0 | Foundation | 3 weeks | Everything builds, nothing runs |
| 1 | Core Types + REPL | 4 weeks | Type C++, get results |
| 2 | Math Library CPU | 8 weeks | Full CPU math coverage |
| 3 | CUDA Backend | 6 weeks | GPU dispatch working |
| 4 | SIMD Layer | 3 weeks | CPU paths SIMD-optimised |
| 5 | Qt6 IDE | 6 weeks | Full IDE functional |
| 6 | Distributed | 6 weeks | Multi-node computation |
| 7 | Frameworks | 6 weeks | All 5 frameworks integrated |
| 8 | Special Functions | 6 weeks | Full DLMF catalogue |
| 9 | Replace Libraries | ongoing | Own implementations |
| 10 | Hardening | 5 weeks | Production quality |

Total to first shipping-quality release: approximately 53 weeks of AI-driven development.

---

## 15.2 Phase 0 — Foundation (Weeks 1–3)

**Goal:** Everything compiles. Nothing runs yet.

**Deliverables:**
- CMake scaffold for all platforms builds cleanly
- Clang plugin compiles and loads (one test diagnostic working)
- Qt6 window opens and closes
- mimalloc linked and verified active (via mimalloc stats)
- `Matrix<double>` type — construction only, no operations
- `version.hpp` generated correctly
- CI pipeline running — build stage only

**Exit criterion:** `cmake --build` succeeds on Linux and Windows. Plugin loads into REPL. Empty window opens.

---

## 15.3 Phase 1 — Core Types + REPL (Weeks 4–7)

**Goal:** Type C++ in the REPL, get correct results.

**Deliverables:**
- Clang-REPL embedded with full preamble loader
- All plugin rules implemented and compliance tests passing
- `Matrix<double>`, `Tensor`, `Sparse`, `Sym` types with construction and print
- `AlignedAllocator` and `ArenaAllocator` wired into Matrix
- Basic linalg: `lu`, `qr`, `solve`, `matmul`, `norm`, `eye`, `zeros`, `ones`
- Console REPL in Qt: input, output, history
- SystemTopology detection at startup
- Unit tests for all implemented functions

**Exit criterion:**
```cpp
Matrix<double> A = {{4,3},{6,3}};
auto [L, U] = lu(A).value();
U.print();
// Expected: [ 6 3 ; 0 0.5 ]
```

---

## 15.4 Phase 2 — Math Library CPU (Weeks 8–15)

**Goal:** All 31 math modules implemented with CPU backends.

**Deliverables (by module order):**

*Weeks 8–9: linalg completion*
- All decompositions: `svd`, `eig`, `schur`, `chol`, `ldl`, `hess`, `bidiag`
- All solvers: iterative (cg, bicgstab, gmres), direct sparse
- All matrix functions: `expm`, `logm`, `sqrtm`, `funm`

*Weeks 9–10: fft + signal*
- Own FFT: Cooley-Tukey, Bluestein, Rader, split-radix
- All DCT/DST types, DHT, CZT
- Wavelet transforms (DWT, CWT)
- Full filter design suite
- Spectral analysis: Welch, MUSIC, ESPRIT

*Weeks 10–11: stats + probability*
- All descriptive statistics
- All hypothesis tests
- All regression models
- All 50+ distributions with full parameterisation

*Weeks 11–12: optimisation + ODE + PDE*
- All 6 optimisation categories
- All ODE solvers including stiff methods
- FD/FV/spectral PDE solvers

*Weeks 12–13: polynomial + symbolic (SymEngine transitional)*
- FFT-based polynomial multiply, half-GCD
- Symbolic differentiation, integration, solving
- Laplace/Fourier/Z transforms symbolic

*Weeks 13–14: remaining domains*
- Number theory, combinatorics, graph theory
- Computational geometry
- Control theory, information theory
- Financial mathematics
- Quantum mechanics operators

*Week 15: cross-module*
- Numerical accuracy tests against all reference values
- Coverage audit

**Exit criterion:** All numerical accuracy tests pass. All 31 modules have unit test coverage.

---

## 15.5 Phase 3 — CUDA Backend (Weeks 16–21)

**Goal:** GPU dispatch working. CPU and GPU give identical results.

**Deliverables:**
- `DeviceBuffer`, `PinnedBuffer`, `DevicePair`, `StreamPool` implemented
- cuBLAS dispatch for matmul, triangular solve, BLAS-3
- cuSOLVER dispatch for LU, QR, SVD, eigenvalues
- cuFFT dispatch for all FFT variants
- cuSPARSE for sparse matrix-vector operations
- All custom CUDA kernels: elementwise, reduction, transpose, conv2d, fill
- Load balancer with NVML monitoring
- Multi-GPU via NCCL (if hardware available; stub otherwise)
- Dispatch thresholds calibrated per operation

**Exit criterion:**
```cpp
auto A = randn<double>(5000, 5000, GPU{});
auto B = randn<double>(5000, 5000, GPU{});
auto C = matmul(A, B);   // dispatches to cuBLAS
// Benchmark: 10x faster than CPU path for this size
```

---

## 15.6 Phase 4 — SIMD Layer (Weeks 22–24)

**Goal:** CPU paths are SIMD-optimised.

**Deliverables:**
- ISA detection at startup (CPUID, `/proc/cpuinfo`)
- All elementwise operations: AVX-512, AVX2, NEON paths
- AVX-512 DGEMM micro-kernel
- Runtime dispatch table for all SIMD functions
- Portable SIMD API (`ms::simd::`) working
- Intrinsic typedefs (`f64x8`, `f32x16`, etc.) available in REPL
- Benchmarks showing SIMD speedup: ≥ 4x scalar for large arrays

**Exit criterion:** Benchmark shows AVX-512 elementwise exp is ≥ 4x scalar throughput. CPU DGEMM is within 80% of OpenBLAS performance.

---

## 15.7 Phase 5 — Qt6 IDE (Weeks 25–30)

**Goal:** Full IDE functional.

**Deliverables:**
- Editor with clangd LSP: completion, hover, jump-to-definition
- Plugin diagnostics as underlines in editor gutter
- Workspace variable inspector live-updating after each execution
- 2D plot bridge: `plot()`, `scatter()`, `hist()`, `spy()`, `imshow()`
- 3D surface via `surf()` using OpenGL
- File browser
- Session save/restore
- Hardware monitor bar (GPU%, CPU%, RAM, MPI rank)
- Settings panel
- Dark theme (Qt Fusion dark)

**Exit criterion:** Full workflow — open file, edit, run in REPL, see plot, inspect variable — end to end without error.

---

## 15.8 Phase 6 — Distributed (Weeks 31–36)

**Goal:** Multi-node computation correct and performant.

**Deliverables:**
- `MPIContext` fully implemented with all collectives
- `DistMatrix` with Block and BlockCyclic distributions
- Distributed solve, SVD, eigenvalues via SLATE
- GPU-direct MPI with host-staging fallback
- NUMA allocator and thread pinning working
- hwloc topology detection driving all assignments
- 4-rank CI tests all passing
- `mathscript-server` binary for headless ranks

**Exit criterion:**
```cpp
auto mpi = MPIContext{};
auto dA  = DistMatrix<double>::scatter(A, mpi).value();
auto x   = distributed::solve(dA, b, mpi).value();
// Result matches single-rank solve to 1e-10
```

---

## 15.9 Phase 7 — Frameworks (Weeks 37–42)

**Goal:** All 5 frameworks fully integrated.

**Deliverables:**
- GRIA: `compute_alpha`, GF(2^n) ops, CA analysis, LFSR analysis, dispatch hints
- Izaac: full 12-application suite, session RNG integration
- CyphaDIF: full `DifModel`, NIG primitives, CUDA dispatch via ms:: layer
- CellAI: `CellMemory`, Hebbian updates, SSM kernels, CyphaDIF bridge
- AXIOM: full evolution engine, primitive registry over ms:: namespace
- All framework unit tests passing
- Framework integration tests (GRIA → dispatch, Izaac → randn)

**Exit criterion:**
```cpp
// End-to-end: AXIOM finds an algorithm using GRIA fitness
auto axiom = Axiom{EvolutionConfig{.population_size=50}, primitives};
auto algo  = axiom.evolve(
    [&](auto& a) { return axiom.gria_fitness(a, training_data); },
    [](auto& a)  { return a.fitness > 0.45; }
).value();
```

---

## 15.10 Phase 8 — Special Functions (Weeks 43–48)

**Goal:** Full DLMF catalogue implemented own code.

**Deliverables (ordered by complexity):**
- Error functions, Fresnel, Dawson (Week 43)
- Gamma family, Beta family, Bernoulli/Euler numbers (Week 43)
- Bessel J/Y/I/K, spherical Bessel, zeros (Week 44)
- Airy, Struve, Kelvin, Anger-Weber (Week 44)
- Hypergeometric 0F1/1F1/2F1, Whittaker, Tricomi (Week 45)
- Meijer G, Fox H (Week 45)
- All orthogonal polynomial families (Week 46)
- Elliptic integrals and elliptic functions (all 12 Jacobi) (Week 46)
- Theta functions, Weierstrass elliptic functions (Week 47)
- Zeta functions, polylogarithm, Clausen (Week 47)
- Mathieu, spheroidal wave, parabolic cylinder (Week 47)
- Heun functions (all 5 confluent forms) (Week 48)
- Painlevé transcendents I–VI via ODE solver (Week 48)
- Numerical accuracy against DLMF reference data

**Exit criterion:** All special function numerical accuracy tests pass against NIST DLMF reference values.

---

## 15.11 Phase 9 — Replace Libraries (Ongoing from Week 20)

Replace all Class A transitional libraries with own implementations. This runs in parallel with later phases.

| Library | Replacement target | Priority |
|---|---|---|
| OpenBLAS | Own BLAS + LAPACK (SIMD-optimised) | High |
| SymEngine | Own symbolic CAS engine | Medium |
| SLATE | Own distributed linalg (post-SLATE) | Low |
| oneTBB | Own work-stealing thread pool | Low |
| hwloc | Own topology detection | Low |

**OpenBLAS replacement strategy:**
1. Own DGEMM (AVX-512 micro-kernel from Phase 4) + packing
2. Own DSYRK, DTRSM (triangular solve), DPOTRF (Cholesky) using own DGEMM
3. Own LAPACK routines building on own BLAS
4. Run same numerical accuracy tests — must match or exceed OpenBLAS

---

## 15.12 Phase 10 — Hardening (Weeks 49–53)

**Goal:** Production quality. Ready for 1.0.

**Deliverables:**
- Coverage ≥ 85% line
- All fuzz targets run ≥ 24 hours — no crashes
- Unsafe surface report: all sites reviewed and approved
- ORC JIT v2 backend option implemented and tested
- 3D plotting via OpenGL polished
- Documentation: API reference from headers, architecture docs
- Windows installer (NSIS/WiX)
- Linux packages (deb, rpm)
- Performance: all benchmarks within 10% of theoretical peak
- Memory: no leaks under Valgrind/Dr. Memory on test suite

**Exit criterion:** Version 1.0.0 tag. All CI stages green. Unsafe surface report approved. Packages buildable on clean machines.

### 15.12.1 Implementation progress

Waves 1 through 216 incrementally implemented this plan; each wave's specific additions are tracked in full detail in `CHANGELOG.md` rather than duplicated here.

**Current status:**
- Phase 10 hardening substantially underway (see `docs/RELEASE.md`'s tag-criteria list for gate details)
- Version 1.0.0 released; CI green on Windows and Linux
- ~91% line coverage (90% gate); Valgrind memcheck; libFuzzer smoke (7 targets)
- **369 CTest suites, all passing** (as of Wave 216)
- Wave 207: full documentation overhaul (README/ARCHITECTURE/API/USER_GUIDE/CONTRIBUTING/RELEASE/CHANGELOG/master-plan all rewritten or refreshed), plus 8 spec-vs-implementation gaps closed (special spherical harmonics, signal coherence, stats partial correlation/VIF, ML PR-AUC, finance historical VaR, graph min arborescence, poly_roots rewritten via companion-matrix eigenvalues, quantum Schmidt decomposition)
- Wave 208: 8 more gaps closed in previously-untouched modules (optim conjugate gradient, control step_info, image hough_circles, tensorops NMF, graph k-core decomposition, signal Chirp Z-Transform, ML IsolationForest, geo polygon triangulation)
- Wave 209: 8 more gaps closed (numthy Carmichael lambda, combo Eulerian numbers, bignum modular inverse, diffgeo torsion, compress Golomb-Rice coding, ODE exponential Euler/ETD1, topo witness complex, symbolic sym_expand)
- Wave 210: 8 more gaps closed (pde Helmholtz solver, cplx Cauchy principal value, info transfer entropy, special Kummer U, stats weighted statistics, finance Merton credit-risk model, ML spectral clustering, graph transitive closure)
- Wave 211: 8 more gaps closed (prob Rayleigh distribution, fft frequency-axis helpers, linalg Sylvester equation solver, geo Sutherland-Hodgman polygon clipping, numthy multiplicative order, cypha NIG moments, core::units compile-time-checked sqrt, simd sum reduction)
- Wave 212: 8 more gaps closed (domain lcm/extended_gcd, izaac exponential mechanism, gria CA hamming distance, cellai boltzmann weights, axiom mse/rmse fitness, core session_exponential, cuda device_memory_free, distributed allreduce_max/min)
- Wave 213: 8 more gaps closed (cpu BLAS L1/L2, core nextafter/sparse_add/Tensor reshape/MatMul expr, interp list_session_objects, runtime parallel_for, memory Arena bytes_used)
- Wave 214: 8 more gaps closed (signal sosfilt, prob Logistic distribution, stats kde, optim rmsprop, control lqe, compress ANS coding, special lambert_w, error ValueOutOfRange)
- Wave 215: 8 more gaps closed (image watershed, quantum purity, ode trapezoidal, combo restricted_partitions, finance heston_call, geo poly_union, poly poly_factor, ml AdaBoost)
- Wave 216: 8 more gaps closed (signal cheby1, finance sabr_call, image slic, optim adadelta, graph eccentricity, symbolic sym_collect, pde pde_laplace_2d, bignum bigint_next_prime)
- Unsafe surface audit + delta baseline (blocking in CI); install/package smoke (DEB/RPM/NSIS/WiX)
- ORC JIT scalar expr + native matrix/scalar dispatch; OpenGL PlotSurfWidget
- docs/API.md, docs/ARCHITECTURE.md, docs/RELEASE.md; scripts/pre_release.sh + tag checklists

**Remaining / known gaps:**
- 24h fuzz marathon (`fuzz-24h.yml` workflow_dispatch — manual step)
- Full ORC JIT v2 matrix LLVM IR lowering (post-1.0 enhancement)
- Windows installer / Linux packages (post-1.0 packaging)
- **REPL surface completeness:** two batches of 33 simple scalar bindings (Waves 194–195) covered numthy/poly/finance/special/prob; the large majority of library functions added since Wave 187 remain unbound (especially stateful/vector/callback-argument signatures, and entire modules like ml/image/tensorops/izaac beyond current bindings)
- HOSVD/Tucker share one REPL session-object kind (`TuckerDecomposition`) — introspection cannot distinguish algorithm by kind alone (cosmetic)
- **ms::signal:** cheby1 closed (Wave 216); advanced IIR/FIR design (cheby2/ellip/bessel/remez), spectral analysis (periodogram/music/esprit/pburg), emd/hht/vmd, modulation family
- **ms::geo:** polygon boolean ops (poly_intersect/diff) — robust non-convex union/intersection deferred; convex `poly_union` MVP closed (Wave 215); convex_hull_3d closed (Wave 193)
- **ms::graph:** planarity/embedding (Boyer-Myrvold), Blossom matching — substantial dedicated efforts
- **ms::finance:** SABR put and related extensions (SABR call closed Wave 216; Heston call Wave 215; Monte Carlo, Markowitz, lookback, Black-Litterman, etc. now closed)
- **ms::bignum:** APFloat/APComplex with full transcendental support — fundamental-rewrite scale
- **ms::image:** SIFT/SURF/ORB, graph-cut, marching-cubes (watershed/SLIC closed Waves 215–216)
- **ms::poly:** full factorization over R/C/Fp (rational-root subset closed Wave 196)
- **ms::izaac:** military namespace, full generic MPC beyond secret sharing
- **ms::crypto:** no standalone module planned; primitives routed through ms::izaac
- Exotic spec entries (Painlevé, Heun, Meijer G/Fox H, Mathieu, FEM, CFD) deliberately descoped
- ODE formula-bridge, stateful-class REPL, Axiom/PrimitiveRegistry, and tensor-decomposition REPL gaps resolved (Waves 179–187); no single large concrete gap remains — future work likely continues hardening, benchmarks, REPL bindings, and smaller API gaps

---

## 15.13 Milestone Summary

| Milestone | Version | Criteria |
|---|---|---|
| REPL works | 0.1.0 | Phase 1 complete |
| Math library complete | 0.2.0 | Phase 2 complete |
| GPU working | 0.3.0 | Phase 3 complete |
| IDE complete | 0.4.0 | Phase 5 complete |
| Distributed working | 0.5.0 | Phase 6 complete |
| Frameworks integrated | 0.6.0 | Phase 7 complete |
| Special functions | 0.7.0 | Phase 8 complete |
| Own math core | 0.9.0 | Phase 9 majority complete |
| Shipping | 1.0.0 | Phase 10 complete |


---

## Part 16: Math Runtime Implementation Guide

---

## 16.1 Function Implementation Template

Every math function follows the same implementation pattern. This is the template the AI build system uses for every function in every module.

```cpp
// include/ms/linalg/solve.hpp
namespace ms::linalg {

/// Solve the linear system A x = b for x.
///
/// @param A   Coefficient matrix (n × n)
/// @param b   Right-hand side (n × k)
/// @param policy  Execution policy — CPU{}, GPU{}, AUTO{}
/// @return    Solution x (n × k), or error
///
/// @errors
///   DimensionMismatch — if A is not square or b rows ≠ A rows
///   SingularMatrix    — if A is singular (condition > 1/eps)
///   DeviceError       — if GPU execution fails
///
/// @complexity  O(n³) — LU factorisation
/// @references  Golub & Van Loan, "Matrix Computations", 4th ed., §3.2
template<typename S>
[[nodiscard]] Result<Matrix<S>>
solve(const Matrix<S>& A, const Matrix<S>& b,
      ExecPolicy policy = AUTO{});

} // namespace ms::linalg
```

Implementation dispatch:

```cpp
// src/linalg/solve.cpp
template<typename S>
Result<Matrix<S>> ms::linalg::solve(
    const Matrix<S>& A, const Matrix<S>& b, ExecPolicy policy)
{
    // 1. Validate inputs
    if (A.rows() != A.cols())
        return ms::unexpected(DimensionMismatch{
            A.rows(), A.cols(), A.rows(), A.rows()});
    if (b.rows() != A.rows())
        return ms::unexpected(DimensionMismatch{
            b.rows(), b.cols(), A.rows(), b.cols()});

    // 2. Dispatch
    const auto decision = runtime::decide(
        A.rows() * A.cols(), OpClass::LinearSolve, policy, g_topology);

    if (decision.backend == runtime::Backend::CUDA) {
        return cuda_solve(A, b, decision.cuda_device);
    } else {
        return cpu_solve(A, b, decision.n_threads);
    }
}
```

---

## 16.2 Algorithm Selection Guide

The table below specifies which algorithm to use for each function. This is the authoritative specification for the AI build system.

### Linear Algebra Algorithms

| Function | Algorithm | Complexity | Reference |
|---|---|---|---|
| matmul (CPU large) | Goto-style tiling + AVX-512 microkernel | O(n³) | Goto & van de Geijn 2008 |
| matmul (GPU) | cuBLAS DGEMM | O(n³) | NVIDIA cuBLAS docs |
| lu | LU with partial pivoting (Gaussian elimination) | O(n³) | Golub & Van Loan §3.4 |
| qr (dense) | Householder reflections | O(mn²) | Golub §5.2 |
| qr (tall-thin) | TSQR (parallel) | O(mn²/p) | Demmel 2012 |
| svd | Golub-Reinsch + bidiagonalisation | O(mn²) | Golub §8.6 |
| eig (general) | Francis QR algorithm (implicit double shift) | O(n³) | Watkins 2007 |
| eig (symmetric) | Divide-and-conquer (D&C) | O(n³) | Gu & Eisenstat 1995 |
| chol | Right-looking Cholesky | O(n³/3) | Golub §4.2 |
| solve (square) | LU + triangular solve | O(n³) | Golub §3.2 |
| solve (overdetermined) | QR factorisation | O(mn²) | Golub §5.3 |
| pinv | SVD with threshold | O(mn²) | Golub §5.5 |
| expm | Padé approximation + scaling | O(n³) | Higham 2005 |

### FFT Algorithms

| Function | Algorithm | Complexity | Reference |
|---|---|---|---|
| fft (power-of-2) | Cooley-Tukey DIT, split-radix | O(n log n) | Cooley & Tukey 1965 |
| fft (prime n) | Rader's algorithm | O(n log n) | Rader 1968 |
| fft (composite n) | Mixed-radix FFT | O(n log n) | Bluestein 1970 |
| fft (arbitrary n) | Bluestein (chirp-Z) | O(n log n) | Bluestein 1970 |
| fft2 | Row-column decomposition | O(mn log mn) | — |
| dct-II | Via length-2n FFT | O(n log n) | Makhoul 1980 |
| stft | Windowed FFT with overlap | O(n/H × N log N) | Allen 1977 |

### Polynomial Algorithms

| Function | Algorithm | Complexity | Reference |
|---|---|---|---|
| polymul (small) | Direct convolution | O(n²) | — |
| polymul (large) | FFT convolution | O(n log n) | Schönhage 1971 |
| polygcd | Half-GCD algorithm | O(n log² n) | Knuth TAOCP §4.6 |
| polyroots | Companion matrix eigenvalues | O(n³) | Edelman & Murakami 1995 |
| eval_multi | Subproduct tree | O(n log² n) | Bostan et al. |
| polyfactor (Q) | Berlekamp-Zassenhaus | O(n³ log n) | Cohen §3.5 |

---

## 16.3 Numerical Precision Policy

**Default:** `double` (64-bit IEEE 754)

**For precision-critical functions (special functions, symbolic evaluation):** Use compensated summation (Kahan) or higher-intermediate precision.

**Kahan summation for reductions:**
```cpp
template<typename T>
T kahan_sum(std::span<const T> data) {
    T sum = 0, c = 0;
    for (T x : data) {
        T y = x - c;
        T t = sum + y;
        c   = (t - sum) - y;
        sum = t;
    }
    return sum;
}
```

**Pairwise summation for large arrays (better parallelism):**
```cpp
template<typename T>
T pairwise_sum(const T* data, size_t n) {
    if (n <= 8) {
        T s = 0;
        for (size_t i = 0; i < n; ++i) s += data[i];
        return s;
    }
    return pairwise_sum(data, n/2) + pairwise_sum(data + n/2, n - n/2);
}
```

---

## 16.4 Error Bound Documentation

Every function documents its error bound. Example:

```cpp
/// Computes the matrix exponential exp(A).
///
/// @accuracy
///   Uses Padé approximation of order [13/13].
///   Backward error bounded by u × ||A|| where u = machine epsilon.
///   For well-conditioned A, forward error ≈ condition(A) × u.
///   See: Higham, "Functions of Matrices", SIAM 2008, §10.3
///
/// @note
///   For matrices with large norm, scaling and squaring is applied:
///   compute exp(A/2^s) then square s times, where s = max(0, 1 + floor(log2(||A||_1)))
```

---

## 16.5 Transitional Library Interface

While transitional libraries (OpenBLAS, SymEngine) are active, they are accessed only through a thin interface layer. This makes replacement surgical — the interface is the same, the implementation changes.

```cpp
// src/runtime/cpu/blas_impl.hpp
namespace ms::cpu::blas {

// These signatures are fixed — implementations change
void dgemm(char transa, char transb,
           int m, int n, int k,
           double alpha, const double* A, int lda,
           const double* B, int ldb,
           double beta, double* C, int ldc);

void dsyrk(char uplo, char trans,
           int n, int k,
           double alpha, const double* A, int lda,
           double beta, double* C, int ldc);

void dtrsm(char side, char uplo, char transa, char diag,
           int m, int n,
           double alpha, const double* A, int lda,
           double* B, int ldb);

// ... all needed BLAS routines

} // namespace ms::cpu::blas
```

The implementation file calls OpenBLAS today:

```cpp
// src/runtime/cpu/blas_impl_openblas.cpp
void ms::cpu::blas::dgemm(...) {
    [[ms::unsafe("OpenBLAS C API — transitional implementation")]]
    ::cblas_dgemm(CblasColMajor, ...);
}
```

When own BLAS is ready:

```cpp
// src/runtime/cpu/blas_impl_own.cpp
void ms::cpu::blas::dgemm(...) {
    own::blocked_dgemm_avx512(...);
}
```

One file swap. All callers unchanged.


