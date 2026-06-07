# MathScript HPC Computer Algebra System

A high-performance Computer Algebra System built in C++23 with CPU math libraries, runtime dispatch, and a console REPL.

## Project Status: Phase 2 (Math Library CPU) — In Progress

### Completed Components

- **Build system:** Native Windows MSVC + Ninja, GoogleTest via FetchContent
- **Core types:** Matrix, Tensor, Sparse, Scalar, Sym
- **Linear algebra:** matmul, lu, qr, chol, solve, expm, trace, det, norm, eye, zeros, ones, transpose
- **Math modules:** fft, stats, prob, optim, signal, special (CPU implementations)
- **Runtime:** Topology detection, thread pool, dispatch layer
- **Executables:** `mathscriptc`, `mathscript-repl` (console mode)
- **Unit tests:** 5 linalg test suites — all passing

### Build Instructions (Native Windows)

#### Prerequisites
- Visual Studio 2022/2026 Build Tools with C++ workload
- CMake 3.28+
- Ninja (included with Strawberry Perl or VS)

#### Build
```powershell
.\build.ps1
```

Or manually:
```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=amd64 && cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON && cmake --build build-msvc --config Release"
```

#### Run tests
```powershell
ctest --test-dir build-msvc --output-on-failure
```

#### Run REPL
```powershell
.\build-msvc\bin\mathscript-repl.exe
```

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Complete |
| 1 | Core Types + REPL | Complete (console REPL) |
| 2 | Math Library CPU | In Progress |
| 3-10 | GPU, Qt6, Distributed, etc. | Pending |

## Documentation

See `mathscript-master-plan.md` for the complete 10-phase delivery plan.
