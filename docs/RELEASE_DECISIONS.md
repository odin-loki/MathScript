# MathScript 1.0 scope

These items are **decided**, not remaining work. A gap chosen not to close is not a 1.0 gap.

See [`RELEASE.md`](RELEASE.md) for tag criteria.

## Stubs

| Location | Decision |
|----------|----------|
| NCCL (`include/ms/cuda/nccl.hpp`, `src/cuda/nccl.cpp`) | Documented stubs. Multi-GPU NCCL is post-1.0. |
| `src/interp/jit_orc_stub.cpp` | Leave — this *is* the non-LLVM JIT backend. |
| CUDA solver `"not implemented"` paths | Return `Result` errors; not a silent success. Not tag-blocking once honest. |
| GUI placeholders in `MainWindow.cpp` | Out of 1.0. |
| `axiom.cpp` placeholders | Not 1.0-blocking. |
| Isolated `TODO` comments in `repl_engine.cpp` | Not 1.0-blocking. |

## Post-1.0 (not remaining)

| Item | 1.0 surface |
|------|----------|
| Scalable multi-node MPI linear algebra | Block/gather `dist_*` only |
| Full IDE (LSP, debugger, rendered LaTeX) | Separate product |
| Full NCCL multi-GPU | Stubs |
| Weighted blossom matching | Unweighted matching ships |
| APFloat/APComplex transcendentals | Rewrite-scale |
| SIFT/SURF/ORB, graph-cut, marching cubes | Not in tree |
| Boyer–Myrvold planarity | `is_planar_k5_k33_check` is the honest partial |
