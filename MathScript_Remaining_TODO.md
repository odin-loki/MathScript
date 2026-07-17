# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-17 (reconciled against `main` @ Wave 224 — **374 CTest suites**, MSVC green, **22-bench smoke OK**, profiling **complete**)

---

## Performance profiling ✅ COMPLETE (Waves 218–224)

Seven profiling waves optimized all identified hot paths. No further profiling waves scheduled unless new bottlenecks are discovered.

| Wave | Focus |
|------|--------|
| 218 | FFT convolve, moving_average, iterative FFT, SIMD sub/abs, kde, matmul |
| 219 | welch/spectrogram, rfft, poly batch, median_filter, FMA dot, percentile |
| 221 | xcorr, sosfilt, savgol, conv2, SHA, FEM Thomas, norm_l2 |
| 222 | coherence/spectrogram buffers, periodogram, BFS, SLIC, REPL eval, sum_squares |
| 223 | fft2 dims, 2D FFT conv2, watershed, gaussian blur, welch/filtfilt/resample |
| 224 | medfilt2, boxfilter, imresize, bilateral, Dijkstra, stats quantile, 22-bench smoke, MSVC baselines |

**Remaining perf infra (CI-only):** Linux `bench_regression.sh --write-baseline` to fill null medians in `linux-gcc13.json`.

---

## Next: Feature work (Wave 225+)

| Wave | Focus |
|------|--------|
| **225** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **226–227** | sym transforms + CUDA/MPI/plugin |
| **228–232** | GUI polish + REPL bindings |
| **233+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
