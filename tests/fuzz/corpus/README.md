# libFuzzer seed corpora

Each subdirectory holds optional seed inputs for one fuzz target. libFuzzer
recursively loads every file under the directory passed to `-corpus_dir=`.

## Layout

```
corpus/
  fuzz_special_fns/   # seeds for tests/fuzz/fuzz_special_fns.cpp
  fuzz_matrix_ops/    # seeds for tests/fuzz/fuzz_matrix_ops.cpp
  fuzz_repl_input/    # seeds for tests/fuzz/fuzz_repl_input.cpp
  fuzz_sym_parser/    # seeds for tests/fuzz/fuzz_sym_parser.cpp
  fuzz_poly_ops/      # seeds for tests/fuzz/fuzz_poly_ops.cpp
  fuzz_bignum/        # seeds for tests/fuzz/fuzz_bignum.cpp
  fuzz_mpi_message/   # seeds for tests/fuzz/fuzz_mpi_message.cpp
```

Directories may be empty; each target should have at least one small seed under its folder. `fuzz_repl_input/` ships text seeds; other targets ship `seed.bin` byte seeds. CI smoke, nightly, and fuzz-24h pass `-corpus_dir=tests/fuzz/corpus/<target>` when the directory exists.
runs can point `-corpus_dir=tests/fuzz/corpus/<target>` to improve coverage
faster.

## Adding seeds

- Keep files small (typically under 256 bytes).
- Prefer inputs that exercise distinct code paths (e.g. small matrices, short
  REPL lines, low-degree polynomials).
- Binary seeds are fine; libFuzzer treats file contents as raw bytes.

Example local run:

```bash
./build-fuzz/tests/fuzz/fuzz_sym_parser \
  -corpus_dir=tests/fuzz/corpus/fuzz_sym_parser \
  -max_total_time=60
```
