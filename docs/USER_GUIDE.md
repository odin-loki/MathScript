# MathScript User Guide

MathScript ships a line-oriented **REPL** (read–eval–print loop) that lets you call the library interactively: assign variables, run linear algebra, statistics, ODE solvers, plotting, and hundreds of other functions without writing C++. This guide is for anyone opening the REPL for the first time. For the complete function catalogue see [`docs/API.md`](API.md); for building from source see [`docs/CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Starting the REPL

After a successful build, the console REPL binary is `mathscript-repl` (e.g. `build-msvc/bin/mathscript-repl.exe` on Windows, `build-linux/bin/mathscript-repl` on Linux).

**Interactive session** — type commands at the `ms>` prompt; type `exit` or `quit` to leave:

```powershell
mathscript-repl
```

**One-shot evaluation** — run a command and exit (repeatable `-e` / `--eval`):

```powershell
mathscript-repl -e "x = 42"
mathscript-repl -e "A = [1, 2; 3, 4]" -e "det(A)"
```

**Load a saved session first** — useful with `-e` so the process exits after eval instead of blocking in interactive mode:

```powershell
mathscript-repl --load session.ms -e "vars"
```

Other useful flags (see `mathscript-repl --help`):

| Flag | Purpose |
|------|---------|
| `-e`, `--eval <cmd>` | Execute one REPL line and exit (repeatable) |
| `--load <file.ms>` | Load session before eval or interactive mode |
| `--eval-file <path>` | Run script lines from file, then enter interactive mode |
| `--debug` | Trace each command (timing, state diff, parse kind) to stderr |
| `--jit` | Use LLVM ORC backend when linked (see [JIT mode](#jit-mode)) |
| `--jit-stats` | Log JIT dispatch path per command (requires `--jit`) |
| `-h`, `--help` | Show usage |
| `--version` | Print version and build metadata |

Inside the REPL, type `help` for a compact command and function summary, `vars` to list session variables, and `version` for build info.

---

## Basic syntax

### Variable assignment

Scalars and matrices are stored in the session by name:

```
ms> x = 2.5
ms> A = [1, 2; 3, 4]
ms> y = x / 2
ms> z = sin(0)
```

Scalar expressions support `+`, `-`, `*`, `/`, parentheses, unary minus, and libm-style calls such as `sin`, `cos`, `sqrt`, and `pow`:

```
ms> c = sqrt(b)
ms> y = erf(0.5)
```

### Matrix literals

Matrices use square brackets. **Rows** are separated by semicolons (`;`); **columns** within a row by commas (`,`):

```
[1, 2; 3, 4]       # 2×2 matrix
[1, 2, 3, 4]       # 1×4 row vector
[1; 2; 3; 4; 5]    # 5×1 column vector
[]                 # empty matrix
```

All rows must have the same width; mismatched rows produce an error.

### Function calls and results

Most library functions are invoked as `name(arg1, arg2, …)`. Many can be used either as bare calls (printing a formatted result) or on the right-hand side of an assignment:

```
ms> det(A)
ms> x = solve(A, B)
ms> U, S, V = svd(A)
```

Multi-target assignments (`L, U = lu(M)`, `Q, R = qr(M)`) store each output matrix under its variable name.

Bare function calls and assignments print their result to stdout when non-empty. Use `vars` to inspect what is in the session without re-printing values:

```
ms> vars
x = 2.5
A (2x2)
```

Type `clear` to reset all scalars, matrices, plots, and session objects.

---

## Calling library functions

The REPL exposes the same numerical routines documented in [`docs/API.md`](API.md). Below are small, verified examples across several domains.

### Linear algebra

```
ms> A = [3, 1; 1, 2]
ms> B = [1; 1]
ms> x = solve(A, B)
ms> det(A)
ms> trace(A)
ms> C = matmul(A, A)
ms> U, S, V = svd(A)
```

`solve(A, B)` solves the linear system \(Ax = b\). For the example above, `x` is approximately `[0.2; 0.4]`.

For large sparse or distributed setups, **`dist_cg(A, b)`** runs a conjugate-gradient linear solve (stub-safe: gathers to rank 0 and solves locally when MPI is off or size is 1):

```
ms> S = [4, 1; 1, 3]
ms> rhs = [1; 2]
ms> xcg = dist_cg(S, rhs)
```

Related distributed helpers: `dist_solve(A, b)` (direct gather + `solve`), `dist_gmres(A, b)`, `dist_jacobi(A, b)`, `dist_matmul(A, B)` (row-block GEMM), and `mpi` / `mpi_rank()` / `mpi_size()` / `mpi_allreduce_sum(x)` for backend introspection.

### Statistics

Vector arguments are typically column vectors (`;`-separated):

```
ms> m = stats_mean([1; 2; 3; 4; 5])
ms> stats_mean([1; 2; 3; 4; 5])
ms> sd = stats_stddev([1; 2; 3; 4; 5])
ms> r = stats_correlation([1; 2; 3; 4; 5], [2; 4; 6; 8; 10])
```

`stats_mean` of `[1; 2; 3; 4; 5]` prints `3`; `stats_stddev` prints approximately `1.58114`.

### ODE solvers (formula bridge)

Several solvers take the ODE right-hand side as a **quoted formula string** rather than a REPL variable. The string is parsed by the symbolic engine (`sym_parse`) into a callable \(f(t, y)\). For a scalar ODE \(y' = y\) with \(y(0)=1\) on \([0, 1]\):

```
ms> ode_rk4("y", 0, 1, 1, 100)
```

Arguments are: `"formula"`, `t0`, `y0`, `t_end`, `steps`. The output is a trajectory of `[t, y]` rows; for this exponential growth problem the last row is approximately `[1, 2.718…]`.

Adaptive RK45 adds tolerance arguments:

```
ms> ode_rk45("y", 0, 1, 1, 1e-6, 1e-9)
```

For systems, separate component formulas with semicolons inside the quoted string and pass a column initial condition:

```
ms> ode_rk4_vec("y1; -y0", 0, [1, 0], 1, 1000)
```

Here `y0` and `y1` refer to the first and second state components. More elaborate expressions (e.g. `"y - t*t"`, `" -10*y"`) work when they parse cleanly; malformed formulas return a parse error.

### Signal / FFT

```
ms> fft([1, 2, 3, 4])
```

Returns formatted FFT magnitudes for a 1×N row vector (2-D matrices are rejected).

### Cryptography (hex I/O)

Several crypto helpers take **hex-encoded** byte strings and return hex output. **`crypto_hkdf_sha256(hex_ikm, hex_salt, hex_info, len)`** performs HKDF-SHA256 extract/expand (RFC 5869) and returns `len` bytes of key material as hex:

```
ms> crypto_hkdf_sha256("0b0b0b0b0b0b0b0b0b0b0b", "000102030405060708090a0b0c", "f0f1f2f3f4f5f6f7f8f9", 42)
```

Other Wave 231+ bindings include AES-128/256 CBC, ChaCha20, AES-GCM, ChaCha20-Poly1305, X25519, PBKDF2, and Ed25519 (`crypto_ed25519_keypair` / `sign` / `verify`) — see [`docs/API.md`](API.md).

---

## Stateful objects (session-object registry)

Some subsystems return **handles** to long-lived C++ objects stored in a per-session registry. Creation functions follow the `*_new(handle, …)` pattern; subsequent calls take the handle as the first argument.

### Bloom filter

Bloom filters require an RNG seed before creation:

```
ms> izaac seed 42
ms> bloom_new(bf, 100, 0.01)
ms> bloom_insert(bf, "alpha")
ms> bloom_check(bf, "alpha")
true
```

### Tensor decomposition

CP/Tucker decompositions create typed handles you can reconstruct from later:

```
ms> tensorops_decompose_cp(cp1, [1, 2; 3, 4], 2)
ms> tensorops_reconstruct_cp(cp1)
ms> tensorops_decompose_tucker(tk1, [1, 2; 3, 4], [2, 2])
ms> tensorops_reconstruct_tucker(tk1)
```

### Introspection

List active handles and their types:

```
ms> session_objects()
bf bloom
cp1 cpdecomposition
tk1 tuckerdecomposition
```

Remove one handle:

```
ms> session_object_clear(cp1)
```

Other registry-backed families include `cluster_new`, `tokenbucket_new`, `cellmemory_new`, and `difmodel_new` — see `help` and [`docs/API.md`](API.md) for the full set.

---

## Plotting

Plot commands store a **plot series** in the session. Supported forms include:

| Command | Description |
|---------|-------------|
| `plot([y…])` | Line plot (y vs. index) |
| `plot([x…], [y…])` | Line plot with explicit x |
| `scatter([x…], [y…])` | Scatter plot |
| `hist([…])` | Histogram (bar plot) |
| `imshow(matrix)` | Heatmap |
| `spy(matrix)` | Sparsity pattern |
| `surf([z…])` or `surf([x…], [y…], [z…])` | Surface / height field |

Examples:

```
ms> plot([1, 4, 2, 8])
ms> plot([0, 1], [5, 10])
ms> hist([1, 2, 2, 10])
ms> scatter([0, 1], [5, 10])
ms> imshow([1, 2; 3, 4])
ms> surf([1, 2; 3, 4])
```

### Console vs. GUI

In **`mathscript-repl`**, plot commands print a short status line plus an **ASCII preview** of the data (sampled rows/columns, sparsity dots for `spy`, etc.). Use:

- **`show`** — re-print the current plot preview
- **`saveplot <file.txt>`** — write the preview text to a file

There is no native GUI window in the console REPL.

When built with **`MS_BUILD_GUI=ON`**, the **`mathscript-gui`** IDE (Qt6) runs the same REPL engine but renders plots in **`PlotWidget`** (2-D) and **`PlotSurfWidget`** (3-D surfaces), with PNG export from the File menu. The REPL input bar accepts the same plot commands.

**Wave 238–240 GUI additions:**

| Feature | How to use |
|---------|------------|
| Find in output | **Edit → Find in Output** (**Ctrl+F**); **Find Next in Output** (**F3**) |
| Find in script / goto line | **Edit → Find in Script** / **Go to Line…** (**Ctrl+G**) |
| Plot panel toggle | **View → Show Plot Panel** — show/hide the plot dock |
| Line numbers / cursor | Script editor gutter; status bar **Ln X, Col Y** |
| Export last result as LaTeX | **File → Export Last Result as LaTeX…** — plain-text `.tex` |
| Export command history | **File → Export Command History…** — save REPL input history to a text file |
| Up/Down command history | **Up-arrow / Down-arrow** in the REPL input bar (draft recall on Down past newest entry) |

Earlier GUI waves added syntax highlighting, layout persistence, variable inspector, cooperative **Stop**, status-bar GPU info, **Ctrl+Enter** Run, Clear Output (**Ctrl+L**), Open Recent, font zoom, and About dialog — all optional when `MS_BUILD_GUI=ON`.

Named matrices work as arguments: `scatter(X, Y)`, `imshow(M)`, `spy(S)`.

---

## Sessions

Save and restore scalars, matrices, the current plot state, and **command history** to a `.ms` session file.

**Inside the REPL:**

```
ms> x = 7
ms> A = [1, 2; 3, 4]
ms> plot([1, 4, 2, 8])
ms> save mysession.ms
ms> clear
ms> load mysession.ms
ms> vars
```

**At the command line** (before interactive mode or `-e` eval):

```powershell
mathscript-repl --load mysession.ms
```

Session files are text-based (`# MathScript session` header, `scalar` / `matrix` lines, optional `plot` metadata, optional `history` lines). They round-trip scalars, matrices (including empty `[]`), plot series, and REPL command history.

Export history without a full session save:

```
ms> export history myhistory.txt
ms> save_history myhistory.txt
```

Both meta-commands write one REPL command per line (aliases).

**`load` replaces the entire session** with the saved state. To run a script of REPL commands without clearing variables, use **`run_file`** or its alias **`source`**:

```
ms> x = 1
ms> run_file setup.ms
ms> source more.ms
```

Script files use the same line format as `--eval-file`: one command per line; empty lines and `#` comments are skipped. Unlike `load`, existing scalars and matrices stay in the session unless the script assigns over them.

---

## Scripting

### `run_file` / `source` (interactive REPL)

Inside an interactive session, **`run_file path.ms`** (alias **`source path.ms`**) executes each non-comment line as a REPL command and keeps the current session. This matches **`mathscript-repl --eval-file`** but can be invoked mid-session. Use **`load`** when you want to restore a saved session snapshot instead.

### `mathscript-repl --eval-file`

Run a script file line-by-line (empty lines and `#` comments are skipped), then drop into interactive mode unless `-e` is also given:

```powershell
mathscript-repl --eval-file setup.ms
```

Each line is a single REPL command — there is no block syntax or semicolon-separated statements on one line.

Example `setup.ms`:

```
# Initialize a small linear algebra session
A = [3, 1; 1, 2]
B = [1; 1]
x = solve(A, B)
det(A)
```

Combine with `--load`, `--debug`, or `--jit` as needed.

### `mathscriptc`

**`mathscriptc`** is a dedicated batch runner: pass a `.ms` file as the sole positional argument and it executes every non-comment line, printing output and exiting with a non-zero status on the first error:

```powershell
mathscriptc setup.ms
mathscriptc --version
```

Unlike `mathscript-repl`, `mathscriptc` does not support `--jit`, `--load`, or interactive mode — it always uses the interpreted REPL engine and always exits when the file finishes.

---

## JIT mode

When MathScript is built with **`-DMS_BUILD_JIT=ON`** (LLVM 18 ORC LLJIT linked), pass **`--jit`** to `mathscript-repl`:

```powershell
mathscript-repl --jit -e "x = 1 + 2"
mathscript-repl --jit --jit-stats -e "y = sin(x)" -e "A = matmul(B, C)"
```

What `--jit` accelerates today:

| Form | Path |
|------|------|
| Scalar literal assignment (`x = 3.14`) | LLVM JIT compile |
| Scalar expression assignment (`y = a * sin(b) + 2`) | LLVM JIT compile (libm calls, parentheses, unary `-`) |
| Matrix call assignment (`C = matmul(A, B)`) | Native C++ dispatch (not LLVM-compiled) |
| Multi-matrix assignment (`L, U = lu(A)`) | Native C++ dispatch |
| Everything else (plot commands, ODE calls, session objects, bare function calls, `help`, …) | **REPL interpreter fallback** |

If LLVM init or smoke-compile fails, the backend silently delegates to the standard interpreter (`orc-jit-llvm-fallback`). Add **`--jit-stats`** to log whether each line took the JIT, native-dispatch, or fallback path.

JIT does not change numerical results — it only affects how scalar assignments are executed.

---

## Where to go next

- **[`docs/API.md`](API.md)** — exhaustive REPL function reference grouped by module (linalg, stats, signal, ML, graph, special functions, …).
- **[`docs/ARCHITECTURE.md`](ARCHITECTURE.md)** — how the REPL, JIT backend, session registry, and GUI fit into the wider library.
- **[`docs/CONTRIBUTING.md`](CONTRIBUTING.md)** — build options, test commands, coverage, and JIT build instructions.
- **[`CHANGELOG.md`](../CHANGELOG.md)** — recent features, new REPL bindings, and breaking changes.

Type **`help`** at the `ms>` prompt anytime for an inline summary of syntax, plotting, ODE formula strings, and the most common function forms.
