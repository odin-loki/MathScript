# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Wave-by-wave implementation history (thousands of entries) is in [`docs/WAVES.md`](docs/WAVES.md). Current architecture, API, and release criteria are in [`docs/`](docs/).

## [Unreleased]

### Features completed

Every entry that `docs/RELEASE_DECISIONS.md` listed as a deliberately-deferred
stub has been closed; see that file for the before/after table.

- Graph: `max_weight_matching` / `max_weight_matching_value` (Edmonds' primal-dual blossom, verified against exhaustive brute force on random graphs), and exact planarity via `is_planar` (Left-Right criterion) with `planar_embedding` and `kuratowski_subgraph`. The old heuristic remains as `is_planar_k5_k33_check`.
- Geo: `marching_cubes` / `marching_cubes_mesh` (full 256-case Lorensen-Cline tables), `marching_squares`, `mesh_surface_area`, `mesh_volume`. Sphere area and volume reproduce 4*pi*r^2 and 4/3*pi*r^3 to 0.17% and 0.30% on a 48^3 grid.
- Image: `graph_cut_segment` / `grabcut_segment` / `min_cut_value`; `fast_corners`, `orb_detect_and_compute`, `sift_detect_and_compute` and descriptor matching. SURF is deliberately not implemented.
- Bignum: `APFloat` and `APComplex` with a full arbitrary-precision transcendental set. pi, e and ln2 reproduce their standard expansions to 40+ digits.
- Distributed: a `dist_ops` communication layer, SUMMA matmul, and row-distributed Krylov solvers replacing the gather-to-one-rank path (which remains the documented fallback).
- CUDA: real NCCL communicator management and collectives behind `MS_HAS_NCCL`; the default build keeps its identity semantics.
- ML: real Barnes-Hut t-SNE replacing the dense O(n^2) stub whose perplexity search never converged on a target entropy.
- Frameworks: Axiom's `evaluation` / `selection` / `mutation` Syms carry real per-individual provenance instead of three constants.
- Geo: `poly_boolean` and its `poly_union_general` / `poly_intersect_general` / `poly_diff_general` / `poly_symmetric_diff_general` wrappers are a general two-polygon clipper for arbitrary simple polygons: concave operands, results that split into several disjoint pieces, and results containing holes are all exact. The pre-existing `poly_union` / `poly_intersect` / `poly_diff` stay as the documented convex MVPs that return a convex-hull over-approximation. Exposed in the REPL as `geo_boolean_union` / `geo_boolean_intersect` / `geo_boolean_diff` / `geo_boolean_xor`, returning `(x, y, contour_index)` rows.
- Linalg: the blocked Schur-Parlett (Davies-Higham) the header recorded as not implemented. `logm`, `sinm` and `cosm` now serve repeated and clustered eigenvalues by grouping them, reordering the Schur form with Givens swaps, evaluating each diagonal block by its Taylor series about the group mean, and filling the off-diagonal blocks from triangular Sylvester solves. The new `funm_taylor(A, coefficients)` does the same for a caller-supplied `f`, which plain `funm` cannot: `f` alone does not determine `f(A)` at a repeated eigenvalue.
- `mathscript-server` is a real SPMD compute node (`--script`, `-e`, `--serve`, one Interpreter per rank) rather than a heartbeat loop that ignored argv.
- REPL: a bare name or expression now prints its value. `x`, `A`, `1 + 2`, `sqrt(2)` and `x / 2 + 1` were all rejected with "could not parse" -- the only way to look at a variable was to assign it somewhere else first. The fallback runs last, after every command and every assignment form has declined the line, so it shadows nothing and a line that is not an expression still reports the parse error.
- REPL: `erfc`, `gamma`, `zeta`, `fresnel_c` and `fresnel_s` were reachable only in the bare-call printing form; the scalar-expression evaluator did not know them, so `y = gamma(4)` failed as an unknown function while the REPL's own index documented `gamma(x)` and `zeta(s)`. All five now work in both forms, and `erfc` / `fresnel_c` / `fresnel_s` were added to the index.
- REPL: the libm scalar set is complete. It had `sinh` but not `asinh`, `log10` but not `log2`, and `sqrt` but not `cbrt`; `log2`, `exp2`, `expm1`, `log1p`, `cbrt`, `asinh`, `acosh`, `atanh`, `round`, `trunc`, `hypot(x,y)` and `fmod(x,y)` now join them, and `help` lists the whole set instead of "sin, cos, sqrt, pow, min, max, ...".
- REPL: matrix accessors. The interpreter had over a thousand functions that produce matrices and no way to look inside one -- no shape, no element, no row, no column. `mat_rows(A)`, `mat_cols(A)`, `mat_numel(A)` and `mat_at(A,i,j)` return scalars; `mat_row(A,i)`, `mat_col(A,j)`, `mat_reshape(A,rows,cols)` and `mat_submatrix(A,r0,c0,rows,cols)` return matrices. Indices are 0-based, and an index that is fractional, negative or past the end is a reported error rather than a read past the buffer.
- REPL: a matrix call written without a target now prints its result as `_` **and stores it there**, so the next line can use it. This goes through the matrix-call registry, so it covers every matrix-returning callee and its accepted arities at once, including the constructors `zeros`, `ones`, `eye`, `rand`, `randn` and `linspace`, which had no no-assignment form at all. Six callees -- `matmul`, `tensorops_matmul`, `tensorops_einsum`, `signal_conv2`, `ml_mat_mul` and `dist_matmul` -- kept hand-written branches that predated the registry and shadowed it, printing under an invented `C` and storing nothing; the golden corpus found them and they now go the same way as the rest.
- ML model packing: three out-of-bounds accesses in the REPL's model serialisers, all of them reachable from an ordinary `ml_*_fit` call. The NaiveBayes packer sized the matrix `max(n_features, 1)` columns wide and then wrote the header at column 1, so a single-feature model corrupted the heap; the KNN packer sized it `n_features + 1` wide and wrote the header at column 2; and the LDA packer sized its per-class loop by `classes.size()`, which `LDA::fit` fills before it gives up on a single-class problem, so it indexed the empty `discrim_const`. The LDA path now reports "expected at least two distinct class labels in y" instead of returning a model that was never fitted, and the QDA packer got the same consistency check.
- ML ensemble sizes are bounded. `n_trees`, `n_estimators` and `max_depth` arrived from the command line unchecked, so `ml_random_forest_fit(X, y, 3000000000)` and `ml_gradient_boosting_fit(X, y, 3000000000)` grew trees until the process was killed, and `ml_isolation_forest_fit(X, 1e18)` asked for an allocation that aborts rather than reports under `-fno-exceptions`. The caps are 10000 members, depth 512, and a forest sample of 1000000, each reported as a `DomainError`.
- REPL size arguments are checked on the double, before the cast and before the allocation. Fourteen commands ended the process outright when given a large count -- `fem_poisson1d/2d/3d`, `cfd_advection1d/2d/3d`, `numthy_farey`, `impad`, `imresize`, `hough_lines`, `hough_circles`, `ml_pca_fit`, `ml_pca_fit_transform` and `ml_kmeans_fit` -- because the guard in front of each was `const int n_i = static_cast<int>(n_d); if (n_i < 0 || n_d != n_i)`. That cast is undefined behaviour for a double outside `int`'s range rather than a wrap, so it is not a value to test; an `int` that fits is not an allocation that fits (`fem_poisson1d(100000000)` asks for 800 MB); and a bound on each extent is not a bound on their product (`imresize(A, 100000, 100000)` is ten billion elements). With `-fno-exceptions` the `std::bad_alloc` reached `std::terminate` and the process was gone with nothing on either stream. `ExtentBudget` in `src/interp/matrix_call.hpp` reads extents one at a time and divides the existing 262144-element REPL budget down as it goes, so the cap lands on the product; `ml_pca_fit` and `ml_kmeans_fit` get a shape relationship instead, because there are only `min(samples, features)` principal components and k clusters need k points; and `numthy_farey` counts its sequence exactly with a totient sieve, since the length is quadratic in the argument.
- Every integer argument the REPL takes is decided on the double, not after the cast. The size arguments above were the ones that ended the process; the same `static_cast<int>` sat in front of all 256 integer arguments, and it is undefined behaviour for a double outside `int`'s range rather than a wrap. A second family needed a different bound: `legendre_p(1750000000, 0.5)` *returns*, because an order allocates nothing and instead drives a recurrence of one step per unit inside a REPL that cannot interrupt one. `checked_int_argument` bounds those at 10000000 -- the number `repl_engine_internal.cpp` has used for a matrix index and a matrix count since the accessors were added. It is a work bound and not an accuracy one. The same helper refuses truncation, which is the half with no undefined behaviour in it: `bessel_j(1.5, 1)` used to answer as though 1 had been written. Removing the cast made 109 conditions unreachable (`if (n < 0 || n_d != n)` cannot reach its second half once the double has been checked) and they are removed with it; 42 tests that asserted a combined message now assert the per-argument one, which names which argument. Commands whose work is *quadratic* in the argument -- `finance_binomial_call` at 38000 steps, `tensorops_decompose_nmf` at rank 2200 -- are not covered by a linear bound and are still open, and so is the separate `uint64_t` conversion family, where one blanket cap would be wrong: `numthy_is_prime(1e18)` is a legitimate question with a fast answer and `numthy_prime_nth(1e18)` is the same magnitude and not.
- `image::impad` wrote about 4.29 GB past a null base on a large pad, and a guard at the REPL boundary does not reach it. `img.rows + 2*pad` is `int` arithmetic: at `pad >= (INT_MAX - 2) / 2` it overflows to a negative, `Image`'s constructor clamps a non-positive extent to an empty image, and the copy loop -- bounded by the source's extents rather than the destination's -- kept going and wrote an index near 2^30 into a zero-length vector. A negative pad reached the same write from the other end, through `Image::at`'s conversion to `size_t`. Both are refused inside the library now, which returns an empty image the way the rest of the file does with an argument it cannot honour. `image::imresize` had the same class of defect in its destination index, `(r * nc + c) * channels` in `int`, which wraps negative once the output passes INT_MAX elements -- 46341 x 46341 single-channel is already past it, and that is an image this type can hold.
- `compress::lz77_decode` read roughly four billion bytes past its buffer on a malformed token stream, and AddressSanitizer in CI found it rather than any local run. A token's `offset` is a distance back from the end of what has been decoded so far and comes straight from the caller's data; `out.size() - t.offset` is unsigned, so an offset larger than the output wrapped to an index near 2^64. It is reached from `lz77_decode_vec(M3)` -- a 3x3 matrix of small numbers -- and `test_repl_malformed_sweep` had been handing it one all along -- it failed both ASan runs that finished, and the runs before those were cancelled by the next push, so how far back it goes is not established. A stream that back-references a byte it never emitted is refused outright now, since no prefix of it is meaningful either, and `lz77_decode_vec` names the offending token first. It also stops truncating the token fields: `static_cast<uint16_t>(70000)` is 4464, so an offset past the type's range used to become a different, valid offset and decode silently to the wrong bytes.
- REPL: `medfilt2(M)` and `boxfilter(M)` take the arity the arity table has always said they take, with the default `image::medfilt2` declares in its own signature rather than an invented one. `imgaussfilt(M)` and `laplacian_of_gaussian(M)` answer that arity with a diagnostic instead of `assign: unsupported matrix call`, because sigma is not a setting on a Gaussian blur, it is the blur.
- REPL: `bigint("495")` with no assignment target. The command existed only in its assignment form, so a bare line naming it fell through every reading and came back "could not read '...' as a matrix call, a matrix constructor, or a scalar expression" -- for a valid literal as well as for `bigint("495.0")`, whose whole problem is one character and who was never told which. Both forms reach the same reporting parse now.

### Symbolic correctness

An audit drove the REPL over the standard integral, Laplace, Fourier, Mellin, Hankel
and Z-transform tables. Most rows declined, and four inputs came back with a wrong
answer and no error. The wrong answers are the serious half:

- `sym_parse` bound unary minus tighter than `^`, so `-t^2` parsed as `(-t)^2`. It
  evaluated to `+9` at `t = 3` where every convention gives `-9`, `-2^2` gave `4`, and
  `-t^0.5` gave `NaN` because it was taking a real root of a negative number.
  Exponentiation now binds tighter, and the exponent is still parsed as a unary, so
  `2^-3` and right-associative `2^3^2` are unchanged.
- `sym_ztransform("3*2^n")` returned `z/(z - 6)` instead of `3z/(z - 2)`: the matcher
  folded a leading coefficient into the base of the geometric sequence, moving the
  pole. Every scaled geometric sequence was affected.
- `sym_solve_linear` discarded any term it could not read as linear rather than
  refusing. `x + sin(y) - 1` solved to `x = 1` instead of `x = 1 - sin(y)`, `x - exp(a)`
  to `x = -0`, and `x^2 + x - 1` was answered as though the quadratic term were absent.
  Terms free of the unknowns are now carried as constants, and a term that is genuinely
  non-linear refuses the system.
- The unsupported sentinel leaked through the linearity rules. They recurse into each
  operand and reassemble whatever comes back, so one unsupported term inside a
  supported sum produced a tree that was part transform and part `d/dt(...)`, which the
  root-only check reported as a success. Failure now propagates outward at every
  linearity site, and `sym_is_unsupported` scans the whole tree rather than the root.

The table gaps are closed by stating the general rule instead of adding rows:

- Antiderivatives take any first-degree argument `u = a*x + b`, which is the linear case
  of the substitution rule, and cover `Sqrt`, `Log`, `Tan` and `Exp` node types the
  switch had no case for at all. `1/x`, `exp(x)`, `sqrt(x)`, `x/2`, `x^(-2)`, `sin(2x)`,
  `1/(2x+1)` and `log(x)` all integrate now; a small integer power of a non-linear base
  is expanded and integrated term by term.
- `sym_laplace` gained the first shifting theorem, `L{exp(a t) g(t)} = G(s - a)`, and
  frequency differentiation, `L{t^n g(t)} = (-1)^n G^(n)(s)`. Between them those two
  statements supply every s-shifted and every `t^n`-weighted row -- `t*exp(2t)`,
  `exp(-t)sin(3t)`, `t*sin(2t)`, `t^2*exp(-t)` -- with no per-row matcher.
- `sym_ilaplace` is keyed on three denominator families, `(s-a)^n`, `(s-a)^2 + b^2` and
  `s^2 - a^2`, with the numerator taken as an arbitrary `p*s + q`. `a = 0` recovers the
  unshifted rows, so the shifting theorem costs no extra matcher, and the hyperbolic row
  arrives for the first time. The previous table required each numerator to equal the
  constant its canonical row carries, so `2/(s^2+4)` inverted and `1/(s^2+4)` did not.
- The Fourier pair had no linearity at all -- `exp(-2t^2)` transformed and
  `3*exp(-2t^2)` did not. It has Add, Sub, Neg and constant Mul/Div now, matches a
  Gaussian however its coefficient is spelled, gained the forward Lorentzian row, and
  reads its own printed output back (the inverse demanded agreement to 1e-9 from a
  spectrum printed to six decimals).
- The Z-transform gained the same linearity, the `n^k` family via `(-z d/dz)^k`, the
  sampled sine and cosine rows, `exp(a n)` as a geometric sequence, the `z + a` pole
  spelling, and `z/(z-a)^2` on the inverse side.
- `try_get_const_value` folds constant arithmetic, so a coefficient written as `2*3` or
  `1/2` counts as the constant it is.

Two more silently-wrong results, and the rest of the tables:

- `pi` and `e` parsed as free variables. An unbound variable evaluates to zero, so
  `sym_eval("pi")` returned `0.000000` and `sym_eval("2*pi*r")` returned 0 for every
  `r`, with nothing reporting that a symbol was missing. They are constants now, and a
  name that merely begins with one of them is still a variable.
- `sym_dsolve` matched only the multiplied spelling `k*y` and not the divided one
  `y/k`, so `dy/dx = y/2` -- exponential growth written with a time constant, the
  commonest first-order ODE there is -- declined while `dy/dx = 0.5*y` solved. It also
  read the exponent of `y^n` with an `op == Const` test, and a negative literal is
  `Neg(Const)`, so `y^(-1)` never reached the rule the header's own table row promises
  for every `n != 1`. All eight audited ODEs solve now and are checked by substituting
  the solution back into the equation.
- `sym_mellin` required the numerator to be exactly 1, the constant exactly 1 and the
  power exactly `t`. One rule, `M{c/(a + t^n)}(s) = (c/n) a^(s/n - 1) pi / sin(pi s/n)`,
  covers the whole rational column; the shifting rule `M{t^a f(t)}(s) = M{f}(s + a)`
  covers `t/(1+t)`; `(1+t)^-m` and `log(1+t)` are added. `sym_imellin` compares its
  `pi` to a tolerance matched to the six-decimal printer instead of for equality.

Two wrong answers in the REPL's own scalar evaluator, found while scoping the number
formatting:

- A leading unary sign was applied to the whole expression. Both `eval_literal_arith`
  and `eval_scalar_expr_impl` tested `expr.front() == '-'` before looking for a binary
  operator, so `-4 + 1` was read as `-(4 + 1)` and evaluated to **-5**, `-4 - 1` to
  **-3**, and with `x = 4`, `-x + y` to **-6**. The operator is found first now, and a
  leading sign is unary only when there is no binary operator to bind to.
- An exponent literal could not appear in arithmetic. The top-level operator scan did
  not know that `+` or `-` can be the sign of an exponent, so `1e-09 * 2` was split at
  the minus and reported as "could not parse". `vars` already printed small magnitudes
  in exponent form, so a session could show a value it would not then accept back.

`sym_expand` collects like terms. It multiplied out over the expression tree and never
put like terms back together, so `(x+1)^3` came out as eight products rather than four
terms and `(x+1)^8` as 256 rather than nine -- and that is why nested powers exploded.
Expansion now runs on a canonical polynomial form where multiplication merges like
terms as it produces them, so an intermediate is never larger than the answer:
`((x+1)^8)^8` is the degree-64 binomial, sixty-five terms, and completes in 8 ms where
it previously did not return at all. The size ceiling that had been added to stop the
hang is now a ceiling on the size of the answer rather than on a projection of the
uncollected product.

Atoms in that form are identified by structure, not by printed text. `sym_to_string`
renders a constant with six decimals, so `sin(1.0000001*x)` and `sin(1.0000002*x)`
print identically; keying atoms by their printed form would merge them and expand
their difference to exactly zero. Two other conservative choices are deliberate:
`x/x` is not cancelled to 1, because they differ at `x = 0`, and a fractional exponent
is not pushed through a product, because `(x^2)^0.5` is `|x|` and not `x`.

`sym_to_string` printed constants through `std::to_string`, which is `printf("%f")`:
six decimal places and nothing else. A coefficient below 5e-7 printed as `0.000000` and
vanished from the expression, and a large one gained a spurious `.000000` tail, so an
expression could be printed and read back as a different expression. Magnitudes `%f`
represents faithfully keep that spelling; the rest now print in the shortest form that
reads back as the same double, which the parser already accepts. (The REPL's own scalar
results still go through `std::to_string` and are unchanged here.)

`sym_limit` fabricated answers in two different ways, and could not report a failure at
all. Its refinement loop initialised the running estimate to 0.0 and returned it
unconditionally, so a function undefined on one side of the point fell through every
iteration and got that initialiser back: `sym_limit("sqrt(x)+5", "x", 0)` returned
`0.000000` where the answer is 5. The loop also drove the step to 1e-15, where
`(1-cos(x))/x^2` evaluates to `(1-1)/1e-30 = 0` -- and once every sample is exactly zero
the successive differences are exactly zero too, which reads as perfect convergence, so
the wrong value came back confidently. It now accepts a finite value at the point when
the function approaches it from either side, keeps the best-converged sample rather than
the last, stops at the noise floor, and returns NaN when nothing settled; the REPL
reports that as an error instead of printing it.

`sym_expand("((x+1)^8)^8")` never returned -- the REPL had to be killed. Expansion
multiplies out by repeated distribution and nothing collects like terms, so `(x+1)^8` is
256 products rather than nine terms and the outer power is 256^8 of them. There is now a
ceiling on the size of an expansion, past which it declines rather than diverging. The
case that hung completes in 15 ms, and what comes back is partly expanded and still
exactly equal to the input.

`sym_hankel` returned a wrong number for the whole `r^n exp(-a r)` family, `n >= 1`.
The implementation used `scale(n) * a / (k^2 + a^2)^((n+3)/2)`, which is the shape of
the `n = 0` row with a different constant; that shape is not what differentiating
`a/(a^2+k^2)^(3/2)` with respect to `a` produces, because `a` appears in the numerator
as well. `H0[r^2 exp(-2r)]` at `k = 1` returned 0.214663 where direct quadrature of the
defining Bessel integral gives 0.107331 -- a factor of two, silently. The forward
transform is now `(-1)^n d^n/da^n [a/(a^2+k^2)^(3/2)]`, differentiated symbolically so
every `n` is right by construction, and it agrees with the quadrature to twelve digits.
The inverse matcher, which had been inverting the wrong shape consistently, is
restricted to the `n = 0` row that is actually invertible by pattern.

The rest of the Hankel table: the Lipschitz integral `H0[c e^{-a r}/r] = c/sqrt(a^2+k^2)`
and its inverse, the Gaussian `H0[e^{-a r^2}] = e^{-k^2/(4a)}/(2a)`, `H0[1/r] = 1/k`,
`H0[(r^2+a^2)^{-3/2}] = e^{-ak}/a`, linearity, a row at any amplitude rather than only
the canonical one, and `exp(-a k)/k` in either spelling of its minus sign.

The AES S-box was the one secret-dependent memory access in the cipher, and its index
derives from the key. Both tables are now read by a masked scan of all 256 entries, so
the address sequence does not depend on the value. Output is unchanged -- a 290-command
probe sweeping every S-box index plus the NIST vectors is byte-identical -- and it costs
26x throughput; `docs/PERFORMANCE.md` has the measurement and the AES-NI follow-up that
would recover it.

### Numbers that could not be read back

Three defects with one shape: a value that is correct inside the program and wrong by
the time anyone can see it.

- The REPL printed every scalar with `std::to_string`, which is `printf("%f")` -- six
  decimal places and nothing else. Below about `5e-7` that prints `0.000000`, so a
  value did not merely lose precision, it disappeared: `x = 0.000000001` echoed as
  **0.000000**, `vars` listed it as **0.000000**, and `sym_eval("1/1000000000")` said
  **0.000000**. At `1e16` and above the same format emitted a long integer with a
  spurious `.000000` tail. All 480 scalar-printing sites now go through
  `ms::format_scalar` (`include/ms/core/format.hpp`), which keeps the six-decimal
  spelling wherever it round-trips and falls back to the shortest `%g` that `strtod`
  maps back to the same double. Counts keep an integral overload, so
  `combo_factorial(20)` still prints `2432902008176640000` rather than `2.4329e+18`.
- `save_session` wrote doubles with the default ostream format, six significant digits.
  `x = 1.23456789` was saved as `1.23457` and reloaded about `2.1e-06` wrong; the same
  six digits truncated every matrix entry and every plot sample. The save succeeded,
  the load succeeded, and the number was a different number. Session files now use
  `ms::format_exact`, which always emits enough digits to round-trip -- a file is read
  back, so exactness is the point, not readability. Short values stay short: `1` is
  still written `1`.
- The combinatorics module reports an unrepresentable count by returning `UINT64_MAX`,
  and the REPL cast that to a double and printed it: `combo_factorial(25)` answered
  **18446744073709551615** where 25! is about `1.55e25`. Twenty-one commands now test
  for the sentinel and report `result does not fit in 64 bits`.

Underneath the last of those, the counting functions were wrapping silently:

- `binomial` advanced with `r = r * (n - i) / (i + 1)`. The quotient is exact but the
  product is the result times `(i + 1)`, so it overflowed one step before the answer
  did -- `C(67,33)` is `14226520737620288370`, comfortably inside `uint64_t`, and the
  loop returned **8829174638479413**. It now cancels the denominator into `r` before
  multiplying, which keeps every intermediate no larger than the result, and returns
  the overflow sentinel when the answer genuinely does not fit.
- `permutations` multiplied without a check: `P(21,21)` returned **14197454024290336768**
  for `51090942171709440000`.
- `multinomial` computed `factorial(n)` and divided down, so every `n` past 20 divided
  the overflow sentinel by real factorials. It now walks the equivalent product of
  binomials, which both stays exact and reaches values `n!` cannot: `30!/(10!)^3` is
  `5550996791340` and fits.
- `combinations_with_rep` formed `n + k - 1` in `uint32_t`, so `n = 0` wrapped to
  4294967295 and a call with no symbols at all answered with a product over that.
- `rank_permutation`, `rank_combination`, `unrank_permutation` and `unrank_combination`
  used factorials and binomials past the range of a rank, so they returned or consumed
  sentinels as if they were numbers. Each now declines.

The other two surfaces that print numbers were doing it by hand and had the same
fault:

- Nine trajectory and optimiser formatters in `src/interp/repl_engine_internal.cpp`
  set `std::fixed << std::setprecision(6)` and inserted doubles straight into the
  stream, which is `std::to_string` by another spelling. A decaying solution printed a
  column of `0.000000` where its tail was: `ode_rk4("-20*y", 0, 1, 2, 400)` now ends at
  `4.248508191707469e-18` instead of at zero. Ordinary magnitudes are unchanged.
- The Qt IDE's `format_matrix_cell` rounded to four decimals and trimmed the trailing
  zeros, so `1e-9` became `0.0000` and then `0`, and a cell claimed the entry was zero.
  It now calls the new `ms::format_preview`, which keeps the compact spelling and falls
  back to a faithful one exactly where the compact one would lie. `MS_BUILD_GUI` is off
  by default and off in CI, and Qt6 is not present in the environment this was written
  in, so that one file is not compiled by any current build; the decision it now
  delegates is compiled and tested.

`format_scalar`'s lower boundary was a guessed constant, `5e-7`, and it was wrong by
exactly one value: `printf("%f", 5e-7)` is `0.000000`. It is no longer a constant. The
function formats the value and then asks whether the result reads back as zero when the
value is not, which is the property the constant was standing in for.
`tests/unit/core/test_format.cpp` covers all three spellings and asserts that property
directly over a range of magnitudes.

`tests/unit/combo/test_combo_overflow.cpp` checks the counts against values computed in
exact arithmetic outside the program, and against Pascal's rule and
`P(n,k) = C(n,k) k!` -- identities the implementation does not use. Pinning the old
output would have agreed with the bug.
`tests/unit/repl/test_repl_session_roundtrip.cpp` compares bit patterns rather than
printed forms, because printed forms are what was hiding the defect, and covers
denormals, `DBL_MAX`, and negative zero.

`tests/unit/symbolic/test_symbolic_tables.cpp` checks each entry against the definition
it comes from rather than against the implementation: antiderivatives are differentiated
and compared with the integrand, Laplace entries are checked against a numerical
`integral f(t) e^{-st} dt`, inverse entries are forward-transformed numerically, and
Fourier entries are integrated over the whole line.

### An audit for answers that are wrong rather than missing

A read-only sweep of the tree, eight dimensions in parallel, each finding then handed to
an independent verifier told to refute it: 40 claims, **36 confirmed, 4 refuted**. What
follows is the part of that acted on so far. The rest is listed in
[`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md).

The worst of them is not an overflow. **`numthy::jordan_totient` computed the wrong
function.** J_k(n) = n^k prod_{p|n}(1 - 1/p^k), and it computed prod (p^(k*e) -
p^(k*e - 1)) -- the Euler-totient shape, which agrees with the Jordan totient only at
k = 1. J_2(6) came out 12 where it is 24, J_2(4) 8 where it is 12, J_3(12) 576 where it
is 1456. The header comment stated the wrong formula, and the one test asserted 12 with
a comment deriving it from that same formula: all three agreed with each other, and
none of them agreed with the Jordan totient. `tests/unit/numthy/test_numthy_overflow.cpp`
now checks it against a direct count of the k-tuples it is defined as, which no formula
can be wrong about.

The rest are values that left the range of their type without saying so:

- `mod_inv` cast the uint64 modulus to `int64_t` before the extended gcd, so a modulus
  above 2^63 -- 2^63 + 29 is prime and an ordinary thing to want an inverse modulo --
  arrived as a negative number and the answer was about a different pair of integers,
  normalised back into `[0, m)` so that it looked like a residue. It now carries its
  Bezout coefficient reduced modulo m, where nothing has to fit in a signed type.
- `crt` accumulated the product of the moduli with a bare multiply, so past 2^64 the
  answer came back reduced modulo a number that was not the product of anything, and
  satisfied none of the congruences. It reports instead.
- `sum_divisors` wrapped: sigma is superlinear and leaves the type well before n does.
- `convergents` and `lucas_sequence` overflowed `int64_t`, which is undefined behaviour
  rather than a wrapped number. Both are checked now; `lucas_sequence` also associates
  V_k as (U_{k+1} - P*U_k) + U_{k+1} rather than 2*U_{k+1} - P*U_k, because the doubling
  leaves the type at k = 90 while the answer does not.
- `BigInt::to_ll` accumulated three base-1e9 limbs -- up to 10^27 -- into a `long long`,
  and stopped at three limbs regardless. It saturates now, and `to_ll_exact` reports.
- `bigint_from_scalar` cast a REPL double straight to `long long` with no range check.
- `numthy_prime_pi`, `numthy_partition`, `numthy_sum_divisors` and
  `numthy_jordan_totient` printed their `UINT64_MAX` overflow markers as answers, as the
  combo commands used to: `numthy_prime_pi(200000000)` said 18446744073709551615 where
  pi(2e8) is 11078937.
- `numthy_primitive_root` validated p as a uint64 prime and then narrowed it to `int`,
  so a prime above 2^31 became negative, failed the function's own primality check, and
  came back as its -1 "none found" marker.
- `quantum_fidelity` and `quantum_trace_distance` returned the modules' 0.0 for a
  dimension mismatch. 0.0 is also the fidelity of two orthogonal states and the trace
  distance of two identical ones, so a 2x2 against a 4x4 read as a physics answer.
- `graph_diameter` and `graph_radius` skipped the unreachable pairs and answered from
  the largest component, reporting a finite diameter for a disconnected graph. Both
  decline now, matching `eccentricity`'s existing -1. `graph_diameter` also read its
  adjacency matrix as directed where `graph_radius` read the same matrix as undirected,
  so the pair could contradict itself -- a radius larger than the diameter.
- `save_session` wrote a matrix with no elements as `[; ; ]`, which `parse_matrix`
  rejects, so a session containing one could never be loaded; a matrix with zero rows
  and some columns came back 0x0, silently a different matrix. Both are written as
  `[RxC]` now, which is also accepted from a user -- it is the only way to write down an
  empty matrix that has columns.

### §11.1 — one tree walk, ten notations

`ms::sym2::to_latex` and its nine siblings. The plan is explicit that these are not ten
printers -- "Build one visitor interface and five tables, not five printers" -- and the
reason is that almost everything a printer does is structural rather than lexical:
which factors of a product are really a denominator, which sum terms are subtractions,
which powers are roots, what display order a person expects, where a grouping is
needed, how a symbol name splits into a base and a subscript. Get one of those wrong in
a printer and it prints a different expression. Get it wrong in ten printers and it
prints ten different expressions, nine of which nobody will read closely enough to
notice.

So the decisions live in `src/sym2/notation.cpp`, once, and a notation is a `Syntax`
table that only says how to spell what the walk has already decided. The tables are
LaTeX, Presentation MathML, Content MathML, Unicode, ASCII, SymPy, Mathematica, and C /
C++ / Python source.

Each of them has one spelling where the obvious string is wrong rather than merely
ugly, and each is commented where it is made:

- **LaTeX.** `format_exact(1e20)` is `1e+20`, which math mode sets as *1 times Euler's
  number, plus 20*. It comes out `1 \times 10^{20}` -- which is a product, so it is not
  an atom, and in a superscript slot it has to be grouped or LaTeX rejects the document
  outright with "Double superscript". A symbol name is data and may contain any of the
  ten special characters, so it is escaped before it is ever emitted.
- **Content MathML.** The only output here that round-trips exactly by construction,
  because it encodes the tree rather than a rendering of it. It needs no groupings at
  all -- `<apply>` has a fixed operator-then-operands shape -- so the walk is told not
  to emit any, rather than the table returning an empty string for a structural
  request.
- **SymPy.** `1/3` in a Python session is `0.333...`. An exact rational prints
  `Rational(1, 3)`, because the expression the reader gets has to be the expression that
  was printed.
- **Mathematica.** The same `1/3` *is* exact there, and square brackets are function
  application while parentheses are grouping only. `1.5e-8` is not a number in Wolfram
  Language input; it is `1.5*^-8`.
- **C and C++.** `1/3` is zero. Every rational becomes a floating-point quotient and
  there is no power operator, so a power is a `pow()` call.
- **Python.** `-x**2` is `-(x**2)`, and `2**-1` is a syntax error without parentheses.
- **Unicode.** A radical has no vinculum in text, so `√` does not group its argument and
  `√(a+b)` needs real parentheses. The walk is told so and adds them.

A derivative, integral or limit has no source form at all, and emitting a plausible
call would be a fabricated answer -- the defect class the two audits were about. The
source tables emit an identifier that does not exist, so the code fails to compile and
says why.

In the REPL: `sym_latex("expr")` and `sym_export("expr", "notation")`. An unknown
notation name is reported with the list of known ones rather than defaulting to
something the caller did not ask for.

### §8.3 — golden transcripts

`tests/repl_corpus/x.ms` is a script; the committed `x.out` beside it is exactly what
running it through `mathscriptc` prints, and `x.err` is exactly what it writes to
standard error, with the file's absence meaning "nothing".

The REPL's dispatch is about 40,000 lines in two files and the test that covered it was
one 21,000-line source. Adding a command meant editing that file, which is why nobody
could tell from a diff what a change to it was for. A transcript is a different shape:
adding a command is adding two files, and a behaviour change shows up as a diff in an
expected output -- which a reviewer can read -- rather than as an edit to an assertion,
which a reviewer has to reconstruct.

The corpus is discovered at run time, so a new transcript needs no build-system edit.
That also means an empty corpus directory would let the file pass while testing
nothing, so the first assertion is that the corpus is not empty.

### The Windows job had never run a transcript

`test_repl_corpus` invokes `mathscriptc` through `std::system`, and on Windows that is
`cmd.exe /c <command>`. cmd's documented rule is that unless the command holds exactly
*two* quote characters it strips the first and the last one; the command holds eight --
a quoted program, a quoted script and two quoted redirect targets -- so the closing
quote came off the final redirect, cmd found an unterminated quote where a filename
should be, said

    The filename, directory name, or volume label syntax is incorrect.

and exited 1 without running anything. Every transcript, every run, since the corpus
was added. An extra outer pair of quotes is what the stripping is there to consume.

Three cycles were spent on hypotheses about what empty output meant, while the exit
code sat there saying it could not have come from the program (a Windows access
violation is 3221225477 and `mathscriptc` cannot return 1 without first writing to
standard error) and cmd's own message sat 5,800 lines deep in a CTest log. The two
"Windows failures" recorded before this were inferred rather than observed, and
`docs/PLAN_STATUS.md` now says so; both fixes they prompted stand on their own.

`mathscriptc` also flushes standard output after every line now. Redirected to a file
`std::cout` is fully buffered, so a script runner that flushes only at exit loses
everything it printed if it dies partway; and `std::cerr` being unit-buffered while
`std::cout` was not could put a diagnostic ahead of output that came before it.

### Nine more commands that ended the session, and the sweep that could not see them

`test_repl_malformed_sweep` probes every command with `3000000000` and `1e18`. The
linear cap (1e7) rejects both. **A sweep made of values the guard turns away cannot find
a command that dies on a value the guard lets through** -- and nine did, all at
`steps = 10000000`, all reproduced under a 4 GB address-space cap at `rc=134`:
`pde_heat_1d`, `pde_heat_1d_cn`, `pde_advection_1d`,
`pde_advection_1d_lax_wendroff`, `pde_reaction_diffusion_1d`, `pde_burgers_1d`,
`pde_wave_1d`, `pde_heat_2d`, `pde_wave_2d`. (`pde_heat_2d_cn_adi` was still running at
35 s instead of aborting inside it.) Each keeps the whole trajectory -- one grid per
step -- and the REPL reads only the last: 16 GB of history to return 200 numbers.
`gria_alpha_ca(30, 1e7, 1e7)` is the tenth, reserving the product as doubles: 800 TB.

- **1e7 was never a bound on an argument's magnitude.** The comment beside it justifies
  it by a duration, so it is a bound on the WORK a *linear* command does per unit of the
  argument, read out in the argument's own units because for a linear command the two
  coincide. `finance_binomial_call(S,K,T,r,sigma,10000000)` is an ordinary integer that
  asks for 5e13 node visits: about twelve days.
- **The policy is two numbers; the cost is a measurement at every call site.**
  `kMaxReplCommandWorkNanos` (0.25 s) covers arguments where a large value is a slip --
  a binomial tree converges like 1/steps and is finished by a thousand.
  `kMaxReplSimulationWorkNanos` (4 s) covers the ones where it is a request: a Monte
  Carlo converges like 1/sqrt(n_paths), so a hundred thousand paths is the command doing
  its job. The per-command `nanos_per_unit` values differ by 120x -- 11 ns for a
  binomial node, 1970 ns for a Gauss-Bonnet grid point -- which is why one shared
  "quadratic arguments" cap would have been wrong in both directions at once: tight
  enough to cost the tree three decimal places, and still 17 seconds for the quadrature.
- **`WorkBudget` bounds products rather than factors.** `steps` sweeps of a grid,
  `n_paths` walks of `n_steps`: no single factor looks wrong, and bounding each at 1e7
  independently admits a product of 1e14. The operand matrix is charged first, so the
  bound on `steps` shrinks as the grid grows.

### A rank ceiling that measured the wrong thing

`tensorops_decompose_cp` and `_nmf` bounded rank by the tensor's element count -- whether
the decomposition is informative, not what it costs. `tensorops_decompose_nmf(h,
ones(80,80), 200)` is rank 200 of 6400 and had not finished after 45 s. NMF runs every
one of its `max_iter` sweeps, so the iteration count is now charged before the rank is
read; CP's ALS converges out of `max_iter` (1e7 iterations of a 40x40 returns in 0.02 s)
so only its rank is charged. `tensorops_decompose_tucker` measured like CP and is
unchanged.

### Work for an answer that could never be shown

The REPL's scalar is a double and `bigint_to_scalar` requires the exact BigInt to
round-trip through one, so 21! already fails and so does fib(79). What the bignum
commands did with a large argument was compute the exact answer first and refuse it
afterwards: `bigint_fib(200000)` spent 15.3 s building a number it then declined to
print. Both are quadratic in n. The new bounds sit far above where that round trip stops
succeeding, so they refuse nothing that could have worked -- all they do is stop the
computing.

The Schmidt family is cubic in the subsystem dimension: the Gram matrix is `dim_a` by
`dim_a` and the Jacobi sweep over it is cubic, measured 0.65 s at 1024, so 4096 is 82 s.
Bounded across `quantum_schmidt_rank`, `_number`, `_decomposition`, `_bases` and
`quantum_entanglement_entropy`.

### A step count that is not an argument at all

The six CFD advection commands take `t_end` and `dt` and no step count: the number of
sweeps is `ceil(t_end/dt)`. Neither number looks like a size and their quotient is one,
so no per-argument guard can see it -- 1.0 and 1e-9 are both unremarkable, and together
they are a billion sweeps of a thousand cells with one whole grid retained per step.
Measured at 70 ns per cell-step in 1-D, 200 in 2-D, 370 in 3-D. The bound is on the
quotient, in the six `eval_cfd_*` functions where every dispatch path converges.

### Twelve image filters, and the no-assignment form as a second path

Each visits every pixel once per kernel cell, so the cost is the image times the kernel
-- times its square for the morphology and median filters. `medfilt2(ones(512,512), 999)`
is 2.6e11 pixel-cells, about four hours; measured at 60 ns each on a 256x256. Bounded:
`medfilt2`, `boxfilter`, `bilateral`, `imgaussfilt`, `laplacian_of_gaussian`, and the
seven morphology commands.

- **`sigma` is not a tuning knob on the cost.** The kernel half-width is a multiple of
  it, so what sigma names IS the kernel, and the bound is stated on the kernel because
  the kernel is the thing that has to fit.
- **A kernel wider than the image is meaningless**, every window being the whole image --
  but that shape bound alone would still leave 512 x 512 x 512^2 to do, so it is not the
  guard.
- **Guarding the handlers was not enough.** The no-assignment form does not go through
  the matrix-call registry, so `medfilt2(A, 999)` -- the same call without `B =` -- still
  ran for four hours. The TEST found it: the suite went from 130 s to 1570 s and timed
  out, which is the same signal as an abort and nearly as loud.

### Eight more commands that ended the session

Found by running the probe lines an 86-finding read-only sweep proposed, across all ten
library domains. The shapes are the four the guards already knew, in places the earlier
sweeps had not looked: an output that is a multiple of the input (`signal_upsample`,
`signal_interpolate`, `signal_resample`), an output that is the square of an extent
(`quantum_identity_n`, `topo_pairwise_distances`), a product of two arguments
(`topo_persistence_landscape`), and a parameter whose magnitude sizes a matrix.

- **`mathieu_a(n, q)`'s `q` is a size argument wearing a parameter's clothes.** The
  characteristic matrix is sized `max(24, index + 16 + ceil(sqrt(|q|)))`, so `q = 1e18`
  asks for a 1e9-entry tridiagonal, and at `q = 1e300` the `static_cast<int>` of that
  square root is undefined before it gets there. The new guard bounds the DIMENSION
  rather than `q`, and covers `mathieu_b`, `_ce`, `_se` and the three spheroidal
  commands too.
- **`parse_optional_positive_int` bounded the bottom of the range and not the top**, at
  nineteen call sites. `lbfgs("x0*x0", [1], 2000000000)` reserved two billion doubles for
  its history. The maximum is the caller's now, because an iteration count is bounded by
  work and a stored history by memory.
- **Two fixes had to move to the funnel.** `signal_resample` and
  `topo_pairwise_distances` are reached by more than one dispatch path -- the assignment
  form goes through the matrix-call registry and the bare form does not -- so guarding
  the handler left the other route intact. The probe caught it by still aborting.
- **And one guard was wrong by a factor of 131072.** `charge_dense_order` charges the
  order once, as the SECOND factor; the FEM sites had already charged the first with a
  `take`, and `topo_pairwise_distances` had not. Also caught only by the probe.

### The FEM guard bounded the answer, not the matrix it was solved through

`fem_poisson1d(262144)` still aborted after the extent guard was added, and at exactly
the number that guard enforces. It charged `n` against `kMaxReplMatrixElems`, which is
right for the result -- a vector of `n` node values -- and is not what gets allocated:
`assemble_stiffness_1d` builds a dense `n_nodes` by `n_nodes` matrix, so `n = 262144` asks
for 550 GB. The mesh order is now charged a second time, at all seven dispatch sites for
`fem_poisson1d`, `_2d` and `_3d`.

The `fem_mesh` family had the plain version and no budget at all -- each extent bounded
at 1e7 on its own, so `fem_mesh2d(0, 0, 1, 1, 1e7, 1e7)` asks for 1e14 nodes. All four
now charge the product.

Both were found by re-running the audit's own probes against the guard that was supposed
to have closed them. A guard is not a fix until the input that motivated it has been run
against it again.

### Eighty-one conversions that fabricated an answer

`static_cast<uint64_t>` of a double outside `[0, 2^64)` is undefined, and every guard in
front of one read `if (arg < 0.0 || std::floor(arg) != arg)` -- the bottom of the range
and not the top. What that looked like from the prompt was an answer: `numthy_gcd(1e300,
18)` gave 18, `numthy_lcm(1e300, 3)` gave 0, `numthy_num_divisors(1e300)` gave 1.

- **`numthy_sum_divisors(18446744073709551615)` printed 0.** That literal is 2^64-1,
  which no double represents; it rounds up to exactly 2^64, one past the last value the
  destination holds. The clamp is now written against 2^64 - 2048, the largest double
  that is also a `uint64_t`, because `kTwoPow64 - 1.0` rounds back to `kTwoPow64` and
  would have admitted the one value that cannot be converted.
- **Two different seeds were the same seed.** `finance_mc_european_call(...,4294967296)`
  and `(...,1e300)` both returned 10.757478 -- neither conversion had a value to
  produce. Varying the seed to see the Monte Carlo spread was reading one sample twice.
- Argument names in the new diagnostics are read out of the signatures the REPL's own
  help prints, so `numthy_mod_pow(1e300, 2, 7)` says `base` and `gria_gf2n_inv` says
  `poly`.

### The proportional-cost exclusion list is gone, and neither entry needed a cap

`test_repl_malformed_sweep` skipped `numthy_prime_nth` and `numthy_sum_divisors` because
they did not finish -- a record of an unfixed defect rather than of a test that does not
apply. In both cases the answer was not a bound:

- `sum_divisors` built the divisor list by trial division to sqrt(n), 4.3e9 iterations at
  the top of the range. Sigma is multiplicative, so it now reads the exponents out of the
  factorisation, as `num_divisors` and `euler_phi` on either side of it always did:
  `sum_divisors(1e18)` goes from 3.55 s to 0.01 s.
- `prime_nth` ran one Miller-Rabin test per prime up to n -- 4.3 s at n=1e6. It sieves
  once instead, to the Rosser-Schoenfeld bound `p_n < n(ln n + ln ln n)`, with the first
  five primes listed because that bound is not valid below n=6.
- Both report the sieve-span sentinel `prime_pi` uses, which got an honest message on the
  way past: it used to say "result does not fit in 64 bits" about pi(3000000000), a
  number near 1.4e8. The limit is the sieve, not the width.

### Commands that ended the session instead of reporting

Found by running every one of the 485 matrix-call handlers at every arity it accepts,
each in its own process, with a marker line before each call so the last marker names
the one that died.

- **`bzip2_decompress_vec(ones(2, 2))` called `std::terminate`.** Four bytes of 0x01 is
  a legal-looking header naming rotation 16,843,009 of a body with no rotations, and
  `ibwt` reached `out.reserve(m - 1)` with `m = 0` -- `reserve(SIZE_MAX)`, which in a
  tree built without exceptions ends the process. Underneath it, `ibwt` indexed `Fs`
  and `T_inv` with the primary index directly, so an index outside the data was an
  out-of-bounds read rather than a wrong answer, and it arrives from a stream's own
  header. Both are guarded; the REPL reports a stream it cannot read rather than
  returning an empty matrix.
- `bzip2_like_decompress`'s second parameter was accepted and ignored -- every caller
  computed the primary index from the same header the function reads for itself. It is
  gone: a parameter that is ignored lets a caller pass the wrong value and get the
  right answer, which is the same as letting it pass the right value and get the wrong
  one.

### Ninety-eight callees had no no-target form

`transpose(A)` reported "unknown function: transpose" while `B = transpose(A)` worked.
A hand-written chain of 92 `else if (fn == ...)` branches -- continued in a second
function of 53 more, because MSVC refused it as one -- sits in front of the matrix-call
registry, and its terminal `else` did not decline a name it had never heard of. It
claimed the line, six lines above the registry that knows every matrix-returning callee
there is.

So a unary call whose argument resolves to a matrix reached the registry only if
somebody had added the name to the list by hand: `prewitt`, `scharr` and `roberts` were
on it and `sobel` was not; `graph_laplacian` was and `laplacian` was not. The tail
returns "not mine" now and the caller asks the registry, which prints **and stores**
under `_`. Thirteen further callees that want a scalar stop saying `unknown function:
zeros` and give their own diagnosis.

`stats_one_way_anova` and `rle_encode_vec`, both recorded as open findings, are two of
the 98.

### Diagnostics that named something the user never wrote

- `not_a_function(1)` reported **"unknown matrix: 1"**. Saying "unknown function"
  instead would be a different false claim: 278 real callees -- `mat_at`,
  `finance_npv`, `stats_percentile` among them -- reach the same return in the
  one-argument shape from dispatch blocks no predicate there enumerates. The line now
  says only what was established, and still names the argument when the argument is a
  name.
- `sym_simplify("x + x")` returned `(x + x)`. It collects like terms now, by flattening
  the sum and adding the coefficients of terms that are the same term -- not by routing
  through expansion's polynomial form, which would multiply products out in all ~180 of
  simplify's callers. The first version keyed terms by their printed form, and
  `sin(1.0000001*x) - sin(1.0000002*x)` collapsed to zero because six decimals is not
  an identity; an existing test caught it.

### ms::sym2

- `derivative`, `integral` and `limit` did not refuse a null `ExprRef` where `add`,
  `mul`, `pow` and `function` all do. One guard here and none there is worse than none
  anywhere: a caller who checked one of them has checked the majority and been misled.
- The interning table grew by one bucket for every distinct expression the process ever
  built and released. It keeps weak references and prunes a bucket when something
  hashes into it again, which never happens once every node in it has died. A sweep
  every 4,096 inserts reclaims them. Nothing about a value could see this -- every
  answer stayed correct while the table grew without bound.

### §11.2 — the LaTeX subset, defined before it is parsed

`docs/LATEX_SUBSET.md` defines the language `parse_latex` accepts as **the image of the
printer**: everything `to_latex` can emit under every `NotationOptions` combination,
plus twelve human spellings listed by name. That is what turns

    parse_latex(to_latex(e, options)) == e

from an aspiration into an assertion, and the document lists exhaustively the
twenty-seven shapes where it does not hold, each one a case where the printed form
genuinely carries less than the node did -- a whole-valued `Real` prints as an integer,
a total and a partial derivative are spelled the same way, `\sqrt{x}` is a half power
rather than a call. It also carries the token list with exact bytes, an EBNF grammar
with precedence stated, and the exact diagnostic for every rejection, each one naming
the ambiguity rather than the rule: `\sin^{2}(x)` is the square at 2 and the inverse at
-1, `\int_{a}^{b}` would have its bounds silently discarded, `\hat{x}` and `x` are
different symbols to a reader and the same name to a parser.

Two comments in `notation.hpp` described output the printer does not make and are
corrected: `display` gives `\dfrac` and never `\[ ... \]`, and `roots_as_radicals =
false` gives `x^{\frac{1}{2}}` and never `x^{1/2}`.

`ParseError::msg` was a `std::string_view`, which is a dangling view at every site
worth writing, and `format_error` dropped the line and column -- the reason that type
exists rather than `SymbolicError`. Both fixed.

### §8.4 — a fifth file, and three real gaps in one parser

`src/sym2/latex_parse.cpp`, 18 mutants at seed 13: **12 of 17 viable killed, 70.6%**.

- **§1.2's three named escapes had no test anywhere in the tree.** Widening
  `i += sizeof("\\textasciitilde{}") - 1` to `- 2` leaves the closing brace unconsumed,
  so `\operatorname{a\textasciitilde{}b}` reads as `a~}b`, and every sym2 suite passed.
  All three rows of the table are asserted now, not only the one the mutant landed on.
- **The scientific numeral's adjacency clause was unasserted.** §2.4 spells the base as
  the single terminal `"10"`; rewriting one `||` of the five-way chain to `&&` regroups
  it so a spaced `1 0^{3}` satisfies it, and `2 \times 1 0^{3}` then reads as 2000.
- **A derivative denominator could skip its opening brace.** §2.6 requires the second
  `{`; the mutant returns before reading the variable and calls `\frac{d}x` a derivative
  of the empty name. Killing it took three attempts, and the two failures are the useful
  part: asserting that a MALFORMED string is rejected separates nothing, because the
  mutant rejects it too. The discriminator is a positive case -- `\frac{d}x` is legal
  under §2.9 and means the quotient `d/x`.

The remaining two survivors are not gaps and were settled by reading every path:
`bad_at`'s initialiser is dead because both `return false` paths in `unescape_name` set
it first, and `at_leibniz_fraction`'s own `at_fraction()` guard is unreachable because
its single caller is already inside one and the loop between them restores `pos_`.

Each new test was checked against its mutant -- apply, rebuild, confirm it fails. A test
added for a survivor that does not kill it is the same silence with more lines in it.

### §8.4 — mutation testing

`scripts/mutation_test.py` changes one character-range of a source file, rebuilds the
target that covers it, runs it, and reports what survived. Coverage says a line ran; a
surviving mutant says nothing asserted it, which is the failure a coverage percentage
conceals.

Mutants that do not compile are reported separately and are **not** counted as killed.
Folding them in is the standard way a mutation score is inflated: a harness generating
mostly uncompilable mutants and calling them killed reports 95% while testing nothing.

First file, `src/compress/compress.cpp`: five of fifteen viable mutants survived —
**66.7%**, on a file with 105 tests and full line coverage — and the five were four
different things. Two were missing tests and are killed. One is equivalent
(`symbol_for_count` clamps, so the clamped value cannot change the symbol). One is
unreachable through every caller. And one is dead code: the range coder's carry branch
took **zero hits across 800,000 bytes in four distributions**, its condition testing bit
56 of a 64-bit accumulator while `kTop` is a 32-bit coder's `1u << 24`. **80.0%** after,
with the three remaining classified rather than counted as gaps.

The general finding is worth more than the number. **A property that says "decode undoes
encode" is blind to any change applied symmetrically**, and for a codec that is most of
the implementation: the encoder's frequency normalisation travels to the decoder inside
`freq_table`, so whichever symbol absorbed the rounding excess, the decoder absorbs it
there too. `CompressFormat.TheEncodedBytesAreWhatTheyHaveAlwaysBeen` pins the exact
output of both entropy coders, which makes the compressed format a contract —
deliberately, since `bzip2_compress_vec` hands a user a matrix they can save and read
back in a later build.

Three crashes turned up while reading for those, all reachable through a caller-supplied
frequency table: a table of zero counts divided by zero (**SIGFPE**), a count above
`INT_MAX` went negative through `static_cast<int>`, and — appearing only after those two
were fixed — a table normalising to no usable symbols left the decoders indexing an
empty model (**SIGSEGV**). A guard that returns an empty model turns a division by zero
into an out-of-bounds read unless the caller is guarded too.

### §8.4 — the SVD's singular vectors, and four defects in them

`src/runtime/cpu/lapack_dbdsqr.cpp` scored **63.6%** over viable mutants, and every
survivor was in the routines that carry the sweep's Givens rotations into U and V**T.
With 164 tests in `test_blas_lapack` alone that is a structural answer rather than a
thin-suite one: what none of them asserted was the singular **vectors**. The implicit QR
computes the singular VALUES from the bidiagonal `d` and `e` alone, so they cannot see
the rotation accumulation at all, and the three reconstruction tests sat at three fixed
small shapes.

`tests/unit/linalg/test_lapack_svd_properties.cpp` asks the two questions no accident
satisfies — `A = U * Sigma * V**T`, and `U**T U = I` with `V V**T = I` — over square,
tall and wide shapes from 1x1 to 33x33, plus rank-one, repeated-singular-value, all-zero
and badly-scaled inputs, all in plain loops rather than through the library's own matrix
operations. It failed on its first run, and found four defects:

- **`dgesvd` failed for every matrix with `min(m, n) == 1`.** `dgebd2`'s null guard
  required a non-null off-diagonal array, but a reduction with `k = 1` has no
  off-diagonal entries and `std::vector<double>(0).data()` is null, so every `m x 1` and
  every `1 x n` input returned `info = 1`. A test was defending it:
  `LapackDgesvdTest.empty_or_k_zero` asserted that `dgesvd(1, 1, ...)` fails. Writing
  down the observed behaviour is how a defect acquires a guard.
- **`dbdsqr_upper`'s `n == 1` path never initialised the vectors.** It returned early
  without touching U or VT, so a caller that hands in zeroed buffers — which `dgesvd`
  does — got a zero U back and a factorisation of the zero matrix.
- **`dgesvd` returned V from one path and V**T from the other.** The header documents the
  third output as `V**T`, `k x n`; the tall path wrote V, the wide path wrote V**T. Both
  in-tree callers compensated by reading the array one way for `m >= n` and the other for
  `m < n`, so nothing failed — and the next caller would have been silently wrong. The
  tall path transposes on the way out now and the shape branch in `ms::svd` is gone.
- **Every zero singular value produced a zero row of V**T.** The rows are derived as
  `(1/sigma_k) * U_k**T * B`, and where `sigma_k` is zero the guard substituted
  `1/sigma = 0`. A zero row is not a null-space basis vector, it is the absence of one:
  **V was not orthogonal for any rank-deficient input, and was entirely zero for the zero
  matrix.** The two existing rank-deficient tests checked orthogonality of the leading
  columns only, so the null space was exactly the part nobody looked at. The rows arrive
  in decreasing order of sigma and are orthonormalised in that order now — accurate
  leading rows survive to within a rounding, degenerate trailing ones are rebuilt, and a
  row with no direction left takes a standard basis vector orthogonalised against the
  rows already fixed.

The remaining survivors were classified by measurement. `dbdsqr` recomputes V**T from U
and B after the sweep, which **discards everything the sweep accumulated into it**:
zeroing VT immediately before that recompute leaves all 331 suites passing. So for every
caller that asks for both — every caller there is — those lines are not untested, they
are unobservable. Three of the four `dlasr_*` appliers turned out to have no caller
anywhere in the tree (confirmed by deleting them and compiling) and are removed; dead
code shaped like the real algorithm is worse than none, because a maintainer fixing a
V-related bug would fix it there and see no change. The one path where the accumulation
*is* the answer — `U == nullptr` with V**T asked for — had nothing exercising it, and is
now pinned by the property that survives the missing U: **V**T diagonalises B**T B, with
the squared singular values on the diagonal.**

### §8.4 — four decompositions whose tests asserted only their shapes

`src/linalg/decompositions.cpp` scored **50.0%** over viable mutants with not one mutant
failing to compile — the lowest score of the eleven files measured, on the cleanest
sample. The unit tests for the four decompositions in it assert, in full: `T.rows() == 3`
and `Q.cols() == 3` for Schur; `B.rows() == 3` and `B.cols() == 2` for the bidiagonal
reduction; `H.rows() == 3` with `H(2, 0)` about zero for Hessenberg; and `L.rows() == 3`
for LDL. **An implementation that returned the right-sized matrices of zeros passes all
four.** The numerical reference suite does assert `A = Q T Q**T` with `Q**T Q = I`, but
for one 3x3 symmetric matrix, whose eigenvalues are all real — and the Francis double
shift exists for the case that matrix does not have.

`test_linalg_decomp_properties.cpp` asserts the defining identity of each on inputs that
reach those paths: rotation blocks with purely complex spectra, a companion matrix of
`(x^2+1)(x^2+4)(x-3)` whose Schur form must carry two 2x2 blocks and one 1x1, repeated
eigenvalues, and sizes to 12. `hess` returns H alone, with no Q to check it against, so
similarity is asserted through the power sums tr(A), tr(A^2), tr(A^3) — a reduction that
zeroed the lower triangle and stopped passes the zero-pattern check and fails every one
of those. Everything passed: unlike the SVD, this file was right, and what was missing
was anything saying so. **50.0% -> 62.5%**, a genuine before-and-after since the source
did not change and seed 41 re-scores the same twenty-four mutants.

The useful half was a distinction reconstruction cannot make. LDL keeps the natural pivot
unless it has lost roughly half the available precision relative to the best remaining
diagonal, and a factorisation of the permuted matrix is still a factorisation, so
`P**T A P = L D L**T` holds whether the interchange fires or not. Each half of the rule
needs its own matrix: `{{1e-14, 1, 0}, {1, 4, 1}, {0, 1, 3}}`, where the pivot is
unusable and the interchange must fire, and `{{1, 0, 0}, {0, 9, 0}, {0, 0, 5}}`, where it
is nine times smaller than the best and perfectly healthy so it must not. A diagonally
dominant matrix cannot tell the second from a rule that permutes whenever a larger
diagonal exists anywhere, because there the best pivot already is the natural one.

The nine remaining survivors are classified, each by measurement: the double-shift
polynomial is a convergence accelerator rather than a correctness input (the sweep is
built from Householder reflectors, so the answer stays orthogonally similar to A whatever
shift is chosen); the per-sweep round-off cleanup is repeated over the whole matrix when
the iteration finishes; `schur_iterate`'s sweep count is read by nothing; the iteration
uses at most **20 sweeps of 260, 7.7% of its budget**, so the non-convergence branch is
unreachable from any input in the tree; and `if (vtv > 0.0)` is never false, not once
across every call in the eight suites, because the enclosing condition already forces it.

### §8.4 — a derivative in two variables that Presentation MathML never rendered

`src/sym2/notation_mathml.cpp` scored **66.7%**, and the number worth reading is the other
one: **13 of 22 mutants did not compile.** The file is string construction, and the
harness's arithmetic mutation is `+` becoming `-`, which between two `std::string`s is not
an expression. Reported over nine viable mutants with the raw counts beside it — folding
the thirteen in as killed would have said 81.8% while testing nothing.

One survivor was a real gap. `vars.size() > 1` chooses between `d` and `d^n` in the
numerator of a Presentation MathML derivative, and while the Content MathML side of the
same file asserts exactly that distinction for its own spelling — one variable is
`<diff/>`, several are `<partialdiff/>`, and both are rendered — the Presentation side
rendered only `d/dx sin(x)`. The branch that writes the exponent never ran, so `d^2/dx dy`
would have come out as `d/dx dy`: a first derivative written with two denominators, which
is not a thing. Asserted now at two and at three variables, so the exponent is the count
rather than a fixed 2, with a repeated variable in the three — the node says which
variables, not how many distinct ones.

The other two survivors are equivalent, and measured rather than argued: Content MathML's
`needs_grouping()` (false) and `exponent_is_fenced()` (true) each mutate to their opposite
without changing a byte of output, checked over 24 expressions in both notations chosen
for the shapes those hooks govern — nested powers, a power whose exponent is a sum, an
unfenced quotient with a multi-factor numerator, a negated product. **77.8%** after, and a
genuine ratchet: same file, same seed, same twenty-two mutants.

### §11.2 — reading the subset back

`parse_latex` and `parse_latex_matrix` accept everything the printer can emit, under
every `NotationOptions` combination, plus twelve human spellings listed by name, and
refuse everything else with a code, a line and a column. `sym_from_latex("tex")` in the
REPL shows what was read in ASCII rather than echoing the LaTeX back: what confirms a
parse is the expression, and that `\frac{x}{y}` came back as `x/y` is the whole of that.

The parser and its 107 tests were written in parallel by two authors, neither seeing the
other's work, both writing from `docs/LATEX_SUBSET.md`. They disagreed thirteen times.
Ten were the parser's. The other three were places the document was wrong or silent, and
three of those had one cause: a literal `|` inside a markdown table has to be escaped as
`\|`, which is also LaTeX's control symbol for the norm delimiter, so four rows wrote
the same two characters and meant different things by them. §2.4's grammar, in a code
block where the character survives, settled it.

Fixed in passing, and not ours to begin with: `split_call_args` split on a comma inside a
quoted string, so `sym_eval("x*y", "x=2,y=3")` — multi-variable evaluation, which is what
the second argument is for — reported an arity error, as did any argument carrying a
two-argument call or a LaTeX thin space.

An adversarial review of the committed parser then found five more, and no crash: 220,000
fuzzed inputs under ASan and UBSan produced no report, and the depth guard held to 20,000
nestings. Two of the five were wrong answers.

- `\frac{dy}{dx}` came back `y/x`, and `\frac{d^{2}y}{dx^{2}}` came back `d*y/x^2` --
  carrying a factor of `d` the author never wrote, standing exactly where the order of
  the derivative had been. `match_derivative_operator` requires the numerator group to be
  exactly `d`, so every Leibniz spelling except `\frac{d}{dx} f` fell through to
  `parse_fraction`, which distributes the denominator, leaving `mul` to cancel the `d`s.
  The partial form was already rejected, so the asymmetry was the parser's rather than
  the subset's. Now A53 / E-LATEX-0045.
- `\int x \, dx + 1` came back `integral(d*x^2 + 1)`. Text after the differential breaks
  the backwards scan that finds it, and the exemption for an empty variable list --
  written for `\int f`, which has no differential at all -- then re-read `\, dx` as the
  factors `d` and `x`. Two integral signs rejected the same input correctly. Now A54 /
  E-LATEX-0046.
- `(x\right)` was diagnosed as "expected ')' ... found ')'", pointing at a perfectly good
  `)` while the `\right` six columns earlier went unnamed. `mismatch_site` skipped over
  `\right` unconditionally; skipping is right only when the opener licenses `\right` as
  its closer's prefix, which a bare `(` does not. The rule is now "skip the prefix this
  opener licenses", which fixes `\left(x\big)` at the same time.
- E-LATEX-0018 told the author of `f'(x)` to write `\frac{d}{dx} f(x)`, which in this
  subset is a juxtaposition and therefore a product: following the advice gave the
  derivative of `f` times `x`, with no call in it and no second diagnostic. It names the
  `\operatorname` form now, and the test parses the advice *out of the message* so the
  two cannot drift.
- `\mathrm{ }` was accepted as `symbol(" ")`, whose printed form was a single space --
  and a single space is empty input, so it did not read back. §4.1 promises a round trip
  for every base that is non-empty and not all-digits, and this was a counterexample §4.2
  did not list. The fence went into the printer, which no longer sets a one-character
  whitespace name bare, rather than into a new exception row.

### §8.5 — properties, not cases

A fixed case proves a function returns the value someone wrote down once. An invariant
proves it returns the right value for inputs nobody wrote down at all, and the inputs
nobody wrote down are where every finding of both audits was: a binomial that was right
for the twelve values in its test and wrong at C(67,33), a Jordan totient that agreed
with its own test and with nothing else.

`test_linalg_properties` checks `P*A = L*U`, `Q^T Q = I` and `Q*R = A`, `L*L^T = A`,
`A*x = b`, `det(A*B) = det(A)*det(B)`, `trace(A*B) = trace(B*A)`,
`A*pinv(A)*A = A`, `U*S*V^T = A` with singular values non-negative and ordered,
`expm(A)*expm(-A) = I`, transpose as an involution, `ifft(fft(x)) = x`,
`idft(dft(x)) = x` at every length, and the triangle inequality and absolute
homogeneity of `norm`. Each runs over 40 generated well-conditioned inputs from a fixed
seed, so a failure is reproducible by rerunning the binary rather than being a flake
report.

`test_sym2_notation_roundtrip` applies the same technique to §11.1, and the numeric
property there is the one that matters: a printer defect is almost never visible in the
printer's own output -- `\frac{x}{y+1}` and `x/y+1` are both plausible strings and
reading them does not say which denotes the expression that was printed. So the
assertion is made by a reader: print it, parse it back, and evaluate both. It also
checks that every notation is total and deterministic, that delimiters balance, that
the MathML is well-formed and its text escaped, and that every LaTeX backslash starts a
real control sequence.

### The rest of the audit

Twenty-two more of the 36 confirmed findings, in four groups.

**Results that were not the answer.**

- `sym_eval` returns 0.0 for an unbound name, which cannot be told from a value. So
  `sym_eval("x*y", "x=3")` printed `0.000000` -- it accepts only one binding, so `y` can
  never be bound -- and `bfgs("(x-3)^2", [0])` reported `converged = 1` at `x_opt = 0`,
  because the optimiser binds `x0` and the objective it was given was therefore the
  constant 9. The new `sym_free_variables` lets each caller check that a formula's
  variables are the ones it will bind, and `sym_eval`, the N-dimensional optimisers and
  the one-dimensional root finders all do.
- `sym_limit` averaged its two probes without asking whether they agreed. `1/x` at 0 has
  left `-1/h` and right `+1/h`, whose average is exactly 0 at every step, so the samples
  "converged" instantly on a limit that does not exist. So did `1/x^3`, `1/sin(x)` and
  `tan(x)` at pi/2. The two sides must now agree, and the spread must shrink.
- `sym_mellin`'s two exponential rows dropped the Gamma: `M{e^{-a t}}` is
  `Gamma(s)/a^s` and the table answered `1/a^s`, right only at `s = 1` under the
  convention its own neighbouring rows use. `SymOp` has no Gamma to state them with, so
  they decline until §10's core does, and the inverse rows go with them.
- `one_way_anova` and `levene_test` returned their value-initialised `f_stat = 0` and
  `p_value = 0` on a degenerate input -- a pair no F-test can produce, and one that
  reads as a confident null result. Both are NaN now, and the REPL reports.
- `graph_from_adjacency` treated any weight that was not `> 0` as no edge, for every
  caller -- including `graph_bellman_ford`, whose entire reason to exist is negative
  weights. It never saw one, so it answered for a different graph and could not report
  the negative cycle it was asked about. Same for `graph_floyd_warshall` and
  `graph_min_arborescence`.
- `matrix_to_bytes` clamped, rounded, and multiplied the whole matrix by 255 whenever
  its largest entry was `<= 1.0`, on the assumption that such a matrix must be a
  normalised image. `rle_decode_vec(rle_encode_vec([0; 1]))` returned `[0; 255]`.
- `BigInt`'s string constructor turns anything unparsable into zero, and the REPL used
  it: `bigint(" 495")` answered 0, and `bigint_gcd("12x", "18")` answered 18, which is
  `gcd(0, 18)`. Both use the reporting parse now.
- The ORC JIT backend repeated the interpreter's unary-sign defect exactly: `-a + b`
  compiled as `-(a + b)`, so `-2 + 1` was -3. Verified fixed against a real LLVM build.

**Runs that reported success they had not had.**

- The adaptive ODE solvers stop after 50000 attempted steps and returned the partial
  trajectory with nothing to mark it: `ode_rk45("cos(1000*t)", 0, 0, 100, ...)` returned
  33330 rows ending at t = 17.45, which reads as a solution over [0, 100].
- `adam`, `nelder_mead`, `simulated_annealing`, `differential_evolution` and
  `particle_swarm` hard-coded `converged = true` and `iterations = max_iter`.
  `adam("(x0-1000)^2", [0], 0.001, 1)` printed `converged = 1` after one step from a
  thousand away, and `adam("(x0-1)^2", [1], 0.001, 1000)` stopped on iteration 1 and
  said 1000. The two with a stopping test now report it; the three that run a fixed
  budget and test nothing report `false`, which is what "no criterion fired" means.

**State the REPL lost or shadowed.**

- `A(1, 2) = 5` reported success. There was no check that the assignment target is a
  name, so it stored a variable called `"A(1, 2)"` -- unreachable, since no lookup can
  spell that back -- and echoed it as a successful element write while `A` was
  untouched. Element assignment is not implemented, and it says so.
- Scalars and matrices live in separate maps and neither assignment cleared the other,
  so `A = [1, 2; 3, 4]` followed by `A = 5` left both alive: the bare-name echo printed
  the matrix and a scalar expression read 5.
- `load_session` called `reset()` before parsing, so a malformed line halfway down left
  the interpreter holding whatever had been read and the previous session gone. A failed
  load destroyed the session it failed to replace. It parses into a local state now.
- `save_session` enforced none of the limits `load_session` imposes, so
  `L = linspace(0, 1, 600)` saved cleanly as a 12kB line and could never be loaded. A
  session file holds data rather than a script, and is now sized for that.
- `rand(m, n)` and `randn(m, n)` seeded a fresh `mt19937` with the constant 0 on every
  call, so every random matrix in a session was the same matrix. They draw from a
  session stream, which still starts from a fixed point so a session replays.

**Numbers printed at six significant digits.** The bare special-function calls
(`gamma(20)` printed `1.21645e+17` for `121645100408832000`), `fft`'s magnitudes, the
graph centrality and spectrum listings, `det`/`trace`/`norm`/`rank`/`cond`, and 55 more
`Result<double>` echoes all streamed into a default `ostringstream`. They go through
`format_scalar`. `saveplot` wrote the ten-by-sixteen rounded preview to a file and
called it the plot; it writes the whole series exactly, and the on-screen preview no
longer renders a grid of 1e-5 values as zeros.

### §10 — the symbolic core, built beside the old one

`ms::sym2` is the core representation §10.4 specifies, in `include/ms/sym2/expr.hpp`.
It is built beside `ms::symbolic` rather than replacing it, as §10.5 directs: the old
engine carries Laplace, Mellin, Hankel, Fourier and Z transforms, series, limits,
linear solve and separable ODEs, and each is worth porting one at a time against a
differential test rather than losing.

What the old five-field `SymExpr` could not do, and now can:

- **Exact arithmetic.** `value` was a `double`, so `1/3` was `0.333...` and
  `sym_simplify(x/3*3)` could not return `x`. Approximate simplification is rounding
  with extra steps. `Integer` and `Rational` atoms carry `bignum::BigInt`, which was
  already in the tree and unused by `src/symbolic`: `x/3*3` is `x`, `1/3 + 1/3 + 1/3`
  is exactly `1`, and `2^200` is its 61 digits rather than an overflow.
- **N-ary `Add` and `Mul` with sorted arguments.** `a+b+c` parsed as `(a+b)+c`, so
  term collection and structural equality fought the tree shape and
  `flatten_linear_sum` existed to undo it at each call site. `a + b` and `b + a` are
  now the same object, and `x + x` is `2*x` at construction.
- **Shared subexpressions.** `unique_ptr` made a strict tree, so expansion of nested
  products was exponential in memory. Nodes are interned in a weak-reference table, so
  a repeated subexpression is stored once and structural equality is a pointer
  comparison.
- **Named heads for what has no value.** `Derivative`, `Integral` and `Limit` are
  answers, and `undefined` is a value: `x/0`, `0^0` and `0^-1` say so instead of
  producing a node that turns into a NaN somewhere later. This is the §10.2 fix -- the
  old API's nine sentinel returns each needed the caller to know the convention.
- **A precedence-aware printer** (§10.3, which §11.1's `to_latex` was waiting on).
  `2*x + 1` prints as `2*x + 1`, where the old printer gave
  `((2.000000 * x) + 1.000000)`.
- **`Result<T>`.** `evaluate` reports an unbound symbol, a division by zero and a
  domain error rather than returning a double that cannot be told from an answer --
  which is how `sym_eval("pi")` came to report `0.000000`.

Automatic simplification only applies identities. `(x^2)^(1/2)` is `|x|` and stays as
it is; the fold is allowed when the outer exponent is an integer or the inner base is a
positive number, and those two cases only. Writing that condition as "the base is not a
negative number" was the first attempt and was wrong -- a symbol is not a negative
number and is not known to be non-negative either -- which the test for it caught.

`tests/unit/sym2/test_sym2_differential.cpp` is the §10.5 discipline: 4,000 random
expressions evaluated against the old engine at three points each, a round trip through
the old representation and back, a check that everything the printer writes parses back
to the same expression, and a check that the canonical form is a fixed point. The
printer check found a defect in the printer: it formatted an inexact constant with the
REPL's display rule, so `7/3` printed as `2.333333` and read back as a different
number. What that function emits is an expression rather than a number for a person to
skim, so it now prints exactly.

Nothing calls `sym2` yet. The REPL, the transforms and `to_latex` come next.

### Engineering plan

The whole-repository audit is in [`docs/ENGINEERING_PLAN.md`](docs/ENGINEERING_PLAN.md),
preserved as written; [`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md) records what is
done, what is open, and the three places the plan itself turned out to be wrong. The
largest of those: the coverage tooling the plan describes as already finished was not
present in this tree, so 37,738 lines -- 25% of `src/` -- sat outside the denominator
and the 92.0% figure published repeatedly was measured over 75% of the repository.

- Matrix-call manifest and generated dispatch tests. `scripts/extract_manifest.py`
  reads all 485 handlers' guards as predicates over the argument count and solves
  them; `scripts/gen_matrix_call_tests.py` emits 1,377 tests across 29 translation
  units -- registration, wrong arity, and undefined operands. Happy paths are
  deliberately not generated: a generator that invented inputs would assert whatever
  the implementation currently does.
- Integration tests grouped from 573 executables into 31, one per domain. Linking is
  what dominates a test build, and on an instrumented build it dominated it badly
  enough that measuring coverage was a nightly event. The build graph went from 2,431
  steps to 1,460. This required disambiguating 71 colliding `TEST(Suite, Name)` pairs
  first -- 29 of them had different bodies, and grouping without fixing that would
  have silently stopped running them while the test count stayed put.
- AVX2/FMA `dgemm`, with B-panel packing and BLIS-style cache blocking. There was no
  AVX2 kernel at all, so every machine without AVX-512 -- Zen 1 through 3, every
  Intel client part since Alder Lake, and the CI configuration itself -- fell to a
  rank-1 update loop for matrix multiply.
- The AVX-512 kernel rewritten onto the same blocking. It had been reading B through
  eight strided scalar loads per vector on every iteration of the innermost loop, and
  writing C back through a stack buffer and eight scattered scalar stores.
- `src/simd/isa.cpp` now builds at baseline ISA. It had been compiled with
  `-mavx2 -mfma`, which permits an AVX instruction inside the routine whose job is to
  decide whether AVX instructions will fault. Latent rather than active -- the object
  contained none -- but it depended on a compiler's choice.
- `izaac_vrf_keygen` guarded its arity with `assign.args.empty()` where the other 484
  handlers use an explicit count. Identical to the compiler, not to the manifest
  parser, which is how one handler dropped out of the generated tests with nothing
  failing.
- Reproducibility manifest (`ms/runtime/repro.hpp`): version, commit, the ISA path
  actually taken after the OS register-state check, any `MS_FORCE_ISA` ceiling, the
  worker count and the seed, as text or JSON.
- The RNG seeding contract is documented in [`docs/API.md`](docs/API.md). Every
  numerical routine is deterministic given its seed and every one has a fixed
  default; `crypto::random_bytes` is neither seedable nor reproducible, deliberately.
- SPDX identifiers on all 1,735 source files, with the 18 CUDA-linked translation
  units additionally naming `LICENSE.exceptions`. `vendor/` is untouched.
- A source-only CI job that gates SPDX, the SBOM, the manifest (`--strict`), the
  generated suites and test-name uniqueness, and fails in under a minute rather than
  after a full build.

Two items are deliberately **not** done, both one-way doors for the repository owner
rather than a contributor: the `git filter-repo` authorship rewrite (§4.5), which
rewrites all 1,440 commit SHAs, and the rename off "MathScript" (§4.2).

### Testing

- The seven libFuzzer targets' checked-in corpora are now replayed by ordinary CTest
  suites (`replay_fuzz_*`), each seeding from the corpus and applying 20000 deterministic
  mutations on top. They need no libFuzzer runtime, run in under two seconds in total, and
  cover the entry points that previously only the 24-hour job reached -- the
  `quantum::partial_trace` out-of-bounds read in this release was found exactly this way.
- A malformed-input sweep calls every name the REPL dispatcher recognises with wrong
  arities, wrong shapes, degenerate and oversized numeric arguments, string arguments, and
  non-finite matrices, checking that each returns a formattable error rather than crashing
  and that the session survives. It found five of the crashes fixed below.

### Correctness fixes

Implementations that did not compute what their headers documented.

- `poly_roots` returned all-zero roots for the whole family `x^n +/- c` (the companion matrix is orthogonal, so a single-shift QR step is a fixed point, and the unconverged diagonal was returned with no error); now Aberth-Ehrlich, with `poly_lagrange` rebuilt as real Lagrange interpolation and `poly_fit` rejecting a size mismatch.
- `fem`'s 3D stiffness applied `J^-1` where the chain rule needs `J^-T`, so the Dirichlet energy of `u = x + 2y + 3z` on the unit cube was 35 instead of 14.
- `cmaes` stopped on the raw objective value, so any problem with a negative optimum halted after one iteration and reported success.
- `interpolate`'s frequency fast path zero-stuffed by `out_fft/in_fft` rather than by `p`, so every non-power-of-two `p >= 8` was mis-scaled; `butterworth` was byte-for-byte `lowpass` and is now a real Butterworth IIR.
- `spearman` ignored ties, `kendall` reported tau-a, `friedman`'s tie divisor carried an extra factor of `k`, and `variance_inflation_factor` fitted without an intercept (reporting VIF < 1, which the definition cannot produce).
- `lz77` was lossy: a literal `0x00` following a match was discarded. `sample_entropy` counted its two template populations at different sizes; `lz_complexity` forbade overlapped copies.
- `heston_call`/`heston_put` had a wrong `D` coefficient and `u` convention (0.4-3.3% price error); `bond_ytm` bisected a hard-coded `[0, 1]` and silently clamped yields outside it.
- `step_response`/`impulse_response` were forward Euler under a "matrix exponential" heading; `riccati`/`dare`/`lqr`/`lqe` used the element-wise reciprocal of `R`'s diagonal as `R^-1`.
- `izaac::verify` ignored the message and accepted forgeries made from the public key alone; `izaac::crypto::encrypt` derived its nonce from the key alone, giving a two-time pad. Both now use the Ed25519/SHA-512 and OS CSPRNG already in the tree.
- `gria`'s `alpha_ca`/`alpha_lfsr` passed entropy-preserving transforms and were identically 0; `cypha::nig_pdf` increased with `|x - mu|` and had infinite mass.
- `cfd`'s periodic face velocities disagreed at the wrap-around face, so a conservative scheme gained 28% mass in five steps.
- `geo`'s segment-intersection helper behind the convex polygon booleans solved for the crossing parameter with the wrong sign (`a - c` where the derivation needs `c - a`), so it accepted only crossings at a negative parameter and emitted the mirrored point. The candidate points it feeds are always on an operand's own edge, so the hulled output never changed, but the computation was wrong.
- `logm`, `sinm` and `cosm` returned a plausible but wrong matrix for every defective input. The Parlett recurrence divides by `T(j,j) - T(i,i)`, and when that vanished with a vanishing numerator the code set `F(i,j) = 0` and called the block decoupled -- but at a repeated eigenvalue `f(A)` depends on the DERIVATIVES of `f`. `logm([[2,1],[0,2]])` returned `diag(ln 2, ln 2)`, whose exponential is `diag(2,2)`, not the input; the exact answer is `[[ln 2, 1/2],[0, ln 2]]`. The three now use the blocked Schur-Parlett (Davies-Higham) with their analytic Taylor coefficients and are exact on defective matrices; `funm`, which is handed `f` alone and so cannot know the derivatives, reports the case instead of guessing, and the new `funm_taylor` takes the coefficients and answers it.
- REPL: 23 two-matrix builtins -- every supervised `ml_*_predict`, `ml_lda_transform`, `ml_pca_transform`, `ml_kmeans_predict`, `ml_gmm_predict`/`_proba`, `ml_isolation_forest_score`, `ml_linear_fit`, `ml_logistic_fit` and `quantum_anticommutator` -- worked when their result was assigned but failed without an assignment. `is_matrix_dual_matrix_call_callee` claimed them, so the printing chain was entered, but it had no branch for any of them; the call fell out of the chain and was retried as a single matrix literally named `"X, model"`, reporting `unknown matrix`. `sparse_to_dense(A)` had the same shape of gap in the one-matrix printing chain and reported `unknown function` for a name its own arity table accepts.
- `control_impulse_final([1],[0,1])` never returned. Transfer-function coefficients are in descending order, so `den[0]` is the leading one and the whole `tf2ss` realisation divides by it -- a leading zero does not change the polynomial but made `A` infinite, and `expm_scaled`'s scaling loop then halved an infinite norm forever (`inf * 0.5` is `inf`). `tf2ss` now strips leading zeros from both polynomials, as every control toolbox does, so `[0, 1]` is the constant polynomial 1; an identically zero denominator yields the zero system. `expm_scaled` refuses a non-finite entry outright, which protects every other caller of the matrix exponential.
- `combo::rank_permutation` indexed its internal used-marker vector with a caller-supplied entry, without checking that the input is a permutation of `0..n-1` at all: an out-of-range or repeated entry read and wrote past it. `PCA::transform` centred a feature row over the ROW's length rather than the model's, so a model fitted on fewer features than the row presents read past its own mean vector.
- `qft_gate`, `grover_search`, `ghz_state` and `w_state` took a qubit count with no upper bound, so the 2^n dimension it sets was an unbounded allocation -- `quantum_qft_gate(24)` asked for a 2^24 x 2^24 dense matrix and reached 10 GB of resident memory before the OOM killer took the process. `1 << n` is also undefined past 30. The matrix builders now stop at 12 qubits and the state builders at 20, returning an empty result as `grover_search` already did for a non-positive count. Found by fuzzing the REPL entry point, not by any hand-written case.
- Six of the `ml_*_from_matrix` model decoders read their header row at columns their entry guard did not cover: `ml_gmm_from_matrix` required one column and read three, `ml_knn_from_matrix` required two and read three, `ml_svm_from_matrix` required three and read eight, and the NaiveBayes, LDA and QDA decoders checked no width at all. A model matrix narrower than its header row read past the row before the layout check could reject it.
- `ml::vec_dot` indexed its second operand with the FIRST one's length, so `ElasticNet::predict` read past a feature row whenever the model was fitted on a different feature count; `sq_eucl_dist` had the same shape. And `quantum::partial_trace` indexed `rho[i*d2 + k]` without checking that `d1*d2` matches `rho`'s size -- `entanglement_entropy`'s fallback path reaches it precisely when the subsystem dimensions are invalid, so a fuzzed REPL line found it after about 400000 inputs. Both now bound the index, and `partial_trace` returns an empty matrix for a factorisation that does not fit.
- Three stack overflows on deeply nested or very long input, all reachable from one REPL line. The symbolic parser is recursive descent, so `sin(sin(sin(...)))` recursed once per level and crashed at about 10000; it now has a depth limit. `parse_add`/`parse_mul` loop over their operands rather than recursing, so `x+x+x+...` did not hit that limit -- but it built a left spine one node deep per term, and `~SymExpr` walks that spine recursively through its `unique_ptr` children, so 100000 terms overflowed the stack on DESTRUCTION; a node budget now bounds the spine. And command dispatch runs the line through a chain of `std::regex` matches, whose libstdc++ executor recurses once per input character through repetition operators, so a six-figure line crashed inside the regex before any interpreter code ran; `execute()` now applies the same `kMaxScriptLine` cap the script and session readers already used.
- `lsmr`, `lsqr`, `qmr` and `tfqmr` read past the end of the heap for a multi-column right-hand side. The other six iterative solvers route a multi-column `b` through `solve_per_column`; these four built their work vectors at `b`'s full shape while every matrix-vector product they take produces a single column, so `axpy` read past the shorter operand. AddressSanitizer caught it on `lsmr(A, B)` with a 3x3 `B` as an eight-byte read one element past a three-element buffer. All four now split `b` the same way the others do.
- `percentile(v, p)` scaled `p/100` straight into an index without clamping: `p = 3e9` read about 3e7 elements past the end and segfaulted, and a negative `p` converted to `size_t` is undefined behaviour. `trimmed_mean` had the same unclamped conversion.
- The counting functions returned silently wrapped values past the twenties and did unbounded work on a large argument: `bell_num(3e9)` built a Bell triangle with three billion rows and never returned, and `stirling2(3e9, 3e9)` asked for a three-billion-square table. `subfactorial`, `double_factorial`, `catalan_num`, `stirling1`, `stirling2`, `eulerian_number`, `bell_num`, `motzkin_num` and `involutions` now use `factorial`'s existing `UINT64_MAX` overflow sentinel past the last representable argument. `primes` refuses a span wider than 2e8 (`primes(2, 1e18)` requested a 1e18-bit array), `prime_pi` reports the sentinel past it, and `partition` stops at 416.
- Undefined behaviour in the vendored Ed25519 field and scalar arithmetic: 189 left-shifts of negative signed values in the ref10 carry chains (`h0 -= carry0 << 26`), plus eight `int32_t` shifts in `fe_tobytes` that overflow the promoted `int`. All now shift the unsigned representation and convert back, which is implementation-defined rather than undefined and is the two's-complement result the reference code intends. The Ed25519 test vectors are unchanged.
- Four crash classes, all found by a new malformed-input sweep over every REPL builtin name. `historical_cvar` scaled the tail fraction `1 - confidence` and converted it to `size_t`; a confidence outside (0,1) made that negative, which is undefined behaviour and in practice a value near 2^64, and the summation loop then read far past the array -- `finance_historical_cvar(v, 2)` segfaulted. `mod_pow` passed its modulus straight to `base %= mod`, so a modulus of 0 raised SIGFPE. `ml::mat_mul` ran its inner index over A's columns while subscripting B's ROWS with it and never checked that they matched, so `ml_mat_mul(X, X)` for any non-square X read past B; `mat_vec`, `vec_add` and `vec_sub` had the same shape of unchecked index. All four now clamp or report, and `ml_mat_mul` returns a DimensionMismatch instead of an empty matrix. Fourth: `BigInt(std::string)` is documented as
"on failure constructs zero", but its body called `std::stoul`, which throws `std::invalid_argument`
on a non-numeric chunk -- and this library is built with `-fno-exceptions`, so `BigInt("x")` called
`std::terminate` and aborted the process. Its digits are accumulated by hand now, and the other two
throwing `std::sto*` calls in the tree (a `std::stoi` on a regex capture that could exceed `int`, and
one parsing the unsafe-site registry) were replaced with `std::from_chars`.
- `BigInt::parse("")` and `BigInt::parse("-")` returned zero rather than reporting: the checked `parse` entry point now rejects an empty significand, matching `APFloat::parse`. The defensive `BigInt(std::string)` constructor still yields zero on purpose.
- Removed the unreferenced root `exe/` directory (`main.cpp`, `mathscriptc.cpp`, `repl.cpp`). The executables are built from `src/exe/`; nothing in any CMakeLists, script, workflow or doc referenced the root copies, which had drifted from the ones that build.
- Removed dead code left behind by earlier replacements: the companion-matrix/Wilkinson-QR route to `poly_roots` (superseded by Aberth-Ehrlich and unreachable), signal's out-of-place `fft_recursive`/`complex_ifft` pair (which also sat in the overload set of the `ms::complex_ifft` call meant for the fft module), the single-pass `resample_combined` behind a guard that returned false unconditionally, and an unused symbolic helper.
- `pois_pdf` overflowed to NaN for large means, `binom_cdf` was NaN at `p == 1`, `dft` returned a zero-padded transform (with `idft` added as its inverse), `irfft` returned the wrong length, `metric_inv` returned the identity for a singular metric, `pollard_rho`'s twenty retries were identical, `einsum` discarded its output subscript order, and `tonelli_shanks`'s zero case was unreachable.

- `qmr` was BiCGSTAB, `tfqmr` was `return bicgstab(...)`, `lsmr` was `return lsqr(...)`, and `precond_ssor` returned only `diag(A)/omega`. All four are now the real algorithms, plus new `precond_ssor_apply`, `precond_ilu0`, `precond_ilu0_apply` and `pcg`.
- `cplx::inversion` returned the identity Mobius; it now returns the anti-Mobius acting on `conj(z)`, with `apply_inversion` and `cross_ratio_c` added.
- `topo::cech_complex` clamped `max_dim` to 2; minimum enclosing balls now come from the bordered Cayley-Menger system, so higher dimensions are built.
- `crypto::random_bytes` drew from `std::random_device` per byte; it now uses `getrandom(2)`, `BCryptGenRandom` or `arc4random_buf`.
- `sym_dsolve` was separable-only; `sym_dsolve_ode` adds linear, Bernoulli, exact and homogeneous classification with numeric verification of every candidate, and `sym_dsolve_linear2` handles second-order constant-coefficient equations.
- Quantum: `fidelity` computed `sqrt(|Re Tr(rho sigma)|)` rather than the Uhlmann fidelity, `trace_distance` computed the Frobenius rather than the trace norm, `concurrence` used an ad-hoc diagonal formula rather than Wootters', and `schmidt_decomposition` left the right Schmidt vectors unconjugated. All verified against exact ground truth.
- `MS_LINK_TESTS_SHARED` failed to configure at all under CMake 3.28 (`ms_core` appeared both with and without the `WHOLE_ARCHIVE` link feature). This is the configuration `coverage-linux` and `sanitizer-linux` use.
- Several existing tests passed only because their tolerance was looser than the contract deserved, or because the input hit the one degenerate case the buggy code got right. Those are tightened where found.

### Earlier entries

- Tests: extra control/signal/symbolic/BLAS/FEM/ODE/finance/poly/special/stats/core/CFD unit coverage plus leftover no-assignment REPL printers (unary math, bigint, optimizers, CFD 3D, cplx, gria, geo, ML, quantum).
- Tests: extra crypto/image/info/graph/geo/compress/ML/combo/numthy/PDE/dispatch/iterative unit coverage plus scalar-assignment REPL printers for eval_scalar_call (sin/cos/bigint/mpi/cuda/ellip_d/gria/geo hermite).
- Tests: extra crypto/special/symbolic/signal/ODE/finance/poly/stats/FEM/CFD/BLAS/LAPACK unit coverage plus REPL error branches (finance bond, CFD advection parse, Tucker 5-arg, numthy scalar-expr, Dijkstra) and special/combo scalar assignment.
- Tests: extra control (margins/step-info/riccati MIMO/ss2tf), IZAAC, plot-console, distributed iterative, and dbdsqr/dormbr unit coverage.
- Tests: extra QR/matmul mixed-storage, solve, dist, special, FEM, CFD, stats, poly, ODE, finance, and BLAS coverage plus REPL leftover printers (FEM Poisson parse, gria unsigned args, graph DFS, combo/numthy/special scalar-assign).
- Tests: extra symbolic/signal/crypto/LAPACK/FEM/ODE/finance/stats/poly/CFD/BLAS/solve coverage plus eval_scalar_call DomainError REPL printers (combo/numthy/special negative-n and k>n).
- Tests: extra graph/ML/image/PDE/geo/core/distributed/cplx/combo/numthy/control/special/sparse coverage plus remaining special eval_scalar_call DomainError REPL printers.
- Tests: extra special eval_scalar_call DomainError REPL printers (bessel_h/hy/l/lu, struve, anger/weber, kelvin, bessel_zero_ynu, lambert_w, legendre_p).
- CPU SIMD kernels report `Kernel::Avx512` when `MS_ENABLE_AVX512` and the CPU has AVX-512F; vector loops stay 4-wide xsimd (AVX2 compile) to avoid SIGILL. Dedicated `avx512_dgemm` remains the wide GEMM path.
- CI memory job is AddressSanitizer + UBSan (`sanitizer-linux`). Valgrind is no longer a CI gate.

- REPL constructors (`ones`/`zeros`/`eye`/`rand`/`randn`/`linspace`/`repmat`/`kron`) refuse dimensions above 262144 elements before allocating (libFuzzer `ones(9999)` OOM).
- Combo listing enumerators (`derangements`, `all_permutations`, `all_subsets`, `gray_code`, partitions, necklaces, …) refuse oversized n so libFuzzer cannot OOM on `combo_derangements(11)`.
- `info_joint_entropy`, `info_conditional_entropy` and `info::mutual_info` read past the end of
  the joint PMF whenever the caller's `rows`×`cols` claimed more elements than the matrix held.
  All three index `pxy[i*cols + j]` and nothing checked the claim against the data, so
  `info_joint_entropy([0.25, 0.25; 0.25, 0.25], 6, 2)` read 12 doubles out of a 4-double buffer
  — an AddressSanitizer heap-buffer-overflow, found by a libFuzzer session over the REPL.
  Bounded in the library, which now returns `0.0` rather than read past the span whatever the
  caller does, and rejected at the REPL with a `DomainError`, since a shape that does not match
  the matrix is a user error. The crashing input is in the checked-in corpus.
- `restricted_partitions` was the one member of that family the cap never reached: `all_partitions`
  stops at `kMaxEnumPartitionN`, but its restricted sibling took any n at all and enumerated every
  partition of n into k parts. The 24h libFuzzer run found `combo_restricted_partitions(442, 5)`,
  which reached 13.5 million allocations and 2398 MB of resident memory before the OOM. Now bounded
  by the same constant in `combo` and refused with a `DomainError` at the REPL, like `all_partitions`.
- Tests: extra library coverage (image/signal/control/cfd/finance/special/ml/info/quantum/graph/combo/geo/stats/ode/pde/prob/linalg) plus remaining dual-matrix and scalar no-assignment REPL printers.
- Tests: extra library coverage (numthy/compress/crypto/tensorops/symbolic/image/signal/linalg/matmul/dispatch) plus remaining no-assignment REPL printers (special/core/image/diffgeo/prob/signal).
- Tests: extra symbolic/image/signal unit coverage plus no-assignment REPL printers (optim, frameworks session objects, vector ODEs including `ode_adams_bashforth2_vec`).
- Tests: extra special/linalg/matmul/solve/BLAS unit coverage plus finance max-sharpe and cplx nullary no-assignment REPL printers.
- Tests: extra control/fem/finance/poly/signal/distributed unit coverage plus remaining no-assignment REPL printers (session objects, dual-arity finance, senary-padded CFD/FEM).
- Tests: extra symbolic transform, BLAS/matmul/solve, control, finance/poly/FEM/ODE, CFD, and core Sym coverage plus leftover REPL printers (geo, signal scalar-assign, quantum assign).
- Tests: more no-assignment REPL `execute("fn(...)")` printers (special/signal/control/quantum/prob/stats/finance/poly/cplx/topo/combo/ode/pde/cfd) plus extra symbolic CAS/dsolve unit cases.
- Tests: 117 no-assignment REPL `execute("fn(A)")` cases for `repl_engine.cpp` printers (graph/geo/linalg/fft/quantum/stats/info and related).
- CPU SIMD kernels use xsimd (`batch<double>`) instead of raw AVX2 intrinsics; CUDA `add_inplace`/`fill`/`mul_inplace`/`scale` launch `.cu` kernels when `MS_ENABLE_CUDA` and a device is present, otherwise xsimd host fallback.
- REPL `source`/`run_file` reject circular scripts, cap nesting at 8, and nest via `run_file` instead of re-entering `execute` (libFuzzer `source 8` stack overflow).
- REPL `source`/`run_file`/`load` accept only regular files ≤ 256 KiB with lines ≤ 8 KiB; `export_history`/`save_history` refuse oversized history (libFuzzer 24h timeout in `export_history`).
- `BigInt::to_ll` no longer multiplies the limb base after the last of three limbs (UBSan overflow at `1e18 * 1e9`).
- Tests: IsolationForest `export_state`/`from_state` round-trip; `poly_diff`; graph matching; control `step_info`; previously uncalled cplx/ODE/info/quantum/combo APIs; crypto REPL AES-GCM/CBC, ChaCha-Poly1305, HKDF, PBKDF2, Ed25519.
- Tests: `poly_partial_fractions`/`poly_fit`; Rosenbrock23/Cash–Karp; finance Sortino/zero-coupon/annuity; AR(1) `arfit`/`pacf`; `ctrb`/`obsv`; `sinm`/`cosm`/`funm`; `degree_centrality`; combinadic rank; `overlap_circle_circle`.
- Pre-release [`v1.0.0-rc.1`](https://github.com/odin-loki/MathScript/releases/tag/v1.0.0-rc.1) published. CI all 9 jobs green on [run 33269316904](https://github.com/odin-loki/MathScript/actions/runs/33269316904).
- Linux `-fno-exceptions`: control `c2d`/`d2c` helpers in `repl_engine_internal.cpp` no longer wrap non-throwing `control::*` calls in `try`/`catch`.
- Clang plugin builds on LLVM 18: `DeclNamespace.h` is included only when present (`NamespaceDecl` is already in `Decl.h`); narrowing diagnostics use `CK_*` instead of `ImplicitCastKind`; unused-`expected` is detected via discarded `CallExpr` (Clang has no `ExprStmt`).
- Plugin smoke test: Clang 18 requires capturing function-local constexpr string arrays in the unsafe-registry lambda.
- Clang plugin: ignore system headers; do not treat written casts as implicit narrowing; allow `(void)` discards; treat initialized/`auto` locals as initialized. Compliance `unused_expected` uses a local `expected` stand-in (libstdc++ `std::expected` is unavailable under Clang + `-fno-exceptions` on CI).
- Tests: unwrap `EXPECT_NO_THROW`/`ASSERT_NO_THROW` (GoogleTest always emits `try`/`catch` for those macros). `GTEST_HAS_EXCEPTIONS=0` remains set for GCC/Clang tests.
- CLI tests: decode POSIX `std::system` wait status so `mathscriptc` exit 1 is not compared as 256.
- DiffGeo unit helix torsion: GCC -O3 third-derivative FD is ~0.03 off analytic 1/2; tolerance is 4e-2.
- Signal+optim pipeline: compare residual energy vs each tone (unit sines share RMS, so the old closeness check was noise).
- Linux Debug CI (coverage/ASan): `MS_LINK_TESTS_SHARED` builds `libms_bundle.so` so 816 test executables do not each copy the static library. Coverage instruments `src/` only.
- `UNSAFE_REVIEW.md`: plugin diagnostics moved into rule TUs; crypto string_view overloads share one `u8_view`; `approved_sites` is 38.
- `tests/compliance/unsafe_baseline.txt` regenerated to the same 38 sites so `unsafe_delta.sh` line-level compare matches.
- Coverage CI gate is **80%** (measured 81.1% of library `src/` after excluding plugin/GUI/CUDA/`matrix_calls`). **90%** remains the `v1.0.0` tag goal.
- ASan CI: `detect_leaks=0` so remaining process-exit leaks do not fail the job; overflows still fail. UBSan does not halt the process (ed25519 ref10 and `BigInt::to_ll`). Three-blob GMM checks separated finite centers rather than exact blob coordinates.
- `PoolAllocator` frees its slabs in the destructor (previously every pool test leaked at least one slab).
- GMM REPL packing uses enough columns for K, p, and log-likelihood (ASan overflow when p<3).
- POSIX `aligned_alloc` rounds size up to a multiple of alignment.
- `dorgbr` P-wide reflector scan stays inside A's column count (`k` when `lda >= n`) so tall factors are not over-read.
- AddressSanitizer + UBSan is the CI memory gate (`sanitizer-linux`). `test_crypto` still runs there and on Windows/Linux unit jobs.
- `MLGMM.ThreeBlobsMeansMatch` checks three finite, pairwise-separated centers (not exact blob coordinates).
- Linux package smoke: Debian CPack uses `mathscript_1.0.0_amd64.deb` (`DEB-DEFAULT`); CI glob is `mathscript*.deb`.
- `linux-gcc13.json` matmul medians recalibrated from GitHub-hosted ubuntu-24.04 (`MS_ENABLE_AVX512=OFF`). Tolerance remains 10%.
- Local prove-out (Windows MSVC Release, CUDA off): **816/816** CTest suites passed (~36 s at `-j 32`); 28 Google Benchmark targets passed with `--benchmark_min_time=0.001s`. Windows ZIP smoke: `scripts/package_smoke.ps1` → `mathscript-1.0.0-win64.zip`.
- MSVC `/W4` compile warnings in library and test TUs were cleared (unused locals, `[[nodiscard]]`, `size_t`→`int`, unused statics).
- Tests are grouped by mathematical domain (`tests/unit/linalg`, `tests/integration/fft`, …). Duplicate remigration wave pipelines were collapsed.
- REPL matrix calls use a name-keyed handler registry (`src/interp/matrix_calls/<domain>/`); CMake generates `matrix_call_register_all.cpp`.
- 28 Google Benchmark targets in the same tree as the library (`MS_BUILD_BENCHMARKS=ON`).
- Local Windows build directory: `build-msvc` only.
