# Formal verification

Bounded model checking of the parts of this tree where the property is about
*every* input rather than the inputs a test happens to pick — bit-level gating,
integer overflow, division by zero.

## Running it

```bash
bash verification/run.sh          # every harness
cbmc verification/isa_gating.c --unwind 2 --bounds-check --conversion-check
```

## Tool

**ESBMC 8.5.0** and **CBMC 5.95.1**. Every harness is run under both, and both
must succeed.

Two tools rather than one because they disagree usefully. ESBMC caught a defect
in the harnesses themselves that CBMC could not: `__CPROVER_assume` is CBMC's
spelling, and ESBMC treats it as an ordinary undefined call, so the assumption is
silently DROPPED rather than rejected. `miller_rabin_witness.c` therefore passed
under CBMC and reported an arithmetic overflow under ESBMC -- correctly, because
under ESBMC the input was unconstrained. The harnesses now use
`__VERIFIER_assume` via `MS_ASSUME` in `assume.h`, which both tools implement
with the same meaning.

That is worth stating plainly: for a while these harnesses were verifying
something weaker than they appeared to, and only a second tool exposed it.

Installing them:

```bash
apt-get install cbmc
curl -fsSL -o esbmc.zip \
  https://github.com/esbmc/esbmc/releases/download/v8.5/esbmc-linux.zip
unzip -q esbmc.zip && install -m755 release/bin/esbmc /usr/local/bin/esbmc
```

The asset is `esbmc-linux.zip`, not a tarball, and `esbmc-linux-armv8.zip` next to
it is the ARM build -- selecting on the substring "linux" alone picks the wrong
one and fails with `Exec format error`, which is exactly what the first CI run of
`.github/workflows/verify.yml` did.

## What is checked, and what was found

### `isa_gating.c` — the SIGILL fix (plan §6.1)

Extracts the OSXSAVE/XGETBV gate from `src/simd/isa.cpp` and enumerates every
assignment of the six CPUID bits, the OSXSAVE bit and all 2^64 values of XCR0.
Ten properties, all **SUCCESS**:

- no wide path is reported unless the OS saves the register state it uses
  (`XCR0 & 0x6` for AVX/AVX2/FMA, `XCR0 & 0xE6` for AVX-512);
- AVX-512 implies the YMM state, since `0xE6` has `0x6` as a subset — this is
  what keeps the reported set internally consistent;
- nothing is reported that the CPU does not have;
- with OSXSAVE clear, nothing wide is reported at all.

That last one is the property that matters: it is the case where `XGETBV` cannot
even be executed, and it is exactly the configuration — hypervisor, sandbox,
`noxsave` — that produced the crash.

### `miller_rabin_witness.c` — a correction to the plan

Plan §6.2 lists four defects in the old witness expression. Model checking says
**two of the four are not reachable**, and the reason is the order of the
statements:

```cpp
long long nll = nm1.to_ll();
if (nll <= 2) nll = 3;                              // <-- runs first
long long u = 2 + (rng() % (std::abs(nll) - 2));
```

The guard rewrites *every* value `<= 2` — including `LLONG_MIN` and every
negative — to 3 before `std::abs` sees it. So:

- **"`std::abs(nll)` on `LLONG_MIN` is undefined"** — not reachable. CBMC reports
  no signed-overflow on the negation even when the input is pinned to
  `LLONG_MIN`.
- **"if `std::abs(nll) == 2` the expression is `% 0`"** — not reachable. After the
  guard the smallest value reaching `abs` is 3, so the modulus is at least 1.
  CBMC proves `modulus != 0`.

The plan was produced by static inspection and says so (§15.5: *"every claim
about runtime behaviour should be confirmed on real hardware before it is acted
on"*). This is one of those claims, and confirming it is what the tooling is for.

**The other two defects are real**, and they are the ones that made the function
wrong rather than merely ugly:

- `to_ll()` keeps only the low three base-1e9 digits, so for any `n` wider than
  that the witness range is derived from a truncated number unrelated to `n` —
  exactly the inputs an arbitrary-precision primality test exists to serve.
- The generator was seeded with the constant 42, so the witness set was fixed and
  public, and composites that pass it are constructible by anyone reading the
  source.

A third, not in the plan: `rounds = 0` skipped the loop entirely and returned
`true` for every odd number that survived the initial checks.

All three are fixed in `bigint_is_prime`, which now decides `n < 3.317e24`
outright with the first twelve primes as bases, and above that adds
`random_device`-seeded witnesses drawn from the full `[2, n-2]` range using
BigInt arithmetic rather than a truncated `long long`.
`tests/unit/bignum/test_bignum_primality.cpp` covers Carmichael numbers, the
strong pseudoprimes to each initial base set, Mersenne primes past `long long`,
and the `rounds = 0` case.

## Scope and honesty about it

Bounded model checking proves properties of the extracted C fragment, under the
stated unwind bound — not of the C++ in `src/`. Where a harness reproduces logic
by hand, it can drift from the original; each one names the file and function it
mirrors so the pairing is checkable. This is a real technique with real limits,
not a claim that the repository is verified.
