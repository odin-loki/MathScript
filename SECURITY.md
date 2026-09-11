# Security policy

## Reporting a vulnerability

Report privately to **odin.loch@outlook.com.au** with `SECURITY` in the subject.

Please include the affected version or commit, the configuration (which
`MS_ENABLE_*` / `MS_BUILD_*` options were on), and a reproducer if you have one.
A crashing input for one of the fuzz targets under `tests/fuzz/` is ideal.

Do not open a public issue for a vulnerability before it is fixed.

**Expect:** acknowledgement within 5 working days, an assessment within 15, and
agreement with you on a disclosure date before anything is published. Credit is
given unless you ask otherwise.

## Scope

In scope: the library, the REPL, `mathscriptc`, and `mathscript-server`.

`mathscript-server` is the sharpest edge of that surface. It reads commands from
standard input, so anything that pipes untrusted text into it is exposing the
interpreter to untrusted input.

Out of scope: the GUI (`MS_BUILD_GUI`, off by default and not shipped), the Clang
compliance plugin (a build-time developer tool), and anything requiring an
attacker who already has local code execution as the same user.

## What this project does and does not claim

**Not hardened against side channels.** `src/crypto/` implements AES with
data-dependent table lookups for the S-box, `xtime` and `MixColumns`. That is a
cache-timing side channel and it leaks key material to a co-resident attacker.
GCM tag comparison *is* constant-time, and the primitives are tested against the
published FIPS-197, FIPS-180 and RFC 4231 vectors, so the arithmetic is right --
but the implementation is appropriate for a computer algebra system, not for
protecting keys against an adversary sharing your hardware. Do not use it as a
general-purpose cryptographic library. A constant-time AES-NI path with a
bitsliced fallback is planned; until it lands, this limitation is a documented
property rather than a bug.

**Randomness is OS-backed.** `crypto::random_bytes` draws from `getrandom(2)` on
Linux, `BCryptGenRandom` on Windows and `arc4random_buf` on BSD, and fails
loudly rather than falling back to a non-cryptographic source.

**Untrusted input is fuzzed.** Seven libFuzzer targets cover the REPL parser, the
symbolic parser, matrix and polynomial operations, bignum and the MPI message
decoder. Their corpora are replayed on every build by the `replay_fuzz_*` CTest
suites, so a regression is caught even where no libFuzzer runtime is installed.
A 24-hour marathon runs separately (`scripts/fuzz_24h_local.sh`, or
`.github/workflows/fuzz-24h.yml`).

**Memory safety.** The tree is built and tested under AddressSanitizer and
UBSan in CI. It compiles with `-fno-exceptions` and contains no `throw`
statements; errors propagate as `Result<T>`. A Clang plugin enforces the
prohibitions listed in `UNSAFE_REVIEW.md`, and the reviewed unsafe surface is
33 sites, gated against a checked-in baseline.

## Export control

The crypto module implements AES-128/256, AES-GCM, SHA-256/512, HMAC, X25519 and
Ed25519. See `docs/EXPORT_CONTROL.md` for the position on Australian DSGL Part 2
Category 5 and the Defence Trade Controls Act before redistributing.
