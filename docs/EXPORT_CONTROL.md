# Export control position

**Status: undetermined. This document prepares the determination; it does not make one.**

Nothing here is legal advice, and none of it has been reviewed by a lawyer or by
Defence Export Controls. It exists because §4.3 of the engineering plan recorded
that the repository carried no written position at all, which is worse than an
unfavourable one: a reviewer who asks the question currently finds nothing.
What follows is the factual inventory a determination needs, and what is missing
before one can be made.

## Why this applies

MathScript is developed by an Australian entity and the stated strategy includes
supplying Five Eyes agencies. Two regimes are engaged:

- **Defence Trade Controls Act 2012 (Cth)** and the **Defence and Strategic Goods
  List (DSGL)**, Part 2 (Dual-Use), Category 5, Part 2 — "Information Security".
- **Wassenaar Arrangement** Category 5 Part 2, which the DSGL implements, and its
  equivalents wherever the software is received.

The trigger is not the mathematics. It is that the tree implements cryptography
capable of protecting confidentiality, which is what Category 5 Part 2 covers.

## What is actually in the tree

Established by reading `src/crypto/` and `include/ms/crypto/`, not from the
documentation.

| Primitive | Parameters | Purpose in this tree |
|---|---|---|
| AES | 128 and 256-bit keys | Block cipher |
| AES-GCM | 128 and 256-bit keys | Authenticated encryption, constant-time tag comparison |
| SHA-256, SHA-512 | — | Hashing |
| HMAC | over SHA-256 / SHA-512 | Message authentication |
| X25519 | Curve25519 | Key agreement |
| Ed25519 | Curve25519 | Digital signature |

Key lengths matter to the classification: symmetric algorithms above 56 bits and
asymmetric key agreement or signature based on elliptic curves above 112 bits are
the thresholds Category 5 Part 2 uses. **This implementation is above both.**

Also relevant to any determination:

- The cryptography is a **module of a computer algebra system**, not the product's
  purpose. Whether that engages the "ancillary cryptography" carve-out is exactly
  the kind of question a determination has to answer rather than assume.
- `mathscript-server` can be operated over a network, so the software can be
  *supplied* rather than only *exported* as a file. Under the DTCA, an intangible
  supply to a foreign person can itself be a controlled activity.
- Everything is **original work in this repository**, not a wrapper over a
  third-party library. The "publicly available" and "open source" notes below
  turn on that.

## The exemptions that plausibly apply

Each of these is a real path, and each needs to be confirmed rather than assumed.

**Publicly available / open source.** The DSGL and Wassenaar both carve out
software "in the public domain" or generally available to the public. This tree
is published under AGPL-3.0-or-later, which is a strong fit for that language.
The interaction that needs checking is the **dual-licence model**: the same code
is also offered commercially under other terms (`COMMERCIAL-LICENCE.md`). A
determination should address whether offering a proprietary licence alongside the
public one disturbs reliance on the public-availability exemption.

**Ancillary cryptography.** Where cryptography is not the primary function and
supports something else, the classification can differ. The argument here is that
MathScript is a CAS and `src/crypto/` is one module among forty. The counter is
that the module implements full primitives with a public API rather than, say,
verifying an update signature.

**Basic scientific research and published algorithms.** All the primitives are
published standards implemented from published specifications.

## What has to happen before distribution

1. **Get a written determination.** Either a self-assessment documented and kept,
   or an application to Defence Export Controls for a formal classification.
   Their assessment service is free and its output is the artefact a procurement
   review wants to see.
2. **Record the outcome here**, with the date, who made it, and the DSGL entry
   considered.
3. **Decide the packaging consequence.** If the outcome is unfavourable, the
   options are a build without `src/crypto/` (`MS_ENABLE_CRYPTO=OFF` does not yet
   exist and would need adding), a permit, or restricting supply.
4. **Re-check on change.** Adding a primitive, raising a key length, or changing
   who the software is supplied to can change the answer.

## Interim position

Until step 1 is complete, treat the crypto module as **potentially controlled**.
The practical consequence is that the repository stays public under AGPL — which
is the strongest available support for the public-availability exemption — and
that no targeted supply is made to a foreign government or entity without the
determination in hand.

## Not covered here

US EAR and ECCN 5D002, and EU Regulation 2021/821, if the software is ever
re-exported from or supplied into those jurisdictions. Same inventory above,
different regime, separate determination.
