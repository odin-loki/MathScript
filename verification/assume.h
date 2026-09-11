/* Portable assumption across bounded model checkers.
 *
 * __CPROVER_assume is CBMC's spelling. ESBMC does not recognise it and treats it
 * as an ordinary undefined call, so the assumption is silently DROPPED and the
 * harness then verifies something weaker than it appears to. That is how a
 * miller_rabin_witness.c run passed under CBMC and reported an overflow under
 * ESBMC: the input was unconstrained there.
 *
 * __VERIFIER_assume is the SV-COMP spelling and both tools implement it, so it
 * is the one that means the same thing to each. */
#pragma once

#if defined(__CPROVER__) || defined(__CPROVER_H_INCLUDED)
#define MS_ASSUME(cond) __CPROVER_assume(cond)
#else
void __VERIFIER_assume(int);
#define MS_ASSUME(cond) __VERIFIER_assume(cond)
#endif
