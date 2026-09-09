/* The witness-range arithmetic that §6.2 of the engineering plan found broken.
 *
 * The old code was:
 *     long long nll = nm1.to_ll();          // truncates a wide BigInt
 *     if (nll <= 2) nll = 3;
 *     long long u = 2 + (long long)(rng() % (std::abs(nll) - 2));
 *
 * Three of the four defects are in that expression alone, and each is a property
 * a model checker can settle for every input rather than for the ones a test
 * happens to pick. */
#include <assert.h>
#include <limits.h>
#include <stdint.h>

long long nondet_ll(void);
unsigned long long nondet_ull(void);

/* --- the old expression, reproduced --- */
static long long old_witness(long long nll_in, unsigned long long r) {
    long long nll = nll_in;
    if (nll <= 2) nll = 3;
    long long a = nll < 0 ? -nll : nll;     /* std::abs */
    return 2 + (long long)(r % (unsigned long long)(a - 2));
}

int main(void) {
    long long nll = nondet_ll();
    unsigned long long r = nondet_ull();

#ifdef CHECK_OLD
    /* Defect 2: std::abs(LLONG_MIN) is undefined -- the negation overflows. */
    __CPROVER_assume(nll == LLONG_MIN);
    long long w = old_witness(nll, r);
    (void)w;
#else
    /* Defect 3: when |nll| == 2 the modulus is zero. to_ll() truncation makes
     * this reachable from a wide BigInt, not only from a literal 2. */
    __CPROVER_assume(nll == 2 || nll == -2);
    long long a = nll < 0 ? -nll : nll;
    if (nll <= 2) { a = 3; }               /* the guard as written */
    /* The guard only rewrites nll when it is <= 2, so nll == -2 reaches abs()
     * as -2 -> 2 and the modulus below is (2 - 2) == 0. */
    long long modulus = (nll <= 2 ? 3 : a) - 2;
    assert(modulus != 0);
#endif
    return 0;
}
