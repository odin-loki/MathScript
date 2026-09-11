# What the REPL prints, at the boundaries where it used to print something else.
#
# Every scalar went through std::to_string, which is printf("%f"): six decimal places
# and nothing else. Below about 5e-7 that prints 0.000000, so a small value did not
# lose precision, it disappeared.
tiny = 0.000000001
tiny
small = 0.0000005
small
big = 1e16
big
huge = 1.7976931348623157e308
huge
negzero = -0.0
negzero
combo_factorial(20)
combo_factorial(25)
combo_binomial(67, 33)
combo_catalan(34)
# NOT gamma(20). A transcendental printed at full round-trip precision pins the last
# bit of whatever libm the host has, and libms are not required to agree there: glibc
# and MSVC differ by an ulp, which changes the 17th significant digit and so the string.
# That is a spelling the C library chose, not a promise MathScript makes, and a golden
# transcript is the wrong instrument for it. The property that gamma(20) prints all of
# 19! rather than 1.21645e+17 is asserted with a tolerance in test_repl_audit_fixes.
combo_factorial(19)
