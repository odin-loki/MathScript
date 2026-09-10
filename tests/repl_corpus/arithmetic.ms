# Scalar arithmetic, precedence, and the unary minus.
#
# RECORDED GAP: `^` is not an operator in the REPL's scalar evaluator. `2^3` does not
# parse -- `pow(2, 3)` is the only spelling -- although `^` is an operator in every
# symbolic command and in the matrix literal syntax. The three `^` lines below are here
# so the day that is fixed shows up as three lines moving from the .err file to the
# .out file, which is a diff a reviewer can read.
#
# The last of those is here because both backends got it wrong the same way: a leading
# minus was applied to the whole expression, so -2 + 1 was -3. The interpreter and the
# ORC JIT each had their own copy of the defect.
x = 2
y = 3
x + y
x - y
x * y
x / y
x ^ y
-2 + 1
-x + 1
-x - 1
2 - -3
2 * -3
-(x + y)
2 ^ 3 ^ 2
(2 ^ 3) ^ 2
1 + 2 * 3
(1 + 2) * 3
2e3
1.5e-3
