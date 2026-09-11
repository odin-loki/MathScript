# Scalar arithmetic, precedence, and the unary minus.
#
# `^` is here because the corpus found it missing: `2^3` did not parse, and `pow(2, 3)`
# was the only spelling, although `^` was an operator in every symbolic command and in
# the matrix literal syntax. Fixing it moved these three lines out of the .err file,
# which is what the transcript was recorded to show. Two of them are the readings that
# have a plausible alternative: 2^3^2 is 512 and not 64, and -2^2 is -4 and not 4.
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
