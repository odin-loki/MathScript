# The error returns in the dispatch chain, which are the least-covered lines in the
# tree: a chain reaches high line coverage with one branch of each test taken, and the
# branch that is not taken is always the one that reports.
#
# Every line here is a diagnostic, so this transcript is mostly stderr. That is the
# point: the wording of a diagnostic is a contract, and a message that stops naming
# what went wrong is a change a reviewer should see in a diff.
not_a_function(1)
sqrt()
sqrt(1, 2)
mat_at(1, 2)
A = [1, 2; 3]
B = [1, 2; 3, 4]
mat_at(B, 5, 5)
mat_at(B, 0.5, 0)
mat_row(B, -1)
mat_reshape(B, 3, 3)
C = [1, 2; 3, 4]
D = [1, 2, 3]
C + D
C * D
solve(C, D)
chol(C)
1 / 0
log(-1)
sqrt(-1)
sym_diff("x^", "x")
sym_export("x", "klingon")
undefined_variable
