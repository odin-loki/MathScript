# Matrix construction, shape, and the operations a first session reaches for.
#
# RECORDED GAP, visible below: `A + B` and `A * B` are not operators over matrices.
# `matmul(A, B)` is the spelling that works; there is no operator form and no
# elementwise-add function reachable from here.
#
# `transpose(A)` with no target now prints and stores under `_`, along with 97 other
# matrix-returning callees that reached "unknown function" when printed and worked
# when assigned. The hand-written display chain claimed every unary call whose
# argument was a matrix and rejected the names it had not been told about, so the
# registry six lines below it was never asked.
A = [1, 2; 3, 4]
A
B = [5, 6; 7, 8]
A + B
A * B
transpose(A)
det(A)
trace(A)
rank(A)
mat_rows(A)
mat_cols(A)
mat_numel(A)
I = eye(3)
I
Z = zeros(2, 3)
Z
v = [1; 2; 3]
norm(v)
