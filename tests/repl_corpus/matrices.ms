# Matrix construction, shape, and the operations a first session reaches for.
#
# RECORDED GAPS, both visible below:
#   - `A + B` and `A * B` are not operators over matrices. `matmul(A, B)` is the
#     spelling that works; there is no operator form and no elementwise-add function
#     reachable from here.
#   - `transpose(A)` has no no-target form, though `matmul(A, A)` does. The registry
#     that gives every matrix-returning callee a bare form does not reach it.
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
