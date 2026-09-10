# One name holds one value.
#
# Scalars and matrices lived in separate maps and neither assignment cleared the other,
# so a name could be a matrix to one lookup and a scalar to another at the same time.
A = [1, 2; 3, 4]
A
A = 5
A
vars
B = 7
B = [9, 9]
B
vars
