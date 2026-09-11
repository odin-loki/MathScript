# sym_limit on its own, because it is the one symbolic command that works by numeric
# probing rather than by a table, and so the one whose result can depend on how the
# host evaluates a function near a singularity.
sym_limit("sin(x)/x", "x", 0)
sym_limit("(1-cos(x))/x^2", "x", 0)
