# The legacy symbolic surface: the ms::symbolic engine, reached through sym_*.
#
# Split from the notation and limit transcripts so that a failure names which of the
# three it is. Windows produced NO output at all for the combined file -- empty stdout,
# empty stderr, exit 1, which mathscriptc cannot reach through any normal path -- so the
# process died before its buffered stdout was flushed and the transcript could not say
# where.
sym_simplify("x + x")
sym_expand("(x + 1)^2")
sym_diff("x^3", "x")
sym_integrate("x^2", "x")
sym_eval("x*x", "x=3")
sym_collect("2*x + 3*x", "x")
