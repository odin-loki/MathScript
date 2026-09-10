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
gamma(20)
