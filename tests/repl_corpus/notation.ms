# §11.1's notation export, on its own.
#
# This is the newest code in the corpus and the only transcript that emits non-ASCII --
# the Unicode table's superscripts and radicals -- so it is the half most likely to be
# behind a platform difference.
sym_latex("x^2 + 1")
sym_latex("(x + 1)/(y - 1)")
sym_export("x^2 + 1", "latex")
sym_export("x^2 + 1", "unicode")
sym_export("x^2 + 1", "sympy")
sym_export("x^2 + 1", "mathematica")
sym_export("x^2 + 1", "python")
sym_export("x^2 + 1", "c")
sym_export("sqrt(x)", "latex")
sym_export("1/3", "sympy")
sym_export("1/3", "mathematica")
