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
# §11.2 reads it back. The result is shown in the ASCII form, not echoed as LaTeX:
# what confirms a parse is the EXPRESSION, and that `\frac{x}{y}` came back as `x/y`
# is the whole of that.
sym_from_latex("\frac{x}{y}")
sym_from_latex("x^{2} + 1")
sym_from_latex("\sqrt{x}")
sym_from_latex("\int f \, dx")
sym_from_latex("\frac{d}{dx} f")
# And the half a transcript of successes would not show: the subset refuses things, by
# name and position. Both of these are legal LaTeX that means two different expressions.
sym_from_latex("\sin^{2}(x)")
sym_from_latex("1e20")
