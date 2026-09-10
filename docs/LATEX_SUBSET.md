<!--
SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-FileCopyrightText: 2026 Odin Loch
-->

# The LaTeX subset MathScript reads

§11.2 of the engineering plan asks for LaTeX *input*. The plan's own assessment of it
is the reason this document exists rather than a parser alone:

> LaTeX is presentation markup and there is no correct general parser, so the work is
> to define a subset, parse it strictly, and reject everything outside it with a source
> position.

A general LaTeX reader has to guess, and every guess is a chance to hand back an
expression the author did not write -- which is the defect class this project's audits
kept finding, arriving through a new door. So the subset is **defined by the printer**:
`parse_latex` accepts everything `to_latex` can emit, under every `NotationOptions`
combination, plus the twelve human spellings listed in §2.9, and refuses the rest by
name and position.

That definition is what makes the central property testable rather than aspirational:

    parse_latex(to_latex(e, options)) == e

for every expression `e` and every `options`, with §4.2 listing -- exhaustively -- the
shapes where it does not hold and why. The list is short and every entry is a case
where the *printed* form genuinely carries less than the node did.

Everything below was checked against `src/sym2/notation_latex.cpp` and, where reading
the code left a doubt, against the printer itself. The line references are to the
files as they stand.

---

## 0. Scope

The accepted language is **everything `src/sym2/notation_latex.cpp` can emit from a well-formed node, under every `NotationOptions` combination**, plus the small human list in §2.9.

**Well-formed node** (the printer can emit unparseable text from a malformed one, so the grammar must exclude these rather than pretend):

* `Head::Symbol` atom is non-empty (`spell_name` returns `{}` for `""` — notation_latex.cpp:130-132, so the atom vanishes from the output entirely).
* `Head::Function` atom is non-empty (`\operatorname{}(x)` is emitted otherwise — notation_latex.cpp:541; verified).
* `Head::Derivative`/`Head::Integral` `args[1..]` and `Head::Limit` `args[1]` are `Head::Symbol` (they are placed at `kPrecOpen` — notation.cpp:326, :336, :342 — so `derivative(x,{x+y})` emits `\frac{d}{dx + y} x` and `integral(x,{x+y})` emits `\int x \, dx + y`; both verified).
* `Head::Constant` atom is one of the seven named at `notation_syntax.hpp:61`. Any other
  name is spelled `\mathrm{name}` (notation_latex.cpp:315), which is byte-for-byte what
  `symbol("name")` produces, so the two are indistinguishable on the way back.
* `NotationOptions::matrix_environment` ∈ {`pmatrix`,`bmatrix`,`vmatrix`,`Vmatrix`,`matrix`} (it is interpolated verbatim into `\begin{}`/`\end{}`, notation_latex.cpp:456-459, with no escaping).
* `NotationOptions::decimal_separator` ∈ {`.`, `,`}.

---

## 1. Token list — exact bytes

The lexer is a TeX-token lexer, not a regex sweep. Category: control word `\` + `[A-Za-z]+` (maximal munch, trailing spaces after a control **word** are absorbed); control symbol `\` + one non-letter.

### 1.1 Structural / operator terminals

| Token | Exact bytes | Emitted at |
|---|---|---|
| `FRAC` | `\frac` | notation_latex.cpp:513 |
| `DFRAC` | `\dfrac` | :513 (`display`) |
| `SQRT` | `\sqrt` | :370, :372, :386 |
| `CDOT` | `\cdot` | :342 |
| `TIMES` | `\times` | :343, and inside a scientific numeral :276 |
| `THINSPACE` | `\,` (backslash, comma) | :346, :444 |
| `LPAREN` | `(` | :407 |
| `RPAREN` | `)` | :407 |
| `LEFT` | `\left` | :405, :383 |
| `RIGHT` | `\right` | :405, :383 |
| `BAR` | `\|` (U+007C) | :383 (always after `\left`/`\right`) |
| `LBRACE` | `{` | everywhere |
| `RBRACE` | `}` | everywhere |
| `LBRACK` | `[` | :372 (root degree only) |
| `RBRACK` | `]` | :372 |
| `CARET` | `^` | :365, :276 |
| `UNDERSCORE` | `_` | :288, :451 |
| `PLUS` | `+` (space-plus-space in output: `" + "`) | :330 |
| `MINUS` | `-` | :236, :307, :326, :330, :375, and inside `BigInt::to_string` |
| `COMMA` | `,` | :394 (argument separator only) |
| `AMP` | `&` | :468 (matrix cell separator, ` & `) |
| `ROWSEP` | `\\` (two backslashes) | :463 (` \\ `) |
| `OPERATORNAME` | `\operatorname` | :541 |
| `MATHRM` | `\mathrm` | :94-104, :155, :310, :313, :315 |
| `BEGIN` | `\begin` | :475, :477 |
| `END` | `\end` | :475, :477 |
| `INT` | `\int` (emitted as `\int ` with a trailing space, :437) | :437 |
| `LIM` | `\lim` | :451 |
| `TO` | `\to` | :451 |
| `DIGIT` | `0`–`9` | numerals, exponents |
| `LETTER` | `A`–`Z`, `a`–`z` | one-character names, the differential `d` |
| `DOT` | `.` | inside a Real, when `decimal_separator == '.'` |
| `BYTE` | any byte ≥ 0x80 | passed through untouched by `escape` (default case, :70), so a UTF-8 name lands raw inside `\mathrm{}` |

### 1.2 Escape terminals (reverse of `escape`, notation_latex.cpp:49-74)

Exactly ten, and only these ten. The trailing `{}` on the last three **is part of the escape** and must be consumed, not read as an empty group.

```
\#   -> '#'
\$   -> '$'
\%   -> '%'
\&   -> '&'
\_   -> '_'
\{   -> '{'
\}   -> '}'
\textasciitilde{}   -> '~'
\textasciicircum{}  -> '^'
\textbackslash{}    -> '\'
```

### 1.3 Named-glyph terminals (notation_latex.cpp:106-109, matched case-sensitively at :142-146)

```
\hbar   \ell   \infty   \nabla   \partial   \aleph
```

`\infty` is also the spelling of `constant("inf")` (:304) and of a non-finite `Real` (:250).

### 1.4 Greek terminals

Lower case — 30 control words, one per `kGreek` entry (notation_latex.cpp:93-104 via :138). Note `\omicron` and `\varsigma`, which are **not** standard LaTeX control sequences but are what this table emits:

```
\alpha \beta \gamma \delta \epsilon \zeta \eta \theta \iota \kappa
\lambda \mu \nu \xi \omicron \pi \rho \sigma \tau \upsilon
\phi \chi \psi \omega
\varepsilon \vartheta \varpi \varrho \varsigma \varphi
```

Capitals with a control sequence — 11:

```
\Gamma \Delta \Theta \Lambda \Xi \Pi \Sigma \Upsilon \Phi \Psi \Omega
```

Capitals spelled as an upright Latin letter — 13. These are `MATHRM LBRACE <one uppercase letter> RBRACE` and, because a one-character symbol prints bare (`symbol("A")` → `A`, :152-153), a single uppercase Latin letter inside `\mathrm{}` can **only** be a Greek capital:

```
\mathrm{A} \mathrm{B} \mathrm{E} \mathrm{Z} \mathrm{H} \mathrm{I} \mathrm{K}
\mathrm{M} \mathrm{N} \mathrm{O} \mathrm{P} \mathrm{T} \mathrm{X}
```

Inverse map: `A→Alpha B→Beta E→Epsilon Z→Zeta H→Eta I→Iota K→Kappa M→Mu N→Nu O→Omicron P→Rho T→Tau X→Chi`.

### 1.5 Operator-name terminals (`kOperators`, notation_latex.cpp:210-214) — 25, exact and case-sensitive

```
\sin \cos \tan \log \ln \exp \sinh \cosh \tanh
\arcsin \arccos \arctan \sec \csc \cot \min \max \gcd
\det \arg \deg \dim \ker \lg \sup
```

`\inf` is **not** in this list; `inf` is the atom name of the infinity constant. `\coth`, `\Pr`, `\hom`, `\liminf` are not here either and arrive as `\operatorname{…}`.

### 1.6 Numeral terminals

* `INTLIT` = `DIGIT+`, unbounded (a `BigInt` never acquires an exponent — test_sym2_notation_latex.cpp:39-40).
* `DECPOINT` = `.` when the parser is configured `decimal_separator == '.'`; the three bytes `{` `,` `}` when configured `','` (notation_latex.cpp:516-530). `{,}` is **not** a group.
* `DECLIT` = `INTLIT DECPOINT INTLIT`. `format_exact` (format.hpp:106-118) always emits a leading digit, so `.5` and `5.` are never emitted and are rejected.

### 1.7 Whitespace

`SPACE`, `TAB`, `LF`, `CR` are insignificant separators. `THINSPACE` (`\,`) is a **token**, not whitespace, because it is load-bearing in two places: the differential separator (:444) and the only mark between two juxtaposed factors that would fuse (:346). Dropping it turns `2\,10^{n}` into `210^{n}`.

---

## 2. Grammar (EBNF)

### 2.1 Precedence and associativity, loosest first

| Level | Operators | Associativity | Note |
|---|---|---|---|
| 0 | `\frac{d}{dx}…`, `\int…`, `\lim_{…}…` | prefix, body extends **maximally** to the end of the enclosing group | matches `kPrecAdd` at notation.cpp:330, :338, :344 |
| 1 | binary `+`, `-` | left | `sum()` at notation_latex.cpp:320-335 |
| 2 | prefix unary `-` | takes a **Term** (level 3) | `-2 \cdot x` is `-(2·x)`; `-x^{2}` is `-(x^{2})` |
| 3 | `\cdot` `\times` `*` `\ast` `/` `\div`, and **juxtaposition** | left | juxtaposition binds identically to an explicit operator |
| 4 | `^` | right (through braces) | base slot is `kPrecPow+1 == kPrecAtom`, notation.cpp:272 |
| — | `_` | not an operator | only inside a name or after `\lim` |

The reason level 0 is a *statement-position* operator rather than a prefix at level 1: the walker places every sum term and every product factor at `kPrecMul` or tighter (notation.cpp:119, :194), so a calculus operator is **always** parenthesised there — verified `y + (\frac{d}{dx} x)` and `(\frac{d}{dx} x) \cdot y`. It appears bare only where a whole Expression is expected.

### 2.2 Top level

```ebnf
Input        = [ MathOpen ] Expression [ MathClose ] EOF ;
MathOpen     = "$$" | "$" | "\(" | "\[" ;          (* human, §2.9; one outer pair only *)
MathClose    = "$$" | "$" | "\)" | "\]" ;

Expression   = CalculusExpr | Additive ;
```

### 2.3 Additive, multiplicative, power

```ebnf
Additive     = [ "-" ] Term { ( "+" | "-" ) Term } ;

Term         = Factor { ( MulOp Factor ) | Factor } ;      (* the bare Factor is juxtaposition *)
MulOp        = "\cdot" | "\times" | "*" | "\ast" | "/" | "\div" ;

Factor       = Power ;
Power        = Primary [ "^" Argument ] ;                  (* at most one "^" — see A18 *)

Argument     = "{" Expression "}" | SingleToken ;          (* TeX's one-token rule, §2.9 H4 *)
SingleToken  = DIGIT | LETTER | GreekCS | GlyphCS | EscapeCS ;
```

`Power`'s `Primary` deliberately includes `Fraction`, `Root`, `Call` and `Abs`, because all four report `kPrecAtom` and are therefore **not** parenthesised under a `^`. All four shapes are emitted and were verified:

```
\frac{x}{y}^{2}      \frac{1}{x}^{n}      \sqrt{x}^{2}
\sin(x)^{2}          \left| x \right|^{n}
```

### 2.4 Primary

```ebnf
Primary  = RealNumeral | Numeral | Constant | SymbolRef
         | Group | Fraction | Root | Call | Abs ;

Group    = Open Expression Close ;
Open     = "(" | "\left" "(" | BigCS "(" ;
Close    = ")" | "\right" ")" | BigCS ")" ;

Numeral      = INTLIT | DECLIT ;
RealNumeral  = ( INTLIT | DECLIT ) "\times" "10" "^" "{" [ "-" ] INTLIT "}" ;
```

`RealNumeral` is matched **leftmost-longest, before** the multiplicative rule, so `1 \times 10^{20} \times x` under `Multiplication::Cross` lexes as `Real(1e20)` then `\times` then `x` (verified emitted). Semantics: `real(strtod(mantissa "e" exponent))`. `Numeral` with a `DECPOINT` is `real(strtod(…))`; without one it is `integer(BigInt(digits))`.

```ebnf
Fraction = FracCS Argument Argument ;
FracCS   = "\frac" | "\dfrac" | "\tfrac" ;

Root     = "\sqrt" [ "[" Expression "]" ] Argument ;

Call     = FuncHead Open [ Expression { "," Expression } ] Close ;
FuncHead = OperatorCS | "\operatorname" [ "*" ] "{" EscapedName "}" ;
OperatorCS = "\sin" | "\cos" | … ;                    (* the 25 of §1.5 *)

Abs      = AbsOpen Expression AbsClose ;
AbsOpen  = "\left" "|" | "\lvert" | "|" ;
AbsClose = "\right" "|" | "\rvert" | "|" ;
```

A `Call`'s `Open`/`Close` must be parentheses and must agree in kind (`\left(` pairs only with `\right)`). Zero arguments are legal — `\operatorname{f}()` and `\max()` are emitted (:400 with an empty `arguments`; verified). The argument separator is exactly `,` and is never affected by `decimal_separator` (:392-396).

### 2.5 Names and symbols

```ebnf
SymbolRef  = NameAtom [ "_" Argument ] ;
NameAtom   = LETTER | EscapeCS | GreekCS | GlyphCS | UprightName ;
UprightName= "\mathrm" "{" EscapedText "}" ;
EscapedText= { LETTER | DIGIT | BYTE | EscapeCS | <any byte except \ { } $ & # % ^ ~ _> } ;
```

Name reconstruction, inverting `spell_name` (notation_latex.cpp:129-156) and `split_subscript` (notation.cpp:351-360):

1. Un-spell the base: a single letter → itself; `EscapeCS` → the raw character; `\mathrm{…}` → its contents with the ten escapes reversed; a Greek control word → its lowercase name; a Greek capital control word or `\mathrm{<capital>}` → the Capitalised name; a `GlyphCS` → its bare name; `INTLIT` → the digits.
2. If a `_ Argument` follows, un-spell the argument the same way (it is a **name**, not an expression: `x_{\pi}` is the name `x_pi`, `x_{\mathrm{max}}` is `x_max`, `x_{12}` is `x_12` — all verified).
3. The atom is `base` or `base "_" subscript`. Note `\mathrm{x\_a}_{1}` → `x_a_1` (verified) — the escaped underscore in the base must be un-escaped **before** rejoining.
4. Brace matching inside `\mathrm{}` must skip backslash-escaped braces exactly as `is_upright_word` does (notation_latex.cpp:172-176): `\mathrm{a\{b}` is one group.

```ebnf
Constant = "\pi" | "e" | "i" | "\infty" | "-" "\infty"
         | "\mathrm" "{" "NaN" "}" | "\mathrm" "{" "undefined" "}" ;
```

These win over `SymbolRef` **at expression level only**; in a subscript `Argument` they are names (§2.5 step 2).

### 2.6 Calculus operators

```ebnf
CalculusExpr = DerivativeOp+ Expression
             | IntegralExpr
             | LimitOp Expression ;

DerivativeOp = FracCS "{" DiffD "}" "{" DiffD Variable "}" ;
DiffD        = "d" | "\mathrm" "{" "d" "}" ;
Variable     = SymbolRef ;                         (* well-formedness, §0 *)

IntegralExpr = IntSign+ Expression { "\," DiffD Variable } ;
IntSign      = "\int" | "\iint" | "\iiint" | "\int" "\limits" ;

LimitOp      = "\lim" [ "\limits" ] "_" "{" Variable ToCS Expression "}" ;
ToCS         = "\to" | "\rightarrow" | "\longrightarrow" ;
```

**Derivative.** A run of adjacent `DerivativeOp`s is **one flat** `Head::Derivative` with the variables in written order; whitespace between them is not significant, which is exactly why the flat and nested encodings cannot be told apart (verified: flat = `\frac{d}{dx}\frac{d}{dy} x \cdot y`, nested = `\frac{d}{dx} \frac{d}{dy} x \cdot y` — one space).

**Integral, differential-scan algorithm** (this is the part that cannot be done with a plain recursive descent):

1. On seeing `IntSign+`, record `n` = total sign count (`\iint` counts 2).
2. Scan forward to the end of the enclosing group (matching `{}`, `()`, `\left`/`\right`), tracking depth.
3. Walk **backwards** from that end, consuming maximal `\, DiffD Variable` units at depth 0. Stop at the first unit that does not match — in particular, `\, d` **at end of input with no variable after it is not a differential** (this is what saves `\int \mathrm{aa}\,d`, a real emitted string).
4. Let `k` = units consumed. Require `k == n`, or (`k == 0` and `n == 1`) — the latter is the empty-variable-list form `\int x^{2}`, which the printer emits and which round-trips (notation_latex.cpp:434). Otherwise diagnose `E-LATEX-0031`.
5. The span between the signs and the first differential is the integrand, parsed as an `Expression`.

**Limit.** The point is an `Expression` that must contain no top-level `\to`. The body is unfenced and greedy: `\lim_{x \to 0} x + y` is `limit(x+y, x, 0)` (verified).

### 2.7 Matrix — a separate entry point

`Head` has no matrix member (expr.hpp:51-64); `to_latex(const Matrix<double>&)` never goes through `walk()` (notation.cpp:425-437). So this is `parse_latex_matrix` returning `Matrix<double>`, and a matrix inside an expression is rejected.

```ebnf
MatrixDoc = "\begin" "{" Env "}" [ Row { RowSep Row } ] "\end" "{" Env "}" ;
Env       = "pmatrix" | "bmatrix" | "vmatrix" | "Vmatrix" | "matrix" ;
Row       = Cell { "&" Cell } ;
RowSep    = "\\" ;
Cell      = [ "-" ] ( RealNumeral | Numeral ) | "\infty" | "-" "\infty"
          | "\mathrm" "{" "NaN" "}" ;
```

Exact separators: cells are joined by ` & `, rows by ` \\ `, one space after `\begin{ENV}` and one before `\end{ENV}` — **except** that an empty body emits `\begin{ENV}\end{ENV}` with **no** spaces (notation_latex.cpp:474-476; verified for 0×0, 3×0 and 0×3, all three identical). The opening and closing `Env` must be the same string. Rows must be rectangular.

A cell is not a plain float regex: at only `1e8` it becomes `1 \times 10^{8}` (verified), and with `decimal_separator = ','` it becomes `0{,}5` (verified).

### 2.8 What `display`, `sized_delimiters`, `roots_as_radicals` and `multiplication` cost the parser

Nothing structural — the grammar above accepts every setting simultaneously. Two documentation errors are worth repeating: `display` produces **only** `\frac`→`\dfrac` (notation_latex.cpp:513), never `\[ … \]` (that comes unconditionally from `latex_document`, notation_api.cpp:166-175), contradicting notation.hpp:69; and `roots_as_radicals = false` produces `x^{\frac{1}{2}}`, never `x^{1/2}` (verified), contradicting notation.hpp:84.

### 2.9 The human list — complete, and nothing else

| # | Accepted | Meaning |
|---|---|---|
| H1 | `\tfrac` | ≡ `\frac` (`\dfrac` is printer output, not a concession) |
| H2 | `\bigl \bigr \Bigl \Bigr \biggl \biggr \Biggl \Biggr \big \Big \bigg \Bigg` before `(` `)` `\|` | pure delimiter spelling, no meaning |
| H3 | `\lvert X \rvert`, `\|X\|` | ≡ `\left\| X \right\|` = `function("abs",{X})` |
| H4 | braceless single-token argument: `x^2`, `\frac12`, `\sqrt2`, `x_1`, `\frac\alpha\beta` | TeX's one-token rule |
| H5 | `\;` `\:` `\!` `\quad` `\qquad` `\thinspace` `\ ` `~` | whitespace (never emitted, so no conflict with `\,`) |
| H6 | `\rightarrow` `\longrightarrow`; `\lim\limits` | ≡ `\to`; ≡ `\lim` |
| H7 | `\iint` `\iiint` `\int\limits`; `\mathrm{d}` in a differential position | 2/3 `\int`; ≡ `\int`; ≡ `d` |
| H8 | `\operatorname*{name}`; `\operatorname{sin}` | ≡ `\operatorname{name}`; ≡ `\sin` (fold `\operatorname{X}` and `\X` to one name before comparing) |
| H9 | one outer `$…$`, `$$…$$`, `\(…\)`, `\[…\]` | stripped |
| H10 | `%` to end of line; U+2212; literal Unicode Greek letters; U+221E | comment; `-`; their control words; `\infty` |
| H11 | `/` and `\div` | infix division, multiplication precedence, left-associative |
| H12 | `*` and `\ast` | ≡ `\cdot` |

---

## 3. The ambiguity table

Diagnostics are quoted verbatim; `L:C` is the line and column of the offending token.

### 3.1 Accepted with a stated meaning

| # | Text | The two readings | Ruling |
|---|---|---|---|
| A1 | `g(x + y)`, `\mathrm{foo}\,(x + y)`, `\operatorname{f}(y)x` | application vs product | **Product.** Application is always marked in printer output — `\operatorname{}` or a `kOperators` control word (:388, :532-542). A bare italic letter or a `\mathrm{}` word before `(` is implicit multiplication. All three strings are emitted (verified). A user who means a call writes `\operatorname{f}(x)`. |
| A2 | `\sqrt{x}` | `Pow(x,1/2)` vs `Function("sqrt",{x})` | **`pow(x, 1/2)`.** Byte-identical producers at :369 and :386 (tests :226, :272). |
| A3 | `\frac{A}{B}` | Rational / Mul-denominator / negative Pow | **`div(A,B)`** in every case. No case analysis: `div(3,4) ≡ rational(3,4)`, `div(1,pow(x,3)) ≡ pow(x,-3)` (verified). |
| A4 | `\frac{d}{dx} f` | derivative vs `d/(d·x)` quotient | **Derivative**, when the numerator group is exactly `d`/`\mathrm{d}` and the denominator is `d` + a Symbol. To write the quotient: `\frac{d}{d \cdot x}`. |
| A5 | `\int f \, dx` | differential vs a juxtaposed `d·x` | **Differential**, by the position rule of §2.6. Note `\,` means multiplication at :346 and delimiter at :444, so the rule keys on position, never on the `\,`. |
| A6 | `\int \int x \cdot y \, dx \, dy` | flat vs nested | **Flat**: `integral(x·y, {x,y})`. The two encodings are byte-identical (verified). |
| A7 | `\frac{d}{dx}\frac{d}{dy} f` / `\frac{d}{dx} \frac{d}{dy} f` | flat vs nested | **Flat**: `derivative(f,{x,y})`. They differ by one space, which math mode discards. |
| A8 | `\pi` `e` `i` `\infty` `\mathrm{NaN}` `\mathrm{undefined}` | Constant vs Symbol vs Real | **Constant**, at expression level. Inside a subscript they are names. |
| A9 | `\mathrm{A}` … `\mathrm{X}` (one uppercase letter) | Greek capital vs symbol `A` | **Greek capital** (`symbol("Alpha")`). A one-letter symbol prints bare (`A`), so the two spellings are distinct (test :91-92). |
| A10 | `-\infty` after a unary minus | `constant("-inf")` vs `neg(constant("inf"))` | **`constant("-inf")`.** Both are emitted for that string (verified). A **binary** minus is unaffected: `x - \infty` is `sub(x, constant("inf"))` — and is emitted (verified). |
| A11 | `1 \times 10^{20}` | Real numeral vs `Mul(1, Pow(10,20))` | **One Real numeral** (§2.4). Costs nothing: the exact integer is written `10^{20}`. |
| A12 | `2 \times x` vs `2 \times 10^{3}` | operator vs numeral | Numeral rule matched first (leftmost-longest); everything else is `Multiplication::Cross`. |
| A13 | `12` | Integer vs Real vs `symbol("12")` | **`integer(12)`.** |
| A14 | `2\,3` / `2 3` | numeral `23` vs product | **Product** `2·3`. A numeral's digits are contiguous; any separator is a factor boundary. Required by `2\,10^{n}` (verified). |
| A15 | `\operatorname{f}(1{,}5, 2{,}5)` | decimal comma vs argument comma | **Braces decide** (:516-530 vs :394). Only accepted when the parser is configured `decimal_separator = ','`. |
| A16 | `\sqrt[n]{x}` with a symbolic degree | — | **`pow(x, 1/n)`.** Never emitted (the degree is always `integer(exponent.den)`, notation.cpp:257), but unambiguous, so accepted. |
| A17 | `-x^{2}` | `(-x)^2` vs `-(x^2)` | **`-(x^{2})`.** Unary minus binds looser than `^` — the same rule task #32 fixed in the REPL parser. |
| A18 | `x^{-y}`, `x^{-\infty}`, `x \cdot y^{-n}` | — | A unary minus inside an exponent is ordinary negation. These **are** emitted (verified); only a *numeric* negative exponent becomes `\frac{1}{…}`. |

### 3.2 Rejected, with the exact diagnostic text

| # | Text | Diagnostic (after `latex:L:C: error[CODE]: `) |
|---|---|---|
| A19 | `x^10`, `x_12` unbraced | `[E-LATEX-0011] a braceless argument takes only the next token: x^10 is x^{1} multiplied by 0 in TeX; write x^{10}` |
| A20 | `x^{2}^{3}` | `[E-LATEX-0012] double superscript; TeX rejects this. Write x^{2 \cdot 3} or (x^{2})^{3}` |
| A21 | `\sin^{2}(x)`, `\sin^{-1}(x)` | `[E-LATEX-0013] a superscript on a function name is the square of the value at 2 and the inverse function at -1; write (\sin(x))^{2}, \arcsin(x), or \frac{1}{\sin(x)}` |
| A22 | `\sin x` | `[E-LATEX-0014] the extent of the argument is not written: \sin x + 1 reads as (\sin x) + 1 or \sin(x + 1); write \sin(x)` |
| A23 | `\log_{2}(x)` | `[E-LATEX-0015] a subscript on a function name is the base of the logarithm here and an index elsewhere; the subset has no based logarithm. Write \frac{\log(x)}{\log(2)}` |
| A24 | `a / b c` | `[E-LATEX-0016] the denominator ends at b or at c depending on the writer; use \frac{a}{bc} or \frac{a}{b} \cdot c` |
| A25 | `n!` | `[E-LATEX-0017] a postfix ! is a factorial, a double factorial when doubled, and a negation in some writing; write \operatorname{factorial}(n)` |
| A26 | `f'(x)`, `\dot{x}`, `\ddot{x}` | `[E-LATEX-0018] a prime or a dot is a derivative with respect to an unwritten variable (and a prime is also a transpose); write \frac{d}{dx} f(x)` |
| A27 | `a \pm b`, `a \mp b` | `[E-LATEX-0019] a \pm b denotes two expressions at once; a MathScript expression is one value` |
| A28 | `=` `<` `>` `\le` `\leq` `\ge` `\neq` `\ne` `\approx` `\equiv` `\sim` `\propto` `\in` `\mid` | `[E-LATEX-0020] the subset parses expressions, not equations; '=' has no expression head. Parse the two sides separately, or write the difference` |
| A29 | `\sum` `\prod` `\bigcup` `\bigcap` `\coprod` `\bigoplus` | `[E-LATEX-0021] \sum_{i=1}^{n} has no expression head in ms::sym2 (see expr.hpp Head); big operators are outside the subset` |
| A30 | `\binom{n}{k}`, `{n \choose k}` | `[E-LATEX-0022] \binom{n}{k} has no expression head; write \operatorname{binomial}(n, k)` |
| A31 | `\int_{a}^{b}`, `\oint` | `[E-LATEX-0023] Head::Integral records no bounds (expr.hpp:62), so limits would be silently discarded; write \int f \, dx` |
| A32 | `\lim_{x \to 0^{+}}`, `\limsup`, `\liminf` | `[E-LATEX-0024] Head::Limit records no direction (expr.hpp:63), so a one-sided limit would be read as a two-sided one` |
| A33 | `\frac{\partial}{\partial x}` | `[E-LATEX-0025] Head::Derivative records no distinction between a partial and a total derivative (notation_latex.cpp:414); write \frac{d}{dx}. \partial alone is the symbol named partial` |
| A34 | `\{ … \}`, `[a, b]`, `[ … ]`, `\langle` `\lfloor` `\lceil` `\lVert` `\|` | `[E-LATEX-0026] \{ \} is a set or a case split, [a, b] is an interval, a list or a matrix row, and \lfloor x \rfloor is a floor; none has an expression head. Use ( ) for grouping and \operatorname{floor}(x) for a floor` |
| A35 | a `\|` carrying a script (`f\big\|_{0}^{1}`), `\left.`, `\right.` | `[E-LATEX-0027] an evaluation bar (a null delimiter with limits) has no expression head` |
| A36 | nested bare `\|` (`\|\|x\|\|` or `\|a\|b\|`) | `[E-LATEX-0028] a bare \| cannot be paired: the open and close delimiter are the same character. Write \left\| … \right\| or \lvert … \rvert` |
| A37 | `\text{}` `\mathbf` `\mathbb` `\mathcal` `\mathfrak` `\mathsf` `\boldsymbol` `\vec` `\hat` `\bar` `\overline` `\tilde` `\underline` `\mathit` `\bm` | `[E-LATEX-0029] decoration is not part of a name in this subset: \hat{x} and x are different symbols to a reader and the same name to the parser. Write x_{hat} or another name` |
| A38 | `\varGamma`, `\upalpha` | `[E-LATEX-0030] \varGamma is a font variant of \Gamma, not a distinct symbol; write \Gamma` |
| A39 | differential/sign count mismatch | `[E-LATEX-0031] N integral signs but M differentials; write one \, dx per \int` |
| A40 | `1e20` | `[E-LATEX-0032] 1e20 is 1 multiplied by Euler's number, plus 20, in math mode (notation_latex.cpp:259-262); write 1 \times 10^{20}` |
| A41 | `\inf` | `[E-LATEX-0033] \inf is the infimum operator, not infinity; write \infty. (\sup is a function name in this subset; \inf is not.)` |
| A42 | `\cfrac` | `[E-LATEX-0034] \cfrac is continued-fraction layout with an optional alignment argument, not a distinct operation; write \frac{a}{b}` |
| A43 | `\newcommand` `\renewcommand` `\def` `\let` `\providecommand` | `[E-LATEX-0035] macro expansion is not performed; TeX is Turing-complete. Expand the macro before parsing` |
| A44 | `\begin{array}` `{cases}` `{aligned}` `{align}` `{equation}` `{gather}` `{split}` `{smallmatrix}` | `[E-LATEX-0036] \begin{array}{cc} carries a column specification, which is presentation rather than structure; use pmatrix, and parse a matrix with the matrix entry point` |
| A45 | a matrix environment inside an expression | `[E-LATEX-0037] a matrix is not an expression in ms::sym2 (expr.hpp has no matrix head); parse it with parse_latex_matrix` |
| A46 | `&` or `\\` outside a matrix | `[E-LATEX-0038] '&' is a matrix cell separator and '\\' a row separator; neither is an expression token` |
| A47 | `\operatorname{}` | `[E-LATEX-0039] an empty function name carries no name to recover` |
| A48 | a subscript on anything that is not a name (`\frac{a}{b}_{1}`, `(x)_{1}`) | `[E-LATEX-0040] a subscript belongs to a symbol name; it has no meaning here` |
| A49 | `{,}` when configured `decimal_separator = '.'` (or a bare `,` inside a numeral) | `[E-LATEX-0041] 1,5 is one number with a decimal comma or two numbers in a list; write 1{,}5 and configure decimal_separator = ','` |
| A50 | `\sqrt{}` | `[E-LATEX-0042] empty radicand` |
| A51 | `\surd`, `\root` | `[E-LATEX-0043] \surd is a glyph with no radicand; write \sqrt{x}` |
| A52 | any other control sequence | `[E-LATEX-0044] unknown control sequence \foo; the accepted subset is documented in docs/LATEX_SUBSET.md` |

---

## 4. The round-trip claim

Write `P(e, o) = to_latex(e, o)` and `R(s, o) = parse_latex(s, o)`.

### 4.1 Guaranteed

> For every well-formed `e` (§0) drawn from the classes below, and for **every** `NotationOptions o` whose `decimal_separator` and `matrix_environment` are legal, `structurally_equal(R(P(e,o), o), e)` holds.

* `Head::Integer` — every value, any number of digits, either sign. `-7` re-reads as `neg(integer(7))`, which **is** `integer(-7)` (verified).
* `Head::Rational` — every value. `-\frac{3}{4}` re-reads as `neg(rational(3,4)) ≡ rational(-3,4)` (verified). `\frac{p}{q}` of two literals re-reads as `div(p,q) ≡ rational(p,q)` (verified).
* `Head::Real` **iff** `ms::format_exact(value)` contains a `.` or an `e` — i.e. the value is finite and not whole-valued below 1e6. Covers `2.5`, `0.3333333333333333`, `0.0001`, `1e20`, `1e-7`, `1e6` (which is already `1 \times 10^{6}`, verified), and every denormal.
* `Head::Symbol` whose atom, after `split_subscript`, satisfies **all** of: base non-empty and not all-digits; base ∉ {`pi`, `e`, `i`, `infty`, `NaN`, `undefined`}; if the base lowercases to a Greek name it is spelled exactly canonically (`alpha` or `Alpha`, `gamma` or `Gamma`) and is not a capitalised `var`-form; the same two conditions on the subscript (a subscript may be `pi`, `e`, `i`, or all digits — it is un-spelled as a name, not a constant). Names with LaTeX specials, with embedded or trailing underscores, and with non-ASCII bytes all round-trip.
* `Head::Constant` — all seven.
* `Head::Add`, `Head::Mul` — always, including display reordering (`walk_add` moves the numeric term to the end, notation.cpp:100-112; `walk_mul` stable-sorts by base name, :166-176). Re-parsing goes back through `add()`/`mul()`, which re-canonicalise, so **argument order is restored**.
* `Head::Pow` — always, including the reciprocal rewrite (`\frac{1}{x^{3}}` ≡ `pow(x,-3)`, verified), the radical rewrite (`\sqrt[3]{x}` → `pow(x,1/3)`), `roots_as_radicals = false` (`x^{\frac{1}{2}}`), a symbolically negative exponent (`x^{-y}`), and an atom-precedence base under a superscript (`\frac{x}{y}^{2}`, `\frac{1}{x}^{n}`).
* `Head::Function` — every name **except** `sqrt` with exactly one argument; every arity including 0; `abs`/1 through the bars; case preserved (`\operatorname{Sin}` ≠ `\sin`).
* `Head::Derivative` with a **non-empty** variable list whose entries are Symbols.
* `Head::Integral` with **any** variable list, including the empty one (`\int f` → `integral(f,{})`), whose entries are Symbols.
* `Head::Limit` whose variable is a Symbol.
* `Matrix<double>` with at least one cell, every finite entry.

### 4.2 NOT guaranteed — the complete list, with the reason

A round-trip test should be written directly from this table; each row is a case that must be asserted **unequal** (or asserted to produce the stated substitute), not skipped.

| # | Expression | Printed | Re-parsed as | Why |
|---|---|---|---|---|
| N1 | `real(v)` where `format_exact(v)` is bare digits — every whole-valued `\|v\| < 1e6`, e.g. `real(1.0)`, `real(2.0)`, `real(123456.0)` | `1` | `integer(1)` | `%g` drops the fractional part (format.hpp:106-118). LaTeX carries no marker for `Head::Real`. Test :59 pins it. `is_exact`, `is_zero` and Mul's zero-annihilation all treat the two differently (expr.hpp:106-107, :122-123). |
| N2 | `real(-0.0)` | `-0` | `integer(0)` | N1, plus the sign of zero is lost. Also `-0.0 < 0.0` is false, so it is `kPrecAtom` and reaches a superscript bare: `-0^{n}` (verified), which LaTeX reads as `-(0^n)`. |
| N3 | `real(+inf)` | `\infty` | `constant("inf")` | notation_latex.cpp:242-250 deliberately routes non-finite Reals through the constant spellings. |
| N4 | `real(-inf)` | `-\infty` | `constant("-inf")` | as N3. (Precedence differs from the Constant: `pow(real(-inf),n)` is `(-\infty)^{n}` and `pow(constant("-inf"),n)` is `-\infty^{n}` — both verified — but both re-parse to the same Constant.) |
| N5 | `real(nan)` | `\mathrm{NaN}` | `constant("nan")` | as N3. |
| N6 | `mul(integer(-1), constant("inf"))` | `-\infty` | `constant("-inf")` | Verified: both nodes print `-\infty`. Ruling A10 picks the Constant. |
| N7 | `symbol("pi")` | `\pi` | `constant("pi")` | `greek_letter` maps the Symbol to `\pi` (:134-141), which is also `constant("pi")` (:294). |
| N8 | `symbol("e")`, `symbol("i")` | `e`, `i` | `constant("e")`, `constant("i")` | A one-letter name prints bare (:152) and so does the constant (:300-302). |
| N9 | `symbol("infty")` | `\infty` | `constant("inf")` | `kNamedGlyphs` (:109, :142-146) vs `constant("inf")` (:304). |
| N10 | `symbol("NaN")`, `symbol("undefined")` | `\mathrm{NaN}`, `\mathrm{undefined}` | `constant("nan")`, `constant("undefined")` | Multi-letter fallback (:155) collides with :310-313. |
| N11 | `symbol(<all digits>)`, e.g. `symbol("12")` | `12` | `integer(12)` | `all_digits` short-circuits the `\mathrm{}` wrapper (:111-121, :149-151) — a subscript convenience applied to the base too. |
| N12 | `symbol("")` | `` (empty) | parse error `E-LATEX-0001` | `spell_name` returns `{}` (:130-132); the atom vanishes. Excluded by §0 well-formedness. |
| N13 | any Greek spelling that is not exactly canonical: `ALPHA`, `aLPHA`, `GaMmA`, `GAMMA` | `\alpha`, `\alpha`, `\Gamma`, `\Gamma` | `symbol("alpha")` / `symbol("Gamma")` | `greek_letter` lowercases the whole name and keys `capital` on `base[0]` only (notation.cpp:373-385). Verified: `symbol("aLPHA")` → `\alpha`, `symbol("GAMMA")` → `\Gamma`. |
| N14 | the six capitalised `var`-forms: `Varepsilon`, `Vartheta`, `Varpi`, `Varrho`, `Varsigma`, `Varphi` | `\mathrm{E}`, `\Theta`, `\Pi`, `\mathrm{P}`, `\Sigma`, `\Phi` | `Epsilon`, `Theta`, `Pi`, `Rho`, `Sigma`, `Phi` | The `var` spellings share one capital (notation_latex.cpp:91-92, :102-103). Verified for `Vartheta`. |
| N15 | N13/N14 **inside a subscript**, e.g. `symbol("x_VARTHETA")` | `x_{\Theta}` | `symbol("x_Theta")` | The subscript goes through the same `spell_name` (:288). |
| N16 | `function("sqrt", {a})` — arity 1 only | `\sqrt{a}` | `pow(a, 1/2)` | :385-387 vs :369; tests :226 and :272 pin both. |
| N16b | any expression with `function("sqrt",{a})` as a **power base** | `\sqrt{a}^{2}` | `a` | N16, then `pow(pow(a,1/2),2)` folds through expr.cpp's `(b^p)^n` rule to `pow(a,1)` = `a`. Double loss. |
| N17 | `function("", args)` | `\operatorname{}(x)` | parse error `E-LATEX-0039` | :541 with an empty escape result. Excluded by §0. |
| N18 | `derivative(e, {})` | `<body only>` | `e` | :424-426 returns the body with no trace of the head. Verified: `derivative(x^2,{})` → `x^{2}`. |
| N19 | nested Derivative, e.g. `derivative(derivative(f,{y}),{x})` | `\frac{d}{dx} \frac{d}{dy} f` | `derivative(f, {x, y})` | Flat and nested differ by one space, which math mode discards (both verified). Note the variable order also inverts between the two encodings. |
| N20 | nested Integral, e.g. `integral(integral(f,{x}),{y})` | `\int \int f \, dx \, dy` | `integral(f, {x, y})` | **Byte-identical** to the flat form (both verified). |
| N21 | `Matrix<double>` with zero cells — 0×0, m×0, 0×n | `\begin{pmatrix}\end{pmatrix}` | 0×0 | The row separator is appended only when the body is non-empty (:462), so empty rows collapse. All three verified identical. |
| N22 | `Multiplication::Cross` + an exact factor `c · 10^k · …` with `\|k\| > 4096` | `2 \times 10^{5000} \times x` | `mul(real(inf), x)` | Verified emitted. Below the `kMaxExactPowerExponent = 4096` cap (expr.cpp:173) the power folds to digits, so this is only reachable past it. Rule A11 claims the `\times 10^{}`. |
| N23 | `Multiplication::Juxtaposition` + an integrand ending in an upright-word factor, then a factor named `d`, then one more factor | `\int \mathrm{aa}\,dx` | `integral(symbol("aa"), {x})` instead of `integral(aa·d·x, {})` | Verified emitted. The factor sort (notation.cpp:166-176) puts `d` early, so this needs a name sorting before `d`; the thin space then makes it look like a differential. |
| N24 | `Head::Constant` outside the seven, e.g. `constant("gamma_E")` | `\mathrm{gamma\_E}` | `symbol("gamma_E")` | :315 is byte-identical to the multi-letter Symbol spelling (:155). Verified. Excluded by §0. |
| N25 | a Derivative/Integral/Limit variable that is not a Symbol | `\frac{d}{dx + y} x`, `\int x \, dx + y` | parse error / wrong tree | Variables are placed at `kPrecOpen` (notation.cpp:326, :336, :342) so they are never fenced. Both verified emitted. Excluded by §0. |
| N26 | `matrix_environment` outside the allow-list, e.g. `"array}{cc"` | `\begin{array}{cc} … \end{array}{cc}` | parse error `E-LATEX-0036` | Interpolated with no validation (:456-459). Excluded by §0. |
| N27 | a null `ExprRef`, or a head `walk` does not recognise | `\mathrm{undefined}` | `constant("undefined")` | notation.cpp:346 and :418-422 emit the same string as the value itself. Right for `undefined()`, a fabrication for the other two. |

Everything not in this table round-trips. In particular these do **not** belong on it, contrary to the surveys: negative Integer and negative Rational atoms; the display reordering of `Add` and `Mul`; `\frac{}{}` of every provenance; `x - 7`; `-2 \cdot x`; `\frac{2 \cdot x}{3}`; `\dfrac` vs `\frac`; `\left(` vs `(`; `x^{\frac{1}{2}}` under `roots_as_radicals = false`; `\int f` with no differential.

---
---

## 5. The error model

A rejection carries a position, because a diagnostic without one sends the reader
through the whole input looking for it. The tree already has the type for this:
`ms::ParseError{line, col, msg}` inside `Error`, with `msg` owning its text so it can
name what was actually found.

```cpp
Result<ExprRef>        parse_latex(std::string_view text, const NotationOptions& o = {});
Result<Matrix<double>> parse_latex_matrix(std::string_view text, const NotationOptions& o = {});
```

Rules:

1. **`line` and `col` are 1-based**, and `col` counts UTF-8 scalar values rather than
   bytes: a symbol name may carry raw non-ASCII (the `default:` case of `escape`,
   notation_latex.cpp:70, passes bytes through untouched), and a byte column would
   point into the middle of a character.
2. **The message names what was expected and what was found**, with the found text
   quoted verbatim and never normalised -- a diagnostic about `\varGamma` says
   `\varGamma` and not `\Gamma`. At end of input it says so rather than quoting
   nothing.
3. **A delimiter mismatch names the opener too.** `\left(` closed by `\right]`, an
   unclosed `{`, a `\begin{pmatrix}` closed by `\end{bmatrix}`: the position is the
   closer and the message carries the opener's position, because the closer is where
   the reader is and the opener is what they have to go back to.
4. **The parser reports the first error and stops.** No recovery and no cascade: a
   strict parser that guesses its way past an error is the failure mode this document
   exists to prevent, and a list of ten errors nine of which are consequences of the
   first is worse than one error.
5. **There are no warnings.** A construct is in the subset or it is not.
6. `options.decimal_separator` selects the decimal terminal. `options.matrix_environment`
   is ignored on input -- all five environments are accepted -- and so are `display`,
   `sized_delimiters`, `multiplication` and `roots_as_radicals`: the grammar accepts
   every setting's output at once, which is what makes the round-trip property
   independent of the options the printer was given.

## 6. Two things this work found in the printer's documentation

Both are comments in `include/ms/sym2/notation.hpp` that describe output the printer
does not produce. They are corrected there; recorded here because a parser written
from the header rather than from the code would have accepted the wrong language.

- `display` was documented as producing `\[ ... \]` as well as display-style
  fractions. It produces only `\frac` -> `\dfrac` (notation_latex.cpp:513). The
  `\[ ... \]` comes from `latex_document`, unconditionally.
- `roots_as_radicals = false` was documented as giving `x^{1/2}`. It gives
  `x^{\frac{1}{2}}`, which is the same expression spelled the way every other
  quotient in this notation is spelled.
