// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §8.5 applied to §11.1: the printer properties, over generated expressions.
//
// The plan names this explicitly -- "The same technique applies to the printer
// round-trips in §11" -- and it is the right technique here for a specific reason. A
// printer defect is almost never visible in the printer's own output. `\frac{x}{y+1}`
// and `x/y+1` are both plausible strings; only one of them denotes the expression that
// was printed, and reading the string does not tell you which. So the assertion has to
// be made by a *reader*: print the expression, read the text back, and ask whether the
// two agree numerically. That is a question with an answer, and it does not depend on
// anyone's taste in parenthesisation.
//
// The one notation MathScript can read back is its own, so that is where the numeric
// property runs. For the other six the assertions are structural -- balance, escaping,
// determinism, non-emptiness -- which is weaker but still catches the failure that
// matters most in markup: a delimiter that was opened and not closed.
//
// This file counts what it skipped and fails if it skipped too much. A property test
// whose generator drifts into inputs the harness cannot evaluate goes green while
// testing nothing, and the green is indistinguishable from the green it gave when it
// worked.

#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ms/sym2/bridge.hpp"
#include "ms/sym2/expr.hpp"
#include "ms/sym2/notation.hpp"

using namespace ms;
using namespace ms::sym2;

namespace {

constexpr int kTrials = 300;
constexpr std::uint64_t kSeedRoot = 0x9E3779B97F4A7C15ull;

/// Generate an expression of at most `depth` nesting levels.
///
/// The atoms are deliberately small and mostly exact. A BigInt past 2^53 or a Real
/// needing seventeen digits would exercise the legacy parser's numeric limits rather
/// than the printer's precedence, and the round trip would fail for a reason that has
/// nothing to do with what is being tested. Those limits have their own tests.
ExprRef generate(std::mt19937_64& rng, int depth) {
    std::uniform_int_distribution<int> pick(0, depth > 0 ? 9 : 3);
    switch (pick(rng)) {
    case 0:
        return integer(std::uniform_int_distribution<int>(-9, 9)(rng));
    case 1: {
        const int num = std::uniform_int_distribution<int>(-9, 9)(rng);
        const int den = std::uniform_int_distribution<int>(1, 9)(rng);
        return rational(bignum::BigInt(num), bignum::BigInt(den));
    }
    case 2: {
        static const char* const kNames[] = {"x", "y", "z"};
        return symbol(kNames[std::uniform_int_distribution<int>(0, 2)(rng)]);
    }
    case 3:
        // A short decimal, which `from_legacy` recovers as an exact rational, so the
        // round trip is exact rather than merely close.
        return real(0.25 * std::uniform_int_distribution<int>(-8, 8)(rng));
    case 4:
        return add({generate(rng, depth - 1), generate(rng, depth - 1)});
    case 5:
        return mul({generate(rng, depth - 1), generate(rng, depth - 1)});
    case 6:
        return sub(generate(rng, depth - 1), generate(rng, depth - 1));
    case 7:
        return neg(generate(rng, depth - 1));
    case 8:
        // Small integer exponents only: a symbolic exponent makes the expression
        // unevaluable at half the environments and the trial would be skipped rather
        // than run.
        return pow(generate(rng, depth - 1),
                   integer(std::uniform_int_distribution<int>(-2, 3)(rng)));
    default: {
        // Only the six the legacy parser has an operator for. `ms::sym2` will hold a
        // `Function` head of any name, and the printer will spell it, but `sym2::parse`
        // goes through the old parser and there is no `SymOp::Tanh` for it to build --
        // so a generated `tanh` would fail the round trip on the parser rather than on
        // the printer, which is not what this property is about. The asymmetry is real
        // and is asserted on its own below.
        static const char* const kFunctions[] = {"sin", "cos", "tan", "exp", "log", "sqrt"};
        return function(kFunctions[std::uniform_int_distribution<int>(0, 5)(rng)],
                        {generate(rng, depth - 1)});
    }
    }
}

std::map<std::string, double> environment(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> value(0.5, 2.0);
    return {{"x", value(rng)}, {"y", value(rng)}, {"z", value(rng)}};
}

/// Whether every delimiter that opens is closed, in order. Applied to the ASCII-family
/// notations, where a parenthesis is a delimiter rather than a character in text.
bool delimiters_balance(const std::string& text, char open, char close) {
    int depth = 0;
    for (const char c : text) {
        if (c == open) {
            ++depth;
        } else if (c == close) {
            if (--depth < 0) {
                return false;
            }
        }
    }
    return depth == 0;
}

/// The MathML subset this project emits uses no attributes containing '<' or '>' and no
/// CDATA, so a tag scan is enough to establish that every element is closed in order.
bool xml_elements_balance(const std::string& text, std::string& unbalanced) {
    std::vector<std::string> open;
    std::size_t at = 0;
    while ((at = text.find('<', at)) != std::string::npos) {
        const std::size_t end = text.find('>', at);
        if (end == std::string::npos) {
            unbalanced = "unterminated tag";
            return false;
        }
        std::string tag = text.substr(at + 1, end - at - 1);
        at = end + 1;
        if (tag.empty()) {
            unbalanced = "empty tag";
            return false;
        }
        if (tag.back() == '/') { // <pi/>, <plus/>
            continue;
        }
        if (tag.front() == '/') {
            const std::string name = tag.substr(1);
            if (open.empty() || open.back() != name) {
                unbalanced = "</" + name + "> closes " +
                             (open.empty() ? std::string("nothing") : open.back());
                return false;
            }
            open.pop_back();
            continue;
        }
        const std::size_t space = tag.find(' ');
        open.push_back(space == std::string::npos ? tag : tag.substr(0, space));
    }
    if (!open.empty()) {
        unbalanced = "<" + open.back() + "> is never closed";
        return false;
    }
    return true;
}

const Notation kAllNotations[] = {
    Notation::Latex,   Notation::PresentationMathml, Notation::ContentMathml,
    Notation::Unicode, Notation::Ascii,              Notation::SymPy,
    Notation::Mathematica, Notation::C,              Notation::Cpp,
    Notation::Python,
};

// --- The properties ------------------------------------------------------------------

// The one that matters: print it, read it back, and ask a reader whether the two are
// the same number. Every precedence defect in the walker or in the ASCII table shows up
// here, and none of them are visible by reading the string.
TEST(Sym2NotationProperties, AsciiReadsBackAsTheSameValue) {
    int compared = 0;
    int skipped_unparsable = 0;
    int skipped_undefined = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        std::mt19937_64 rng{kSeedRoot + static_cast<std::uint64_t>(trial)};
        const ExprRef e = generate(rng, 3);
        const std::map<std::string, double> env = environment(rng);
        const auto original = evaluate(e, env);
        if (!original || !std::isfinite(*original) || std::abs(*original) > 1e12) {
            // A pole, an overflow, or a domain error. Nothing to compare, and forcing a
            // comparison at a value the expression does not have would be inventing one.
            ++skipped_undefined;
            continue;
        }
        const std::string text = to_notation(e, Notation::Ascii);
        const auto reparsed = sym2::parse(text);
        if (!reparsed) {
            ++skipped_unparsable;
            ADD_FAILURE() << "MathScript cannot read back its own printer's output: "
                          << text << " (from " << to_string(e) << ")";
            continue;
        }
        const auto again = evaluate(*reparsed, env);
        ASSERT_TRUE(again.has_value())
            << "reparsed expression does not evaluate: " << text;
        ++compared;
        // The tolerance is relative: the generated expressions reach 1e12, where an
        // absolute epsilon would be meaningless, and go down to 1e-3, where a large one
        // would hide a sign error.
        EXPECT_LT(std::abs(*again - *original), 1e-9 * std::max(1.0, std::abs(*original)))
            << "printed as " << text << "\n  which reads back as " << *again
            << "\n  but denotes " << *original << "\n  original: " << to_string(e);
    }
    // Without this the test passes when the generator produces nothing evaluable, and a
    // vacuous pass looks exactly like a real one.
    EXPECT_GE(compared, kTrials / 3)
        << "only " << compared << " of " << kTrials
        << " trials were actually compared (" << skipped_undefined << " undefined, "
        << skipped_unparsable << " unparsable) -- the property was not exercised";
    EXPECT_EQ(skipped_unparsable, 0);
}

// Every notation produces something, deterministically, for every expression.
TEST(Sym2NotationProperties, EveryNotationIsTotalAndDeterministic) {
    for (int trial = 0; trial < kTrials; ++trial) {
        std::mt19937_64 rng{kSeedRoot + 0x100000ull + static_cast<std::uint64_t>(trial)};
        const ExprRef e = generate(rng, 3);
        for (const Notation notation : kAllNotations) {
            const std::string once = to_notation(e, notation);
            EXPECT_FALSE(once.empty())
                << notation_name(notation) << " produced nothing for " << to_string(e);
            EXPECT_EQ(once, to_notation(e, notation))
                << notation_name(notation) << " is not deterministic for " << to_string(e);
        }
    }
}

// A delimiter opened and not closed is the failure that markup makes easy and that
// reading the output does not reveal.
TEST(Sym2NotationProperties, DelimitersBalance) {
    for (int trial = 0; trial < kTrials; ++trial) {
        std::mt19937_64 rng{kSeedRoot + 0x200000ull + static_cast<std::uint64_t>(trial)};
        const ExprRef e = generate(rng, 3);
        for (const Notation notation : {Notation::Ascii, Notation::Unicode, Notation::SymPy,
                                        Notation::C, Notation::Cpp, Notation::Python}) {
            const std::string text = to_notation(e, notation);
            EXPECT_TRUE(delimiters_balance(text, '(', ')'))
                << notation_name(notation) << ": " << text;
        }
        // Mathematica uses brackets for application and parentheses for grouping, so
        // both have to balance and they are not interchangeable.
        const std::string wolfram = to_notation(e, Notation::Mathematica);
        EXPECT_TRUE(delimiters_balance(wolfram, '(', ')')) << wolfram;
        EXPECT_TRUE(delimiters_balance(wolfram, '[', ']')) << wolfram;
        // LaTeX groups with braces.
        const std::string latex = to_notation(e, Notation::Latex);
        EXPECT_TRUE(delimiters_balance(latex, '{', '}')) << latex;
        EXPECT_TRUE(delimiters_balance(latex, '(', ')')) << latex;
    }
}

TEST(Sym2NotationProperties, MathmlIsWellFormed) {
    for (int trial = 0; trial < kTrials; ++trial) {
        std::mt19937_64 rng{kSeedRoot + 0x300000ull + static_cast<std::uint64_t>(trial)};
        const ExprRef e = generate(rng, 3);
        for (const Notation notation :
             {Notation::PresentationMathml, Notation::ContentMathml}) {
            const std::string text = to_notation(e, notation);
            std::string reason;
            EXPECT_TRUE(xml_elements_balance(text, reason))
                << notation_name(notation) << ": " << reason << "\n" << text;
            // A bare '&' that is not the start of an entity is not XML, and it is what
            // an unescaped symbol name produces.
            for (std::size_t at = text.find('&'); at != std::string::npos;
                 at = text.find('&', at + 1)) {
                const std::size_t semicolon = text.find(';', at);
                EXPECT_NE(semicolon, std::string::npos)
                    << "unescaped '&' in " << notation_name(notation) << ": " << text;
                if (semicolon != std::string::npos) {
                    EXPECT_LT(semicolon - at, 12U)
                        << "'&' with no nearby ';' in " << notation_name(notation) << ": "
                        << text;
                }
            }
        }
    }
}

// A LaTeX control sequence is a backslash followed by letters, or by exactly one
// non-letter. A backslash followed by a digit or by nothing is a document that will not
// compile, and it is what an unescaped symbol name produces.
TEST(Sym2NotationProperties, LatexControlSequencesAreWellFormed) {
    for (int trial = 0; trial < kTrials; ++trial) {
        std::mt19937_64 rng{kSeedRoot + 0x400000ull + static_cast<std::uint64_t>(trial)};
        const ExprRef e = generate(rng, 3);
        const std::string text = to_notation(e, Notation::Latex);
        for (std::size_t at = 0; at < text.size(); ++at) {
            if (text[at] != '\\') {
                continue;
            }
            ASSERT_LT(at + 1, text.size()) << "trailing backslash in " << text;
            const char next = text[at + 1];
            const bool letter = (next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z');
            if (letter) {
                while (at + 1 < text.size() &&
                       ((text[at + 1] >= 'a' && text[at + 1] <= 'z') ||
                        (text[at + 1] >= 'A' && text[at + 1] <= 'Z'))) {
                    ++at;
                }
                continue;
            }
            EXPECT_NE(std::string("\\{}$&#_%^ ,;!").find(next), std::string::npos)
                << "'\\" << next << "' is not a control sequence, in " << text;
            ++at;
        }
    }
}

// The round trip above is restricted to the six functions the legacy parser knows.
// This is the other half of that statement: a Function head outside those six prints
// perfectly well and cannot be read back, and what comes back is a reported error
// rather than a different expression. `bridge.hpp` states this asymmetry; nothing was
// checking it.
TEST(Sym2NotationProperties, AFunctionTheParserDoesNotKnowIsReportedNotMangled) {
    const ExprRef e = function("tanh", {symbol("x")});
    const std::string text = to_notation(e, Notation::Ascii);
    EXPECT_EQ(text, "tanh(x)") << "the printer spells it regardless";
    const auto reparsed = sym2::parse(text);
    EXPECT_FALSE(reparsed.has_value())
        << "read back as " << (reparsed ? to_string(*reparsed) : std::string{})
        << " -- silently becoming a different expression is the failure mode";
}

// The notation names round-trip through their own parser, and an unknown one is
// reported rather than defaulting to something the caller did not ask for.
TEST(Sym2NotationProperties, NotationNamesRoundTrip) {
    for (const Notation notation : kAllNotations) {
        const auto parsed = notation_from_name(notation_name(notation));
        ASSERT_TRUE(parsed.has_value()) << notation_name(notation);
        EXPECT_EQ(static_cast<int>(*parsed), static_cast<int>(notation));
    }
    EXPECT_FALSE(notation_from_name("latex2").has_value());
    EXPECT_FALSE(notation_from_name("").has_value());
    // Case does not matter, because a caller typing a format name is typing prose.
    EXPECT_TRUE(notation_from_name("LaTeX").has_value());
    EXPECT_TRUE(notation_from_name("SymPy").has_value());
}

} // namespace
