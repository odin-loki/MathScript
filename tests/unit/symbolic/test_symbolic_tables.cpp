// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The symbolic tables, checked against the definitions rather than against the
// tables.
//
// An audit drove the REPL over the standard integral, Laplace and Fourier tables and
// found that most rows declined: sym_integrate had no entry for 1/x or exp(x),
// sym_laplace could not transform t*exp(2*t) or a negative coefficient, sym_ilaplace
// required each numerator to be exactly the constant its canonical row carries, and
// the Fourier pair had no linearity at all. The rules that close those gaps are
// general -- the first shifting theorem, frequency differentiation, a first-degree
// numerator over three denominator families -- so asserting the expected closed form
// for each row would mostly be asserting the implementation against itself.
//
// These tests do not do that. Each entry is checked against the definition it comes
// from:
//
//   antiderivatives   differentiate the result and compare with the integrand
//   Laplace           integrate f(t) e^{-st} numerically and compare with F(s)
//   inverse Laplace   forward-transform the result numerically and compare with F(s)
//   Fourier           integrate f(t) cos(omega t) numerically over the whole line
//
// so a table row that is wrong fails here even if it is wrong in exactly the way the
// implementation intends.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

namespace {

SymExpr parse_or_die(const std::string& text) {
    auto parsed = sym_parse(text);
    EXPECT_TRUE(parsed.has_value()) << "parse failed: " << text;
    if (!parsed.has_value()) {
        return sym_const(0.0);
    }
    return std::move(*parsed);
}

double at(const SymExpr& expr, const std::string& var, double value) {
    return sym_eval(expr, {{var, value}});
}

// Composite Simpson. The transforms are checked by evaluating their own defining
// integral, so the quadrature has to be well inside the tolerance the comparison
// uses. Simpson's error falls as h^4; the panel counts below put it several orders
// under the 1e-6 the comparisons ask for, leaving truncation of the infinite tail as
// the only term that matters.
template <typename F>
double simpson(F f, double lo, double hi, int panels) {
    const double h = (hi - lo) / panels;
    double total = f(lo) + f(hi);
    for (int i = 1; i < panels; ++i) {
        total += (i % 2 == 1 ? 4.0 : 2.0) * f(lo + static_cast<double>(i) * h);
    }
    return total * h / 3.0;
}

// L{f}(s) = integral from 0 to infinity of f(t) e^{-st} dt. Callers pick an s far
// enough above the growth rate of f that the integrand is negligible at t = 30.
double laplace_numeric(const SymExpr& f, const std::string& t, double s) {
    return simpson([&](double x) { return at(f, t, x) * std::exp(-s * x); }, 0.0, 30.0, 60000);
}

// F[f](omega) = integral over the line of f(t) e^{-i omega t} dt. Every row of this
// table is even, so the imaginary part vanishes and the cosine integral is the whole
// transform.
double fourier_numeric(const SymExpr& f, const std::string& t, double omega) {
    return simpson([&](double x) { return at(f, t, x) * std::cos(omega * x); }, -30.0, 30.0,
                   120000);
}

// The module writes a two-sided decay without an absolute value: exp(-a*t) in its
// output means exp(-a*|t|), as the source comments say. A numeric check therefore
// cannot evaluate such a result directly -- the integral of the literal expression
// diverges -- so it reads the two parameters back out by evaluation and rebuilds the
// intended function itself. Reading them rather than pattern-matching keeps the test
// independent of how the result happens to be spelled.
struct Decay {
    double scale = 0.0;
    double rate = 0.0;
};

Decay read_decay(const SymExpr& expr, const std::string& var) {
    const double zero = at(expr, var, 0.0);
    const double one = at(expr, var, 1.0);
    return {zero, -std::log(one / zero)};
}

SymExpr clone_for_solve(const SymExpr& expr) {
    // sym_solve_linear takes its equations by const reference but SymExpr is
    // move-only, so a case that needs the expression twice has to copy it.
    auto reparsed = sym_parse(sym_to_string(expr));
    EXPECT_TRUE(reparsed.has_value());
    if (!reparsed.has_value()) {
        return sym_const(0.0);
    }
    return std::move(*reparsed);
}

void expect_supported(const SymExpr& result, const std::string& var, const std::string& label) {
    ASSERT_FALSE(sym_is_unsupported(result, var))
        << label << " declined: " << sym_to_string(result);
}

} // namespace

// ---------------------------------------------------------------------------
// Antiderivatives: differentiate the answer and compare with the integrand.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, AntiderivativesDifferentiateBackToTheIntegrand) {
    struct Case {
        const char* integrand;
        double lo;
        double hi;
    };
    // Sample points avoid the singular or branch points of each integrand, which is
    // why the range is per-case rather than shared.
    const Case cases[] = {
        {"1/x", 0.5, 4.0},          {"exp(x)", -1.0, 2.0},
        {"sqrt(x)", 0.25, 4.0},     {"x/2", -2.0, 2.0},
        {"x^(-2)", 0.5, 4.0},       {"sin(2*x)", -2.0, 2.0},
        {"cos(3*x)", -2.0, 2.0},    {"exp(2*x)", -1.0, 1.5},
        {"exp(-x)", -1.0, 2.0},     {"1/(2*x+1)", 0.5, 4.0},
        {"log(x)", 0.5, 4.0},       {"tan(x)", -1.0, 1.0},
        {"(x+1)^2", -2.0, 2.0},     {"-3*x^2", -2.0, 2.0},
        {"x^3", -2.0, 2.0},         {"1/(x^2)", 0.5, 4.0},
        {"sin(x)", -2.0, 2.0},      {"cos(x)", -2.0, 2.0},
        {"5", -2.0, 2.0},           {"3*x^2 + 2*x", -2.0, 2.0},
        {"sqrt(2*x+1)", 0.25, 3.0}, {"exp(-2*x)", -1.0, 1.5},
        {"1/(x+3)", 0.5, 4.0},      {"x^(-3)", 0.5, 4.0},
    };
    for (const Case& c : cases) {
        const SymExpr integrand = parse_or_die(c.integrand);
        const SymExpr antiderivative = sym_integrate(integrand, "x");
        ASSERT_FALSE(sym_is_unsupported(antiderivative, "x"))
            << "sym_integrate declined " << c.integrand;
        const SymExpr derivative = sym_diff(sym_integrate(integrand, "x"), "x");
        for (int i = 0; i <= 8; ++i) {
            const double x = c.lo + (c.hi - c.lo) * static_cast<double>(i) / 8.0;
            const double want = at(integrand, "x", x);
            const double got = at(derivative, "x", x);
            EXPECT_NEAR(got, want, 1e-9 * std::max(1.0, std::abs(want)))
                << "d/dx of the antiderivative of " << c.integrand << " at x=" << x;
        }
    }
}

TEST(SymbolicTables, IntegrationStillDeclinesWhatItCannotDo) {
    // A genuine chain rule and a product of two var-dependent factors: neither is in
    // the table, and neither should silently produce something.
    for (const char* text : {"exp(x^2)", "sin(x^2)", "x*sin(x)", "sin(x)*cos(x)", "exp(1/x)"}) {
        const SymExpr f = parse_or_die(text);
        EXPECT_TRUE(sym_is_unsupported(sym_integrate(f, "x"), "x"))
            << text << " was not declined: " << sym_to_string(sym_integrate(f, "x"));
    }
}

// ---------------------------------------------------------------------------
// Forward Laplace: integrate the definition.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, ForwardLaplaceMatchesTheDefiningIntegral) {
    struct Case {
        const char* f;
        double s;
    };
    const Case cases[] = {
        {"t*exp(2*t)", 5.0},      {"exp(-t)*sin(3*t)", 2.0},  {"exp(2*t)*cos(3*t)", 5.0},
        {"-5*exp(2*t)", 5.0},     {"t/2", 2.0},               {"t*sin(2*t)", 2.0},
        {"t*cos(3*t)", 2.0},      {"t^2*exp(-t)", 2.0},       {"(t+1)^2", 2.0},
        {"t^2*sin(t)", 2.0},      {"exp(-3*t)", 1.0},         {"sin(2*t)", 1.5},
        {"cos(2*t)", 1.5},        {"t^3", 2.0},               {"exp(-t)*cos(2*t)", 2.0},
        {"3*t*exp(-2*t)", 2.0},   {"t^2/4", 2.0},             {"5", 2.0},
        {"t + t*exp(-t)", 3.0},   {"t^2*exp(-t)/2", 2.0},
    };
    for (const Case& c : cases) {
        const SymExpr f = parse_or_die(c.f);
        const SymExpr transformed = sym_laplace(f, "t", "s");
        ASSERT_FALSE(sym_is_unsupported(transformed, "t")) << "L{" << c.f << "} declined";
        for (const double s : {c.s, c.s + 1.0, c.s + 2.5}) {
            const double want = laplace_numeric(f, "t", s);
            const double got = at(transformed, "s", s);
            EXPECT_NEAR(got, want, 1e-6 * std::max(1.0, std::abs(want)))
                << "L{" << c.f << "} at s=" << s << " gave " << sym_to_string(transformed);
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse Laplace: forward-transform the answer and compare with the input.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, InverseLaplaceForwardTransformsBackToItsInput) {
    struct Case {
        const char* spectrum;
        double s;
    };
    const Case cases[] = {
        {"1/(s^2+4)", 2.0},        {"5/(s^2+9)", 2.0},        {"(2*s)/(s^2+4)", 2.0},
        {"1/(s-2)^2", 5.0},        {"3/(s-1)^3", 4.0},        {"1/(s^2-4)", 5.0},
        {"s/(s^2-4)", 5.0},        {"2/((s-1)^2+4)", 4.0},    {"(s-2)/((s-2)^2+9)", 5.0},
        {"(s+1)/(s^2+1)", 2.0},    {"1/(2*s)", 2.0},          {"1/(3*s^2)", 2.0},
        {"1/s", 2.0},              {"1/(s+3)", 1.0},          {"2/s^3", 2.0},
        {"1/(s^2+1)", 2.0},        {"s/(s^2+1)", 2.0},        {"4/(s^2+4)", 2.0},
        {"1/((s+1)^2+9)", 2.0},    {"6/(s-1)^4", 4.0},
    };
    for (const Case& c : cases) {
        const SymExpr spectrum = parse_or_die(c.spectrum);
        const SymExpr inverted = sym_ilaplace(spectrum, "s", "t");
        ASSERT_FALSE(sym_is_unsupported(inverted, "s")) << c.spectrum << " declined";
        for (const double s : {c.s, c.s + 1.0, c.s + 2.5}) {
            const double want = at(spectrum, "s", s);
            const double got = laplace_numeric(inverted, "t", s);
            EXPECT_NEAR(got, want, 1e-6 * std::max(1.0, std::abs(want)))
                << c.spectrum << " inverted to " << sym_to_string(inverted) << ", whose transform"
                << " at s=" << s << " is not the input";
        }
    }
}

TEST(SymbolicTables, InverseLaplaceDeclinesDistributions) {
    // A numerator that leaves a bare constant after division is a delta: a
    // distribution, not a function. Returning any function for it would be wrong.
    for (const char* text : {"1", "s/(s-2)", "s", "(s^2+1)/(s^2+4)"}) {
        const SymExpr spectrum = parse_or_die(text);
        const SymExpr inverted = sym_ilaplace(spectrum, "s", "t");
        EXPECT_TRUE(sym_is_unsupported(inverted, "s"))
            << text << " was not declined: " << sym_to_string(inverted);
    }
}

TEST(SymbolicTables, LaplacePairRoundTrips) {
    for (const char* text : {"exp(-3*t)", "sin(2*t)", "cos(2*t)", "t^2", "t*exp(-t)",
                             "exp(-t)*sin(3*t)", "exp(2*t)*cos(3*t)", "t^2*exp(-t)"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr forward = sym_laplace(f, "t", "s");
        expect_supported(forward, "t", std::string("L{") + text + "}");
        const SymExpr back = sym_ilaplace(forward, "s", "t");
        ASSERT_FALSE(sym_is_unsupported(back, "s"))
            << "round trip of " << text << " failed at the inverse: " << sym_to_string(forward);
        for (const double t : {0.25, 0.75, 1.5, 2.25}) {
            EXPECT_NEAR(at(back, "t", t), at(f, "t", t), 1e-9 * std::max(1.0, std::abs(at(f, "t", t))))
                << text << " round-tripped to " << sym_to_string(back);
        }
    }
}

// ---------------------------------------------------------------------------
// Fourier: integrate the definition over the whole line.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, ForwardFourierMatchesTheDefiningIntegral) {
    for (const char* text : {"exp(-(t^2))", "exp(-(2*t^2))", "exp(-2*t^2)", "exp(-t^2/2)",
                             "3*exp(-(2*t^2))", "exp(-(2*t^2))+exp(-(3*t^2))",
                             "exp(-(t^2))/4", "2*exp(-(t^2)) - exp(-(2*t^2))"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr spectrum = sym_fourier(f, "t", "w");
        ASSERT_FALSE(sym_is_unsupported(spectrum, "t")) << "F[" << text << "] declined";
        for (const double omega : {0.0, 0.5, 1.0, 2.0}) {
            const double want = fourier_numeric(f, "t", omega);
            const double got = at(spectrum, "w", omega);
            EXPECT_NEAR(got, want, 1e-6 * std::max(1.0, std::abs(want)))
                << "F[" << text << "] at w=" << omega << " gave " << sym_to_string(spectrum);
        }
    }
}

TEST(SymbolicTables, ForwardFourierOfALorentzianIsTheTwoSidedDecay) {
    // 2a/(a^2 + t^2) transforms to 2*pi*exp(-a|omega|). The integrand decays only as
    // 1/t^2, so the tail is truncated rather than negligible: integrating by parts
    // bounds what is left beyond T by g(T)/omega, which at T = 2000 is about 1e-6 --
    // hence the looser tolerance here than for the Gaussian rows.
    const SymExpr f = parse_or_die("4/(4+t^2)");
    const SymExpr spectrum = sym_fourier(f, "t", "w");
    ASSERT_FALSE(sym_is_unsupported(spectrum, "t")) << "the Lorentzian declined";
    const Decay decay = read_decay(spectrum, "w");
    for (const double omega : {0.5, 1.0, 2.0}) {
        const double want =
            simpson([&](double x) { return at(f, "t", x) * std::cos(omega * x); }, -2000.0, 2000.0,
                    200000);
        const double got = decay.scale * std::exp(-decay.rate * std::abs(omega));
        EXPECT_NEAR(got, want, 2e-5 * std::max(1.0, std::abs(want)))
            << "F[4/(4+t^2)] at w=" << omega << " gave " << sym_to_string(spectrum);
    }
}

TEST(SymbolicTables, InverseFourierOfALorentzianIsTheTwoSidedDecay) {
    // The dual direction, checked the same way: rebuild exp(-a|t|) from the reported
    // parameters and transform it forward numerically, which converges quickly.
    struct Case {
        const char* spectrum;
        double omega;
    };
    const Case cases[] = {{"1/(1+w^2)", 1.0}, {"2/(1+w^2)", 1.0}, {"4/(4+w^2)", 1.0},
                          {"1/(4+w^2)", 2.0}, {"6/(9+w^2)", 1.5}};
    for (const Case& c : cases) {
        const SymExpr spectrum = parse_or_die(c.spectrum);
        const SymExpr inverted = sym_ifourier(spectrum, "w", "t");
        ASSERT_FALSE(sym_is_unsupported(inverted, "w")) << c.spectrum << " declined";
        const Decay decay = read_decay(inverted, "t");
        for (const double omega : {0.0, 0.5, c.omega}) {
            const double want = at(spectrum, "w", omega);
            const double got = simpson(
                [&](double x) {
                    return decay.scale * std::exp(-decay.rate * std::abs(x)) * std::cos(omega * x);
                },
                -60.0, 60.0, 240000);
            EXPECT_NEAR(got, want, 1e-6 * std::max(1.0, std::abs(want)))
                << c.spectrum << " inverted to " << sym_to_string(inverted);
        }
    }
}

TEST(SymbolicTables, GaussianFourierPairRoundTripsThroughItsOwnPrintedOutput) {
    // The forward transform prints six decimals, so the spectrum a user copies back
    // out of the REPL carries a scale good to about 1e-6. The inverse used to demand
    // agreement to 1e-9 and so could not read the module's own output.
    for (const char* text : {"exp(-(t^2))", "exp(-(2*t^2))", "exp(-(0.5*t^2))"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr spectrum = sym_fourier(f, "t", "w");
        ASSERT_FALSE(sym_is_unsupported(spectrum, "t")) << "F[" << text << "] declined";
        const SymExpr reparsed = parse_or_die(sym_to_string(spectrum));
        const SymExpr back = sym_ifourier(reparsed, "w", "t");
        ASSERT_FALSE(sym_is_unsupported(back, "w"))
            << text << " did not round-trip: " << sym_to_string(spectrum);
        for (const double t : {0.0, 0.5, 1.0, 2.0}) {
            EXPECT_NEAR(at(back, "t", t), at(f, "t", t), 1e-5)
                << text << " round-tripped to " << sym_to_string(back);
        }
    }
}

// ---------------------------------------------------------------------------
// The sentinel must survive linearity.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, OneUnsupportedTermDeclinesTheWholeExpression) {
    // The linearity rules recurse into each operand and reassemble. Before the scan
    // became recursive, an unsupported operand left a d/dt subterm inside an
    // otherwise-plausible answer and the root-only check called it a success.
    struct Case {
        const char* text;
        const char* var;
        SymExpr (*apply)(const SymExpr&, const std::string&, const std::string&);
    };
    const Case transforms[] = {
        {"t + exp(t^2)", "t", &sym_laplace},
        {"exp(t^2) - t", "t", &sym_laplace},
        {"3*exp(t^2)", "t", &sym_laplace},
        {"exp(t^2)/2", "t", &sym_laplace},
        {"sin(t^2) + cos(t)", "t", &sym_laplace},
    };
    for (const Case& c : transforms) {
        const SymExpr f = parse_or_die(c.text);
        const SymExpr result = c.apply(f, c.var, "s");
        EXPECT_TRUE(sym_is_unsupported(result, c.var))
            << c.text << " leaked a sentinel: " << sym_to_string(result);
    }

    for (const char* text : {"x + exp(x^2)", "exp(x^2) - x", "3*exp(x^2)", "exp(x^2)/2",
                             "sin(x)*cos(x) + x"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr result = sym_integrate(f, "x");
        EXPECT_TRUE(sym_is_unsupported(result, "x"))
            << text << " leaked a sentinel: " << sym_to_string(result);
    }

    for (const char* text : {"1/(s^2+4) + 1", "1 + 1/s", "2/(s^5+1) + 1/s"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr result = sym_ilaplace(f, "s", "t");
        EXPECT_TRUE(sym_is_unsupported(result, "s"))
            << text << " leaked a sentinel: " << sym_to_string(result);
    }

    for (const char* text : {"exp(-(t^2)) + exp(t^3)", "3*exp(t^3)"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr result = sym_fourier(f, "t", "w");
        EXPECT_TRUE(sym_is_unsupported(result, "t"))
            << text << " leaked a sentinel: " << sym_to_string(result);
    }
}

// ---------------------------------------------------------------------------
// Operator precedence, which the tables sit on top of.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, UnarySignBindsLooserThanExponentiation) {
    // -x^n means -(x^n). Parsing it as (-x)^n is not a missing feature but a wrong
    // value: it flips the sign whenever n is even, and turns a fractional exponent
    // into a real root of a negative number, which is NaN.
    struct Case {
        const char* text;
        double at_three;
    };
    const Case cases[] = {
        {"-t^2", -9.0},        {"-t^3", -27.0},     {"-2^2", -4.0},
        {"-t^0.5", -std::sqrt(3.0)}, {"2-t^2", -7.0},     {"-(t^2)", -9.0},
        {"(-t)^2", 9.0},       {"-t*t", -9.0},      {"2^-2", 0.25},
        {"-t^-2", -1.0 / 9.0}, {"2^3^2", 512.0},    {"-t^2+t", -6.0},
    };
    for (const Case& c : cases) {
        const SymExpr parsed = parse_or_die(c.text);
        EXPECT_NEAR(at(parsed, "t", 3.0), c.at_three, 1e-12)
            << c.text << " parsed as " << sym_to_string(parsed);
    }
}

TEST(SymbolicTables, NegatedPowersReachTheTables) {
    // The precedence fix is what lets these spellings be recognised at all: with
    // (-t)^2 in the exponent the Gaussian is exp(+t^2/2), which has no transform, and
    // declining it was the right answer to the wrong question.
    const SymExpr gaussian = parse_or_die("exp(-t^2/2)");
    const SymExpr spectrum = sym_fourier(gaussian, "t", "w");
    ASSERT_FALSE(sym_is_unsupported(spectrum, "t")) << "exp(-t^2/2) declined";
    for (const double omega : {0.0, 0.5, 1.0, 2.0}) {
        const double want = fourier_numeric(gaussian, "t", omega);
        EXPECT_NEAR(at(spectrum, "w", omega), want, 1e-6 * std::max(1.0, std::abs(want)))
            << "F[exp(-t^2/2)] at w=" << omega;
    }
}

// ---------------------------------------------------------------------------
// Linear solving: a term it cannot read must not simply vanish.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, SolveLinearCarriesTermsFreeOfTheUnknowns) {
    // sin(y) and exp(a) are constants with respect to x. They used to be dropped --
    // not refused, dropped -- so "x + sin(y) - 1" solved to x = 1 and "x - exp(a)"
    // to x = -0.
    struct Case {
        const char* equation;
        const char* other_var;
        double other_value;
        double expected;
    };
    const Case cases[] = {
        {"x+sin(y)-1", "y", 0.3, 1.0 - std::sin(0.3)},
        {"x-exp(a)", "a", 0.7, std::exp(0.7)},
        {"x+2*cos(y)-4", "y", 1.1, 4.0 - 2.0 * std::cos(1.1)},
        {"x/a-1", "a", 2.5, 2.5},
    };
    for (const Case& c : cases) {
        const SymExpr equation = parse_or_die(c.equation);
        std::vector<SymExpr> equations;
        equations.push_back(clone_for_solve(equation));
        const auto solved = sym_solve_linear(equations, {"x"});
        ASSERT_TRUE(solved.has_value()) << c.equation << " was refused";
        const auto found = solved->find("x");
        ASSERT_NE(found, solved->end());
        EXPECT_NEAR(at(found->second, c.other_var, c.other_value), c.expected, 1e-12)
            << c.equation << " solved to " << sym_to_string(found->second);
    }
}

TEST(SymbolicTables, SolveLinearReadsFractionalAndBracketedCoefficients) {
    struct Case {
        const char* equation;
        double expected;
    };
    const Case cases[] = {
        {"x/2-3", 6.0},      {"2*(x+1)-8", 3.0},  {"(x+1)/2-3", 5.0},
        {"x/4+1", -4.0},     {"3*(x-2)-3", 3.0},  {"(2*x+1)/3-3", 4.0},
    };
    for (const Case& c : cases) {
        std::vector<SymExpr> equations;
        equations.push_back(parse_or_die(c.equation));
        const auto solved = sym_solve_linear(equations, {"x"});
        ASSERT_TRUE(solved.has_value()) << c.equation << " was refused";
        const auto found = solved->find("x");
        ASSERT_NE(found, solved->end());
        EXPECT_NEAR(sym_eval(found->second, {}), c.expected, 1e-12)
            << c.equation << " solved to " << sym_to_string(found->second);
    }

    // A 2x2 system with the coefficients written as divisions.
    std::vector<SymExpr> system;
    system.push_back(parse_or_die("x/2+y/3-1"));
    system.push_back(parse_or_die("x-y"));
    const auto solved = sym_solve_linear(system, {"x", "y"});
    ASSERT_TRUE(solved.has_value()) << "the fractional 2x2 system was refused";
    EXPECT_NEAR(sym_eval(solved->at("x"), {}), 1.2, 1e-12);
    EXPECT_NEAR(sym_eval(solved->at("y"), {}), 1.2, 1e-12);
}

TEST(SymbolicTables, SolveLinearRefusesWhatIsNotLinear) {
    // Dropping the term it could not read turned these into a different, solvable
    // system. Refusing is the only correct answer.
    struct Case {
        const char* equation;
        std::vector<std::string> vars;
    };
    const Case cases[] = {
        {"x^2+x-1", {"x"}},
        {"x*x-4", {"x"}},
        {"exp(x)-2", {"x"}},
        {"sin(x)-x", {"x"}},
    };
    for (const Case& c : cases) {
        std::vector<SymExpr> equations;
        equations.push_back(parse_or_die(c.equation));
        const auto solved = sym_solve_linear(equations, c.vars);
        EXPECT_FALSE(solved.has_value())
            << c.equation << " was solved as if it were linear";
    }
}
