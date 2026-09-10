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
#include <numbers>

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

// ---------------------------------------------------------------------------
// Separable ODEs, checked by substituting the solution back into the equation.
// ---------------------------------------------------------------------------

namespace {

// Residual of dy/dx = rhs(x, y) at one point, with y' taken by central difference.
// Checking the solution against the equation, rather than against an expected closed
// form, keeps these tests independent of which of several equivalent forms the solver
// happens to produce -- C*exp(log(x)) and C*x are the same solution.
double ode_residual(const SymExpr& rhs, const SymExpr& solution, double x, double c) {
    const std::map<std::string, double> params{{"C", c}, {"k", 3.0}};
    auto value_at = [&](double point) {
        std::map<std::string, double> env = params;
        env["x"] = point;
        return sym_eval(solution, env);
    };
    const double h = 1e-6;
    const double derivative = (value_at(x + h) - value_at(x - h)) / (2.0 * h);
    std::map<std::string, double> env = params;
    env["x"] = x;
    env["y"] = value_at(x);
    return derivative - sym_eval(rhs, env);
}

} // namespace

TEST(SymbolicTables, SeparableOdesSolveTheEquationTheyWereGiven) {
    // Every one of these declined before. The exponential ones were refused because
    // only the multiplied spelling k*y was matched and not the divided one y/k, though
    // a time constant is the ordinary way to write such an equation; the power ones
    // because a negative exponent parses as Neg(Const) and never reached the rule the
    // header's own table row promises for every n != 1.
    const char* equations[] = {
        "y/2",   "-y/2",    "y/x",      "y^(-1)", "1/y",      "y/(x^2)",
        "y^(-2)", "y/k",    "1/(2*y)",  "3/y^2",  "y/(2*x)",  "2*y",
    };
    for (const char* text : equations) {
        const SymExpr rhs = parse_or_die(text);
        const SymExpr solution = sym_dsolve(rhs, "x", "y");
        ASSERT_FALSE(sym_is_unsupported(solution, "x")) << "dy/dx = " << text << " declined";
        for (const double x : {0.7, 1.3, 2.1}) {
            for (const double c : {1.5, 2.0}) {
                const double residual = ode_residual(rhs, solution, x, c);
                EXPECT_NEAR(residual, 0.0, 1e-4)
                    << "dy/dx = " << text << " solved to " << sym_to_string(solution)
                    << ", which does not satisfy it at x=" << x << ", C=" << c;
            }
        }
    }
}

TEST(SymbolicTables, SeparableOdesStillDeclineWhatTheyCannotSolve) {
    // y^1 is the case the power rule excludes (it is the linear equation, handled
    // elsewhere), and the rest are genuinely outside the separable table.
    for (const char* text : {"y*sin(y)", "exp(x^2)", "y^2+y", "y*x*y"}) {
        const SymExpr rhs = parse_or_die(text);
        const SymExpr solution = sym_dsolve(rhs, "x", "y");
        EXPECT_TRUE(sym_is_unsupported(solution, "x"))
            << "dy/dx = " << text << " was answered: " << sym_to_string(solution);
    }
}

// ---------------------------------------------------------------------------
// Mellin: integrate the defining integral, on a mesh that reaches both ends.
// ---------------------------------------------------------------------------

namespace {

// M{f}(s) = integral from 0 to infinity of t^(s-1) f(t) dt, evaluated numerically.
//
// The integral runs over a half-line and its integrand is singular at both ends, so it
// is split at 1 and each half is substituted onto [0, 1] in a way that removes the
// singularity exactly rather than truncating it:
//
//   near 0    f(t) ~ t^alpha,  so with q = s + alpha > 0 and t = v^(1/q),
//             integral from 0 to 1 of t^(s-1) f(t) dt = (1/q) * integral of f(v^(1/q)) v^(-alpha/q) dv
//   near inf  f(t) ~ t^(-beta), so with p = beta - s > 0 and t = 1/w^(1/p),
//             integral from 1 to inf = (1/p) * integral of f(1/w^(1/p)) w^(-beta/p) dw
//
// Both results are bounded on (0, 1]. A further v = z^4 grading concentrates the mesh
// at the origin, which the power cases do not need and the logarithmic one does.
double mellin_numeric(const SymExpr& f, const std::string& t, double s, double alpha,
                      double beta) {
    const double q = s + alpha;
    const double p = beta - s;
    EXPECT_GT(q, 0.0) << "the near-zero exponent puts s outside the strip of convergence";
    EXPECT_GT(p, 0.0) << "the decay exponent puts s outside the strip of convergence";
    const int panels = 200000;
    const double head = simpson(
        [&](double z) {
            const double v = z * z * z * z;
            if (v <= 0.0) {
                return 0.0;
            }
            return 4.0 * z * z * z * at(f, t, std::pow(v, 1.0 / q)) * std::pow(v, -alpha / q);
        },
        0.0, 1.0, panels);
    const double tail = simpson(
        [&](double z) {
            const double w = z * z * z * z;
            if (w <= 0.0) {
                return 0.0;
            }
            return 4.0 * z * z * z * at(f, t, 1.0 / std::pow(w, 1.0 / p)) * std::pow(w, -beta / p);
        },
        0.0, 1.0, panels);
    return head / q + tail / p;
}

} // namespace

TEST(SymbolicTables, MellinEntriesMatchTheDefiningIntegral) {
    struct Case {
        const char* f;
        double s;
        double alpha;  // f ~ t^alpha as t -> 0
        double beta;   // f ~ t^(-beta) as t -> infinity
    };
    // Every one of these declined before: the matcher required the numerator to be
    // exactly 1, the constant exactly 1 and the power exactly t, so linearity, the
    // scaling rule and the whole 1/(1+t^n) column were each refused.
    const Case cases[] = {
        {"1/(1+t)", 0.5, 0.0, 1.0},      {"3/(1+t)", 0.5, 0.0, 1.0},
        {"1/(2+t)", 0.5, 0.0, 1.0},      {"1/(1+t^2)", 1.0, 0.0, 2.0},
        {"1/(1+t^3)", 1.5, 0.0, 3.0},    {"1/(1+t)^2", 0.5, 0.0, 2.0},
        {"t/(1+t)", -0.5, 1.0, 0.0},     {"t^2/(1+t^3)", 0.5, 2.0, 1.0},
        {"2/(3+t^2)", 1.0, 0.0, 2.0},    {"log(1+t)", -0.5, 1.0, 0.0},
    };
    for (const Case& c : cases) {
        const SymExpr f = parse_or_die(c.f);
        const SymExpr transformed = sym_mellin(f, "t", "s");
        ASSERT_FALSE(sym_is_unsupported(transformed, "t")) << "M{" << c.f << "} declined";
        const double want = mellin_numeric(f, "t", c.s, c.alpha, c.beta);
        const double got = at(transformed, "s", c.s);
        EXPECT_NEAR(got, want, 1e-5 * std::max(1.0, std::abs(want)))
            << "M{" << c.f << "} at s=" << c.s << " gave " << sym_to_string(transformed);
    }
}

TEST(SymbolicTables, MellinPairRoundTripsThroughItsOwnOutput) {
    // pi used to parse as a free variable, so sym_imellin("pi/sin(pi*s)") -- a verbatim
    // row of the module's own documented inverse table -- could not be typed at all.
    const SymExpr spectrum = parse_or_die("pi/sin(pi*s)");
    const SymExpr back = sym_imellin(spectrum, "s", "t");
    ASSERT_FALSE(sym_is_unsupported(back, "s")) << "the reflection row declined";
    for (const double t : {0.5, 1.0, 2.5}) {
        EXPECT_NEAR(at(back, "t", t), 1.0 / (1.0 + t), 1e-12);
    }
}

TEST(SymbolicTables, PiAndEAreConstantsNotFreeVariables) {
    // An unbound variable evaluates to zero, so parsing pi as one meant sym_eval("pi")
    // returned 0.000000 and every formula a user wrote pi into was silently wrong.
    EXPECT_NEAR(sym_eval(parse_or_die("pi"), {}), std::numbers::pi, 1e-12);
    EXPECT_NEAR(sym_eval(parse_or_die("e"), {}), std::numbers::e, 1e-12);
    EXPECT_NEAR(sym_eval(parse_or_die("2*pi*r"), {{"r", 3.0}}), 6.0 * std::numbers::pi, 1e-12);
    EXPECT_NEAR(sym_eval(parse_or_die("sin(pi/2)"), {}), 1.0, 1e-12);
    // A name that merely starts with one of them is still a variable.
    EXPECT_NEAR(sym_eval(parse_or_die("pizza"), {{"pizza", 7.0}}), 7.0, 1e-12);
    EXPECT_NEAR(sym_eval(parse_or_die("ex"), {{"ex", 5.0}}), 5.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Hankel: the defining Bessel integral, and the self-inverse property.
// ---------------------------------------------------------------------------

namespace {

// J0, computed here rather than taken from <cmath>. The C++17 mathematical special
// functions are not implemented by Microsoft's standard library, so std::cyl_bessel_j
// does not compile on the MSVC job even though it is standard.
//
// Two regimes: the ascending series, exact to about 1e-11 where its terms do not
// cancel, and the standard asymptotic expansion beyond, good to about 1e-8 at the
// crossover and better further out. That is far inside what the quadrature needs -- the
// integrands it is used on decay like exp(-2r), so everything past the crossover
// contributes less than 1e-7 of the total. BesselJ0MatchesKnownValues pins both regimes
// against published values.
double bessel_j0(double x) {
    x = std::abs(x);
    if (x < 18.0) {
        const double quarter_square = 0.25 * x * x;
        double term = 1.0;
        double total = 1.0;
        for (int m = 1; m < 60; ++m) {
            term *= -quarter_square / (static_cast<double>(m) * static_cast<double>(m));
            total += term;
            if (std::abs(term) < 1e-18 * std::abs(total)) {
                break;
            }
        }
        return total;
    }
    // P and Q for nu = 0, i.e. mu = 4*nu^2 = 0, from
    //   P ~ 1 - (mu-1)(mu-9)/(2!(8x)^2) + (mu-1)(mu-9)(mu-25)(mu-49)/(4!(8x)^4) - ...
    //   Q ~   (mu-1)/(8x) - (mu-1)(mu-9)(mu-25)/(3!(8x)^3) + ...
    // giving 9/128, 11025/98304, 108056025/188743680 and 1/8, 225/3072, 893025/3932160.
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    const double p_series = 1.0 - (9.0 / 128.0) * inv2 + (11025.0 / 98304.0) * inv2 * inv2 -
                            (108056025.0 / 188743680.0) * inv2 * inv2 * inv2;
    const double q_series = -(1.0 / 8.0) * inv + (225.0 / 3072.0) * inv2 * inv -
                            (893025.0 / 3932160.0) * inv2 * inv2 * inv;
    const double phase = x - 0.25 * std::numbers::pi;
    return std::sqrt(2.0 / (std::numbers::pi * x)) *
           (p_series * std::cos(phase) - q_series * std::sin(phase));
}

// H0[f](k) = integral from 0 to infinity of f(r) J0(k r) r dr, evaluated numerically.
// Only rows whose integrand decays exponentially are checked this way; the algebraic
// ones oscillate too slowly for a truncated quadrature to say anything, and are pinned
// by the round trip below instead.
double hankel_numeric(const SymExpr& f, const std::string& r, double k) {
    // The lower limit is a whisker above zero rather than zero: the integrand of a row
    // like exp(-a*r)/r is finite there (the 1/r cancels against the r weight) but is
    // computed as inf * 0. What is skipped is of order 1e-9.
    return simpson([&](double x) { return at(f, r, x) * bessel_j0(k * x) * x; },
                   1e-9, 60.0, 240000);
}

} // namespace

TEST(SymbolicTables, HankelEntriesMatchTheDefiningBesselIntegral) {
    // Each of these declined before. exp(-a*r)/r is the Lipschitz integral, the
    // screened-Coulomb pair; exp(-a*r^2) is the most-cited order-0 pair of all.
    for (const char* text : {"exp(-2*r)", "exp(-2*r)/r", "exp(-2*r^2)", "r*exp(-2*r)",
                             "3*exp(-2*r)", "exp(-(3*r))/r", "exp(-0.5*r^2)"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr transformed = sym_hankel(f, "r", "k");
        ASSERT_FALSE(sym_is_unsupported(transformed, "r")) << "H0[" << text << "] declined";
        for (const double k : {0.5, 1.0, 2.0, 3.5}) {
            const double want = hankel_numeric(f, "r", k);
            EXPECT_NEAR(at(transformed, "k", k), want, 1e-6 * std::max(1.0, std::abs(want)))
                << "H0[" << text << "] at k=" << k << " gave " << sym_to_string(transformed);
        }
    }
}

TEST(SymbolicTables, HankelIsItsOwnInverse) {
    // The order-0 transform is self-inverse, so every forward row is an inverse row
    // with r and k swapped. The two directions had drifted apart: exp(-a*k) had no
    // inverse entry though its forward partner has always existed, exp(-2*k)/k was
    // refused for the spelling of its minus sign, and a row at any amplitude other
    // than the canonical one was refused outright.
    // r*exp(-a*r) is deliberately absent: its forward transform is a derivative of the
    // n = 0 row with respect to a, not a constant over a power of (a^2+k^2), so there
    // is no shape for the inverse to match. The matcher used to claim it, and inverted
    // it consistently with a forward formula that was wrong.
    for (const char* text : {"exp(-2*r)", "exp(-2*r)/r", "1/(r^2+4)^1.5", "1/r",
                             "exp(-2*r^2)"}) {
        const SymExpr f = parse_or_die(text);
        const SymExpr forward = sym_hankel(f, "r", "k");
        ASSERT_FALSE(sym_is_unsupported(forward, "r")) << "H0[" << text << "] declined";
        const SymExpr back = sym_ihankel(forward, "k", "r");
        ASSERT_FALSE(sym_is_unsupported(back, "k"))
            << text << " did not survive the round trip: " << sym_to_string(forward);
        for (const double r : {0.4, 1.0, 2.5}) {
            const double want = at(f, "r", r);
            EXPECT_NEAR(at(back, "r", r), want, 1e-6 * std::max(1.0, std::abs(want)))
                << text << " round-tripped to " << sym_to_string(back);
        }
    }
}

TEST(SymbolicTables, HankelStillDeclinesWhatNeedsASpecialFunction) {
    // 1/(r^2+a^2) is a Bessel K0 and 1 is a distribution; neither is representable in
    // SymExpr, so declining is the right answer, not a gap.
    for (const char* text : {"1/(r^2+4)", "log(r)", "sin(r)"}) {
        const SymExpr f = parse_or_die(text);
        EXPECT_TRUE(sym_is_unsupported(sym_hankel(f, "r", "k"), "r"))
            << text << " was answered: " << sym_to_string(sym_hankel(f, "r", "k"));
    }
}

// ---------------------------------------------------------------------------
// Limits, and the two ways the old estimator invented an answer.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, LimitsWithAOneSidedDomainAreNotZero) {
    // The refinement loop initialised its estimate to 0.0 and only assigned it when
    // BOTH sides evaluated finite. A function undefined on one side of the point never
    // satisfied that, fell through every iteration, and returned the initialiser:
    // sqrt(x) + 5 at 0 came back as 0.000000 where the answer is 5.
    struct Case {
        const char* f;
        double point;
        double expected;
    };
    const Case cases[] = {
        {"sqrt(x)+5", 0.0, 5.0},   {"sqrt(x)", 0.0, 0.0},      {"sqrt(x)*3+1", 0.0, 1.0},
        {"sqrt(x-2)+7", 2.0, 7.0},
    };
    for (const Case& c : cases) {
        EXPECT_NEAR(sym_limit(parse_or_die(c.f), "x", c.point), c.expected, 1e-6) << c.f;
    }
}

TEST(SymbolicTables, LimitsSurviveCatastrophicCancellation) {
    // The old loop drove the step to 1e-15, where (1 - cos(x))/x^2 evaluates to
    // (1 - 1)/1e-30 = 0, and returned that. Worse, once every sample is exactly zero
    // the successive differences are exactly zero too, which reads as perfect
    // convergence -- so the wrong value was returned confidently.
    struct Case {
        const char* f;
        double point;
        double expected;
    };
    const Case cases[] = {
        {"(1-cos(x))/x^2", 0.0, 0.5},   {"sin(x)/x", 0.0, 1.0},
        {"(exp(x)-1)/x", 0.0, 1.0},     {"tan(x)/x", 0.0, 1.0},
        {"(1-cos(x))/x", 0.0, 0.0},     {"(x^2-1)/(x-1)", 1.0, 2.0},
        {"x^2", 3.0, 9.0},              {"(sin(x)-x)/x^3", 0.0, -1.0 / 6.0},
    };
    for (const Case& c : cases) {
        EXPECT_NEAR(sym_limit(parse_or_die(c.f), "x", c.point), c.expected, 1e-5) << c.f;
    }
}

TEST(SymbolicTables, DivergentLimitsReportNoValue) {
    // log(x) at 0 marches off towards minus infinity. The old code returned whichever
    // sample it happened to stop on -- a finite number, indistinguishable from a real
    // limit. NaN is the honest answer, and the REPL turns it into an error.
    for (const char* text : {"log(x)", "1/x^2"}) {
        EXPECT_TRUE(std::isnan(sym_limit(parse_or_die(text), "x", 0.0)))
            << text << " returned a finite value";
    }
}

// ---------------------------------------------------------------------------
// Expansion terminates.
// ---------------------------------------------------------------------------

TEST(SymbolicTables, ExpansionCollectsLikeTermsAndStaysEqualToItsInput) {
    // Expansion used to multiply out over the tree without ever putting like terms
    // back together, so (x+1)^3 was eight products and (x+1)^8 was 256. Nothing
    // collected them, which is why ((x+1)^8)^8 -- 256^8 products distributed
    // pairwise -- never returned and the REPL had to be killed.
    struct Case {
        const char* text;
        int terms;
    };
    const Case cases[] = {
        {"(x+1)^3", 4},   {"(x+1)^2", 3},        {"(x+1)*(x+2)", 3},
        {"(x+1)^8", 9},   {"(x+y)*(x-y)", 2},    {"(x+1)^2*(x-3)", 4},
        {"x*y + x*z", 2}, {"x + x + x", 1},      {"(x+1)^2 - (x+1)^2", 1},
    };
    for (const Case& c : cases) {
        const SymExpr original = parse_or_die(c.text);
        const SymExpr expanded = sym_expand(parse_or_die(c.text));
        // Value first: whatever shape it takes, it has to be the same function.
        for (const double x : {0.3, 1.3, -0.7, 2.1}) {
            const double want = at(original, "x", x);
            EXPECT_NEAR(at(expanded, "x", x), want, 1e-9 * std::max(1.0, std::abs(want)))
                << "sym_expand(" << c.text << ") changed the value at x=" << x;
        }
        // Then the shape: one term per distinct monomial, counted by the separators
        // between them.
        const std::string printed = sym_to_string(expanded);
        int separators = 0;
        for (std::size_t i = 0; i + 2 < printed.size(); ++i) {
            if (printed[i] == ' ' && (printed[i + 1] == '+' || printed[i + 1] == '-') &&
                printed[i + 2] == ' ') {
                ++separators;
            }
        }
        EXPECT_EQ(separators + 1, c.terms) << c.text << " expanded to " << printed;
    }
}

TEST(SymbolicTables, NestedPowersExpandInsteadOfHanging) {
    // ((x+1)^8)^8 is (x+1)^64: sixty-five terms. Distributing pairwise it is 256^8
    // products, which is what did not terminate.
    const SymExpr expanded = sym_expand(parse_or_die("((x+1)^8)^8"));
    const std::string printed = sym_to_string(expanded);
    int separators = 0;
    for (std::size_t i = 0; i + 2 < printed.size(); ++i) {
        if (printed[i] == ' ' && (printed[i + 1] == '+' || printed[i + 1] == '-') &&
            printed[i + 2] == ' ') {
            ++separators;
        }
    }
    EXPECT_EQ(separators + 1, 65) << "expected the degree-64 binomial";
    // The leading coefficients are C(64, k).
    EXPECT_NE(printed.find("x ^ 64"), std::string::npos) << printed.substr(0, 120);
    EXPECT_NE(printed.find("2016.000000"), std::string::npos) << "C(64,2)";
    EXPECT_NE(printed.find("41664.000000"), std::string::npos) << "C(64,3)";

    // Value equality is asserted only where the expanded polynomial can actually be
    // evaluated. It is symbolically exact everywhere, and numerically unusable for
    // x < 0: at x = -0.7 the answer is 0.3^64 = 3.4e-34 while the largest term is
    // 5.6e13, so recovering it needs about 47 digits of cancellation and a double
    // carries 16. That is a property of the degree-64 polynomial, not of this
    // expansion -- evaluating the same coefficients in exact arithmetic and then
    // rounding shows the identical loss. For x > 0 every term is positive, nothing
    // cancels, and the two forms agree.
    const SymExpr original = parse_or_die("((x+1)^8)^8");
    for (const double x : {0.05, 0.3, 1.3, 2.1}) {
        const double want = at(original, "x", x);
        EXPECT_NEAR(at(expanded, "x", x), want, 1e-9 * std::max(1.0, std::abs(want)))
            << "at x=" << x;
    }
}

TEST(SymbolicTables, ExpansionDeclinesRatherThanReturningPartOfTheAnswer) {
    // Eight distinct variables to the eighth power is C(15,7) = 6435 monomials,
    // past the 4096-term ceiling. Past it, expansion hands back what it was given
    // rather than a truncated polynomial -- so the result is still the same
    // function, just not multiplied out.
    const char* wide = "(a+b+c+d+f+g+h+i)^8";
    const SymExpr original = parse_or_die(wide);
    const SymExpr expanded = sym_expand(parse_or_die(wide));
    const std::map<std::string, double> env{{"a", 1.1}, {"b", 0.7}, {"c", 1.3}, {"d", 0.2},
                                            {"f", 0.9}, {"g", 1.7}, {"h", 0.4}, {"i", 1.2}};
    const double want = sym_eval(original, env);
    EXPECT_NEAR(sym_eval(expanded, env), want, 1e-9 * std::max(1.0, std::abs(want)))
        << "declining to expand must not change the value";

    // A quotient by something that is not a number is left alone: cancelling x/x to
    // 1 would differ from x/x at x = 0, and nothing here can rule that point out.
    const SymExpr quotient = sym_expand(parse_or_die("x/x"));
    EXPECT_TRUE(std::isnan(at(quotient, "x", 0.0))) << sym_to_string(quotient);

    // And a fractional exponent is not pushed through a square: (x^2)^0.5 is |x|.
    const SymExpr root = sym_expand(parse_or_die("(x^2)^0.5"));
    EXPECT_NEAR(at(root, "x", -3.0), 3.0, 1e-12) << sym_to_string(root);
}

TEST(SymbolicTables, ExpansionKeepsDistinctAtomsApart) {
    // Atoms are identified by structure, not by printed form. sym_to_string renders
    // a constant with six decimals, so sin(1.0000001*x) and sin(1.0000002*x) print
    // identically; keying atoms by that text would merge them and expand their
    // difference to exactly zero.
    const SymExpr difference = sym_expand(parse_or_die("sin(1.0000001*x) - sin(1.0000002*x)"));
    const double at_two = at(difference, "x", 2.0);
    EXPECT_NE(at_two, 0.0) << "distinct atoms collapsed: " << sym_to_string(difference);
    EXPECT_NEAR(at_two, std::sin(1.0000001 * 2.0) - std::sin(1.0000002 * 2.0), 1e-15);

    // The same expression really does cancel when the atoms are the same.
    const SymExpr cancels = sym_expand(parse_or_die("sin(1.0000001*x) - sin(1.0000001*x)"));
    EXPECT_EQ(at(cancels, "x", 2.0), 0.0) << sym_to_string(cancels);
}

TEST(SymbolicTables, PrintedConstantsReadBackAsThemselves) {
    // sym_to_string went through std::to_string, which is printf("%f") -- six decimal
    // places and nothing else. A coefficient below 5e-7 printed as 0.000000 and simply
    // vanished from the expression; a large one gained a spurious ".000000" tail.
    const double values[] = {1e-9,  1e-7,   -2.5e-8, 1e20,  -1e18,
                             0.0,   2.0,    0.5,     -3.25, 1234.5};
    for (const double value : values) {
        // The value, not the node type: a negative literal reads back as Neg(Const),
        // which is the parser's shape for it and not a round-trip failure.
        const std::string text = sym_to_string(sym_const(value));
        EXPECT_DOUBLE_EQ(sym_eval(parse_or_die(text), {}), value)
            << value << " printed as " << text;
    }
    // The ordinary magnitudes keep the six-decimal spelling the corpus is written in.
    EXPECT_EQ(sym_to_string(sym_const(2.0)), "2.000000");
    EXPECT_EQ(sym_to_string(sym_const(0.5)), "0.500000");
}

TEST(SymbolicTables, BesselJ0MatchesKnownValues) {
    // The quadrature that checks the Hankel table is only as good as its J0, and this
    // one is hand-rolled because MSVC does not ship std::cyl_bessel_j. The reference
    // values were computed from the ascending series in 300-digit arithmetic, so they
    // are independent of the double-precision implementation being checked.
    struct Case {
        double x;
        double j0;
    };
    // Below the crossover: the ascending series.
    const Case series_cases[] = {
        {0.0, 1.0},
        {1.0, 0.76519768655796655},
        {2.404825557695773, 0.0},  // the first zero
        {5.0, -0.17759677131433830},
        {10.0, -0.24593576445134834},
        {15.0, -0.014224472826780773},
        {17.999999, -0.013355993716868143},
    };
    for (const Case& c : series_cases) {
        EXPECT_NEAR(bessel_j0(c.x), c.j0, 1e-9) << "J0(" << c.x << ")";
    }
    // Above it: the asymptotic expansion. 18.000001 and 17.999999 straddle the
    // crossover, so together they pin both regimes at effectively the same argument --
    // which is the real continuity check, since J0 itself moves by 3.8e-7 across that
    // interval and comparing the two sides to each other would only measure its slope.
    const Case asymptotic_cases[] = {
        {18.000001, -0.013355617727097167},
        {20.0, 0.16702466434058315},
        {50.0, 0.055812327669251815},
        {100.0, 0.019985850304223122},
        {150.0, -0.00077409037539429120},
        {210.0, -0.016170877385332898},
        {400.0, -0.038825181530783956},
    };
    for (const Case& c : asymptotic_cases) {
        EXPECT_NEAR(bessel_j0(c.x), c.j0, 1e-9) << "J0(" << c.x << ")";
    }
}
