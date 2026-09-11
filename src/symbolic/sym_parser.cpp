// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#include "ms/symbolic/symbolic.hpp"
#include <numbers>

#include <charconv>
#include <cctype>
#include <cstdlib>
#include <string_view>

namespace ms {

namespace {

std::expected<SymExpr, SymParseError> sym_expr_ok(SymExpr expr) {
    return std::expected<SymExpr, SymParseError>(std::in_place, std::move(expr));
}

class SymParser {
public:
    explicit SymParser(const std::string& text) : text_(text) {}

    std::expected<SymExpr, SymParseError> parse() {
        skip_ws();
        if (at_end()) {
            return std::unexpected(SymParseError{"empty expression", 0});
        }
        auto expr = parse_expr();
        if (!expr) {
            return std::unexpected(expr.error());
        }
        skip_ws();
        if (!at_end()) {
            return std::unexpected(
                SymParseError{"unexpected trailing input", pos_});
        }
        return sym_expr_ok(std::move(*expr));
    }

private:
    // A recursive-descent parser recurses once per nesting level, so an input like
    // sin(sin(sin(...))) or x+x+x+... turns unbounded input depth into unbounded stack
    // use: sym_parse of 10000 nested calls, or 100000 chained additions, overflowed the
    // stack and crashed with SIGSEGV. The limit is far above any expression a person
    // writes and well below the depth that exhausts a default 8 MB stack.
    static constexpr int kMaxDepth = 256;

    // parse_add and parse_mul LOOP over their operands rather than recursing, so a chain
    // like x+x+x+... does not hit kMaxDepth -- but it still builds a left spine one node
    // deep per term, and ~SymExpr walks that spine recursively through its unique_ptr
    // children. 100000 chained additions therefore overflowed the stack on destruction
    // rather than during the parse. Bounding the node count bounds that spine too.
    static constexpr int kMaxNodes = 10000;

    const std::string& text_;
    size_t pos_ = 0;
    int depth_ = 0;
    int nodes_ = 0;

    // Called for every node the parse produces.
    bool budget_exhausted() { return ++nodes_ > kMaxNodes; }

    // Increments on construction and decrements on destruction, so every return path out
    // of a parse function unwinds the count.
    class DepthGuard {
      public:
        explicit DepthGuard(SymParser& p) : parser_(p) { ++parser_.depth_; }
        ~DepthGuard() { --parser_.depth_; }
        DepthGuard(const DepthGuard&) = delete;
        DepthGuard& operator=(const DepthGuard&) = delete;
        bool too_deep() const { return parser_.depth_ > kMaxDepth; }

      private:
        SymParser& parser_;
    };

    bool at_end() const { return pos_ >= text_.size(); }

    char peek() const {
        return at_end() ? '\0' : text_[pos_];
    }

    char get() {
        if (at_end()) {
            return '\0';
        }
        return text_[pos_++];
    }

    void skip_ws() {
        while (!at_end() && std::isspace(static_cast<unsigned char>(peek()))) {
            ++pos_;
        }
    }

    SymParseError make_error(const std::string& message) const {
        return SymParseError{message, pos_};
    }

    std::expected<void, SymParseError> expect(char ch, const std::string& message) {
        skip_ws();
        if (peek() != ch) {
            return std::unexpected(make_error(message));
        }
        ++pos_;
        return {};
    }

    std::expected<SymExpr, SymParseError> parse_expr() {
        DepthGuard guard(*this);
        if (guard.too_deep()) {
            return std::unexpected(make_error("expression nested too deeply"));
        }
        return parse_add();
    }

    std::expected<SymExpr, SymParseError> parse_add() {
        auto left = parse_mul();
        if (!left) {
            return std::unexpected(left.error());
        }
        while (true) {
            skip_ws();
            const char op = peek();
            if (op != '+' && op != '-') {
                break;
            }
            ++pos_;
            if (budget_exhausted()) {
                return std::unexpected(make_error("expression has too many terms"));
            }
            auto right = parse_mul();
            if (!right) {
                return std::unexpected(right.error());
            }
            if (op == '+') {
                left = sym_expr_ok(sym_add(std::move(*left), std::move(*right)));
            } else {
                left = sym_expr_ok(sym_sub(std::move(*left), std::move(*right)));
            }
        }
        return sym_expr_ok(std::move(*left));
    }

    std::expected<SymExpr, SymParseError> parse_mul() {
        auto left = parse_unary();
        if (!left) {
            return std::unexpected(left.error());
        }
        while (true) {
            skip_ws();
            const char op = peek();
            if (op != '*' && op != '/') {
                break;
            }
            ++pos_;
            if (budget_exhausted()) {
                return std::unexpected(make_error("expression has too many terms"));
            }
            auto right = parse_unary();
            if (!right) {
                return std::unexpected(right.error());
            }
            if (op == '*') {
                left = sym_expr_ok(sym_mul(std::move(*left), std::move(*right)));
            } else {
                left = sym_expr_ok(sym_div(std::move(*left), std::move(*right)));
            }
        }
        return sym_expr_ok(std::move(*left));
    }

    // Unary sign binds LOOSER than exponentiation: -x^2 is -(x^2), as it is in every
    // maths text and every CAS. Parsing it the other way round -- which is what this
    // grammar did when the unary level sat below the power level -- is not a missing
    // feature but a wrong answer with no error attached: -t^2 evaluated to +9 at
    // t = 3, -2^2 to 4, and -t^0.5 to NaN, because (-t)^0.5 is a real root of a
    // negative number. The exponent itself is still parsed as a unary, so 2^-3 keeps
    // working and ^ stays right-associative.
    std::expected<SymExpr, SymParseError> parse_unary() {
        DepthGuard guard(*this);
        if (guard.too_deep()) {
            return std::unexpected(make_error("expression nested too deeply"));
        }
        skip_ws();
        if (peek() == '+') {
            ++pos_;
            return parse_unary();
        }
        if (peek() == '-') {
            ++pos_;
            auto operand = parse_unary();
            if (!operand) {
                return std::unexpected(operand.error());
            }
            return sym_expr_ok(sym_neg(std::move(*operand)));
        }
        return parse_pow();
    }

    std::expected<SymExpr, SymParseError> parse_pow() {
        if (budget_exhausted()) {
            return std::unexpected(make_error("expression has too many terms"));
        }
        auto left = parse_primary();
        if (!left) {
            return std::unexpected(left.error());
        }
        skip_ws();
        if (peek() == '^') {
            ++pos_;
            if (budget_exhausted()) {
                return std::unexpected(make_error("expression has too many terms"));
            }
            auto right = parse_unary();
            if (!right) {
                return std::unexpected(right.error());
            }
            left = sym_expr_ok(sym_pow(std::move(*left), std::move(*right)));
        }
        return sym_expr_ok(std::move(*left));
    }

    std::expected<std::string, SymParseError> parse_identifier() {
        skip_ws();
        if (!std::isalpha(static_cast<unsigned char>(peek())) && peek() != '_') {
            return std::unexpected(make_error("expected identifier"));
        }
        const size_t start = pos_;
        while (std::isalnum(static_cast<unsigned char>(peek())) || peek() == '_') {
            ++pos_;
        }
        return text_.substr(start, pos_ - start);
    }

    std::expected<SymExpr, SymParseError> parse_number() {
        skip_ws();
        const size_t start = pos_;
        if (peek() == '.') {
            ++pos_;
            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                return std::unexpected(make_error("expected digits after decimal point"));
            }
            while (std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        } else if (std::isdigit(static_cast<unsigned char>(peek()))) {
            while (std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
            if (peek() == '.') {
                ++pos_;
                while (std::isdigit(static_cast<unsigned char>(peek()))) {
                    ++pos_;
                }
            }
        } else {
            return std::unexpected(make_error("expected number"));
        }

        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') {
                ++pos_;
            }
            if (!std::isdigit(static_cast<unsigned char>(peek()))) {
                return std::unexpected(make_error("expected exponent digits"));
            }
            while (std::isdigit(static_cast<unsigned char>(peek()))) {
                ++pos_;
            }
        }

        const std::string_view token(text_.data() + start, pos_ - start);
        double value = 0.0;
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec != std::errc{} || ptr != token.data() + token.size()) {
            return std::unexpected(make_error("invalid numeric literal"));
        }
        return sym_expr_ok(sym_const(value));
    }

    std::expected<SymExpr, SymParseError> parse_function_call(const std::string& name) {
        if (auto paren = expect('(', "expected '(' after function name"); !paren) {
            return std::unexpected(paren.error());
        }
        auto arg = parse_expr();
        if (!arg) {
            return std::unexpected(arg.error());
        }
        if (auto close = expect(')', "expected ')' after function argument"); !close) {
            return std::unexpected(close.error());
        }

        if (name == "sin") {
            return sym_expr_ok(sym_sin(std::move(*arg)));
        }
        if (name == "cos") {
            return sym_expr_ok(sym_cos(std::move(*arg)));
        }
        if (name == "tan") {
            return sym_expr_ok(sym_tan(std::move(*arg)));
        }
        if (name == "exp") {
            return sym_expr_ok(sym_exp(std::move(*arg)));
        }
        if (name == "log") {
            return sym_expr_ok(sym_log(std::move(*arg)));
        }
        if (name == "sqrt") {
            return sym_expr_ok(sym_sqrt(std::move(*arg)));
        }
        return std::unexpected(make_error("unknown function: " + name));
    }

    std::expected<SymExpr, SymParseError> parse_primary() {
        DepthGuard guard(*this);
        if (guard.too_deep()) {
            return std::unexpected(make_error("expression nested too deeply"));
        }
        if (budget_exhausted()) {
            return std::unexpected(make_error("expression has too many terms"));
        }
        skip_ws();
        if (peek() == '(') {
            ++pos_;
            auto expr = parse_expr();
            if (!expr) {
                return std::unexpected(expr.error());
            }
            if (auto close = expect(')', "expected ')'"); !close) {
                return std::unexpected(close.error());
            }
            return sym_expr_ok(std::move(*expr));
        }

        if (std::isdigit(static_cast<unsigned char>(peek())) || peek() == '.') {
            return parse_number();
        }

        if (std::isalpha(static_cast<unsigned char>(peek())) || peek() == '_') {
            auto ident = parse_identifier();
            if (!ident) {
                return std::unexpected(ident.error());
            }
            skip_ws();
            if (peek() == '(') {
                return parse_function_call(*ident);
            }
            // pi and e are constants, not free variables. Parsed as variables they
            // reached sym_eval unbound, and an unbound variable evaluates to zero --
            // so sym_eval("pi") returned 0.000000 and sym_eval("2*pi*r") returned 0
            // for every r, with nothing reporting that a symbol was missing. It also
            // meant the transform tables, which compare a coefficient against
            // std::numbers::pi, could not match an expression a user had typed pi into.
            if (*ident == "pi") {
                return sym_expr_ok(sym_const(std::numbers::pi));
            }
            if (*ident == "e") {
                return sym_expr_ok(sym_const(std::numbers::e));
            }
            return sym_expr_ok(sym_var(std::move(*ident)));
        }

        if (at_end()) {
            return std::unexpected(make_error("unexpected end of expression"));
        }
        return std::unexpected(make_error(std::string("invalid character: ") + peek()));
    }
};

} // namespace

std::expected<SymExpr, SymParseError> sym_parse(const std::string& text) {
    return SymParser(text).parse();
}

} // namespace ms
