#include "ms/symbolic/symbolic.hpp"

#include <cctype>
#include <cstdlib>

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
    const std::string& text_;
    size_t pos_ = 0;

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

    std::expected<SymExpr, SymParseError> parse_expr() { return parse_add(); }

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
        auto left = parse_pow();
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
            auto right = parse_pow();
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

    std::expected<SymExpr, SymParseError> parse_pow() {
        auto left = parse_unary();
        if (!left) {
            return std::unexpected(left.error());
        }
        skip_ws();
        if (peek() == '^') {
            ++pos_;
            auto right = parse_pow();
            if (!right) {
                return std::unexpected(right.error());
            }
            left = sym_expr_ok(sym_pow(std::move(*left), std::move(*right)));
        }
        return sym_expr_ok(std::move(*left));
    }

    std::expected<SymExpr, SymParseError> parse_unary() {
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
        return parse_primary();
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

        const std::string token = text_.substr(start, pos_ - start);
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) {
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
            return sym_expr_ok(sym_var(*ident));
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
