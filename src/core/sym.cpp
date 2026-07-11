// MathScript Symbolic Expression Implementation

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>

#include "ms/core/sym.hpp"

namespace ms {
namespace {

struct ExprNode {
    virtual ~ExprNode() = default;
    virtual double eval(const std::map<std::string, double>& env) const = 0;
};

struct NumberNode final : ExprNode {
    double value;

    explicit NumberNode(double v) : value(v) {}

    double eval(const std::map<std::string, double>&) const override { return value; }
};

struct VarNode final : ExprNode {
    std::string name;

    explicit VarNode(std::string n) : name(std::move(n)) {}

    double eval(const std::map<std::string, double>& env) const override {
        const auto it = env.find(name);
        return it != env.end() ? it->second : 0.0;
    }
};

struct UnaryNode final : ExprNode {
    char op = '-';
    std::unique_ptr<ExprNode> child;

    double eval(const std::map<std::string, double>& env) const override {
        const double v = child->eval(env);
        return op == '-' ? -v : v;
    }
};

struct BinaryNode final : ExprNode {
    char op = '+';
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;

    double eval(const std::map<std::string, double>& env) const override {
        const double a = left->eval(env);
        const double b = right->eval(env);
        switch (op) {
        case '+':
            return a + b;
        case '-':
            return a - b;
        case '*':
            return a * b;
        case '/':
            return b == 0.0 ? 0.0 : a / b;
        default:
            return 0.0;
        }
    }
};

struct FuncNode final : ExprNode {
    std::string name;
    std::unique_ptr<ExprNode> arg;

    double eval(const std::map<std::string, double>& env) const override {
        const double v = arg->eval(env);
        if (name == "sin") {
            return std::sin(v);
        }
        if (name == "cos") {
            return std::cos(v);
        }
        if (name == "exp") {
            return std::exp(v);
        }
        if (name == "log") {
            return v > 0.0 ? std::log(v) : 0.0;
        }
        if (name == "sqrt") {
            return v >= 0.0 ? std::sqrt(v) : 0.0;
        }
        if (name == "tanh") {
            return std::tanh(v);
        }
        return 0.0;
    }
};

bool is_function_name(const std::string& name) {
    return name == "sin" || name == "cos" || name == "exp" || name == "log" || name == "sqrt"
           || name == "tanh";
}

class Parser {
public:
    explicit Parser(const std::string& src) : src_(src) {}

    std::unique_ptr<ExprNode> parse() {
        auto node = parse_expr();
        skip_ws();
        if (pos_ != src_.size()) {
            failed_ = true;
            return nullptr;
        }
        return node;
    }

    bool failed() const { return failed_; }

private:
    const std::string& src_;
    size_t pos_ = 0;
    bool failed_ = false;

    void skip_ws() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
    }

    bool match(char c) {
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    std::unique_ptr<ExprNode> parse_expr() {
        auto left = parse_term();
        if (!left) {
            return nullptr;
        }
        while (true) {
            skip_ws();
            if (pos_ >= src_.size()) {
                break;
            }
            const char op = src_[pos_];
            if (op != '+' && op != '-') {
                break;
            }
            ++pos_;
            auto right = parse_term();
            if (!right) {
                failed_ = true;
                return nullptr;
            }
            auto bin = std::make_unique<BinaryNode>();
            bin->op = op;
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<ExprNode> parse_term() {
        auto left = parse_unary();
        if (!left) {
            return nullptr;
        }
        while (true) {
            skip_ws();
            if (pos_ >= src_.size()) {
                break;
            }
            const char op = src_[pos_];
            if (op != '*' && op != '/') {
                break;
            }
            ++pos_;
            auto right = parse_unary();
            if (!right) {
                failed_ = true;
                return nullptr;
            }
            auto bin = std::make_unique<BinaryNode>();
            bin->op = op;
            bin->left = std::move(left);
            bin->right = std::move(right);
            left = std::move(bin);
        }
        return left;
    }

    std::unique_ptr<ExprNode> parse_unary() {
        skip_ws();
        if (match('-')) {
            auto child = parse_unary();
            if (!child) {
                failed_ = true;
                return nullptr;
            }
            auto node = std::make_unique<UnaryNode>();
            node->child = std::move(child);
            return node;
        }
        return parse_primary();
    }

    std::unique_ptr<ExprNode> parse_primary() {
        skip_ws();
        if (pos_ >= src_.size()) {
            failed_ = true;
            return nullptr;
        }

        if (match('(')) {
            auto node = parse_expr();
            if (!node || !match(')')) {
                failed_ = true;
                return nullptr;
            }
            return node;
        }

        if (std::isdigit(static_cast<unsigned char>(src_[pos_]))
            || (src_[pos_] == '.' && pos_ + 1 < src_.size()
                && std::isdigit(static_cast<unsigned char>(src_[pos_ + 1])))) {
            char* end = nullptr;
            const double value = std::strtod(src_.c_str() + pos_, &end);
            if (end == src_.c_str() + pos_) {
                failed_ = true;
                return nullptr;
            }
            pos_ = static_cast<size_t>(end - src_.c_str());
            return std::make_unique<NumberNode>(value);
        }

        if (std::isalpha(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_') {
            const size_t start = pos_;
            while (pos_ < src_.size()
                   && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
                ++pos_;
            }
            const std::string ident = src_.substr(start, pos_ - start);
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == '(' && is_function_name(ident)) {
                ++pos_;
                auto arg = parse_expr();
                if (!arg || !match(')')) {
                    failed_ = true;
                    return nullptr;
                }
                auto node = std::make_unique<FuncNode>();
                node->name = ident;
                node->arg = std::move(arg);
                return node;
            }
            return std::make_unique<VarNode>(ident);
        }

        failed_ = true;
        return nullptr;
    }
};

} // namespace

Sym::Sym(const char* expr) {
    expr_ = expr ? expr : "";
    has_value_ = false;
}

Sym::Sym(double value) {
    value_ = value;
    expr_ = "";
    has_value_ = true;
}

std::string Sym::wrap_operand(const Sym& sym) {
    if (sym.has_value_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", sym.value_);
        return buf;
    }
    if (sym.expr_.empty()) {
        return "(0)";
    }
    return "(" + sym.expr_ + ")";
}

Sym& Sym::operator+=(const Sym& other) {
    expr_ = wrap_operand(*this) + "+" + wrap_operand(other);
    has_value_ = false;
    return *this;
}

Sym& Sym::operator-=(const Sym& other) {
    expr_ = wrap_operand(*this) + "-" + wrap_operand(other);
    has_value_ = false;
    return *this;
}

Sym& Sym::operator*=(const Sym& other) {
    expr_ = wrap_operand(*this) + "*" + wrap_operand(other);
    has_value_ = false;
    return *this;
}

Sym& Sym::operator/=(const Sym& other) {
    expr_ = wrap_operand(*this) + "/" + wrap_operand(other);
    has_value_ = false;
    return *this;
}

Sym Sym::operator+(const Sym& other) const {
    Sym result = *this;
    result += other;
    return result;
}

Sym Sym::operator-(const Sym& other) const {
    Sym result = *this;
    result -= other;
    return result;
}

Sym Sym::operator*(const Sym& other) const {
    Sym result = *this;
    result *= other;
    return result;
}

Sym Sym::operator/(const Sym& other) const {
    Sym result = *this;
    result /= other;
    return result;
}

double Sym::eval(const std::map<std::string, double>& env) const {
    if (has_value_) {
        return value_;
    }
    if (expr_.empty()) {
        return 0.0;
    }
    Parser parser(expr_);
    const std::unique_ptr<ExprNode> tree = parser.parse();
    if (!tree || parser.failed()) {
        return 0.0;
    }
    return tree->eval(env);
}

std::string Sym::to_string() const {
    if (has_value_) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", value_);
        return buf;
    }
    return expr_;
}

} // namespace ms
