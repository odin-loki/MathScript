#include "ms/distributed/mpi_context.hpp"
#include "ms/frameworks/axiom/axiom.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/version.hpp"
#include "ms/interp/plot_console.hpp"
#include "ms/core/operations.hpp"
#include "ms/fft/fft.hpp"
#include "ms/linalg/linalg.hpp"
#include "ms/runtime/dispatch.hpp"
#include "ms/runtime/load_balancer.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/simd/simd.hpp"
#include "ms/special/special.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>

namespace ms::interp {

namespace {

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool parse_number(const std::string& text, double& value) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end == text.c_str() + text.size();
}

std::optional<std::vector<std::string>> split_call_args(const std::string& cmd) {
    const auto open = cmd.find('(');
    if (open == std::string::npos) {
        return std::nullopt;
    }
    const auto close = cmd.rfind(')');
    if (close == std::string::npos || close <= open) {
        return std::nullopt;
    }
    std::vector<std::string> args;
    std::string current;
    int depth = 0;
    for (size_t i = open + 1; i < close; ++i) {
        const char c = cmd[i];
        if (c == '[') {
            ++depth;
            current += c;
        } else if (c == ']') {
            --depth;
            current += c;
        } else if (c == ',' && depth == 0) {
            args.push_back(trim_copy(current));
            current.clear();
        } else {
            current += c;
        }
    }
    args.push_back(trim_copy(current));
    return args;
}

bool is_identifier(const std::string& text) {
    if (text.empty()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!std::isalpha(first) && text.front() != '_') {
        return false;
    }
    for (unsigned char c : text) {
        if (!std::isalnum(c) && c != '_') {
            return false;
        }
    }
    return true;
}

bool parse_scalar_operand(const std::string& text, ScalarOperand& out) {
    const std::string token = trim_copy(text);
    double value = 0.0;
    if (parse_number(token, value)) {
        out.is_literal = true;
        out.literal = value;
        out.name.clear();
        return true;
    }
    if (is_identifier(token)) {
        out.is_literal = false;
        out.literal = 0.0;
        out.name = token;
        return true;
    }
    return false;
}

bool is_binary_minus(const std::string& expr, size_t index) {
    size_t j = index;
    while (j > 0 && std::isspace(static_cast<unsigned char>(expr[j - 1]))) {
        --j;
    }
    if (j == 0) {
        return false;
    }
    const char prev = expr[j - 1];
    return prev != '+' && prev != '-' && prev != '*' && prev != '/' && prev != '(';
}

std::optional<std::pair<size_t, char>> find_top_level_op(const std::string& expr, const char* ops) {
    int depth = 0;
    std::optional<std::pair<size_t, char>> last;
    for (size_t i = 0; i < expr.size(); ++i) {
        const char c = expr[i];
        if (c == '(') {
            ++depth;
        } else if (c == ')' && depth > 0) {
            --depth;
        } else if (depth == 0) {
            for (const char* p = ops; *p != '\0'; ++p) {
                if (c == *p) {
                    if (c == '-' && !is_binary_minus(expr, i)) {
                        continue;
                    }
                    last = std::pair{i, *p};
                }
            }
        }
    }
    return last;
}

std::optional<std::pair<size_t, char>> find_scalar_binop(const std::string& rhs) {
    if (auto add_sub = find_top_level_op(rhs, "+-")) {
        return add_sub;
    }
    return find_top_level_op(rhs, "*/");
}

std::string strip_outer_parens(std::string expr) {
    while (true) {
        expr = trim_copy(expr);
        if (expr.size() < 2 || expr.front() != '(' || expr.back() != ')') {
            return expr;
        }
        int depth = 0;
        bool wraps_all = true;
        for (size_t i = 0; i < expr.size(); ++i) {
            if (expr[i] == '(') {
                ++depth;
            } else if (expr[i] == ')') {
                --depth;
                if (depth == 0 && i + 1 != expr.size()) {
                    wraps_all = false;
                    break;
                }
            }
        }
        if (!wraps_all) {
            return expr;
        }
        expr = expr.substr(1, expr.size() - 2);
    }
}

bool contains_scalar_operator(const std::string& rhs) {
    return find_scalar_binop(rhs).has_value();
}

std::optional<std::pair<std::string, std::string>> parse_scalar_unary_call(const std::string& expr) {
    const std::string text = trim_copy(expr);
    const auto open = text.find('(');
    if (open == std::string::npos || open == 0) {
        return std::nullopt;
    }
    const std::string name = trim_copy(text.substr(0, open));
    if (!is_identifier(name)) {
        return std::nullopt;
    }

    int depth = 0;
    size_t close = std::string::npos;
    for (size_t i = open; i < text.size(); ++i) {
        if (text[i] == '(') {
            ++depth;
        } else if (text[i] == ')') {
            --depth;
            if (depth == 0) {
                close = i;
                break;
            }
        }
    }
    if (close == std::string::npos || close + 1 != text.size()) {
        return std::nullopt;
    }

    const std::string arg = trim_copy(text.substr(open + 1, close - open - 1));
    if (arg.empty()) {
        return std::nullopt;
    }
    return std::pair{name, arg};
}

std::vector<std::string> split_scalar_call_args(const std::string& args_text) {
    std::vector<std::string> args;
    std::string current;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (char c : args_text) {
        if (c == '(') {
            ++paren_depth;
            current += c;
        } else if (c == ')') {
            --paren_depth;
            current += c;
        } else if (c == '[') {
            ++bracket_depth;
            current += c;
        } else if (c == ']') {
            --bracket_depth;
            current += c;
        } else if (c == ',' && paren_depth == 0 && bracket_depth == 0) {
            args.push_back(trim_copy(current));
            current.clear();
        } else {
            current += c;
        }
    }
    args.push_back(trim_copy(current));
    return args;
}

std::optional<std::pair<std::string, std::vector<std::string>>> parse_scalar_call(
    const std::string& expr) {
    const auto unary = parse_scalar_unary_call(expr);
    if (!unary) {
        return std::nullopt;
    }

    std::vector<std::string> args = split_scalar_call_args(unary->second);
    args.erase(std::remove_if(args.begin(), args.end(),
                              [](const std::string& arg) { return arg.empty(); }),
               args.end());
    if (args.empty()) {
        return std::nullopt;
    }
    return std::pair{unary->first, std::move(args)};
}

bool is_scalar_expression_rhs(const std::string& rhs) {
    const std::string text = trim_copy(rhs);
    if (text.empty()) {
        return false;
    }
    double literal = 0.0;
    if (parse_number(text, literal)) {
        return false;
    }
    if (text.front() == '-' || text.front() == '+') {
        return is_scalar_expression_rhs(text.substr(1));
    }
    if (const auto call = parse_scalar_call(text)) {
        const std::string fn = lower(call->first);
        if (fn == "matmul" || fn == "solve" || fn == "transpose" || fn == "chol" || fn == "det" ||
            fn == "trace" || fn == "norm" || fn == "rank" || fn == "cond" || fn == "lu" ||
            fn == "qr" || fn == "svd" || fn == "eig_sym") {
            return false;
        }
    }
    return contains_scalar_operator(text) || parse_scalar_call(text).has_value();
}

Result<double> resolve_scalar_operand(const SessionState& state, const ScalarOperand& operand) {
    if (operand.is_literal) {
        return operand.literal;
    }
    const auto it = state.scalars.find(operand.name);
    if (it == state.scalars.end()) {
        return std::unexpected(DomainError{"resolve", "unknown scalar: " + operand.name});
    }
    return it->second;
}

Result<double> eval_scalar_expr(const SessionState& state, const std::string& expr_text) {
    const std::string expr = trim_copy(strip_outer_parens(expr_text));
    if (expr.empty()) {
        return std::unexpected(DomainError{"eval", "empty expression"});
    }

    if (expr.front() == '-') {
        auto inner = eval_scalar_expr(state, expr.substr(1));
        if (!inner) {
            return std::unexpected(inner.error());
        }
        return -(*inner);
    }
    if (expr.front() == '+') {
        return eval_scalar_expr(state, expr.substr(1));
    }

    if (const auto call = parse_scalar_call(expr)) {
        std::vector<double> arg_values;
        arg_values.reserve(call->second.size());
        for (const auto& arg_text : call->second) {
            auto arg = eval_scalar_expr(state, arg_text);
            if (!arg) {
                return std::unexpected(arg.error());
            }
            arg_values.push_back(*arg);
        }
        return Interpreter::eval_scalar_call(call->first, arg_values);
    }

    ScalarOperand single;
    if (parse_scalar_operand(expr, single)) {
        return resolve_scalar_operand(state, single);
    }

    const auto op_pos = find_scalar_binop(expr);
    if (!op_pos) {
        return std::unexpected(DomainError{"eval", "invalid scalar expression"});
    }

    auto left = eval_scalar_expr(state, expr.substr(0, op_pos->first));
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = eval_scalar_expr(state, expr.substr(op_pos->first + 1));
    if (!right) {
        return std::unexpected(right.error());
    }
    return Interpreter::eval_scalar_op(op_pos->second, *left, *right);
}

void append_unique_var(std::vector<std::string>& vars, const std::string& name) {
    if (std::find(vars.begin(), vars.end(), name) == vars.end()) {
        vars.push_back(name);
    }
}

void collect_scalar_expr_variables(const std::string& expr_text, std::vector<std::string>& vars) {
    const std::string expr = strip_outer_parens(expr_text);
    if (expr.empty()) {
        return;
    }

    if (expr.front() == '-' || expr.front() == '+') {
        collect_scalar_expr_variables(expr.substr(1), vars);
        return;
    }

    if (const auto call = parse_scalar_call(expr)) {
        for (const auto& arg_text : call->second) {
            collect_scalar_expr_variables(arg_text, vars);
        }
        return;
    }

    ScalarOperand single;
    if (parse_scalar_operand(expr, single)) {
        if (!single.is_literal) {
            append_unique_var(vars, single.name);
        }
        return;
    }

    const auto op_pos = find_scalar_binop(expr);
    if (!op_pos) {
        return;
    }
    collect_scalar_expr_variables(expr.substr(0, op_pos->first), vars);
    collect_scalar_expr_variables(expr.substr(op_pos->first + 1), vars);
}

} // namespace

std::string Interpreter::trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

Result<Matrix<double>> Interpreter::parse_matrix(const std::string& text) const {
    std::string s = trim(text);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
        return std::unexpected(DomainError{"parse_matrix", "expected [ ... ]"});
    }
    s = s.substr(1, s.size() - 2);
    if (s.empty()) {
        return Matrix<double>(0, 0);
    }

    std::vector<std::vector<double>> rows;
    std::stringstream row_stream(s);
    std::string row_text;
    while (std::getline(row_stream, row_text, ';')) {
        row_text = trim(row_text);
        if (row_text.empty()) {
            continue;
        }
        std::vector<double> row;
        std::stringstream col_stream(row_text);
        std::string cell;
        while (std::getline(col_stream, cell, ',')) {
            cell = trim(cell);
            double value = 0.0;
            if (!parse_number(cell, value)) {
                return std::unexpected(DomainError{"parse_matrix", "invalid number: " + cell});
            }
            row.push_back(value);
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        return std::unexpected(DomainError{"parse_matrix", "no rows found"});
    }
    const size_t cols = rows[0].size();
    for (const auto& row : rows) {
        if (row.size() != cols) {
            return std::unexpected(DimensionMismatch{rows.size(), row.size()});
        }
    }

    Matrix<double> m(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = rows[i][j];
        }
    }
    return m;
}

bool Interpreter::try_parse_scalar_assignment(const std::string& line, std::string& name, double& value) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    name = trim(cmd.substr(0, eq));
    if (name.empty()) {
        return false;
    }
    return parse_number(trim(cmd.substr(eq + 1)), value);
}

Result<std::string> Interpreter::assign_scalar(const std::string& name, double value) {
    state_.scalars[name] = value;
    return name + " = " + std::to_string(value) + "\n";
}

bool Interpreter::try_parse_scalar_binary_assignment(const std::string& line, ScalarBinaryAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto op_pos = find_scalar_binop(rhs);
    if (!op_pos) {
        return false;
    }
    assign.op = op_pos->second;
    const std::string left_text = trim_copy(rhs.substr(0, op_pos->first));
    const std::string right_text = trim_copy(rhs.substr(op_pos->first + 1));
    if (!parse_scalar_operand(left_text, assign.left) || !parse_scalar_operand(right_text, assign.right)) {
        return false;
    }
    return true;
}

bool Interpreter::try_parse_scalar_expr_assignment(const std::string& line, std::string& name,
                                                     std::string& expr) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    name = trim(cmd.substr(0, eq));
    if (name.empty() || !is_identifier(name)) {
        return false;
    }
    expr = trim(cmd.substr(eq + 1));
    if (expr.empty()) {
        return false;
    }
    double literal = 0.0;
    if (parse_number(expr, literal)) {
        return false;
    }
    return is_scalar_expression_rhs(expr);
}

bool is_matrix_call_callee(const std::string& callee) {
    return callee == "matmul" || callee == "solve" || callee == "transpose" || callee == "chol";
}

bool is_valid_matrix_call_arity(const std::string& callee, size_t arity) {
    if (callee == "matmul" || callee == "solve") {
        return arity == 2;
    }
    if (callee == "transpose" || callee == "chol") {
        return arity == 1;
    }
    return false;
}

bool is_scalar_matrix_call_callee(const std::string& callee) {
    return callee == "det" || callee == "trace" || callee == "norm" || callee == "rank" ||
           callee == "cond";
}

std::vector<std::string> split_comma_list(const std::string& text) {
    std::vector<std::string> parts;
    std::string current;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (char c : text) {
        if (c == '(') {
            ++paren_depth;
            current += c;
        } else if (c == ')') {
            --paren_depth;
            current += c;
        } else if (c == '[') {
            ++bracket_depth;
            current += c;
        } else if (c == ']') {
            --bracket_depth;
            current += c;
        } else if (c == ',' && paren_depth == 0 && bracket_depth == 0) {
            parts.push_back(trim_copy(current));
            current.clear();
        } else {
            current += c;
        }
    }
    parts.push_back(trim_copy(current));
    parts.erase(std::remove_if(parts.begin(), parts.end(),
                               [](const std::string& part) { return part.empty(); }),
                parts.end());
    return parts;
}

bool is_multi_matrix_call_callee(const std::string& callee) {
    return callee == "lu" || callee == "qr" || callee == "svd" || callee == "eig_sym";
}

bool is_valid_multi_matrix_target_count(const std::string& callee, size_t count) {
    if (callee == "qr" || callee == "eig_sym") {
        return count == 2;
    }
    if (callee == "lu" || callee == "svd") {
        return count == 2 || count == 3;
    }
    return false;
}

bool Interpreter::try_parse_multi_matrix_call_assignment(const std::string& line,
                                                           MultiMatrixCallAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    const std::string lhs = trim(cmd.substr(0, eq));
    if (lhs.find(',') == std::string::npos) {
        return false;
    }

    assign.targets = split_comma_list(lhs);
    if (assign.targets.empty()) {
        return false;
    }
    for (const auto& target : assign.targets) {
        if (!is_identifier(target)) {
            return false;
        }
    }

    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_multi_matrix_call_callee(assign.callee)) {
        return false;
    }

    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 1) {
        return false;
    }
    assign.arg = trim_copy(call_args->front());
    if (assign.arg.empty()) {
        return false;
    }

    return is_valid_multi_matrix_target_count(assign.callee, assign.targets.size());
}

bool Interpreter::try_parse_scalar_matrix_call_assignment(const std::string& line,
                                                          ScalarMatrixCallAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_scalar_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || call_args->size() != 1) {
        return false;
    }
    assign.arg = trim_copy(call_args->front());
    return !assign.arg.empty();
}

bool Interpreter::try_parse_matrix_call_assignment(const std::string& line, MatrixCallAssign& assign) {
    const std::string cmd = trim(line);
    const auto eq = cmd.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    assign.target = trim(cmd.substr(0, eq));
    if (assign.target.empty() || !is_identifier(assign.target)) {
        return false;
    }
    const std::string rhs = trim(cmd.substr(eq + 1));
    const auto open = rhs.find('(');
    if (open == std::string::npos) {
        return false;
    }
    assign.callee = lower(trim_copy(rhs.substr(0, open)));
    if (!is_matrix_call_callee(assign.callee)) {
        return false;
    }
    const auto call_args = split_call_args(rhs);
    if (!call_args || !is_valid_matrix_call_arity(assign.callee, call_args->size())) {
        return false;
    }
    assign.args.clear();
    for (const auto& arg : *call_args) {
        const std::string trimmed = trim_copy(arg);
        if (trimmed.empty()) {
            return false;
        }
        assign.args.push_back(trimmed);
    }
    return true;
}

std::vector<std::string> Interpreter::list_scalar_expr_variables(const std::string& expr) {
    std::vector<std::string> vars;
    collect_scalar_expr_variables(expr, vars);
    return vars;
}

Result<double> Interpreter::eval_scalar_call(const std::string& name,
                                             const std::vector<double>& args) {
    const std::string fn = lower(name);
    if (args.size() == 1) {
        const double arg = args[0];
        if (fn == "sin") {
            return std::sin(arg);
        }
        if (fn == "cos") {
            return std::cos(arg);
        }
        if (fn == "tan") {
            return std::tan(arg);
        }
        if (fn == "asin") {
            return std::asin(arg);
        }
        if (fn == "acos") {
            return std::acos(arg);
        }
        if (fn == "atan") {
            return std::atan(arg);
        }
        if (fn == "sinh") {
            return std::sinh(arg);
        }
        if (fn == "cosh") {
            return std::cosh(arg);
        }
        if (fn == "tanh") {
            return std::tanh(arg);
        }
        if (fn == "sqrt") {
            return std::sqrt(arg);
        }
        if (fn == "abs") {
            return std::fabs(arg);
        }
        if (fn == "exp") {
            return std::exp(arg);
        }
        if (fn == "log") {
            return std::log(arg);
        }
        if (fn == "log10") {
            return std::log10(arg);
        }
        if (fn == "floor") {
            return std::floor(arg);
        }
        if (fn == "ceil") {
            return std::ceil(arg);
        }
    }
    if (args.size() == 2) {
        if (fn == "pow") {
            return std::pow(args[0], args[1]);
        }
        if (fn == "min") {
            return std::fmin(args[0], args[1]);
        }
        if (fn == "max") {
            return std::fmax(args[0], args[1]);
        }
        if (fn == "atan2") {
            return std::atan2(args[0], args[1]);
        }
    }
    return std::unexpected(DomainError{"eval", "unknown scalar function: " + name});
}

Result<double> Interpreter::eval_scalar_op(char op, double left, double right) {
    switch (op) {
    case '+':
        return left + right;
    case '-':
        return left - right;
    case '*':
        return left * right;
    case '/':
        if (right == 0.0) {
            return std::unexpected(DomainError{"divide", "division by zero"});
        }
        return left / right;
    default:
        return std::unexpected(DomainError{"eval", "unsupported operator"});
    }
}

Result<std::string> Interpreter::assign_scalar_binary(const ScalarBinaryAssign& assign) {
    auto resolve = [this](const ScalarOperand& operand) -> Result<double> {
        return resolve_scalar_operand(state_, operand);
    };

    auto left = resolve(assign.left);
    if (!left) {
        return std::unexpected(left.error());
    }
    auto right = resolve(assign.right);
    if (!right) {
        return std::unexpected(right.error());
    }
    auto value = eval_scalar_op(assign.op, *left, *right);
    if (!value) {
        return std::unexpected(value.error());
    }
    return assign_scalar(assign.target, *value);
}

Result<std::string> Interpreter::assign_scalar_expr(const std::string& name, const std::string& expr) {
    auto value = eval_scalar_expr(state_, expr);
    if (!value) {
        return std::unexpected(value.error());
    }
    return assign_scalar(name, *value);
}

Result<std::string> Interpreter::assign_matrix_call(const MatrixCallAssign& assign) {
    auto resolve_operand = [this](const std::string& text) -> Result<Matrix<double>> {
        auto matrix = resolve_matrix(text);
        if (matrix) {
            return matrix;
        }
        return parse_matrix(text);
    };

    Result<Matrix<double>> result = std::unexpected(DomainError{"assign", "unsupported matrix call"});
    if (assign.callee == "matmul" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = matmul(*left, *right);
    } else if (assign.callee == "solve" && assign.args.size() == 2) {
        auto left = resolve_operand(assign.args[0]);
        if (!left) {
            return std::unexpected(left.error());
        }
        auto right = resolve_operand(assign.args[1]);
        if (!right) {
            return std::unexpected(right.error());
        }
        result = solve(*left, *right);
    } else if (assign.callee == "transpose" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        const auto transposed = transpose(*matrix);
        Matrix<double> stored(transposed.rows(), transposed.cols());
        for (size_t i = 0; i < transposed.rows(); ++i) {
            for (size_t j = 0; j < transposed.cols(); ++j) {
                stored(i, j) = transposed(i, j);
            }
        }
        result = stored;
    } else if (assign.callee == "chol" && assign.args.size() == 1) {
        auto matrix = resolve_operand(assign.args[0]);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        result = chol(*matrix);
    }
    if (!result) {
        return std::unexpected(result.error());
    }

    state_.matrices[assign.target] = *result;
    std::ostringstream out;
    out << assign.target << " =\n";
    print_matrix(out, *result);
    return out.str();
}

Result<std::string> Interpreter::assign_scalar_matrix_call(const ScalarMatrixCallAssign& assign) {
    auto matrix = resolve_matrix(assign.arg);
    if (!matrix) {
        matrix = parse_matrix(assign.arg);
    }
    if (!matrix) {
        return std::unexpected(matrix.error());
    }

    Result<double> value = std::unexpected(DomainError{"assign", "unsupported scalar matrix call"});
    if (assign.callee == "det") {
        value = det(*matrix);
    } else if (assign.callee == "trace") {
        value = trace(*matrix);
    } else if (assign.callee == "norm") {
        value = norm(*matrix);
    } else if (assign.callee == "rank") {
        auto ranked = rank(*matrix);
        if (!ranked) {
            return std::unexpected(ranked.error());
        }
        value = static_cast<double>(*ranked);
    } else if (assign.callee == "cond") {
        value = cond(*matrix);
    }
    if (!value) {
        return std::unexpected(value.error());
    }
    return assign_scalar(assign.target, *value);
}

Result<std::string> Interpreter::assign_multi_matrix_call(const MultiMatrixCallAssign& assign) {
    auto matrix = resolve_matrix(assign.arg);
    if (!matrix) {
        matrix = parse_matrix(assign.arg);
    }
    if (!matrix) {
        return std::unexpected(matrix.error());
    }

    std::ostringstream out;
    if (assign.callee == "lu") {
        auto factored = lu(*matrix);
        if (!factored) {
            return std::unexpected(factored.error());
        }
        const auto& [L, U, P] = *factored;
        state_.matrices[assign.targets[0]] = L;
        state_.matrices[assign.targets[1]] = U;
        out << assign.targets[0] << " =\n";
        print_matrix(out, L);
        out << assign.targets[1] << " =\n";
        print_matrix(out, U);
        if (assign.targets.size() == 3) {
            state_.matrices[assign.targets[2]] = P;
            out << assign.targets[2] << " =\n";
            print_matrix(out, P);
        }
        return out.str();
    }

    if (assign.callee == "qr") {
        auto factored = qr(*matrix);
        if (!factored) {
            return std::unexpected(factored.error());
        }
        const auto& [Q, R] = *factored;
        state_.matrices[assign.targets[0]] = Q;
        state_.matrices[assign.targets[1]] = R;
        out << assign.targets[0] << " =\n";
        print_matrix(out, Q);
        out << assign.targets[1] << " =\n";
        print_matrix(out, R);
        return out.str();
    }

    if (assign.callee == "svd") {
        auto decomp = svd(*matrix);
        if (!decomp) {
            return std::unexpected(decomp.error());
        }
        state_.matrices[assign.targets[0]] = decomp->U;
        state_.matrices[assign.targets[1]] = decomp->S;
        out << assign.targets[0] << " =\n";
        print_matrix(out, decomp->U);
        out << assign.targets[1] << " =\n";
        print_matrix(out, decomp->S);
        if (assign.targets.size() == 3) {
            state_.matrices[assign.targets[2]] = decomp->V;
            out << assign.targets[2] << " =\n";
            print_matrix(out, decomp->V);
        }
        return out.str();
    }

    if (assign.callee == "eig_sym") {
        auto decomp = eig_sym(*matrix);
        if (!decomp) {
            return std::unexpected(decomp.error());
        }
        state_.matrices[assign.targets[0]] = decomp->values;
        state_.matrices[assign.targets[1]] = decomp->vectors;
        out << assign.targets[0] << " =\n";
        print_matrix(out, decomp->values);
        out << assign.targets[1] << " =\n";
        print_matrix(out, decomp->vectors);
        return out.str();
    }

    return std::unexpected(DomainError{"assign", "unsupported multi matrix call"});
}

Result<Matrix<double>> Interpreter::resolve_matrix(const std::string& name) const {
    const auto it = state_.matrices.find(name);
    if (it == state_.matrices.end()) {
        return std::unexpected(DomainError{"resolve", "unknown matrix: " + name});
    }
    return it->second;
}

void print_matrix(std::ostream& out, const Matrix<double>& m) {
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < m.rows(); ++i) {
        out << "  [";
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << m(i, j);
        }
        out << "]\n";
    }
}

std::string matrix_to_line(const Matrix<double>& m) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < m.rows(); ++i) {
        if (i > 0) {
            out << "; ";
        }
        for (size_t j = 0; j < m.cols(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << m(i, j);
        }
    }
    out << "]";
    return out.str();
}

const char* plot_kind_name(PlotSeries::Kind kind) {
    switch (kind) {
    case PlotSeries::Kind::Line:
        return "line";
    case PlotSeries::Kind::Bar:
        return "bar";
    case PlotSeries::Kind::Scatter:
        return "scatter";
    case PlotSeries::Kind::Heatmap:
        return "heatmap";
    case PlotSeries::Kind::Spy:
        return "spy";
    case PlotSeries::Kind::Surface3D:
        return "surface3d";
    }
    return "line";
}

std::optional<PlotSeries::Kind> parse_plot_kind(const std::string& text) {
    const std::string kind = lower(trim_copy(text));
    if (kind == "line") {
        return PlotSeries::Kind::Line;
    }
    if (kind == "bar") {
        return PlotSeries::Kind::Bar;
    }
    if (kind == "scatter") {
        return PlotSeries::Kind::Scatter;
    }
    if (kind == "heatmap") {
        return PlotSeries::Kind::Heatmap;
    }
    if (kind == "spy") {
        return PlotSeries::Kind::Spy;
    }
    if (kind == "surface3d" || kind == "surface") {
        return PlotSeries::Kind::Surface3D;
    }
    return std::nullopt;
}

std::vector<double> parse_double_list(const std::string& text) {
    std::vector<double> values;
    std::stringstream stream(text);
    std::string cell;
    while (std::getline(stream, cell, ',')) {
        cell = trim_copy(cell);
        if (cell.empty()) {
            continue;
        }
        double value = 0.0;
        if (parse_number(cell, value)) {
            values.push_back(value);
        }
    }
    return values;
}

void write_double_list(std::ostream& out, const char* label, const std::vector<double>& values) {
    if (values.empty()) {
        return;
    }
    out << label;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    out << '\n';
}

Result<std::string> Interpreter::set_plot(const Matrix<double>& xs, const Matrix<double>& ys,
                                          PlotSeries::Kind kind) {
    const size_t n = ys.rows() * ys.cols();
    if (n == 0) {
        return std::unexpected(DomainError{"plot", "empty data"});
    }
    std::vector<double> x;
    std::vector<double> y;
    y.reserve(n);
    if (xs.rows() == 0 && xs.cols() == 0) {
        x.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            x.push_back(static_cast<double>(i));
        }
    } else {
        const size_t xn = xs.rows() * xs.cols();
        if (xn != n) {
            return std::unexpected(DimensionMismatch{xn, n});
        }
        x.reserve(n);
        if (xs.cols() == 1) {
            for (size_t i = 0; i < xs.rows(); ++i) {
                x.push_back(xs(i, 0));
            }
        } else {
            for (size_t i = 0; i < xs.cols(); ++i) {
                x.push_back(xs(0, i));
            }
        }
    }
    if (ys.cols() == 1) {
        for (size_t i = 0; i < ys.rows(); ++i) {
            y.push_back(ys(i, 0));
        }
    } else {
        for (size_t i = 0; i < ys.cols(); ++i) {
            y.push_back(ys(0, i));
        }
    }
    state_.plot.x = std::move(x);
    state_.plot.y = std::move(y);
    state_.plot.kind = kind;
    state_.plot.matrix_rows = 0;
    state_.plot.matrix_cols = 0;
    state_.plot.nnz = 0;
    state_.plot.grid = Matrix<double>{};
    state_.plot.valid = true;
    const char* label = kind == PlotSeries::Kind::Scatter ? "scatter" : "plot";
    return std::string{label} + " updated (" + std::to_string(state_.plot.y.size()) + " points)\n" +
           format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_heatmap(const Matrix<double>& m) {
    if (m.rows() == 0 || m.cols() == 0) {
        return std::unexpected(DomainError{"imshow", "empty matrix"});
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Heatmap;
    state_.plot.matrix_rows = m.rows();
    state_.plot.matrix_cols = m.cols();
    state_.plot.nnz = 0;
    state_.plot.grid = m;
    state_.plot.valid = true;
    return "imshow updated (" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ")\n" +
           format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_spy(const Matrix<double>& m) {
    size_t count = 0;
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            if (m(i, j) != 0.0) {
                ++count;
            }
        }
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Spy;
    state_.plot.matrix_rows = m.rows();
    state_.plot.matrix_cols = m.cols();
    state_.plot.nnz = count;
    state_.plot.grid = m;
    state_.plot.valid = true;
    return "spy updated (" + std::to_string(count) + " nonzeros)\n" + format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_surf(const Matrix<double>& z) {
    if (z.rows() == 0 || z.cols() == 0) {
        return std::unexpected(DomainError{"surf", "empty matrix"});
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Surface3D;
    state_.plot.matrix_rows = z.rows();
    state_.plot.matrix_cols = z.cols();
    state_.plot.nnz = 0;
    state_.plot.grid = z;
    state_.plot.valid = true;
    return "surf updated (" + std::to_string(z.rows()) + "x" + std::to_string(z.cols()) +
           " grid)\n" + format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_surf(const Matrix<double>& x, const Matrix<double>& y,
                                               const Matrix<double>& z) {
    if (x.rows() != y.rows() || x.cols() != y.cols() || x.rows() != z.rows() ||
        x.cols() != z.cols()) {
        return std::unexpected(DimensionMismatch{x.rows() * x.cols(), z.rows() * z.cols()});
    }
    if (z.rows() == 0 || z.cols() == 0) {
        return std::unexpected(DomainError{"surf", "empty grid"});
    }
    state_.plot.x.clear();
    state_.plot.y.clear();
    state_.plot.kind = PlotSeries::Kind::Surface3D;
    state_.plot.matrix_rows = z.rows();
    state_.plot.matrix_cols = z.cols();
    state_.plot.nnz = 0;
    state_.plot.grid = z;
    state_.plot.valid = true;
    return "surf updated (" + std::to_string(z.rows()) + "x" + std::to_string(z.cols()) +
           " mesh)\n" + format_plot_preview(state_.plot);
}

Result<std::string> Interpreter::set_plot_bars(std::vector<double> x, std::vector<double> y) {
    if (x.size() != y.size() || x.empty()) {
        return std::unexpected(DimensionMismatch{x.size(), y.size()});
    }
    state_.plot.x = std::move(x);
    state_.plot.y = std::move(y);
    state_.plot.kind = PlotSeries::Kind::Bar;
    state_.plot.grid = Matrix<double>{};
    state_.plot.valid = true;
    return "histogram updated (" + std::to_string(state_.plot.y.size()) + " bins)\n" +
           format_plot_preview(state_.plot);
}

Result<void> Interpreter::save_session(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        return std::unexpected(DomainError{"save", "cannot open: " + path});
    }
    out << "# MathScript session\n";
    for (const auto& [name, value] : state_.scalars) {
        out << "scalar " << name << " = " << value << "\n";
    }
    for (const auto& [name, matrix] : state_.matrices) {
        out << "matrix " << name << " = " << matrix_to_line(matrix) << "\n";
    }
    if (state_.plot.valid) {
        out << "plot meta valid=1 kind=" << plot_kind_name(state_.plot.kind)
            << " rows=" << state_.plot.matrix_rows << " cols=" << state_.plot.matrix_cols
            << " nnz=" << state_.plot.nnz << "\n";
        write_double_list(out, "plot x ", state_.plot.x);
        write_double_list(out, "plot y ", state_.plot.y);
        if (state_.plot.grid.rows() > 0 && state_.plot.grid.cols() > 0) {
            out << "plot grid = " << matrix_to_line(state_.plot.grid) << "\n";
        }
    }
    return {};
}

Result<void> Interpreter::load_session(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected(DomainError{"load", "cannot open: " + path});
    }
    reset();
    PlotSeries pending_plot{};
    bool have_plot = false;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("plot meta ", 0) == 0) {
            const std::string meta = trim(line.substr(10));
            have_plot = meta.find("valid=1") != std::string::npos;
            if (const auto kind_pos = meta.find("kind="); kind_pos != std::string::npos) {
                const auto kind_end = meta.find(' ', kind_pos + 5);
                const std::string kind_text = meta.substr(kind_pos + 5, kind_end - (kind_pos + 5));
                if (const auto kind = parse_plot_kind(kind_text)) {
                    pending_plot.kind = *kind;
                }
            }
            if (const auto rows_pos = meta.find("rows="); rows_pos != std::string::npos) {
                pending_plot.matrix_rows =
                    static_cast<size_t>(std::strtoull(meta.c_str() + rows_pos + 5, nullptr, 10));
            }
            if (const auto cols_pos = meta.find("cols="); cols_pos != std::string::npos) {
                pending_plot.matrix_cols =
                    static_cast<size_t>(std::strtoull(meta.c_str() + cols_pos + 5, nullptr, 10));
            }
            if (const auto nnz_pos = meta.find("nnz="); nnz_pos != std::string::npos) {
                pending_plot.nnz =
                    static_cast<size_t>(std::strtoull(meta.c_str() + nnz_pos + 4, nullptr, 10));
            }
            continue;
        }
        if (line.rfind("plot x ", 0) == 0) {
            pending_plot.x = parse_double_list(trim(line.substr(7)));
            continue;
        }
        if (line.rfind("plot y ", 0) == 0) {
            pending_plot.y = parse_double_list(trim(line.substr(7)));
            continue;
        }
        if (line.rfind("plot grid = ", 0) == 0) {
            auto grid = parse_matrix(trim(line.substr(12)));
            if (!grid) {
                return std::unexpected(grid.error());
            }
            pending_plot.grid = *grid;
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string lhs = trim(line.substr(0, eq));
        const std::string rhs = trim(line.substr(eq + 1));
        const auto space = lhs.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        const std::string kind = lower(lhs.substr(0, space));
        const std::string name = trim(lhs.substr(space + 1));
        if (kind == "scalar") {
            double value = 0.0;
            if (!parse_number(rhs, value)) {
                return std::unexpected(DomainError{"load", "invalid scalar: " + name});
            }
            state_.scalars[name] = value;
        } else if (kind == "matrix") {
            auto matrix = parse_matrix(rhs);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
            state_.matrices[name] = *matrix;
        }
    }
    if (have_plot) {
        pending_plot.valid = true;
        state_.plot = std::move(pending_plot);
    }
    return {};
}

Result<std::string> Interpreter::execute(const std::string& line) {
    const std::string cmd = trim(line);
    if (cmd.empty()) {
        return std::string{};
    }
    state_.history.push_back(cmd);

    const std::string lcmd = lower(cmd);
    if (lcmd == "help") {
        return std::string{
            "Commands:\n"
            "  help, vars, version, show, saveplot, clear, topology, simd, dispatch, balance, gpu, mpi, frameworks, exit\n"
            "  izaac seed <n>   gria(M)   axiom evolve\n"
            "  save <file.ms>  load <file.ms>\n"
            "  name = [1, 2; 3, 4]     matrix assignment\n"
            "  name = 3.14              scalar assignment\n"
            "  name = x + 2             scalar expression (+, -, *, /; () precedence)\n"
            "  name = sin(x) pow(x,2)   scalar calls (sin, cos, sqrt, pow, min, max, ...)\n"
            "  name = matmul(A, B)      matrix multiply assignment\n"
            "  name = solve(A, B)       linear solve assignment\n"
            "  name = transpose(A)      transpose assignment\n"
            "  name = chol(A)           Cholesky factor assignment\n"
            "  name = det(A)            scalar matrix call (det, trace, norm, rank, cond)\n"
            "  L, U, P = lu(A)          LU factors (L, U only also supported)\n"
            "  Q, R = qr(A)             QR factors\n"
            "  U, S, V = svd(A)         SVD factors (U, S only also supported)\n"
            "  D, V = eig_sym(A)        symmetric eigenvalues and eigenvectors\n"
            "  plot([y...])  plot([x...], [y...])  scatter([x...], [y...])  hist([...])\n"
            "  imshow(matrix)  spy(matrix)  surf(matrix)\n"
            "  surf([x...], [y...], [z...])  show  saveplot <file.txt>\n"
            "  det(A), trace(A), norm(A), rank(A), cond(A)\n"
            "  lu(A), qr(A), chol(A), solve(A,B), matmul(A,B), eig_sym(A), svd(A)\n"
            "  fft([1,2,3,4])           vector FFT magnitude\n"
            "  erf(x), gamma(x), bessel_j0(x), spherical_jn(n,x)\n"
            "  kelvin_ber(0,x), struve_h(n,x), bessel_zero_jnu(nu,n)\n"
            "  kummer_m(a,b,x), hypergeo_2f1(a,b,c,x), whittaker_m(k,mu,x)\n"
            "  jacobi_p(n,a,b,x), ellip_k(k), jacobi_sn(u,k)\n"
            "  theta3(z,q), zeta(s), polylog(n,z), mathieu_ce(n,q,x)\n"
            "  heun_g(a,q,alpha,beta,gamma,delta,z), painleve1(x,y0,yp0)\n"
            "  legendre_p(n,x), beta(a,b)\n"};
    }
    if (lcmd == "version") {
        std::ostringstream out;
        out << "mathscriptc " << ms::VERSION_STRING
            << " (commit " << ms::BUILD_COMMIT
            << ", built " << ms::BUILD_DATE << ")\n";
        return out.str();
    }
    if (lcmd == "show") {
        if (!state_.plot.valid) {
            return std::unexpected(DomainError{"show", "no plot in session"});
        }
        return format_plot_preview(state_.plot);
    }
    if (lcmd.rfind("saveplot ", 0) == 0) {
        if (!state_.plot.valid) {
            return std::unexpected(DomainError{"saveplot", "no plot in session"});
        }
        const std::string path = trim(cmd.substr(9));
        if (path.empty()) {
            return std::unexpected(DomainError{"saveplot", "missing path"});
        }
        std::ofstream out(path);
        if (!out) {
            return std::unexpected(DomainError{"saveplot", "cannot write: " + path});
        }
        out << format_plot_preview(state_.plot);
        return "saved plot preview to " + path + "\n";
    }
    if (lcmd == "vars") {
        std::ostringstream out;
        for (const auto& [name, value] : state_.scalars) {
            out << name << " = " << value << "\n";
        }
        for (const auto& [name, matrix] : state_.matrices) {
            out << name << " (" << matrix.rows() << "x" << matrix.cols() << ")\n";
        }
        if (state_.scalars.empty() && state_.matrices.empty()) {
            out << "(empty session)\n";
        }
        return out.str();
    }
    if (lcmd == "clear") {
        reset();
        return std::string{"session cleared\n"};
    }
    if (lcmd == "topology") {
        const auto topo = detect_topology();
        std::ostringstream out;
        out << "cores=" << topo.total_cores()
            << " threads=" << topo.total_threads()
            << " gpus=" << topo.total_gpus << "\n";
        return out.str();
    }
    if (lcmd == "simd") {
        const auto info = simd::dispatch_info();
        return "ISA: " + simd::isa_summary(info.isa) + "\n";
    }
    if (lcmd == "gpu") {
        std::ostringstream out;
        out << "cuda=" << (has_cuda() ? "yes" : "no")
            << " devices=" << get_gpu_count() << "\n";
        if (get_gpu_count() > 0) {
            out << "device0=" << get_gpu_model(0) << "\n";
        }
        return out.str();
    }
    if (lcmd == "dispatch") {
        const auto d = decide(512, ExecPolicy::AUTO);
        std::ostringstream out;
        out << "auto@512 backend="
            << (d.backend == Backend::CUDA ? "cuda" : "cpu")
            << " threads=" << d.n_threads << "\n";
        return out.str();
    }
    if (lcmd == "balance") {
        const auto lb = balance(1024, ExecPolicy::AUTO);
        std::ostringstream out;
        out << "backend=" << (lb.backend == Backend::CUDA ? "cuda" : "cpu")
            << " device=" << lb.cuda_device
            << " threads=" << lb.cpu_threads << "\n";
        return out.str();
    }
    if (lcmd.rfind("save ", 0) == 0) {
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) {
            return std::unexpected(DomainError{"save", "missing path"});
        }
        auto saved = save_session(path);
        if (!saved) {
            return std::unexpected(saved.error());
        }
        return "saved session to " + path + "\n";
    }
    if (lcmd.rfind("load ", 0) == 0) {
        const std::string path = trim(cmd.substr(5));
        if (path.empty()) {
            return std::unexpected(DomainError{"load", "missing path"});
        }
        auto loaded = load_session(path);
        if (!loaded) {
            return std::unexpected(loaded.error());
        }
        return "loaded session from " + path + "\n";
    }

    if (lcmd == "frameworks") {
        return std::string{
            "frameworks: GRIA (gria matrix ops), Izaac (seed), AXIOM (evolve)\n"
            "  gria(M)  izaac seed N  axiom evolve\n"};
    }
    if (lcmd.rfind("izaac seed ", 0) == 0) {
        const std::string arg = trim(cmd.substr(11));
        double seed_value = 0.0;
        if (!parse_number(arg, seed_value)) {
            return std::unexpected(DomainError{"izaac", "expected numeric seed"});
        }
        izaac::seed_session(static_cast<uint64_t>(seed_value));
        return "izaac session seeded\n";
    }
    if (lcmd == "axiom evolve") {
        auto registry = axiom::PrimitiveRegistry::build_from_ms_namespace();
        axiom::Axiom engine(axiom::EvolutionConfig{.population_size = 20, .max_generations = 25}, registry);
        ColMatrix<double> data{{1, 2}, {3, 4}, {5, 6}};
        const auto best = engine
                                .evolve(
                                    [&](const axiom::Algorithm& a) {
                                        return engine.gria_fitness(a, data);
                                    },
                                    [](const axiom::Algorithm& a) { return a.fitness > 0.45; })
                                .value();
        std::ostringstream out;
        out << "axiom best fitness=" << best.fitness << " alpha-target fit\n";
        return out.str();
    }

    if (lcmd == "mpi") {
        static ms::distributed::MPIContext mpi = ms::distributed::init(0, nullptr);
        std::ostringstream out;
        out << "backend=" << ms::distributed::backend_name(mpi)
            << " rank=" << ms::distributed::rank(mpi)
            << " size=" << ms::distributed::size(mpi) << "\n";
        return out.str();
    }

    const auto eq = cmd.find('=');
    if (eq != std::string::npos) {
        const std::string lhs = trim(cmd.substr(0, eq));
        const std::string rhs = trim(cmd.substr(eq + 1));
        if (lhs.empty()) {
            return std::unexpected(DomainError{"assign", "missing variable name"});
        }

        double scalar = 0.0;
        if (parse_number(rhs, scalar)) {
            return assign_scalar(lhs, scalar);
        }

        MultiMatrixCallAssign multi_call{};
        if (try_parse_multi_matrix_call_assignment(cmd, multi_call)) {
            return assign_multi_matrix_call(multi_call);
        }

        MatrixCallAssign matrix_call{};
        if (try_parse_matrix_call_assignment(cmd, matrix_call)) {
            return assign_matrix_call(matrix_call);
        }

        ScalarMatrixCallAssign scalar_matrix_call{};
        if (try_parse_scalar_matrix_call_assignment(cmd, scalar_matrix_call)) {
            return assign_scalar_matrix_call(scalar_matrix_call);
        }

        if (is_scalar_expression_rhs(rhs)) {
            return assign_scalar_expr(lhs, rhs);
        }

        auto matrix = parse_matrix(rhs);
        if (!matrix) {
            return std::unexpected(matrix.error());
        }
        state_.matrices[lhs] = *matrix;
        std::ostringstream out;
        out << lhs << " =\n";
        print_matrix(out, *matrix);
        return out.str();
    }

    static const std::regex plot_binary(
        R"(plot\s*\(\s*(\[[^\]]+\])\s*,\s*(\[[^\]]+\])\s*\))",
        std::regex::icase);
    static const std::regex scatter_binary(
        R"(scatter\s*\(\s*(\[[^\]]+\])\s*,\s*(\[[^\]]+\])\s*\))",
        std::regex::icase);
    std::smatch match;
    if (std::regex_match(cmd, match, plot_binary)) {
        auto xs = parse_matrix(match[1].str());
        auto ys = parse_matrix(match[2].str());
        if (!xs || !ys) {
            return std::unexpected(DomainError{"plot", "invalid x/y vectors"});
        }
        return set_plot(*xs, *ys);
    }
    if (std::regex_match(cmd, match, scatter_binary)) {
        auto xs = parse_matrix(match[1].str());
        auto ys = parse_matrix(match[2].str());
        if (!xs || !ys) {
            return std::unexpected(DomainError{"scatter", "invalid x/y vectors"});
        }
        return set_plot(*xs, *ys, PlotSeries::Kind::Scatter);
    }

    static const std::regex heptary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\))",
        std::regex::icase);
    if (std::regex_match(cmd, match, heptary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "heun_g") {
            double a = 0.0;
            double q = 0.0;
            double alpha = 0.0;
            double beta = 0.0;
            double gamma = 0.0;
            double delta = 0.0;
            double z = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), q) ||
                !parse_number(trim(match[4].str()), alpha) || !parse_number(trim(match[5].str()), beta) ||
                !parse_number(trim(match[6].str()), gamma) || !parse_number(trim(match[7].str()), delta) ||
                !parse_number(trim(match[8].str()), z)) {
                return std::unexpected(
                    DomainError{"heun_g", "expected heun_g(a,q,alpha,beta,gamma,delta,z)"});
            }
            return std::to_string(heun_g(a, q, alpha, beta, gamma, delta, z)) + "\n";
        }
    }

    static const std::regex quaternary(
        R"((\w+)\(([^,]+),([^,]+),([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, quaternary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "hypergeo_2f1" || fn == "jacobi_p") {
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            double d = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), b) ||
                !parse_number(trim(match[4].str()), c) || !parse_number(trim(match[5].str()), d)) {
                return std::unexpected(DomainError{fn, "expected numeric arguments"});
            }
            if (fn == "hypergeo_2f1") {
                return std::to_string(hypergeo_2f1(a, b, c, d)) + "\n";
            }
            return std::to_string(jacobi_p(static_cast<int>(a), b, c, d)) + "\n";
        }
    }

    static const std::regex ternary(R"((\w+)\(([^,]+),([^,]+),([^)]+)\))", std::regex::icase);
    if (lower(cmd).starts_with("surf(") && !cmd.empty() && cmd.back() == ')') {
        const auto args = split_call_args(cmd);
        if (args && (args->size() == 1u || args->size() == 3u)) {
            const auto resolve_arg = [this](const std::string& arg) -> Result<Matrix<double>> {
                auto matrix = parse_matrix(arg);
                if (!matrix) {
                    matrix = resolve_matrix(arg);
                }
                return matrix;
            };
            if (args->size() == 1u) {
                auto z = resolve_arg((*args)[0]);
                if (!z) {
                    return std::unexpected(z.error());
                }
                return set_plot_surf(*z);
            }
            auto X = resolve_arg((*args)[0]);
            if (!X) {
                return std::unexpected(X.error());
            }
            auto Y = resolve_arg((*args)[1]);
            if (!Y) {
                return std::unexpected(Y.error());
            }
            auto Z = resolve_arg((*args)[2]);
            if (!Z) {
                return std::unexpected(Z.error());
            }
            return set_plot_surf(*X, *Y, *Z);
        }
    }
    if (std::regex_match(cmd, match, ternary)) {
        const std::string fn = lower(match[1].str());
        if (fn == "kummer_m" || fn == "whittaker_m" || fn == "mathieu_ce" || fn == "painleve1") {
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            if (!parse_number(trim(match[2].str()), a) || !parse_number(trim(match[3].str()), b) ||
                !parse_number(trim(match[4].str()), c)) {
                return std::unexpected(DomainError{fn, "expected numeric arguments"});
            }
            if (fn == "kummer_m") {
                return std::to_string(kummer_m(a, b, c)) + "\n";
            }
            if (fn == "whittaker_m") {
                return std::to_string(whittaker_m(a, b, c)) + "\n";
            }
            if (fn == "mathieu_ce") {
                return std::to_string(mathieu_ce(static_cast<int>(a), b, c)) + "\n";
            }
            return std::to_string(painleve1(a, b, c)) + "\n";
        }
    }

    static const std::regex binary(R"((\w+)\(([^,]+),([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, binary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg_a = trim(match[2].str());
        const std::string arg_b = trim(match[3].str());

        if (fn == "spherical_jn") {
            double n = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"spherical_jn", "expected spherical_jn(n,x)"});
            }
            return std::to_string(spherical_jn(static_cast<int>(n), x)) + "\n";
        }

        if (fn == "kelvin_ber") {
            double nu = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, nu) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"kelvin_ber", "expected kelvin_ber(nu,x)"});
            }
            return std::to_string(kelvin_ber(static_cast<int>(nu), x)) + "\n";
        }

        if (fn == "struve_h") {
            double nu = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, nu) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"struve_h", "expected struve_h(nu,x)"});
            }
            return std::to_string(struve_h(static_cast<int>(nu), x)) + "\n";
        }

        if (fn == "bessel_zero_jnu") {
            double nu = 0.0;
            double n = 0.0;
            if (!parse_number(arg_a, nu) || !parse_number(arg_b, n)) {
                return std::unexpected(DomainError{"bessel_zero_jnu", "expected bessel_zero_jnu(nu,n)"});
            }
            return std::to_string(bessel_zero_jnu(static_cast<int>(nu), static_cast<int>(n))) + "\n";
        }

        if (fn == "jacobi_sn") {
            double u = 0.0;
            double k = 0.0;
            if (!parse_number(arg_a, u) || !parse_number(arg_b, k)) {
                return std::unexpected(DomainError{"jacobi_sn", "expected jacobi_sn(u,k)"});
            }
            return std::to_string(jacobi_sn(u, k)) + "\n";
        }

        if (fn == "theta3") {
            double z = 0.0;
            double q = 0.0;
            if (!parse_number(arg_a, z) || !parse_number(arg_b, q)) {
                return std::unexpected(DomainError{"theta3", "expected theta3(z,q)"});
            }
            return std::to_string(theta3(z, q)) + "\n";
        }

        if (fn == "polylog") {
            double n = 0.0;
            double z = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, z)) {
                return std::unexpected(DomainError{"polylog", "expected polylog(n,z)"});
            }
            return std::to_string(polylog(static_cast<int>(n), z)) + "\n";
        }

        if (fn == "beta") {
            double a = 0.0;
            double b = 0.0;
            if (!parse_number(arg_a, a) || !parse_number(arg_b, b)) {
                return std::unexpected(DomainError{"beta", "expected numeric arguments beta(a,b)"});
            }
            return std::to_string(beta_func(a, b)) + "\n";
        }

        if (fn == "legendre_p") {
            double n = 0.0;
            double x = 0.0;
            if (!parse_number(arg_a, n) || !parse_number(arg_b, x)) {
                return std::unexpected(DomainError{"legendre_p", "expected legendre_p(n,x)"});
            }
            return std::to_string(legendre_p(static_cast<int>(n), x)) + "\n";
        }

        if (fn == "plot" || fn == "scatter" || fn == "solve" || fn == "matmul") {
            auto A = resolve_matrix(arg_a);
            if (!A) {
                A = parse_matrix(arg_a);
            }
            auto B = resolve_matrix(arg_b);
            if (!B) {
                B = parse_matrix(arg_b);
            }
            if (A && B) {
                if (fn == "plot") {
                    return set_plot(*A, *B);
                }
                if (fn == "scatter") {
                    return set_plot(*A, *B, PlotSeries::Kind::Scatter);
                }
                if (fn == "solve") {
                    auto x = solve(*A, *B);
                    if (!x) {
                        return std::unexpected(x.error());
                    }
                    std::ostringstream out;
                    out << "x =\n";
                    print_matrix(out, *x);
                    return out.str();
                }
                auto C = matmul(*A, *B);
                if (!C) {
                    return std::unexpected(C.error());
                }
                std::ostringstream out;
                out << "C =\n";
                print_matrix(out, *C);
                return out.str();
            }
        }
    }

    static const std::regex unary(R"((\w+)\(([^)]+)\))", std::regex::icase);
    if (std::regex_match(cmd, match, unary)) {
        const std::string fn = lower(match[1].str());
        const std::string arg = trim(match[2].str());

        if (fn == "plot") {
            auto ys = parse_matrix(arg);
            if (!ys) {
                ys = resolve_matrix(arg);
            }
            if (!ys) {
                return std::unexpected(ys.error());
            }
            Matrix<double> empty(0, 0);
            return set_plot(empty, *ys);
        }

        if (fn == "erf" || fn == "erfc" || fn == "gamma" || fn == "bessel_j0" || fn == "fresnel_c" ||
            fn == "fresnel_s" || fn == "ellip_k" || fn == "zeta") {
            double value = 0.0;
            if (!parse_number(arg, value)) {
                return std::unexpected(DomainError{"special", "expected numeric argument"});
            }
            std::ostringstream out;
            if (fn == "erf") {
                out << erf(value);
            } else if (fn == "erfc") {
                out << erfc(value);
            } else if (fn == "gamma") {
                out << gamma_func(value);
            } else if (fn == "bessel_j0") {
                out << bessel_j0(value);
            } else if (fn == "fresnel_c") {
                out << fresnel_c(value);
            } else if (fn == "fresnel_s") {
                out << fresnel_s(value);
            } else if (fn == "ellip_k") {
                out << ellip_k(value);
            } else {
                out << zeta(value);
            }
            out << "\n";
            return out.str();
        }

        if (fn == "gria") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            std::vector<double> flat;
            for (size_t i = 0; i < data->rows(); ++i) {
                for (size_t j = 0; j < data->cols(); ++j) {
                    flat.push_back((*data)(i, j));
                }
            }
            const double alpha = gria::compute_alpha<double>(flat, [](std::span<const double> input) {
                std::vector<double> out;
                out.reserve(input.size());
                for (double v : input) {
                    out.push_back(v * v);
                }
                return out;
            });
            std::ostringstream out;
            out << "alpha=" << alpha << " class="
                << (gria::classify(alpha) == gria::ComputeClass::Critical ? "critical"
                    : gria::classify(alpha) == gria::ComputeClass::Irreversible ? "irreversible"
                                                                                  : "reversible")
                << "\n";
            return out.str();
        }

        if (fn == "imshow") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            return set_plot_heatmap(*data);
        }

        if (fn == "spy") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            return set_plot_spy(*data);
        }

        if (fn == "hist") {
            auto data = parse_matrix(arg);
            if (!data) {
                data = resolve_matrix(arg);
            }
            if (!data) {
                return std::unexpected(data.error());
            }
            std::vector<double> values;
            values.reserve(data->rows() * data->cols());
            for (size_t i = 0; i < data->rows(); ++i) {
                for (size_t j = 0; j < data->cols(); ++j) {
                    values.push_back((*data)(i, j));
                }
            }
            if (values.empty()) {
                return std::unexpected(DomainError{"hist", "empty data"});
            }
            const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
            const double vmin = *min_it;
            const double vmax = *max_it;
            constexpr size_t bins = 10;
            std::vector<double> counts(bins, 0.0);
            const double width = (vmax - vmin) > 0.0 ? (vmax - vmin) / static_cast<double>(bins) : 1.0;
            for (double value : values) {
                size_t idx = static_cast<size_t>((value - vmin) / width);
                if (idx >= bins) {
                    idx = bins - 1;
                }
                counts[idx] += 1.0;
            }
            std::vector<double> centers(bins);
            for (size_t i = 0; i < bins; ++i) {
                centers[i] = vmin + (static_cast<double>(i) + 0.5) * width;
            }
            return set_plot_bars(std::move(centers), std::move(counts));
        }

        if (fn == "fft") {
            auto vec = parse_matrix(arg);
            if (!vec || vec->cols() != 1) {
                if (vec && vec->rows() == 1) {
                    Matrix<double> col(vec->cols(), 1);
                    for (size_t i = 0; i < vec->cols(); ++i) {
                        col(i, 0) = (*vec)(0, i);
                    }
                    vec = col;
                } else if (!vec) {
                    return std::unexpected(vec.error());
                } else {
                    return std::unexpected(DomainError{"fft", "expected vector [a,b,c,...]"});
                }
            }
            std::vector<double> data(vec->rows());
            for (size_t i = 0; i < vec->rows(); ++i) {
                data[i] = (*vec)(i, 0);
            }
            auto spectrum = fft(data);
            if (!spectrum) {
                return std::unexpected(spectrum.error());
            }
            std::ostringstream out;
            out << "fft magnitudes:\n";
            for (size_t i = 0; i < spectrum->size(); ++i) {
                out << "  [" << i << "] " << std::abs((*spectrum)[i]) << "\n";
            }
            return out.str();
        }

        auto matrix = parse_matrix(arg);
        if (matrix) {
            state_.matrices["_"] = *matrix;
        } else {
            matrix = resolve_matrix(arg);
            if (!matrix) {
                return std::unexpected(matrix.error());
            }
        }

        std::ostringstream out;
        if (fn == "det") {
            auto result = det(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "trace") {
            auto result = trace(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "norm") {
            auto result = norm(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "rank") {
            auto result = rank(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "cond") {
            auto result = cond(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << *result << "\n";
        } else if (fn == "eig_sym") {
            auto eig_result = eig_sym(*matrix);
            if (!eig_result) {
                return std::unexpected(eig_result.error());
            }
            out << "eigenvalues:\n";
            for (size_t i = 0; i < eig_result->values.rows(); ++i) {
                out << "  " << eig_result->values(i, 0) << "\n";
            }
        } else if (fn == "svd") {
            auto s = svd(*matrix);
            if (!s) {
                return std::unexpected(s.error());
            }
            out << "singular values:\n";
            for (size_t i = 0; i < s->S.rows(); ++i) {
                out << "  " << s->S(i, 0) << "\n";
            }
        } else if (fn == "lu") {
            auto result = lu(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            const auto& [L, U, P] = *result;
            out << "L =\n";
            print_matrix(out, L);
            out << "U =\n";
            print_matrix(out, U);
            out << "P =\n";
            print_matrix(out, P);
        } else if (fn == "qr") {
            auto result = qr(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            const auto& [Q, R] = *result;
            out << "Q =\n";
            print_matrix(out, Q);
            out << "R =\n";
            print_matrix(out, R);
        } else if (fn == "chol") {
            auto result = chol(*matrix);
            if (!result) {
                return std::unexpected(result.error());
            }
            out << "L =\n";
            print_matrix(out, *result);
        } else {
            return std::unexpected(DomainError{"repl", "unknown function: " + fn});
        }
        return out.str();
    }

    return std::unexpected(DomainError{"repl", "could not parse: " + cmd});
}

} // namespace ms::interp
