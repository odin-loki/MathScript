#include "ms/interp/jit_backend_impl.hpp"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <cctype>
#include <cstdlib>
#include <map>
#include <optional>
#include <vector>
#include <algorithm>

namespace ms::interp {

namespace {

void initialize_llvm_targets_once() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    initialized = true;
}

std::string trim_copy(const std::string& text) {
    return Interpreter::trim(text);
}

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

bool parse_number(const std::string& text, double& out) {
    if (text.empty()) {
        return false;
    }
    char* end = nullptr;
    out = std::strtod(text.c_str(), &end);
    return end != text.c_str() && *end == '\0';
}

bool is_identifier(const std::string& text) {
    if (text.empty()) {
        return false;
    }
    if (!(std::isalpha(static_cast<unsigned char>(text[0])) || text[0] == '_')) {
        return false;
    }
    for (size_t i = 1; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!(std::isalnum(c) || text[i] == '_')) {
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
    args.erase(std::remove_if(args.begin(), args.end(),
                              [](const std::string& arg) { return arg.empty(); }),
               args.end());
    return args;
}

std::optional<std::pair<std::string, std::vector<std::string>>> parse_scalar_call(
    const std::string& expr) {
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

    std::vector<std::string> args = split_scalar_call_args(text.substr(open + 1, close - open - 1));
    if (args.empty()) {
        return std::nullopt;
    }
    return std::pair{name, std::move(args)};
}

const char* libm_symbol_for_scalar_call(const std::string& name, size_t arity) {
    const std::string fn = lower_copy(name);
    if (arity == 1) {
        if (fn == "sin") {
            return "sin";
        }
        if (fn == "cos") {
            return "cos";
        }
        if (fn == "tan") {
            return "tan";
        }
        if (fn == "asin") {
            return "asin";
        }
        if (fn == "acos") {
            return "acos";
        }
        if (fn == "atan") {
            return "atan";
        }
        if (fn == "sinh") {
            return "sinh";
        }
        if (fn == "cosh") {
            return "cosh";
        }
        if (fn == "tanh") {
            return "tanh";
        }
        if (fn == "sqrt") {
            return "sqrt";
        }
        if (fn == "abs") {
            return "fabs";
        }
        if (fn == "exp") {
            return "exp";
        }
        if (fn == "log") {
            return "log";
        }
        if (fn == "log10") {
            return "log10";
        }
        if (fn == "floor") {
            return "floor";
        }
        if (fn == "ceil") {
            return "ceil";
        }
        return nullptr;
    }
    if (arity == 2) {
        if (fn == "pow") {
            return "pow";
        }
        if (fn == "min") {
            return "fmin";
        }
        if (fn == "max") {
            return "fmax";
        }
        if (fn == "atan2") {
            return "atan2";
        }
    }
    return nullptr;
}

llvm::Value* emit_binop(llvm::IRBuilder<>& builder, char op, llvm::Value* left, llvm::Value* right) {
    switch (op) {
    case '+':
        return builder.CreateFAdd(left, right);
    case '-':
        return builder.CreateFSub(left, right);
    case '*':
        return builder.CreateFMul(left, right);
    case '/':
        return builder.CreateFDiv(left, right);
    default:
        return left;
    }
}

struct ScalarEmitCtx {
    llvm::Function* function = nullptr;
    llvm::Module* module = nullptr;
    llvm::IRBuilder<>* builder = nullptr;
    llvm::Type* double_ty = nullptr;
    const std::map<std::string, unsigned>* param_map = nullptr;
};

llvm::Function* get_libm_fn(ScalarEmitCtx& ctx, const char* symbol, size_t arity) {
    if (auto* existing = ctx.module->getFunction(symbol)) {
        return existing;
    }
    std::vector<llvm::Type*> params(arity, ctx.double_ty);
    auto* fn_type = llvm::FunctionType::get(ctx.double_ty, params, false);
    return llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage, symbol, ctx.module);
}

llvm::Value* emit_scalar_call(ScalarEmitCtx& ctx, const std::string& name,
                              const std::vector<llvm::Value*>& args) {
    const char* symbol = libm_symbol_for_scalar_call(name, args.size());
    if (!symbol) {
        return nullptr;
    }
    llvm::Function* callee = get_libm_fn(ctx, symbol, args.size());
    return ctx.builder->CreateCall(callee, args);
}

llvm::Value* emit_scalar_expr(ScalarEmitCtx& ctx, const std::string& expr_text) {
    const std::string expr = trim_copy(strip_outer_parens(expr_text));
    if (expr.empty()) {
        return nullptr;
    }

    if (expr.front() == '-') {
        llvm::Value* inner = emit_scalar_expr(ctx, expr.substr(1));
        if (!inner) {
            return nullptr;
        }
        return ctx.builder->CreateFNeg(inner);
    }
    if (expr.front() == '+') {
        return emit_scalar_expr(ctx, expr.substr(1));
    }

    if (const auto call = parse_scalar_call(expr)) {
        std::vector<llvm::Value*> arg_values;
        arg_values.reserve(call->second.size());
        for (const auto& arg_text : call->second) {
            llvm::Value* arg_val = emit_scalar_expr(ctx, arg_text);
            if (!arg_val) {
                return nullptr;
            }
            arg_values.push_back(arg_val);
        }
        return emit_scalar_call(ctx, call->first, arg_values);
    }

    ScalarOperand single;
    if (parse_scalar_operand(expr, single)) {
        if (single.is_literal) {
            return llvm::ConstantFP::get(ctx.double_ty, single.literal);
        }
        const auto it = ctx.param_map->find(single.name);
        if (it == ctx.param_map->end()) {
            return nullptr;
        }
        return ctx.function->getArg(it->second);
    }

    const auto op_pos = find_scalar_binop(expr);
    if (!op_pos) {
        return nullptr;
    }

    llvm::Value* left = emit_scalar_expr(ctx, expr.substr(0, op_pos->first));
    llvm::Value* right = emit_scalar_expr(ctx, expr.substr(op_pos->first + 1));
    if (!left || !right) {
        return nullptr;
    }
    return emit_binop(*ctx.builder, op_pos->second, left, right);
}

bool compile_const_fn(llvm::orc::LLJIT& jit, const std::string& symbol, double value) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("ms_jit_const", *ctx);
    llvm::IRBuilder<> builder(*ctx);
    auto* fn_type = llvm::FunctionType::get(builder.getDoubleTy(), {}, false);
    auto* function =
        llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage, symbol, module.get());
    auto* entry = llvm::BasicBlock::Create(*ctx, "entry", function);
    builder.SetInsertPoint(entry);
    builder.CreateRet(llvm::ConstantFP::get(builder.getDoubleTy(), value));

    llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(ctx));
    return !jit.addIRModule(std::move(tsm));
}

bool compile_scalar_expr_fn(llvm::orc::LLJIT& jit, const std::string& symbol,
                            const std::string& expr, const std::vector<std::string>& params) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("ms_jit_expr", *ctx);
    llvm::IRBuilder<> builder(*ctx);
    llvm::Type* double_ty = builder.getDoubleTy();

    std::map<std::string, unsigned> param_map;
    for (unsigned i = 0; i < params.size(); ++i) {
        param_map[params[i]] = i;
    }

    std::vector<llvm::Type*> param_types(params.size(), double_ty);
    auto* fn_type = llvm::FunctionType::get(double_ty, param_types, false);
    auto* function =
        llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage, symbol, module.get());
    auto* entry = llvm::BasicBlock::Create(*ctx, "entry", function);
    builder.SetInsertPoint(entry);

    ScalarEmitCtx emit_ctx{function, module.get(), &builder, double_ty, &param_map};
    llvm::Value* result = emit_scalar_expr(emit_ctx, expr);
    if (!result) {
        return false;
    }
    builder.CreateRet(result);

    llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(ctx));
    return !jit.addIRModule(std::move(tsm));
}

std::optional<double> call_jit_fn0(llvm::orc::LLJIT& jit, const std::string& symbol) {
    auto sym = jit.lookup(symbol);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return std::nullopt;
    }
    using Fn = double (*)();
    const auto addr = static_cast<uintptr_t>(sym->getAddress());
    return reinterpret_cast<Fn>(addr)();
}

std::optional<double> call_jit_fn1(llvm::orc::LLJIT& jit, const std::string& symbol, double a) {
    auto sym = jit.lookup(symbol);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return std::nullopt;
    }
    using Fn = double (*)(double);
    const auto addr = static_cast<uintptr_t>(sym->getAddress());
    return reinterpret_cast<Fn>(addr)(a);
}

std::optional<double> call_jit_fn2(llvm::orc::LLJIT& jit, const std::string& symbol, double a,
                                   double b) {
    auto sym = jit.lookup(symbol);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return std::nullopt;
    }
    using Fn = double (*)(double, double);
    const auto addr = static_cast<uintptr_t>(sym->getAddress());
    return reinterpret_cast<Fn>(addr)(a, b);
}

std::optional<double> call_jit_fn3(llvm::orc::LLJIT& jit, const std::string& symbol, double a,
                                   double b, double c) {
    auto sym = jit.lookup(symbol);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return std::nullopt;
    }
    using Fn = double (*)(double, double, double);
    const auto addr = static_cast<uintptr_t>(sym->getAddress());
    return reinterpret_cast<Fn>(addr)(a, b, c);
}

std::optional<double> call_jit_fn4(llvm::orc::LLJIT& jit, const std::string& symbol, double a,
                                   double b, double c, double d) {
    auto sym = jit.lookup(symbol);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return std::nullopt;
    }
    using Fn = double (*)(double, double, double, double);
    const auto addr = static_cast<uintptr_t>(sym->getAddress());
    return reinterpret_cast<Fn>(addr)(a, b, c, d);
}

std::optional<double> call_jit_dynamic(llvm::orc::LLJIT& jit, const std::string& symbol,
                                       const std::vector<double>& args) {
    switch (args.size()) {
    case 0:
        return call_jit_fn0(jit, symbol);
    case 1:
        return call_jit_fn1(jit, symbol, args[0]);
    case 2:
        return call_jit_fn2(jit, symbol, args[0], args[1]);
    case 3:
        return call_jit_fn3(jit, symbol, args[0], args[1], args[2]);
    case 4:
        return call_jit_fn4(jit, symbol, args[0], args[1], args[2], args[3]);
    default:
        return std::nullopt;
    }
}

std::optional<std::vector<double>> collect_expr_args(const SessionState& state,
                                                     const std::vector<std::string>& params) {
    std::vector<double> args;
    args.reserve(params.size());
    for (const auto& name : params) {
        const auto it = state.scalars.find(name);
        if (it == state.scalars.end()) {
            return std::nullopt;
        }
        args.push_back(it->second);
    }
    return args;
}

bool jit_compile_smoke(llvm::orc::LLJIT& jit) {
    if (!compile_const_fn(jit, "ms_jit_smoke", 42.0)) {
        return false;
    }
    const auto value = call_jit_fn0(jit, "ms_jit_smoke");
    return value.has_value() && *value == 42.0;
}

class OrcJitLlvmBackend final : public JitBackend {
public:
    OrcJitLlvmBackend() {
        initialize_llvm_targets_once();
        auto jit = llvm::orc::LLJITBuilder().create();
        if (jit) {
            jit_ = std::move(*jit);
            can_compile_ = jit_compile_smoke(*jit_);
        }
    }

    Result<std::string> execute(const std::string& line) override {
        if (jit_ && can_compile_) {
            std::string name;
            double value = 0.0;
            if (Interpreter::try_parse_scalar_assignment(line, name, value)) {
                const std::string symbol = "ms_jit_c_" + std::to_string(++jit_sym_counter_);
                if (compile_const_fn(*jit_, symbol, value)) {
                    const auto compiled = call_jit_fn0(*jit_, symbol);
                    if (compiled.has_value()) {
                        return interp_.assign_scalar(name, *compiled);
                    }
                }
            }

            MatrixCallAssign matrix_call{};
            if (Interpreter::try_parse_matrix_call_assignment(line, matrix_call)) {
                return interp_.assign_matrix_call(matrix_call);
            }

            MultiMatrixCallAssign multi_call{};
            if (Interpreter::try_parse_multi_matrix_call_assignment(line, multi_call)) {
                return interp_.assign_multi_matrix_call(multi_call);
            }

            ScalarMatrixCallAssign scalar_matrix_call{};
            if (Interpreter::try_parse_scalar_matrix_call_assignment(line, scalar_matrix_call)) {
                return interp_.assign_scalar_matrix_call(scalar_matrix_call);
            }

            std::string expr;
            if (Interpreter::try_parse_scalar_expr_assignment(line, name, expr)) {
                const auto params = Interpreter::list_scalar_expr_variables(expr);
                const auto args = collect_expr_args(interp_.state(), params);
                if (args) {
                    const std::string symbol = "ms_jit_e_" + std::to_string(++jit_sym_counter_);
                    if (compile_scalar_expr_fn(*jit_, symbol, expr, params)) {
                        const auto compiled = call_jit_dynamic(*jit_, symbol, *args);
                        if (compiled.has_value()) {
                            return interp_.assign_scalar(name, *compiled);
                        }
                    }
                }
            }
        }
        return interp_.execute(line);
    }

    std::string backend_name() const override { return jit_ ? "orc-jit" : "orc-jit-llvm-fallback"; }

    const SessionState& state() const override { return interp_.state(); }

    void reset() override {
        interp_.reset();
        jit_sym_counter_ = 0;
    }

    JitCapabilities capabilities() const override {
        if (jit_ && can_compile_) {
            return {true, true,
                    "LLVM ORC LLJIT linked; scalar/matrix call dispatch; REPL fallback otherwise"};
        }
        if (jit_) {
            return {true, false, "LLVM ORC LLJIT linked; smoke compile failed; REPL fallback"};
        }
        return {true, false, "LLVM ORC init failed; delegates to REPL"};
    }

private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    bool can_compile_ = false;
    size_t jit_sym_counter_ = 0;
    Interpreter interp_;
};

} // namespace

std::unique_ptr<JitBackend> create_orc_jit_llvm_backend() {
    return std::make_unique<OrcJitLlvmBackend>();
}

} // namespace ms::interp
