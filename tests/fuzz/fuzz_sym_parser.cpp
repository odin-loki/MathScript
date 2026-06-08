#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

#include "ms/symbolic/symbolic.hpp"

namespace {

double read_f64(const uint8_t* data, size_t size, size_t offset) {
    if (offset + sizeof(double) > size) {
        return 0.0;
    }
    double value = 0.0;
    std::memcpy(&value, data + offset, sizeof(double));
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return std::max(-8.0, std::min(8.0, value));
}

struct BuildCtx {
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    int depth = 0;
    int nodes = 0;
};

uint8_t next_byte(BuildCtx& ctx) {
    if (ctx.pos >= ctx.size) {
        return 0;
    }
    return ctx.data[ctx.pos++];
}

ms::SymExpr build_expr(BuildCtx& ctx) {
    constexpr int kMaxDepth = 5;
    constexpr int kMaxNodes = 24;

    if (ctx.depth >= kMaxDepth || ctx.nodes >= kMaxNodes) {
        return ms::sym_const(read_f64(ctx.data, ctx.size, ctx.pos % ctx.size));
    }
    ++ctx.nodes;

    const uint8_t tag = next_byte(ctx) % 10;
    switch (tag) {
    case 0:
    case 1:
        return ms::sym_const(read_f64(ctx.data, ctx.size, (ctx.pos * 3) % ctx.size));
    case 2:
        return ms::sym_var((next_byte(ctx) & 1) ? "x" : "y");
    case 3: {
        ++ctx.depth;
        auto left = build_expr(ctx);
        auto right = build_expr(ctx);
        --ctx.depth;
        return ms::sym_add(std::move(left), std::move(right));
    }
    case 4: {
        ++ctx.depth;
        auto left = build_expr(ctx);
        auto right = build_expr(ctx);
        --ctx.depth;
        return ms::sym_mul(std::move(left), std::move(right));
    }
    case 5: {
        ++ctx.depth;
        auto arg = build_expr(ctx);
        --ctx.depth;
        return ms::sym_sin(std::move(arg));
    }
    case 6: {
        ++ctx.depth;
        auto arg = build_expr(ctx);
        --ctx.depth;
        return ms::sym_cos(std::move(arg));
    }
    case 7: {
        ++ctx.depth;
        auto arg = build_expr(ctx);
        --ctx.depth;
        return ms::sym_exp(std::move(arg));
    }
    case 8: {
        ++ctx.depth;
        auto arg = build_expr(ctx);
        --ctx.depth;
        const double v = read_f64(ctx.data, ctx.size, (ctx.pos * 5) % ctx.size);
        return ms::sym_log(ms::sym_const(std::max(0.1, std::abs(v) + 0.1)));
    }
    default: {
        ++ctx.depth;
        auto base = build_expr(ctx);
        const double exp = read_f64(ctx.data, ctx.size, (ctx.pos * 7) % ctx.size);
        --ctx.depth;
        return ms::sym_pow(std::move(base), ms::sym_const(std::max(-3.0, std::min(3.0, exp))));
    }
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 16 || size > 256) {
        return 0;
    }

    const std::string var = (data[0] & 1) ? "x" : "y";
    const std::map<std::string, double> env{
        {"x", read_f64(data, size, 8)},
        {"y", read_f64(data, size, 16)},
    };

    BuildCtx ctx{data, size, 0, 0, 0};
    auto expr = build_expr(ctx);
    auto simplified = ms::sym_simplify(std::move(expr));

    ctx.pos = 0;
    ctx.depth = 0;
    ctx.nodes = 0;
    expr = build_expr(ctx);
    auto differentiated = ms::sym_diff(ms::sym_simplify(std::move(expr)), var);
    (void)ms::sym_eval(simplified, env);
    (void)ms::sym_eval(differentiated, env);
    (void)ms::sym_to_string(simplified);
    (void)ms::sym_to_string(differentiated);
    return 0;
}
