#pragma once

#include "rule_context.hpp"

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

namespace ms::plugin::rules {

class CastRules {
public:
    explicit CastRules(RuleEnvironment env);

    void registerDiagnostics(clang::DiagnosticsEngine& diag);

    bool visitExprStmt(clang::ExprStmt* stmt);
    bool visitCStyleCastExpr(clang::CStyleCastExpr* expr);
    bool visitCXXConstCastExpr(clang::CXXConstCastExpr* expr);
    bool visitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr* expr);
    bool visitImplicitCastExpr(clang::ImplicitCastExpr* expr);
    bool visitBinaryOperator(clang::BinaryOperator* expr);
    bool visitGotoStmt(clang::GotoStmt* stmt);

private:
    RuleEnvironment env_;
    unsigned diag_cstyle_cast_ = 0;
    unsigned diag_const_cast_ = 0;
    unsigned diag_reinterpret_ = 0;
    unsigned diag_goto_ = 0;
    unsigned diag_ptr_arith_ = 0;
    unsigned diag_narrowing_ = 0;
    unsigned diag_signed_unsigned_ = 0;
    unsigned diag_unused_expected_ = 0;
};

} // namespace ms::plugin::rules
