#pragma once

#include "rule_context.hpp"

#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

namespace ms::plugin::rules {

class ExceptionRules {
public:
    explicit ExceptionRules(RuleEnvironment env);

    void registerDiagnostics(clang::DiagnosticsEngine& diag);

    bool visitCXXThrowExpr(clang::CXXThrowExpr* expr);
    bool visitCXXTryStmt(clang::CXXTryStmt* stmt);

private:
    RuleEnvironment env_;
    unsigned diag_throw_ = 0;
    unsigned diag_catch_ = 0;
};

} // namespace ms::plugin::rules
