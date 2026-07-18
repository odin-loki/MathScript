#pragma once

#include "rule_context.hpp"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"

namespace ms::plugin::rules {

class MemoryRules {
public:
    explicit MemoryRules(RuleEnvironment env);

    void registerDiagnostics(clang::DiagnosticsEngine& diag);

    bool visitFieldDecl(clang::FieldDecl* decl);
    bool visitVarDecl(clang::VarDecl* decl);
    bool visitCXXNewExpr(clang::CXXNewExpr* expr);
    bool visitCXXDeleteExpr(clang::CXXDeleteExpr* expr);
    bool visitCallExpr(clang::CallExpr* expr);
    bool visitCXXConstructExpr(clang::CXXConstructExpr* expr);
    bool visitCXXMemberCallExpr(clang::CXXMemberCallExpr* expr);

private:
    RuleEnvironment env_;
    unsigned diag_new_ = 0;
    unsigned diag_delete_ = 0;
    unsigned diag_malloc_ = 0;
    unsigned diag_vla_ = 0;
    unsigned diag_uninit_ = 0;
    unsigned diag_stored_span_ = 0;
    unsigned diag_volatile_sync_ = 0;
    unsigned diag_owning_raw_ptr_ = 0;
    unsigned diag_raw_thread_ = 0;
    unsigned diag_raw_mutex_lock_ = 0;
    unsigned diag_detach_ = 0;
};

} // namespace ms::plugin::rules
