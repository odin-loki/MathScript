#include "exception_rules.hpp"

#include "clang/Basic/Diagnostic.h"

namespace ms::plugin::rules {

ExceptionRules::ExceptionRules(RuleEnvironment env) : env_(env) {}

void ExceptionRules::registerDiagnostics(clang::DiagnosticsEngine& diag) {
    diag_throw_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'throw' is prohibited outside [[ms::unsafe]].");
    diag_catch_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'try'/'catch' is prohibited outside [[ms::unsafe]].");
}

bool ExceptionRules::visitCXXThrowExpr(clang::CXXThrowExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_throw_);
    return true;
}

bool ExceptionRules::visitCXXTryStmt(clang::CXXTryStmt* stmt) {
    if (!stmt || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(stmt->getBeginLoc(), diag_catch_);
    return true;
}

} // namespace ms::plugin::rules
