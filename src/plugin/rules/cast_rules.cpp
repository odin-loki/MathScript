#include "cast_rules.hpp"

#include "clang/AST/OperationKinds.h"
#include "clang/Basic/Diagnostic.h"

namespace ms::plugin::rules {

CastRules::CastRules(RuleEnvironment env) : env_(env) {}

void CastRules::registerDiagnostics(clang::DiagnosticsEngine& diag) {
    diag_cstyle_cast_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] C-style cast is prohibited outside [[ms::unsafe]]. "
        "Use static_cast, const_cast, or reinterpret_cast instead.");
    diag_const_cast_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'const_cast' is prohibited outside [[ms::unsafe]].");
    diag_reinterpret_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'reinterpret_cast' is prohibited outside [[ms::unsafe]].");
    diag_goto_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'goto' is prohibited outside [[ms::unsafe]].");
    diag_ptr_arith_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] raw pointer arithmetic is prohibited outside [[ms::unsafe]].");
    diag_narrowing_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] implicit narrowing conversion is prohibited outside [[ms::unsafe]]. "
        "Use static_cast or ms::narrow instead.");
    diag_signed_unsigned_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] signed/unsigned comparison is prohibited outside [[ms::unsafe]].");
    diag_unused_expected_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] discarding std::expected/ms::Result is prohibited outside [[ms::unsafe]]. "
        "Check the return value or propagate the error.");
}

bool CastRules::visitExprStmt(clang::ExprStmt* stmt) {
    if (!stmt || env_.unsafe_depth > 0) {
        return true;
    }
    const auto* call =
        clang::dyn_cast<clang::CallExpr>(stmt->getExpr()->IgnoreParenImpCasts());
    if (!call) {
        return true;
    }
    const clang::QualType ret = call->getCallReturnType(env_.Ctx);
    if (!typeIsExpected(ret)) {
        return true;
    }
    env_.CI.getDiagnostics().Report(call->getBeginLoc(), diag_unused_expected_);
    return true;
}

bool CastRules::visitCStyleCastExpr(clang::CStyleCastExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_cstyle_cast_);
    return true;
}

bool CastRules::visitCXXConstCastExpr(clang::CXXConstCastExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_const_cast_);
    return true;
}

bool CastRules::visitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_reinterpret_);
    return true;
}

bool CastRules::visitImplicitCastExpr(clang::ImplicitCastExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    const auto kind = expr->getCastKind();
    if (kind == clang::ImplicitCastKind::FloatingToIntegral) {
        env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_narrowing_);
        return true;
    }
    if (kind != clang::ImplicitCastKind::IntegralCast) {
        return true;
    }
    const clang::QualType dst = expr->getType().getCanonicalType();
    const clang::QualType src = expr->getSubExpr()->getType().getCanonicalType();
    if (!src->isIntegralOrEnumerationType() || !dst->isIntegralOrEnumerationType()) {
        return true;
    }
    if (env_.Ctx.getIntWidth(src) > env_.Ctx.getIntWidth(dst)) {
        env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_narrowing_);
    }
    return true;
}

bool CastRules::visitBinaryOperator(clang::BinaryOperator* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    const auto op = expr->getOpcode();

    if (clang::BinaryOperator::isComparisonOp(op)) {
        const clang::QualType lhs =
            expr->getLHS()->IgnoreImpCasts()->getType().getCanonicalType();
        const clang::QualType rhs =
            expr->getRHS()->IgnoreImpCasts()->getType().getCanonicalType();
        if ((typeIsSignedIntegral(lhs) && typeIsUnsignedIntegral(rhs)) ||
            (typeIsUnsignedIntegral(lhs) && typeIsSignedIntegral(rhs))) {
            env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_signed_unsigned_);
            return true;
        }
    }

    if (op != clang::BO_Add && op != clang::BO_Sub) {
        return true;
    }
    const clang::Expr* lhs = expr->getLHS()->IgnoreParenImpCasts();
    const clang::Expr* rhs = expr->getRHS()->IgnoreParenImpCasts();
    const auto is_ptr = [](const clang::Expr* e) {
        return e && e->getType()->isPointerType();
    };
    if (!is_ptr(lhs) && !is_ptr(rhs)) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_ptr_arith_);
    return true;
}

bool CastRules::visitGotoStmt(clang::GotoStmt* stmt) {
    if (!stmt || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(stmt->getBeginLoc(), diag_goto_);
    return true;
}

} // namespace ms::plugin::rules
