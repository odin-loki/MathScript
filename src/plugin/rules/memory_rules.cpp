#include "memory_rules.hpp"

#include "clang/AST/OperationKinds.h"
#include "clang/Basic/Diagnostic.h"

namespace ms::plugin::rules {

MemoryRules::MemoryRules(RuleEnvironment env) : env_(env) {}

void MemoryRules::registerDiagnostics(clang::DiagnosticsEngine& diag) {
    diag_new_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'new' is prohibited outside [[ms::unsafe]]. "
        "Use ms::make_unique or ms::make_shared instead.");
    diag_delete_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'delete' is prohibited outside [[ms::unsafe]]. "
        "Use RAII containers instead.");
    diag_malloc_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'malloc'/'calloc'/'realloc'/'free' are prohibited "
        "outside [[ms::unsafe]]. Use RAII containers or ms allocators instead.");
    diag_vla_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] variable-length arrays are prohibited outside [[ms::unsafe]].");
    diag_uninit_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] uninitialized local variables are prohibited outside [[ms::unsafe]].");
    diag_stored_span_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] storing std::span past callee lifetime is prohibited outside [[ms::unsafe]].");
    diag_volatile_sync_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'volatile' for synchronization is prohibited outside [[ms::unsafe]].");
    diag_owning_raw_ptr_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] owning raw pointer members are prohibited outside [[ms::unsafe]]. "
        "Use smart pointers or containers.");
    diag_raw_thread_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] raw std::thread construction is prohibited outside [[ms::unsafe]]. "
        "Use ms::thread wrappers or a thread pool.");
    diag_raw_mutex_lock_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] manual mutex.lock()/try_lock() is prohibited outside [[ms::unsafe]]. "
        "Use std::lock_guard or std::unique_lock.");
    diag_detach_ = diag.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "[ms-profile] 'std::thread::detach' is prohibited outside [[ms::unsafe]].");
}

bool MemoryRules::visitFieldDecl(clang::FieldDecl* decl) {
    if (!decl || env_.unsafe_depth > 0) {
        return true;
    }
    const clang::QualType type = decl->getType();
    if (typeIsStdSpan(type)) {
        env_.CI.getDiagnostics().Report(decl->getBeginLoc(), diag_stored_span_);
        return true;
    }
    if (type->isPointerType() && !type.isConstQualified()) {
        env_.CI.getDiagnostics().Report(decl->getBeginLoc(), diag_owning_raw_ptr_);
    }
    return true;
}

bool MemoryRules::visitVarDecl(clang::VarDecl* decl) {
    if (!decl || env_.unsafe_depth > 0) {
        return true;
    }
    if (decl->getType()->isVariableArrayType()) {
        env_.CI.getDiagnostics().Report(decl->getBeginLoc(), diag_vla_);
        return true;
    }
    if (decl->getType().isVolatileQualified() && decl->hasLocalStorage()) {
        env_.CI.getDiagnostics().Report(decl->getBeginLoc(), diag_volatile_sync_);
        return true;
    }
    if (clang::isa<clang::ParmVarDecl>(decl)) {
        return true;
    }
    if (!decl->hasLocalStorage() || decl->isImplicit()) {
        return true;
    }
    if (decl->hasInit() || decl->getInit() != nullptr) {
        return true;
    }
    if (decl->getType()->isReferenceType()) {
        return true;
    }
    env_.CI.getDiagnostics().Report(decl->getBeginLoc(), diag_uninit_);
    return true;
}

bool MemoryRules::visitCXXNewExpr(clang::CXXNewExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_new_);
    return true;
}

bool MemoryRules::visitCXXDeleteExpr(clang::CXXDeleteExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_delete_);
    return true;
}

bool MemoryRules::visitCallExpr(clang::CallExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    const auto* callee = expr->getDirectCallee();
    if (!callee) {
        return true;
    }
    const llvm::StringRef name = callee->getName();
    if (name != "malloc" && name != "calloc" && name != "realloc" && name != "free") {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_malloc_);
    return true;
}

bool MemoryRules::visitCXXConstructExpr(clang::CXXConstructExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    const auto* ctor = expr->getConstructor();
    if (!ctor) {
        return true;
    }
    const auto* record = ctor->getParent();
    if (!record || record->getName() != "thread" || !recordIsInStdNamespace(record)) {
        return true;
    }
    env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_raw_thread_);
    return true;
}

bool MemoryRules::visitCXXMemberCallExpr(clang::CXXMemberCallExpr* expr) {
    if (!expr || env_.unsafe_depth > 0) {
        return true;
    }
    const auto* method = expr->getMethodDecl();
    if (!method) {
        return true;
    }
    const llvm::StringRef name = method->getName();
    if (name == "detach") {
        env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_detach_);
        return true;
    }
    if (name == "lock" || name == "try_lock") {
        const auto* record = method->getParent();
        if (recordIsMutexType(record)) {
            env_.CI.getDiagnostics().Report(expr->getBeginLoc(), diag_raw_mutex_lock_);
        }
    }
    return true;
}

} // namespace ms::plugin::rules
