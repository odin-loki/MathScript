// MathScript C++ Profile Enforcement Plugin
// Registers with Clang and runs all profile rules on every translation unit

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclNamespace.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "unsafe_registry.hpp"

namespace ms::plugin {

namespace {

bool attrIsMsUnsafe(const clang::Attr* attr) {
    if (!attr) {
        return false;
    }
    const llvm::StringRef spelling = attr->getSpelling();
    if (spelling == "unsafe" || spelling.contains("ms::unsafe")) {
        return true;
    }
    if (const auto* annotate = clang::dyn_cast<clang::AnnotateAttr>(attr)) {
        return llvm::StringRef(annotate->getAnnotation()).contains("ms::unsafe");
    }
    return false;
}

bool declHasMsUnsafe(const clang::Decl* decl) {
    if (!decl) {
        return false;
    }
    for (const auto* attr : decl->attrs()) {
        if (attrIsMsUnsafe(attr)) {
            return true;
        }
    }
    return false;
}

bool lineContainsMsUnsafe(const clang::SourceManager& sm, clang::FileID fid,
                          unsigned line_num) {
    const auto line_start = sm.translateLineCol(fid, line_num, 1);
    if (line_start.isInvalid()) {
        return false;
    }
    bool invalid = false;
    const llvm::StringRef buf = sm.getBufferData(fid, &invalid);
    if (invalid) {
        return false;
    }
    const auto offset = sm.getDecomposedLoc(sm.getFileLoc(line_start)).second;
    if (offset >= buf.size()) {
        return false;
    }
    const char* cursor = buf.data() + offset;
    const char* const end = buf.data() + buf.size();
    while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        ++cursor;
    }
    const llvm::StringRef line(buf.data() + offset, cursor - (buf.data() + offset));
    return line.contains("ms::unsafe") || line.contains("MS_UNSAFE");
}

bool functionSourceHasMsUnsafe(const clang::FunctionDecl* fn,
                               const clang::ASTContext& ctx) {
    if (!fn) {
        return false;
    }
    const auto& sm = ctx.getSourceManager();
    const auto begin = sm.getExpansionLoc(fn->getBeginLoc());
    if (begin.isInvalid()) {
        return false;
    }
    const auto fid = sm.getFileID(begin);
    const auto fn_line = sm.getSpellingLineNumber(begin);
    for (unsigned line = fn_line; line > 0 && line + 2 >= fn_line; --line) {
        if (lineContainsMsUnsafe(sm, fid, line)) {
            return true;
        }
    }
    return false;
}

bool declIsMsUnsafeContext(const clang::Decl* decl, const clang::ASTContext& ctx) {
    if (!decl) {
        return false;
    }
    if (declHasMsUnsafe(decl)) {
        return true;
    }
    if (const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
        return functionSourceHasMsUnsafe(fn, ctx);
    }
    return false;
}

bool stmtHasMsUnsafe(const clang::AttributedStmt* attr_stmt) {
    if (!attr_stmt) {
        return false;
    }
    for (const auto* attr : attr_stmt->getAttrs()) {
        if (attrIsMsUnsafe(attr)) {
            return true;
        }
    }
    return false;
}

std::string extractQuotedReason(llvm::StringRef text) {
    const auto open = text.find('"');
    if (open == llvm::StringRef::npos) {
        return {};
    }
    const auto close = text.find('"', open + 1);
    if (close == llvm::StringRef::npos) {
        return {};
    }
    return text.slice(open + 1, close).str();
}

std::string extractReasonFromAttr(const clang::Attr* attr) {
    if (!attr) {
        return {};
    }
    if (const auto* annotate = clang::dyn_cast<clang::AnnotateAttr>(attr)) {
        const llvm::StringRef annotation = annotate->getAnnotation();
        if (std::string reason = extractQuotedReason(annotation); !reason.empty()) {
            return reason;
        }
        return annotation.str();
    }
    return extractQuotedReason(attr->getSpelling());
}

std::string extractReasonFromSourceLine(const clang::SourceManager& sm,
                                        clang::FileID fid, unsigned line_num) {
    const auto line_start = sm.translateLineCol(fid, line_num, 1);
    if (line_start.isInvalid()) {
        return {};
    }
    bool invalid = false;
    const llvm::StringRef buf = sm.getBufferData(fid, &invalid);
    if (invalid) {
        return {};
    }
    const auto offset = sm.getDecomposedLoc(sm.getFileLoc(line_start)).second;
    if (offset >= buf.size()) {
        return {};
    }
    const char* cursor = buf.data() + offset;
    const char* const end = buf.data() + buf.size();
    while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        ++cursor;
    }
    const llvm::StringRef line(buf.data() + offset, cursor - (buf.data() + offset));
    if (std::string reason = extractQuotedReason(line); !reason.empty()) {
        return reason;
    }
    return "unspecified";
}

std::string extractReasonFromDecl(const clang::Decl* decl, const clang::ASTContext& ctx) {
    if (!decl) {
        return {};
    }
    for (const auto* attr : decl->attrs()) {
        if (!attrIsMsUnsafe(attr)) {
            continue;
        }
        if (std::string reason = extractReasonFromAttr(attr); !reason.empty()) {
            return reason;
        }
    }

    const auto& sm = ctx.getSourceManager();
    const auto begin = sm.getExpansionLoc(decl->getBeginLoc());
    if (begin.isInvalid()) {
        return "unspecified";
    }
    const auto fid = sm.getFileID(begin);
    const auto decl_line = sm.getSpellingLineNumber(begin);
    for (unsigned line = decl_line; line > 0 && line + 2 >= decl_line; --line) {
        if (!lineContainsMsUnsafe(sm, fid, line)) {
            continue;
        }
        if (std::string reason = extractReasonFromSourceLine(sm, fid, line);
            !reason.empty()) {
            return reason;
        }
    }
    return "unspecified";
}

void recordUnsafeSite(const clang::Decl* decl, const clang::ASTContext& ctx) {
    if (!decl) {
        return;
    }
    const auto& sm = ctx.getSourceManager();
    const auto begin = sm.getExpansionLoc(decl->getBeginLoc());
    if (begin.isInvalid()) {
        return;
    }
    const auto presumed = sm.getPresumedLoc(begin);
    if (presumed.isInvalid()) {
        return;
    }
    const std::string file = presumed.getFilename();
    const int line = static_cast<int>(presumed.getLine());
    const std::string reason = extractReasonFromDecl(decl, ctx);
    UnsafeRegistry::instance().record_unsafe(file, line, reason, "");
}

void recordUnsafeSite(clang::SourceLocation loc, const std::string& reason,
                      const clang::ASTContext& ctx) {
    (void)ctx;
    const auto& sm = ctx.getSourceManager();
    const auto begin = sm.getExpansionLoc(loc);
    if (begin.isInvalid()) {
        return;
    }
    const auto presumed = sm.getPresumedLoc(begin);
    if (presumed.isInvalid()) {
        return;
    }
    UnsafeRegistry::instance().record_unsafe(
        presumed.getFilename(), static_cast<int>(presumed.getLine()), reason, "");
}

bool recordIsInStdNamespace(const clang::CXXRecordDecl* record) {
    if (!record) {
        return false;
    }
    const clang::DeclContext* ctx = record->getDeclContext();
    while (ctx) {
        if (const auto* ns = clang::dyn_cast<clang::NamespaceDecl>(ctx)) {
            if (ns->isStdNamespace()) {
                return true;
            }
        }
        ctx = ctx->getParent();
    }
    return false;
}

bool recordIsMutexType(const clang::CXXRecordDecl* record) {
    if (!record) {
        return false;
    }
    const llvm::StringRef name = record->getName();
    return name == "mutex" || name == "recursive_mutex" || name == "timed_mutex" ||
           name == "shared_mutex";
}

bool typeIsSignedIntegral(clang::QualType type) {
    return type.getCanonicalType()->isSignedIntegerOrEnumerationType();
}

bool typeIsUnsignedIntegral(clang::QualType type) {
    return type.getCanonicalType()->isUnsignedIntegerOrEnumerationType();
}

bool typeIsStdSpan(clang::QualType type) {
    const clang::QualType canon = type.getCanonicalType();
    const auto* record = canon->getAsCXXRecordDecl();
    return record && record->getName() == "span" && recordIsInStdNamespace(record);
}

bool typeIsExpected(clang::QualType type) {
    const clang::QualType canon = type.getCanonicalType();
    const auto* record = canon->getAsCXXRecordDecl();
    return record && record->getName() == "expected";
}

class MsProfileVisitor : public clang::RecursiveASTVisitor<MsProfileVisitor> {
    clang::CompilerInstance& CI_;
    clang::ASTContext& Ctx_;
    unsigned unsafe_depth_ = 0;
    unsigned diag_new_;
    unsigned diag_delete_;
    unsigned diag_cstyle_cast_;
    unsigned diag_throw_;
    unsigned diag_malloc_;
    unsigned diag_catch_;
    unsigned diag_const_cast_;
    unsigned diag_goto_;
    unsigned diag_ptr_arith_;
    unsigned diag_reinterpret_;
    unsigned diag_detach_;
    unsigned diag_vla_;
    unsigned diag_narrowing_;
    unsigned diag_signed_unsigned_;
    unsigned diag_raw_thread_;
    unsigned diag_raw_mutex_lock_;
    unsigned diag_uninit_;
    unsigned diag_stored_span_;
    unsigned diag_volatile_sync_;
    unsigned diag_owning_raw_ptr_;
    unsigned diag_unused_expected_;

public:
    MsProfileVisitor(clang::CompilerInstance& CI, clang::ASTContext& Ctx)
        : CI_(CI), Ctx_(Ctx) {
        auto& diag = CI_.getDiagnostics();
        diag_new_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'new' is prohibited outside [[ms::unsafe]]. "
            "Use ms::make_unique or ms::make_shared instead.");
        diag_delete_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'delete' is prohibited outside [[ms::unsafe]]. "
            "Use RAII containers instead.");
        diag_cstyle_cast_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] C-style cast is prohibited outside [[ms::unsafe]]. "
            "Use static_cast, const_cast, or reinterpret_cast instead.");
        diag_throw_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'throw' is prohibited outside [[ms::unsafe]].");
        diag_malloc_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'malloc'/'calloc'/'realloc'/'free' are prohibited "
            "outside [[ms::unsafe]]. Use RAII containers or ms allocators instead.");
        diag_catch_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'try'/'catch' is prohibited outside [[ms::unsafe]].");
        diag_const_cast_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'const_cast' is prohibited outside [[ms::unsafe]].");
        diag_goto_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'goto' is prohibited outside [[ms::unsafe]].");
        diag_ptr_arith_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] raw pointer arithmetic is prohibited outside [[ms::unsafe]].");
        diag_reinterpret_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'reinterpret_cast' is prohibited outside [[ms::unsafe]].");
        diag_detach_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] 'std::thread::detach' is prohibited outside [[ms::unsafe]].");
        diag_vla_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] variable-length arrays are prohibited outside [[ms::unsafe]].");
        diag_narrowing_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] implicit narrowing conversion is prohibited outside [[ms::unsafe]]. "
            "Use static_cast or ms::narrow instead.");
        diag_signed_unsigned_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] signed/unsigned comparison is prohibited outside [[ms::unsafe]].");
        diag_raw_thread_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] raw std::thread construction is prohibited outside [[ms::unsafe]]. "
            "Use ms::thread wrappers or a thread pool.");
        diag_raw_mutex_lock_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] manual mutex.lock()/try_lock() is prohibited outside [[ms::unsafe]]. "
            "Use std::lock_guard or std::unique_lock.");
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
        diag_unused_expected_ = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "[ms-profile] discarding std::expected/ms::Result is prohibited outside [[ms::unsafe]]. "
            "Check the return value or propagate the error.");
    }

    bool TraverseFunctionDecl(clang::FunctionDecl* fn) {
        const bool unsafe = declIsMsUnsafeContext(fn, Ctx_);
        if (unsafe) {
            recordUnsafeSite(fn, Ctx_);
            ++unsafe_depth_;
        }
        const bool result = RecursiveASTVisitor::TraverseFunctionDecl(fn);
        if (unsafe) {
            --unsafe_depth_;
        }
        return result;
    }

    bool TraverseAttributedStmt(clang::AttributedStmt* stmt) {
        const bool unsafe = stmtHasMsUnsafe(stmt);
        if (unsafe) {
            std::string reason = "unspecified";
            for (const auto* attr : stmt->getAttrs()) {
                if (!attrIsMsUnsafe(attr)) {
                    continue;
                }
                if (std::string parsed = extractReasonFromAttr(attr); !parsed.empty()) {
                    reason = std::move(parsed);
                    break;
                }
            }
            recordUnsafeSite(stmt->getBeginLoc(), reason, Ctx_);
            ++unsafe_depth_;
        }
        const bool result = RecursiveASTVisitor::TraverseAttributedStmt(stmt);
        if (unsafe) {
            --unsafe_depth_;
        }
        return result;
    }

    bool TraverseCXXRecordDecl(clang::CXXRecordDecl* decl) {
        const bool unsafe = declHasMsUnsafe(decl);
        if (unsafe) {
            recordUnsafeSite(decl, Ctx_);
            ++unsafe_depth_;
        }
        const bool result = RecursiveASTVisitor::TraverseCXXRecordDecl(decl);
        if (unsafe) {
            --unsafe_depth_;
        }
        return result;
    }

    bool VisitExprStmt(clang::ExprStmt* stmt) {
        if (!stmt || unsafe_depth_ > 0) {
            return true;
        }
        const auto* call =
            clang::dyn_cast<clang::CallExpr>(stmt->getExpr()->IgnoreParenImpCasts());
        if (!call) {
            return true;
        }
        const clang::QualType ret = call->getCallReturnType(Ctx_);
        if (!typeIsExpected(ret)) {
            return true;
        }
        CI_.getDiagnostics().Report(call->getBeginLoc(), diag_unused_expected_);
        return true;
    }

    bool VisitFieldDecl(clang::FieldDecl* decl) {
        if (!decl || unsafe_depth_ > 0) {
            return true;
        }
        const clang::QualType type = decl->getType();
        if (typeIsStdSpan(type)) {
            CI_.getDiagnostics().Report(decl->getBeginLoc(), diag_stored_span_);
            return true;
        }
        if (type->isPointerType() && !type.isConstQualified()) {
            CI_.getDiagnostics().Report(decl->getBeginLoc(), diag_owning_raw_ptr_);
        }
        return true;
    }

    bool VisitCXXNewExpr(clang::CXXNewExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_new_);
        return true;
    }

    bool VisitCXXDeleteExpr(clang::CXXDeleteExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_delete_);
        return true;
    }

    bool VisitCStyleCastExpr(clang::CStyleCastExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_cstyle_cast_);
        return true;
    }

    bool VisitCXXThrowExpr(clang::CXXThrowExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_throw_);
        return true;
    }

    bool VisitCallExpr(clang::CallExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        const auto* callee = expr->getDirectCallee();
        if (!callee) {
            return true;
        }
        const llvm::StringRef name = callee->getName();
        if (name != "malloc" && name != "calloc" && name != "realloc" &&
            name != "free") {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_malloc_);
        return true;
    }

    bool VisitCXXTryStmt(clang::CXXTryStmt* stmt) {
        if (!stmt || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(stmt->getBeginLoc(), diag_catch_);
        return true;
    }

    bool VisitCXXConstCastExpr(clang::CXXConstCastExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_const_cast_);
        return true;
    }

    bool VisitGotoStmt(clang::GotoStmt* stmt) {
        if (!stmt || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(stmt->getBeginLoc(), diag_goto_);
        return true;
    }

    bool VisitBinaryOperator(clang::BinaryOperator* expr) {
        if (!expr || unsafe_depth_ > 0) {
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
                CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_signed_unsigned_);
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
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_ptr_arith_);
        return true;
    }

    bool VisitImplicitCastExpr(clang::ImplicitCastExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        const auto kind = expr->getCastKind();
        if (kind == clang::ImplicitCastKind::FloatingToIntegral) {
            CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_narrowing_);
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
        if (Ctx_.getIntWidth(src) > Ctx_.getIntWidth(dst)) {
            CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_narrowing_);
        }
        return true;
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
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
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_raw_thread_);
        return true;
    }

    bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_reinterpret_);
        return true;
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* expr) {
        if (!expr || unsafe_depth_ > 0) {
            return true;
        }
        const auto* method = expr->getMethodDecl();
        if (!method) {
            return true;
        }
        const llvm::StringRef name = method->getName();
        if (name == "detach") {
            CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_detach_);
            return true;
        }
        if (name == "lock" || name == "try_lock") {
            const auto* record = method->getParent();
            if (recordIsMutexType(record)) {
                CI_.getDiagnostics().Report(expr->getBeginLoc(), diag_raw_mutex_lock_);
            }
        }
        return true;
    }

    bool VisitVarDecl(clang::VarDecl* decl) {
        if (!decl || unsafe_depth_ > 0) {
            return true;
        }
        if (decl->getType()->isVariableArrayType()) {
            CI_.getDiagnostics().Report(decl->getBeginLoc(), diag_vla_);
            return true;
        }
        if (decl->getType().isVolatileQualified() && decl->hasLocalStorage()) {
            CI_.getDiagnostics().Report(decl->getBeginLoc(), diag_volatile_sync_);
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
        CI_.getDiagnostics().Report(decl->getBeginLoc(), diag_uninit_);
        return true;
    }
};

} // namespace

class MsASTConsumer : public clang::ASTConsumer {
    clang::CompilerInstance& CI_;
    unsigned diag_unsafe_audit_note_ = 0;

public:
    explicit MsASTConsumer(clang::CompilerInstance& CI)
        : CI_(CI) {
        diag_unsafe_audit_note_ = CI_.getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Note,
            "[ms-profile] unsafe audit: %0 site(s) recorded in this translation unit");
    }

    void HandleTranslationUnit(clang::ASTContext& ctx) override {
        MsProfileVisitor visitor(CI_, ctx);
        visitor.TraverseDecl(ctx.getTranslationUnitDecl());

        const auto sites = UnsafeRegistry::instance().get_sites();
        if (sites.empty()) {
            return;
        }

#if defined(MS_PLUGIN_AUDIT_DIR)
        (void)UnsafeRegistry::emit_audit_report(MS_PLUGIN_AUDIT_DIR, sites);
#endif

        CI_.getDiagnostics().Report(ctx.getTranslationUnitDecl()->getBeginLoc(),
                                    diag_unsafe_audit_note_)
            << static_cast<int>(sites.size());
    }
};

class MsPlugin : public clang::PluginASTAction {
public:
    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance& CI,
                      llvm::StringRef) override {
        return std::make_unique<MsASTConsumer>(CI);
    }

    bool ParseArgs(const clang::CompilerInstance&,
                   const std::vector<std::string>&) override {
        return true;
    }

    clang::PluginASTAction::ActionType getActionType() override {
        return AddBeforeMainAction;
    }
};

// Register the plugin with Clang
static clang::FrontendPluginRegistry::Add<ms::plugin::MsPlugin>
    X("ms-profile", "MathScript C++ Profile Enforcement");

} // namespace ms::plugin
