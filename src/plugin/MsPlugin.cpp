// MathScript C++ Profile Enforcement Plugin
// Registers with Clang and runs all profile rules on every translation unit

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "rule_context.hpp"
#include "rules/cast_rules.hpp"
#include "rules/exception_rules.hpp"
#include "rules/memory_rules.hpp"
#include "unsafe_registry.hpp"

namespace ms::plugin {

namespace {

class MsProfileVisitor : public clang::RecursiveASTVisitor<MsProfileVisitor> {
    unsigned unsafe_depth_ = 0;
    RuleEnvironment env_;
    rules::MemoryRules memory_rules_;
    rules::ExceptionRules exception_rules_;
    rules::CastRules cast_rules_;

public:
    MsProfileVisitor(clang::CompilerInstance& CI, clang::ASTContext& Ctx)
        : env_{CI, Ctx, unsafe_depth_},
          memory_rules_(env_),
          exception_rules_(env_),
          cast_rules_(env_) {
        auto& diag = CI.getDiagnostics();
        memory_rules_.registerDiagnostics(diag);
        exception_rules_.registerDiagnostics(diag);
        cast_rules_.registerDiagnostics(diag);
    }

    bool TraverseFunctionDecl(clang::FunctionDecl* fn) {
        const bool unsafe = declIsMsUnsafeContext(fn, env_.Ctx);
        if (unsafe) {
            recordUnsafeSite(fn, env_.Ctx);
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
            recordUnsafeSite(stmt->getBeginLoc(), reason, env_.Ctx);
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
            recordUnsafeSite(decl, env_.Ctx);
            ++unsafe_depth_;
        }
        const bool result = RecursiveASTVisitor::TraverseCXXRecordDecl(decl);
        if (unsafe) {
            --unsafe_depth_;
        }
        return result;
    }

    bool VisitExprStmt(clang::ExprStmt* stmt) {
        return cast_rules_.visitExprStmt(stmt);
    }

    bool VisitFieldDecl(clang::FieldDecl* decl) {
        return memory_rules_.visitFieldDecl(decl);
    }

    bool VisitCXXNewExpr(clang::CXXNewExpr* expr) {
        return memory_rules_.visitCXXNewExpr(expr);
    }

    bool VisitCXXDeleteExpr(clang::CXXDeleteExpr* expr) {
        return memory_rules_.visitCXXDeleteExpr(expr);
    }

    bool VisitCStyleCastExpr(clang::CStyleCastExpr* expr) {
        return cast_rules_.visitCStyleCastExpr(expr);
    }

    bool VisitCXXThrowExpr(clang::CXXThrowExpr* expr) {
        return exception_rules_.visitCXXThrowExpr(expr);
    }

    bool VisitCallExpr(clang::CallExpr* expr) {
        return memory_rules_.visitCallExpr(expr);
    }

    bool VisitCXXTryStmt(clang::CXXTryStmt* stmt) {
        return exception_rules_.visitCXXTryStmt(stmt);
    }

    bool VisitCXXConstCastExpr(clang::CXXConstCastExpr* expr) {
        return cast_rules_.visitCXXConstCastExpr(expr);
    }

    bool VisitGotoStmt(clang::GotoStmt* stmt) {
        return cast_rules_.visitGotoStmt(stmt);
    }

    bool VisitBinaryOperator(clang::BinaryOperator* expr) {
        return cast_rules_.visitBinaryOperator(expr);
    }

    bool VisitImplicitCastExpr(clang::ImplicitCastExpr* expr) {
        return cast_rules_.visitImplicitCastExpr(expr);
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr* expr) {
        return memory_rules_.visitCXXConstructExpr(expr);
    }

    bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr* expr) {
        return cast_rules_.visitCXXReinterpretCastExpr(expr);
    }

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* expr) {
        return memory_rules_.visitCXXMemberCallExpr(expr);
    }

    bool VisitVarDecl(clang::VarDecl* decl) {
        return memory_rules_.visitVarDecl(decl);
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
