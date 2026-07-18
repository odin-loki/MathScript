#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclNamespace.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"

#include "unsafe_registry.hpp"

#include <string>

namespace ms::plugin {

struct RuleEnvironment {
    clang::CompilerInstance& CI;
    clang::ASTContext& Ctx;
    unsigned& unsafe_depth;
};

bool attrIsMsUnsafe(const clang::Attr* attr);
bool declHasMsUnsafe(const clang::Decl* decl);
bool lineContainsMsUnsafe(const clang::SourceManager& sm, clang::FileID fid,
                          unsigned line_num);
bool functionSourceHasMsUnsafe(const clang::FunctionDecl* fn,
                               const clang::ASTContext& ctx);
bool declIsMsUnsafeContext(const clang::Decl* decl, const clang::ASTContext& ctx);
bool stmtHasMsUnsafe(const clang::AttributedStmt* attr_stmt);

std::string extractQuotedReason(llvm::StringRef text);
std::string extractReasonFromAttr(const clang::Attr* attr);
std::string extractReasonFromSourceLine(const clang::SourceManager& sm,
                                        clang::FileID fid, unsigned line_num);
std::string extractReasonFromDecl(const clang::Decl* decl, const clang::ASTContext& ctx);

void recordUnsafeSite(const clang::Decl* decl, const clang::ASTContext& ctx);
void recordUnsafeSite(clang::SourceLocation loc, const std::string& reason,
                      const clang::ASTContext& ctx);

bool recordIsInStdNamespace(const clang::CXXRecordDecl* record);
bool recordIsMutexType(const clang::CXXRecordDecl* record);
bool typeIsSignedIntegral(clang::QualType type);
bool typeIsUnsignedIntegral(clang::QualType type);
bool typeIsStdSpan(clang::QualType type);
bool typeIsExpected(clang::QualType type);

} // namespace ms::plugin
