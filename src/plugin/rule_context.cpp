#include "rule_context.hpp"

#include "clang/AST/Stmt.h"

namespace ms::plugin {

namespace {

bool attrIsMsUnsafeImpl(const clang::Attr* attr) {
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

} // namespace

bool attrIsMsUnsafe(const clang::Attr* attr) {
    return attrIsMsUnsafeImpl(attr);
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

} // namespace ms::plugin
