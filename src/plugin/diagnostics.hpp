// MathScript Plugin Diagnostics
// Custom diagnostic IDs and messages for the C++23 profile enforcement

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"

namespace ms::plugin {

enum class DiagnosticID {
    NoRawNew,
    NoMalloc,
    NoOwningRawPtr,
    NoVLA,
    NoCStyleCast,
    NoConstCast,
    NoUnsafeReinterpret,
    Narrowing,
    NoThrow,
    NoCatch,
    UnusedExpected,
    NoUninit,
    NoStoredSpan,
    NoRawThread,
    NoRawMutexLock,
    NoVolatileSync,
    NoDetach,
    NoSignedUnsignedMix,
    NoRawPtrArithmetic,
    NoGoto,
    UnsafeAudit
};

} // namespace ms::plugin

// Diagnostic action handler
void HandleDiagnostic(DiagnosticsEngine& diag, ms::plugin::DiagnosticID id,
                      const char* message, const llvm::SourceLocation& loc);

#pragma clang diagnostic pop