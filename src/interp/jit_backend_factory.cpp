#include "ms/interp/jit_backend.hpp"
#include "ms/interp/jit_backend_impl.hpp"

namespace ms::interp {

namespace {

class ReplBackend final : public JitBackend {
public:
    Result<std::string> execute(const std::string& line) override { return interp_.execute(line); }
    std::string backend_name() const override { return "repl"; }
    const SessionState& state() const override { return interp_.state(); }
    void reset() override { interp_.reset(); }
    JitCapabilities capabilities() const override { return {false, false, "interpreted REPL"}; }

private:
    Interpreter interp_;
};

} // namespace

std::unique_ptr<JitBackend> create_backend(Backend backend) {
    switch (backend) {
    case Backend::OrcJit:
#if defined(MS_JIT_LLVM)
        return create_orc_jit_llvm_backend();
#else
        return create_orc_jit_stub_backend();
#endif
    case Backend::Repl:
    default:
        return std::make_unique<ReplBackend>();
    }
}

} // namespace ms::interp
