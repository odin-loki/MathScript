#include "ms/interp/jit_backend_impl.hpp"

namespace ms::interp {

namespace {

class OrcJitStubBackend final : public JitBackend {
public:
    Result<std::string> execute(const std::string& line) override { return interp_.execute(line); }
    std::string backend_name() const override { return "orc-jit-stub"; }
    const SessionState& state() const override { return interp_.state(); }
    void reset() override { interp_.reset(); }
    JitCapabilities capabilities() const override {
        return {false, false, "ORC JIT stub — delegates to REPL until LLVM ORC v2 is wired"};
    }

private:
    Interpreter interp_;
};

} // namespace

std::unique_ptr<JitBackend> create_orc_jit_stub_backend() {
    return std::make_unique<OrcJitStubBackend>();
}

} // namespace ms::interp
