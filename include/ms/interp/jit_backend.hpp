#pragma once

#include "ms/error/expected.hpp"
#include "ms/interp/repl_engine.hpp"
#include <memory>
#include <string>

namespace ms::interp {

enum class Backend { Repl, OrcJit };

struct JitCapabilities {
    bool llvm_linked;
    bool can_compile;
    std::string notes;
};

class JitBackend {
public:
    virtual ~JitBackend() = default;

    virtual Result<std::string> execute(const std::string& line) = 0;
    virtual std::string backend_name() const = 0;
    virtual const SessionState& state() const = 0;
    virtual void reset() = 0;
    virtual JitCapabilities capabilities() const { return {false, false, "repl interpreter"}; }
};

std::unique_ptr<JitBackend> create_backend(Backend backend = Backend::Repl);

} // namespace ms::interp
