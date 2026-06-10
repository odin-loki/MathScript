#pragma once

#include "ms/interp/jit_backend.hpp"
#include <memory>

namespace ms::interp {

struct JitDispatchStats {
    size_t jit_compiled = 0;
    size_t native_dispatch = 0;
    size_t repl_fallback = 0;
};

std::unique_ptr<JitBackend> create_orc_jit_stub_backend();
#if defined(MS_JIT_LLVM)
std::unique_ptr<JitBackend> create_orc_jit_llvm_backend();
void set_orc_jit_stats_enabled(bool enabled);
const JitDispatchStats& orc_jit_stats();
#endif

} // namespace ms::interp
