#pragma once

#include "ms/interp/jit_backend.hpp"
#include <memory>

namespace ms::interp {

std::unique_ptr<JitBackend> create_orc_jit_stub_backend();
#if defined(MS_JIT_LLVM)
std::unique_ptr<JitBackend> create_orc_jit_llvm_backend();
#endif

} // namespace ms::interp
