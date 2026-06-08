// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: reinterpret_cast allowed inside annotated function")]]
static bool unsafe_reinterpret() {
    int x = 42;
    void* p = reinterpret_cast<void*>(&x);
    return p != nullptr;
}

int main() {
    return unsafe_reinterpret() ? 0 : 1;
}
