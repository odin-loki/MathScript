// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

extern "C" void* malloc(unsigned long);
extern "C" void free(void*);

[[MS_UNSAFE("compliance: malloc allowed inside annotated function")]]
static void unsafe_alloc() {
    void* p = malloc(16);
    free(p);
}

int main() {
    unsafe_alloc();
    return 0;
}
