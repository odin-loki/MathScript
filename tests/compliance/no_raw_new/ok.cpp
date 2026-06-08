// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: raw new allowed inside annotated function")]]
static void unsafe_alloc() {
    int* p = new int(5);
    delete p;
}

int main() {
    unsafe_alloc();
    return 0;
}
