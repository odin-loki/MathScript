// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: const_cast allowed inside annotated function")]]
static int unsafe_const_cast() {
    const int x = 1;
    int* p = const_cast<int*>(&x);
    return *p;
}

int main() {
    return unsafe_const_cast();
}
