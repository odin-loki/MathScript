// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: throw allowed inside annotated function")]]
static void unsafe_throw() {
    throw 42;
}

int main() {
    unsafe_throw();
    return 0;
}
