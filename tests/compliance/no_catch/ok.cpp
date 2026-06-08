// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: try/catch allowed inside annotated function")]]
static int unsafe_try() {
    try {
        return 0;
    } catch (...) {
        return 1;
    }
}

int main() {
    return unsafe_try();
}
