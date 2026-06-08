// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: goto allowed inside annotated function")]]
static int unsafe_goto() {
    goto done;
done:
    return 0;
}

int main() {
    return unsafe_goto();
}
