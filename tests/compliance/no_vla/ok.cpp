// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: VLA allowed inside annotated function")]]
static int unsafe_vla() {
    int n = 3;
    int arr[n];
    arr[0] = 1;
    return arr[0];
}

int main() {
    return unsafe_vla();
}
