// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: pointer arithmetic allowed inside annotated function")]]
static int unsafe_ptr_arith() {
    int arr[2] = {1, 2};
    int* p = arr;
    return *(p + 1);
}

int main() {
    return unsafe_ptr_arith();
}
