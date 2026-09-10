// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

[[MS_UNSAFE("compliance: volatile allowed inside annotated function")]]
static int unsafe_volatile() {
    volatile int flag = 0;
    return flag;
}

int main() {
    return unsafe_volatile();
}
