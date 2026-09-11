// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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
