// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

static int safe_cast(float f) {
    return static_cast<int>(f);
}

[[MS_UNSAFE("compliance: C-style cast allowed inside annotated function")]]
static int unsafe_cast(float f) {
    return (int)f;
}

int main() {
    return safe_cast(1.0f) + unsafe_cast(2.0f);
}
