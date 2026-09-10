// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

#include <thread>

[[MS_UNSAFE("compliance: thread detach allowed inside annotated function")]]
static void unsafe_detach() {
    std::thread worker([] {});
    worker.detach();
}

int main() {
    unsafe_detach();
    return 0;
}
