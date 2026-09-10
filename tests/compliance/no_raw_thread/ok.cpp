// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

#include "ms/unsafe/unsafe.hpp"

#include <thread>

[[MS_UNSAFE("compliance: raw std::thread allowed inside annotated function")]]
static void unsafe_thread() {
    std::thread worker([] {});
    worker.join();
}

int main() {
    unsafe_thread();
    return 0;
}
