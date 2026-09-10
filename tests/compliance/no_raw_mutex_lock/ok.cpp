// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

#include <mutex>

int main() {
    std::mutex m;
    std::lock_guard<std::mutex> guard(m);
    return 0;
}
