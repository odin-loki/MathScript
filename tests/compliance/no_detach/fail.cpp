// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST NOT compile when the ms-profile Clang plugin is active.

#include <thread>

int main() {
    std::thread worker([] {});
    worker.detach();
    return 0;
}
