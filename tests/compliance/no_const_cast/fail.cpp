// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    const int x = 1;
    int* p = const_cast<int*>(&x);
    return *p;
}
