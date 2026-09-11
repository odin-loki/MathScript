// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

int main() {
    int a = -1;
    int b = 1;
    return a < b ? 1 : 0;
}
