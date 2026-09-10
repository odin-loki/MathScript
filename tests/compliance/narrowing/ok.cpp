// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST compile cleanly with the ms-profile Clang plugin active.

int main() {
    long long wide = 42LL;
    const int narrow = static_cast<int>(wide);
    return narrow;
}
