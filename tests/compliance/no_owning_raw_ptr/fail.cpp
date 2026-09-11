// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST NOT compile when the ms-profile Clang plugin is active.

struct Owner {
    int* data;
};

int main() {
    Owner o{};
    return o.data != nullptr ? 1 : 0;
}
