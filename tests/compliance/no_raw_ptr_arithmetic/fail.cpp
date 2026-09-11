// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    int arr[2] = {1, 2};
    int* p = arr;
    return *(p + 1);
}
