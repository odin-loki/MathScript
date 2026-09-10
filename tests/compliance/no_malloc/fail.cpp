// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// MUST NOT compile when the ms-profile Clang plugin is active.

extern "C" void* malloc(unsigned long);
extern "C" void* calloc(unsigned long, unsigned long);
extern "C" void* realloc(void*, unsigned long);
extern "C" void free(void*);

int main() {
    void* p = malloc(16);
    p = calloc(2, 8);
    p = realloc(p, 32);
    free(p);
    return 0;
}
