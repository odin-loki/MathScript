// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    long long wide = 100000000000LL;
    int narrow = wide;
    return narrow;
}
