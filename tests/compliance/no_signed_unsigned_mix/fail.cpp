// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    int a = -1;
    unsigned b = 1U;
    return a < b ? 1 : 0;
}
