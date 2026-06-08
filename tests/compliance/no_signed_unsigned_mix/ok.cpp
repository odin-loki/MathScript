// MUST compile cleanly with the ms-profile Clang plugin active.

int main() {
    int a = -1;
    int b = 1;
    return a < b ? 1 : 0;
}
