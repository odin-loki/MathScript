// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    try {
        return 0;
    } catch (...) {
        return 1;
    }
}
