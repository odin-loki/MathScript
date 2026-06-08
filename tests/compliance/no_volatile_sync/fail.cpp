// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    volatile int flag = 0;
    return flag;
}
