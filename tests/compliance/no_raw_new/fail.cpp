// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    int* p = new int(5);
    delete p;
    return 0;
}
