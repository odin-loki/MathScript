// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    int x = 42;
    void* p = reinterpret_cast<void*>(&x);
    return p != nullptr ? 0 : 1;
}
