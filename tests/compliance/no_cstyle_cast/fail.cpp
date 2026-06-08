// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    float f = 1.0f;
    int x = (int)f;
    return x;
}
