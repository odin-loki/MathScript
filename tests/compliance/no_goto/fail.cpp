// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    goto done;
done:
    return 0;
}
