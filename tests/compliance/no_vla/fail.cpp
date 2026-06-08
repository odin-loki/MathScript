// MUST NOT compile when the ms-profile Clang plugin is active.

int main() {
    int n = 3;
    int arr[n];
    arr[0] = 1;
    return arr[0];
}
