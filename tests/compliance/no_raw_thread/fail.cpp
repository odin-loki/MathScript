// MUST NOT compile when the ms-profile Clang plugin is active.

#include <thread>

int main() {
    std::thread worker([] {});
    worker.join();
    return 0;
}
