// MUST NOT compile when the ms-profile Clang plugin is active.

#include <mutex>

int main() {
    std::mutex m;
    m.lock();
    m.unlock();
    return 0;
}
