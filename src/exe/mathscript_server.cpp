#include "ms/distributed/mpi_context.hpp"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    auto ctx = ms::distributed::init(argc, argv);
    const int r = ms::distributed::rank(ctx);
    const int n = ms::distributed::size(ctx);

    std::cout << "mathscript-server rank " << r << "/" << n
              << " backend=" << ms::distributed::backend_name(ctx) << "\n";

    ms::distributed::barrier(ctx);
    if (r == 0) {
        std::cout << "cluster ready (" << n << " ranks)\n";
    }

    for (int tick = 0; tick < 3; ++tick) {
        ms::distributed::barrier(ctx);
        if (r == 0) {
            std::cout << "heartbeat " << tick << "\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ms::distributed::finalize(ctx);
    return 0;
}
