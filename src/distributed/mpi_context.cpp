#include "ms/distributed/mpi_context.hpp"

#if defined(MS_HAS_MPI) && MS_HAS_MPI
#include <mpi.h>
#endif

namespace ms::distributed {

MPIContext init(int argc, char** argv) {
    MPIContext ctx;
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
    ctx.active = true;
#else
    (void)argc;
    (void)argv;
    ctx.rank = 0;
    ctx.size = 1;
    ctx.active = true;
#endif
    return ctx;
}

void finalize(MPIContext& ctx) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (ctx.active) {
        MPI_Finalize();
    }
#endif
    ctx.active = false;
}

int rank(const MPIContext& ctx) {
    return ctx.rank;
}

int size(const MPIContext& ctx) {
    return ctx.size;
}

std::string backend_name(const MPIContext&) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    return "mpi";
#else
    return "stub";
#endif
}

double allreduce_sum(const MPIContext& ctx, double value) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    double out = 0.0;
    MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
#else
    (void)ctx;
    return value;
#endif
}

void barrier(const MPIContext& ctx) {
#if defined(MS_HAS_MPI) && MS_HAS_MPI
    if (ctx.active) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    (void)ctx;
#endif
}

} // namespace ms::distributed
