#pragma once

#include <string>

namespace ms::distributed {

struct MPIContext {
    int rank = 0;
    int size = 1;
    bool active = false;
};

MPIContext init(int argc, char** argv);
void finalize(MPIContext& ctx);
int rank(const MPIContext& ctx);
int size(const MPIContext& ctx);
std::string backend_name(const MPIContext& ctx);

double allreduce_sum(const MPIContext& ctx, double value);
void barrier(const MPIContext& ctx);

} // namespace ms::distributed
