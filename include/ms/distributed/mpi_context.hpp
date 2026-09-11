// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
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
double allreduce_max(const MPIContext& ctx, double value);
double allreduce_min(const MPIContext& ctx, double value);
double bcast(const MPIContext& ctx, double value);
void barrier(const MPIContext& ctx);

} // namespace ms::distributed
