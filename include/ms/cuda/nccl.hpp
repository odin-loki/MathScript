#pragma once

#include <cstddef>

namespace ms::cuda {

bool nccl_available();
int nccl_device_count();
size_t nccl_comm_size();

} // namespace ms::cuda
