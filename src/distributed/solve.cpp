#include "ms/distributed/solve.hpp"
#include "ms/core/operations.hpp"

namespace ms::distributed {

template<typename S, template<typename> class Alloc>
Result<Matrix<S, StorageOrder::ColMajor, Alloc>> solve(
    const DistMatrix<S, Alloc>& A,
    const DistMatrix<S, Alloc>& b,
    MPIContext& ctx) {
    auto global_A = gather(A, ctx);
    if (!global_A) {
        return std::unexpected(global_A.error());
    }
    auto global_b = gather(b, ctx);
    if (!global_b) {
        return std::unexpected(global_b.error());
    }
    if (rank(ctx) != 0 && size(ctx) > 1) {
        return global_b;
    }
    return ms::solve(*global_A, *global_b);
}

template Result<Matrix<double>> solve(
    const DistMatrix<double>&, const DistMatrix<double>&, MPIContext&);

} // namespace ms::distributed
