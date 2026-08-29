#include <gtest/gtest.h>
#include "ms/distributed/dist_matrix.hpp"
#include "ms/distributed/linalg.hpp"
#include "ms/distributed/mpi_context.hpp"

using namespace ms;
using namespace ms::distributed;

TEST(DistLuTest, distributed_lu_matches_local) {
    auto ctx = init(0, nullptr);
    ColMatrix<double> A{{4, 1, 0}, {1, 3, 2}, {0, 2, 5}};
    auto dA = scatter(A, ctx, Distribution::BlockCyclic).value();
    const auto dist = distributed::lu(dA, ctx).value();
    const auto local = ms::lu(A).value();
    const auto& dist_u = std::get<2>(dist);
    const auto& local_u = std::get<2>(local);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(dist_u(i, j), local_u(i, j), 1e-6);
        }
    }
    finalize(ctx);
}
