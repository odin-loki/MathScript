#include "ms/cpu/lapack.hpp"

#include <cmath>

namespace ms::cpu::lapack {

namespace {

bool is_lower(char uplo) {
    return uplo == 'L' || uplo == 'l';
}

} // namespace

int dpotrf(char uplo, int n, double* A, int lda) {
    if (n <= 0) {
        return 0;
    }
    if (A == nullptr) {
        return 1;
    }
    if (!is_lower(uplo)) {
        return 1;
    }

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            double sum = A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)];
            for (int k = 0; k < j; ++k) {
                sum -= A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] *
                       A[static_cast<std::size_t>(k) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(j)];
            }
            if (i == j) {
                if (!(sum > 0.0)) {
                    return j + 1;
                }
                A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(j)] = std::sqrt(sum);
            } else {
                A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i)] =
                    sum / A[static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(j)];
            }
        }
    }

    return 0;
}

} // namespace ms::cpu::lapack
