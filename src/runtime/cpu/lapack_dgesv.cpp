#include "ms/cpu/lapack.hpp"

namespace ms::cpu::lapack {

int dgesv(int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb) {
    const int info = dgetrf(n, n, A, lda, ipiv);
    if (info != 0) {
        return info;
    }
    dgetrs('N', n, nrhs, A, lda, ipiv, B, ldb);
    return 0;
}

} // namespace ms::cpu::lapack
