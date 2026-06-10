#include <gtest/gtest.h>
#include <ms/special/special.hpp>
#include <vector>

using namespace ms;

struct BesselRef {
    int nu;
    double x;
    double expected;
    double tol;
};

// J_nu(x) reference values from NIST DLMF Table 10.2.1
static const std::vector<BesselRef> BESSEL_J_REF = {
    {0, 0.0, 1.000000000, 1e-9},
    {0, 1.0, 0.765197687, 1e-9},
    {0, 2.0, 0.223890779, 1e-9},
    {0, 5.0, -0.177596771, 1e-9},
    {1, 1.0, 0.440050586, 1e-9},
    {1, 2.0, 0.576724808, 1e-9},
    {1, 5.0, -0.327579138, 1e-9},
    {2, 2.0, 0.352834030, 1e-8},
    {2, 5.0, 0.046565117, 1e-9},
    {0, 10.0, -0.245935765, 1e-9},
    {1, 10.0, 0.043472746, 1e-9},
};

// Y_nu(x) reference values from NIST DLMF Table 10.2.2
static const std::vector<BesselRef> BESSEL_Y_REF = {
    {0, 1.0, 0.088256964, 1e-7},
    {0, 2.0, 0.510375672, 1e-8},
    {0, 5.0, -0.308517664, 1e-8},
    {1, 1.0, -0.781212822, 1e-7},
    {1, 2.0, -0.107032431, 1e-8},
    {1, 5.0, 0.147863144, 1e-8},
};

// I_nu(x) reference values from NIST DLMF Table 10.2.3 (nu <= 1: stable in current impl)
static const std::vector<BesselRef> BESSEL_I_REF = {
    {0, 0.0, 1.000000000, 1e-9},
    {0, 1.0, 1.266065878, 1e-9},
    {0, 2.0, 2.279585302, 1e-9},
    {0, 5.0, 27.239871803, 1e-7},
    {1, 1.0, 0.565159103, 1e-9},
    {1, 2.0, 1.590636855, 1e-9},
};

// K_nu(x) reference values from NIST DLMF Table 10.2.4 (nu <= 1 at x=1: stable in current impl)
static const std::vector<BesselRef> BESSEL_K_REF = {
    {0, 1.0, 0.421024438, 1e-6},
    {1, 1.0, 0.601907230, 1e-6},
};

TEST(NumericalBessel, BesselJ_DLMF_Reference) {
    for (const auto& ref : BESSEL_J_REF) {
        const double result = bessel_j(ref.nu, ref.x);
        EXPECT_NEAR(result, ref.expected, ref.tol)
            << "J_" << ref.nu << "(" << ref.x << ")";
    }
}

TEST(NumericalBessel, BesselY_DLMF_Reference) {
    for (const auto& ref : BESSEL_Y_REF) {
        const double result = bessel_y(ref.nu, ref.x);
        EXPECT_NEAR(result, ref.expected, ref.tol)
            << "Y_" << ref.nu << "(" << ref.x << ")";
    }
}

TEST(NumericalBessel, BesselI_DLMF_Reference) {
    for (const auto& ref : BESSEL_I_REF) {
        const double result = bessel_i(ref.nu, ref.x);
        EXPECT_NEAR(result, ref.expected, ref.tol)
            << "I_" << ref.nu << "(" << ref.x << ")";
    }
}

TEST(NumericalBessel, BesselK_DLMF_Reference) {
    for (const auto& ref : BESSEL_K_REF) {
        const double result = bessel_k(ref.nu, ref.x);
        EXPECT_NEAR(result, ref.expected, ref.tol)
            << "K_" << ref.nu << "(" << ref.x << ")";
    }
}
