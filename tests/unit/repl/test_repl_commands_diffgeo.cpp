#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl/repl_test_helpers.hpp"

using namespace ms::interp;

TEST(ReplCommandsTest, diffgeo_sphere_curvatures) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_gaussian_sphere()");
    expect_contains(interp, "help", "diffgeo_mean_sphere()");

    expect_ok(interp, "K = diffgeo_gaussian_sphere()");
    EXPECT_NEAR(interp.state().scalars.at("K"), 1.0, 0.05);
    expect_contains(interp, "diffgeo_gaussian_sphere()", "\n");

    expect_ok(interp, "H = diffgeo_mean_sphere()");
    EXPECT_NEAR(std::abs(interp.state().scalars.at("H")), 1.0, 0.08);
    expect_contains(interp, "diffgeo_mean_sphere()", "\n");
}

TEST(ReplCommandsTest, diffgeo_principal_curvature_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_principal_curvature_sphere()");

    expect_ok(interp, "k1 = diffgeo_principal_curvature_sphere()");
    EXPECT_NEAR(interp.state().scalars.at("k1"), 1.0, 0.15);

    expect_contains(interp, "diffgeo_principal_curvature_sphere()", "0.999");
}

TEST(ReplCommandsTest, diffgeo_gaussian_curvature_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_gaussian_curvature_sphere(u,v)");

    expect_ok(interp, "Kg = diffgeo_gaussian_curvature_sphere(0.3, 0.7)");
    EXPECT_NEAR(interp.state().scalars.at("Kg"), 1.0, 0.15);

    expect_contains(interp, "diffgeo_gaussian_curvature_sphere(0.3, 0.7)", "0.999");
}

TEST(ReplCommandsTest, diffgeo_mean_curvature_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_mean_curvature_sphere(u,v)");

    expect_ok(interp, "Hm = diffgeo_mean_curvature_sphere(0.3, 0.7)");
    EXPECT_NEAR(interp.state().scalars.at("Hm"), 1.0, 0.15);
}

TEST(ReplCommandsTest, diffgeo_christoffel_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_christoffel_sphere(k,i,j,u,v)");

    expect_ok(interp, "pi = 3.14159265358979323846");
    expect_ok(interp, "G011 = diffgeo_christoffel_sphere(0, 1, 1, pi/4, 0.5)");
    EXPECT_NEAR(interp.state().scalars.at("G011"), -0.5, 0.05);
}

TEST(ReplCommandsTest, diffgeo_geodesic_euclidean) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_geodesic_euclidean(x0,y0,vx,vy,s_end)");

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    ASSERT_GT(interp.state().matrices.count("geo"), 0u);
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(ReplCommandsTest, diffgeo_ricci_scalar_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_ricci_scalar_sphere(u,v)");

    expect_ok(interp, "R = diffgeo_ricci_scalar_sphere(0.3, 1.2)");
    EXPECT_NEAR(interp.state().scalars.at("R"), 2.0, 0.05);
}

TEST(ReplCommandsTest, diffgeo_einstein_scalar_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_einstein_scalar_sphere(u,v)");

    expect_ok(interp, "E = diffgeo_einstein_scalar_sphere(1.047197551196598, 0.523598775598299)");
    EXPECT_NEAR(interp.state().scalars.at("E"), 0.0, 0.05);
}

TEST(ReplCommandsTest, diffgeo_surface_normal_sphere) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_surface_normal_sphere(u,v)");

    expect_ok(interp, "N = diffgeo_surface_normal_sphere(0.5, 0.3)");
    ASSERT_GT(interp.state().matrices.count("N"), 0u);
    const auto& N = interp.state().matrices.at("N");
    double n_norm = 0.0;
    for (size_t i = 0; i < N.rows(); ++i) {
        n_norm += N(i, 0) * N(i, 0);
    }
    EXPECT_NEAR(std::sqrt(n_norm), 1.0, 0.05);
}

TEST(ReplCommandsTest, diffgeo_presets) {
    Interpreter interp;
    expect_contains(interp, "help", "diffgeo_helix_torsion(t[,a[,b]])");
    expect_contains(interp, "help", "diffgeo_sphere_gauss_bonnet([n])");
    expect_contains(interp, "help", "diffgeo_sphere_gauss_bonnet_residual([n])");

    expect_ok(interp, "tau = diffgeo_helix_torsion(0)");
    EXPECT_NEAR(interp.state().scalars.at("tau"), 0.5, 0.05);

    expect_ok(interp, "tau2 = diffgeo_helix_torsion(0, 2, 3)");
    EXPECT_NEAR(interp.state().scalars.at("tau2"), 3.0 / 13.0, 0.05);

    expect_ok(interp, "gb = diffgeo_sphere_gauss_bonnet(200)");
    EXPECT_NEAR(interp.state().scalars.at("gb"), 4.0 * M_PI, 0.05 * 4.0 * M_PI);

    expect_ok(interp, "gbr = diffgeo_sphere_gauss_bonnet_residual(200)");
    EXPECT_NEAR(interp.state().scalars.at("gbr"), 0.0, 0.05 * 4.0 * M_PI);

    expect_ok(interp, "gb_def = diffgeo_sphere_gauss_bonnet()");
    EXPECT_NEAR(interp.state().scalars.at("gb_def"), 4.0 * M_PI, 0.05 * 4.0 * M_PI);
}

TEST(ReplCommandsTest, diffgeo_christoffel_sphere_noassign) {
    Interpreter interp;
    expect_ok(interp, "diffgeo_christoffel_sphere(0, 1, 1, 0.7853981633974483, 0.5)");
    expect_error_contains(interp, "diffgeo_christoffel_sphere(0, 1, 1, 0.5, missing)",
                          "expected diffgeo_christoffel_sphere");
}

TEST(ReplCommandsTest, diffgeo_mean_curvature_sphere_noassign) {
    Interpreter interp;
    expect_ok(interp, "diffgeo_mean_curvature_sphere(0.3, 0.7)");
    expect_error_contains(interp, "diffgeo_mean_curvature_sphere(0.3, missing)",
                          "expected diffgeo_mean_curvature_sphere");
}

TEST(ReplCommandsTest, diffgeo_ricci_scalar_sphere_noassign) {
    Interpreter interp;
    expect_contains(interp, "diffgeo_ricci_scalar_sphere(0.3, 1.2)", "2");
    expect_error_contains(interp, "diffgeo_ricci_scalar_sphere(0.3, missing)",
                          "expected diffgeo_ricci_scalar_sphere");
}

TEST(ReplCommandsTest, diffgeo_einstein_scalar_sphere_noassign) {
    Interpreter interp;
    expect_ok(interp, "diffgeo_einstein_scalar_sphere(1.047197551196598, 0.523598775598299)");
    expect_error_contains(interp, "diffgeo_einstein_scalar_sphere(1.0, missing)",
                          "expected diffgeo_einstein_scalar_sphere");
}
