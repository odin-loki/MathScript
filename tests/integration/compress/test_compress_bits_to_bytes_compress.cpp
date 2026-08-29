
#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/prob/prob.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd << " error: "
                                    << (result ? *result : "unknown");
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(IntegrationCompress,  CompressBitsDiffgeoGeodesic) {
    Interpreter interp;
    expect_contains(interp, "help", "compress_bits_to_bytes");
    expect_contains(interp, "help", "compress_bytes_to_bits");
    expect_contains(interp, "help", "diffgeo_geodesic_euclidean");

    expect_ok(interp, "bits = [1;0;1;0;1;0;1;1;1;1;0;0;1;0;1;1]");
    expect_ok(interp, "bytes = compress_bits_to_bytes(bits)");
    EXPECT_EQ(interp.state().matrices.at("bytes").rows(), 2u);

    expect_ok(interp, "bytes2 = [171; 205]");
    expect_ok(interp, "bits2 = compress_bytes_to_bits(bytes2)");
    EXPECT_EQ(interp.state().matrices.at("bits2").rows(), 16u);

    expect_ok(interp, "geo = diffgeo_geodesic_euclidean(0, 0, 1, 0.5, 1)");
    const auto& geo = interp.state().matrices.at("geo");
    EXPECT_NEAR(geo(geo.rows() - 1, 0), 1.0, 0.05);
    EXPECT_NEAR(geo(geo.rows() - 1, 1), 0.5, 0.05);
}

TEST(IntegrationCompress,  ProbRayleighCdfScalar) {
    Interpreter interp;
    expect_ok(interp, "rc = prob_rayleigh_cdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rc"), ms::rayleigh_cdf(1, 1), 1e-8);
}
