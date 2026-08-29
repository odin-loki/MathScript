
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

TEST(IntegrationTopo,  HuffmanLz77TopoBetti) {
    Interpreter interp;
    expect_contains(interp, "help", "huffman_encode_vec");
    expect_contains(interp, "help", "lz77_encode_vec");
    expect_contains(interp, "help", "topo_betti_curve");

    expect_ok(interp, "M = [97; 98; 99; 97; 97; 98]");
    expect_ok(interp, "HE = huffman_encode_vec(M)");
    EXPECT_GE(interp.state().matrices.at("HE").rows(), 1u);

    expect_ok(interp, "Lz = [97; 98; 99; 97; 98; 99; 97; 98; 99]");
    expect_ok(interp, "T = lz77_encode_vec(Lz)");
    EXPECT_EQ(interp.state().matrices.at("T").cols(), 3u);

    expect_ok(interp, "Dcol = [0, 1, 2; 1, 0, 1; 2, 1, 0]");
    expect_ok(interp, "thr = [0.5; 1.5; 2.5]");
    expect_ok(interp, "bc = topo_betti_curve(Dcol, thr, 1)");
    EXPECT_NEAR(interp.state().matrices.at("bc")(0, 0), 3.0, 1e-8);
    EXPECT_NEAR(interp.state().matrices.at("bc")(1, 0), 1.0, 1e-8);
}

TEST(IntegrationTopo,  ProbRayleighPdfScalar) {
    Interpreter interp;
    expect_ok(interp, "rp = prob_rayleigh_pdf(1, 1)");
    EXPECT_NEAR(interp.state().scalars.at("rp"), ms::rayleigh_pdf(1, 1), 1e-8);
}
