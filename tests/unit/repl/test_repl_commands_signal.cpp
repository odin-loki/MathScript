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

TEST(ReplCommandsTest, signal_moving_average) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_moving_average(x,window)");

    expect_ok(interp, "ma = signal_moving_average([5; 5; 5; 5], 3)");
    ASSERT_GT(interp.state().matrices.count("ma"), 0u);
    for (size_t i = 0; i < interp.state().matrices.at("ma").rows(); ++i) {
        EXPECT_NEAR(interp.state().matrices.at("ma")(i, 0), 5.0, 1e-9);
    }
}

TEST(ReplCommandsTest, signal_upsample_downsample) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_upsample(x,n)");
    expect_contains(interp, "help", "signal_downsample(x,n)");

    expect_ok(interp, "up = signal_upsample([1; 2], 3)");
    ASSERT_GT(interp.state().matrices.count("up"), 0u);
    EXPECT_EQ(interp.state().matrices.at("up").rows(), 6u);
    EXPECT_NEAR(interp.state().matrices.at("up")(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("up")(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("up")(3, 0), 2.0, 1e-12);

    expect_ok(interp, "dn = signal_downsample([1; 2; 3; 4; 5; 6], 2)");
    ASSERT_GT(interp.state().matrices.count("dn"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dn").rows(), 3u);
    EXPECT_NEAR(interp.state().matrices.at("dn")(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("dn")(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("dn")(2, 0), 5.0, 1e-12);
}

TEST(ReplCommandsTest, signal_resample_decimate_interpolate) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_decimate(x,q)");
    expect_contains(interp, "help", "signal_interpolate(x,p)");
    expect_contains(interp, "help", "signal_resample(x,p,q)");

    expect_ok(interp, "dec = signal_decimate([1; 2; 3; 4; 5; 6], 2)");
    ASSERT_GT(interp.state().matrices.count("dec"), 0u);
    EXPECT_EQ(interp.state().matrices.at("dec").rows(), 3u);

    expect_ok(interp, "itp = signal_interpolate([1; 2], 2)");
    ASSERT_GT(interp.state().matrices.count("itp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("itp").rows(), 4u);

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);
}

TEST(ReplCommandsTest, signal_coherence) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_coherence(x,y,fs,nperseg)");

    expect_ok(interp,
              "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    const auto& c = interp.state().matrices.at("c");
    EXPECT_EQ(c.cols(), 2u);
    EXPECT_GT(c.rows(), 0u);
    EXPECT_NEAR(c(0, 0), 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(c(0, 1)));
}

TEST(ReplCommandsTest, signal_filter) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_filter(b,a,x)");

    expect_ok(interp, "b = [1, -1]");
    expect_ok(interp, "a = [1, -0.5]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_filter(b, a, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    const auto& y = interp.state().matrices.at("y");
    EXPECT_EQ(y.cols(), 1u);
    EXPECT_EQ(y.rows(), 5u);
    EXPECT_NEAR(y(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(y(1, 0), 1.5, 1e-12);
    EXPECT_NEAR(y(2, 0), 1.75, 1e-12);
    EXPECT_NEAR(y(3, 0), 1.875, 1e-12);
    EXPECT_NEAR(y(4, 0), 1.9375, 1e-12);
}

TEST(ReplCommandsTest, signal_sosfilt) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_sosfilt(sos,x)");

    // Same DF1 section as SignalSosfiltTest.normalizes_a0_per_section (scaled a0=2).
    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    const auto& y = interp.state().matrices.at("y");
    EXPECT_EQ(y.cols(), 1u);
    EXPECT_EQ(y.rows(), 5u);
    EXPECT_NEAR(y(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(y(1, 0), 1.5, 1e-12);
    EXPECT_NEAR(y(2, 0), 1.75, 1e-12);
    EXPECT_NEAR(y(3, 0), 1.875, 1e-12);
    EXPECT_NEAR(y(4, 0), 1.9375, 1e-12);
}

TEST(ReplCommandsTest, signal_conv2) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_conv2(A,K)");

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(0, 1), 2.0, 1e-12);
    EXPECT_NEAR(C(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);
    EXPECT_NEAR(C(2, 2), 4.0, 1e-12);
}

TEST(ReplCommandsTest, signal_deconv) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_deconv(y,b)");

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    ASSERT_EQ(x.rows(), 3u);
    ASSERT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);
}

TEST(ReplCommandsTest, signal_savgol) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_savgol(x,window_length,polyorder)");

    expect_ok(interp, "x = [2; -1; 3; 0.5; 4; -2; 1.5]");
    expect_ok(interp, "sg = signal_savgol(x, 5, 2)");
    ASSERT_GT(interp.state().matrices.count("sg"), 0u);
    const auto& sg = interp.state().matrices.at("sg");
    EXPECT_EQ(sg.cols(), 1u);
    EXPECT_EQ(sg.rows(), 7u);
    EXPECT_NEAR(sg(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(sg(1, 0), -1.0, 1e-12);
    EXPECT_NEAR(sg(2, 0), 27.0 / 35.0, 1e-12);
}

TEST(ReplCommandsTest, signal_median_filter) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_median_filter(x,window_length)");

    // Hand-computed: x={5,1,3,2,4}, w=3 -> {5,3,2,3,4} (edges unfiltered).
    expect_ok(interp, "x = [5; 1; 3; 2; 4]");
    expect_ok(interp, "mf = signal_median_filter(x, 3)");
    ASSERT_GT(interp.state().matrices.count("mf"), 0u);
    const auto& mf = interp.state().matrices.at("mf");
    EXPECT_EQ(mf.cols(), 1u);
    EXPECT_EQ(mf.rows(), 5u);
    EXPECT_NEAR(mf(0, 0), 5.0, 1e-12);
    EXPECT_NEAR(mf(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(mf(2, 0), 2.0, 1e-12);
    EXPECT_NEAR(mf(3, 0), 3.0, 1e-12);
    EXPECT_NEAR(mf(4, 0), 4.0, 1e-12);
}

TEST(ReplCommandsTest, signal_cheby1) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_cheby1(order,rp_db,cutoff,fs[,type])");

    // scipy.signal.cheby1(2, 1.0, 0.25, fs=2.0, btype='low')
    expect_ok(interp, "ba = signal_cheby1(2, 1.0, 0.25, 2.0)");
    ASSERT_GT(interp.state().matrices.count("ba"), 0u);
    const auto& ba = interp.state().matrices.at("ba");
    EXPECT_EQ(ba.rows(), 2u);
    EXPECT_EQ(ba.cols(), 3u);
    EXPECT_NEAR(ba(0, 0), 0.10255744, 1e-6);
    EXPECT_NEAR(ba(1, 0), 1.0, 1e-12);

    expect_ok(interp, "bah = signal_cheby1(2, 1.0, 0.25, 2.0, 1)");
    ASSERT_GT(interp.state().matrices.count("bah"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bah").rows(), 2u);
}

TEST(ReplCommandsTest, signal_firwin) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_firwin(n_taps,cutoff[,window])");

    expect_ok(interp, "b = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("b"), 0u);
    const auto& b = interp.state().matrices.at("b");
    EXPECT_EQ(b.rows(), 11u);
    EXPECT_EQ(b.cols(), 1u);
    double sum = 0.0;
    for (size_t i = 0; i < b.rows(); ++i) {
        sum += b(i, 0);
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
    EXPECT_EQ(interp.state().matrices.at("bh").cols(), 1u);
}

TEST(ReplCommandsTest, signal_xcorr_xcov_autocorr) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_xcorr(a,b,max_lag)");
    expect_contains(interp, "help", "signal_xcov(a,b,max_lag)");
    expect_contains(interp, "help", "signal_autocorr(x,max_lag)");

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);
    const auto& ac = interp.state().matrices.at("ac");
    EXPECT_EQ(ac.rows(), 5u);
    EXPECT_NEAR(ac(2, 0), 14.25, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    const auto& cv = interp.state().matrices.at("cv");
    EXPECT_EQ(cv.rows(), 5u);
    EXPECT_EQ(cv.cols(), 1u);
    EXPECT_TRUE(std::isfinite(cv(2, 0)));
}

TEST(ReplCommandsTest, signal_lms) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_lms(x,d,filter_length,mu)");
    expect_contains(interp, "help", "signal_lms_weights(x,d,filter_length,mu)");

    // mu=0 keeps zero weights; y[n]=0 and e[n]=d[n].
    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);
    EXPECT_NEAR(ye(2, 1), 1.0, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);
}

TEST(ReplCommandsTest, signal_czt_and_zoom) {
    Interpreter interp;
    // Distinct needles avoid signal_czt / signal_czt_zoom substring collisions.
    expect_contains(interp, "help", "signal_czt(x,m,w_re,w_im,a_re,a_im)");
    expect_contains(interp, "help", "signal_czt_zoom(x,f_start,f_stop,m,fs)");

    // DFT contour: m=N, w=exp(-2*pi*i/N)=(0,-1), a=1 -> matches length-4 DFT DC bin.
    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "Z = signal_czt(x, 4, 0, -1, 1, 0)");
    ASSERT_GT(interp.state().matrices.count("Z"), 0u);
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 4u);
    EXPECT_EQ(Z.cols(), 2u);
    EXPECT_NEAR(Z(0, 0), 10.0, 1e-9);
    EXPECT_NEAR(Z(0, 1), 0.0, 1e-9);

    expect_ok(interp, "zoom = signal_czt_zoom(x, 0.0, 0.5, 8, 4.0)");
    ASSERT_GT(interp.state().matrices.count("zoom"), 0u);
    const auto& zoom = interp.state().matrices.at("zoom");
    EXPECT_EQ(zoom.rows(), 8u);
    EXPECT_EQ(zoom.cols(), 2u);
    EXPECT_TRUE(std::isfinite(zoom(0, 0)));
    EXPECT_TRUE(std::isfinite(zoom(0, 1)));
}

TEST(ReplCommandsTest, signal_convolve) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_convolve(a,b)");

    expect_ok(interp, "c = signal_convolve([1; 2; 3], [1; 1])");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    const auto& conv = interp.state().matrices.at("c");
    EXPECT_EQ(conv.rows(), 4u);
    EXPECT_NEAR(conv(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(conv(1, 0), 3.0, 1e-9);
    EXPECT_NEAR(conv(2, 0), 5.0, 1e-9);
    EXPECT_NEAR(conv(3, 0), 3.0, 1e-9);
}

TEST(ReplCommandsTest, signal_hamming) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_hamming(n)");

    expect_ok(interp, "w = signal_hamming(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("w")(3, 0), 1.0, 0.1);
}

TEST(ReplCommandsTest, signal_hanning) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_hanning(n)");

    expect_ok(interp, "w = signal_hanning(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
    EXPECT_NEAR(interp.state().matrices.at("w")(3, 0), 1.0, 0.1);
}

TEST(ReplCommandsTest, signal_correlate) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_correlate(a,b)");

    expect_ok(interp, "c = signal_correlate([1; 2; 3], [1; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("c"), 0u);
    EXPECT_EQ(interp.state().matrices.at("c").rows(), 5u);
    EXPECT_NEAR(interp.state().matrices.at("c")(2, 0), 1.0, 1e-9);
}

TEST(ReplCommandsTest, signal_blackman) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_blackman(n)");

    expect_ok(interp, "w = signal_blackman(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
    EXPECT_TRUE(std::isfinite(interp.state().matrices.at("w")(3, 0)));
}

TEST(ReplCommandsTest, signal_parzen) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_parzen(n)");

    expect_ok(interp, "w = signal_parzen(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
}

TEST(ReplCommandsTest, signal_triangular) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_triangular(n)");

    expect_ok(interp, "w = signal_triangular(8)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    EXPECT_EQ(interp.state().matrices.at("w").rows(), 8u);
}

TEST(ReplCommandsTest, signal_lowpass) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_lowpass(x,cutoff,fs)");

    expect_ok(interp, "y = signal_lowpass([1; 0; -1; 0], 0.25, 1.0)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 4u);

    expect_ok(interp, "signal_lowpass([1; 0; -1; 0], 0.25, 1.0)");
}

TEST(ReplCommandsTest, signal_butterworth) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_butterworth(x,cutoff,fs)");

    expect_ok(interp, "y = signal_butterworth([1; 0; -1; 0], 0.25, 1.0)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 4u);

    expect_ok(interp, "signal_butterworth([1; 0; -1; 0], 0.25, 1.0)");
}

TEST(ReplCommandsTest, signal_highpass) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_highpass(x,cutoff,fs)");

    expect_ok(interp, "hp = signal_highpass([1; 0; -1; 0], 0.25, 1.0)");
    ASSERT_GT(interp.state().matrices.count("hp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("hp").rows(), 4u);

    expect_ok(interp, "signal_highpass([1; 0; -1; 0], 0.25, 1.0)");
}

TEST(ReplCommandsTest, signal_bandpass) {
    Interpreter interp;
    expect_contains(interp, "help", "signal_bandpass(x,low,high,fs)");

    expect_ok(interp, "bp = signal_bandpass([1; 0; -1; 0], 0.1, 0.3, 1.0)");
    ASSERT_GT(interp.state().matrices.count("bp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bp").rows(), 4u);

    expect_ok(interp, "signal_bandpass([1; 0; -1; 0], 0.1, 0.3, 1.0)");
}

TEST(ReplCommandsTest, quantum_signal_gria) {
    Interpreter interp;

    expect_ok(interp, "psi = [1; 0]");
    expect_ok(interp, "rho = quantum_density_matrix(psi)");
    EXPECT_EQ(interp.state().matrices.at("rho").rows(), 2u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);

    expect_ok(interp, "a = [1; 0; 1]");
    expect_ok(interp, "b = [0; 1; 0]");
    expect_ok(interp, "d = gria_hamming_distance(a, b)");
    EXPECT_TRUE(interp.state().scalars.count("d") > 0);

    expect_ok(interp, "pl = polylog(2, 0.25)");
    EXPECT_TRUE(interp.state().scalars.count("pl") > 0);
}

TEST(ReplCommandsTest, signal_sparse_special) {
    Interpreter interp;

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 3u);

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "db = debye(1, 0.25)");
    EXPECT_TRUE(interp.state().scalars.count("db") > 0);
}

TEST(ReplCommandsTest, signal_ode_special) {
    Interpreter interp;

    expect_ok(interp, "h = signal_firwin(4, 0.3)");
    EXPECT_GT(interp.state().matrices.at("h").rows(), 0u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "ek = ellip_k(0.5)");
    EXPECT_TRUE(interp.state().scalars.count("ek") > 0);
}

TEST(ReplCommandsTest, signal_resample_savgol) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, sosfilt_rk4_sparse_coo) {
    Interpreter interp;

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);

    expect_ok(interp, "traj = ode_rk4(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("traj").rows(), 0u);

    expect_ok(interp, "ri = [0; 1; 2]");
    expect_ok(interp, "ci = [0; 1; 2]");
    expect_ok(interp, "vv = [2; 3; 4]");
    expect_ok(interp, "A = sparse_from_coo(3, 3, ri, ci, vv)");
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 4u);
}

TEST(ReplCommandsTest, step_conv2) {
    Interpreter interp;

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    EXPECT_EQ(interp.state().matrices.at("step").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("step").cols(), 2u);

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    EXPECT_EQ(interp.state().matrices.at("C").rows(), 3u);
}

TEST(ReplCommandsTest, deconv_firwin_xcorr) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    EXPECT_EQ(interp.state().matrices.at("x").rows(), 3u);

    expect_ok(interp, "h = signal_firwin(4, 0.3)");
    EXPECT_GT(interp.state().matrices.at("h").rows(), 0u);

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    EXPECT_EQ(interp.state().matrices.at("xc").rows(), 5u);
}

TEST(ReplCommandsTest, autocorr_lms) {
    Interpreter interp;

    expect_ok(interp, "ac = signal_autocorr([1; -2; 3; 0.5], 2)");
    ASSERT_GT(interp.state().matrices.count("ac"), 0u);

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
}

TEST(ReplCommandsTest, envelope_hilbert) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "env = signal_envelope(x)");
    expect_ok(interp, "h = signal_hilbert(x)");
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 4u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);
}

TEST(ReplCommandsTest, phase_unwrap) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x)");
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);
}

TEST(ReplCommandsTest, coherence_rosenbrock) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, resample_savgol) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, resample_savgol_2) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_2) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_2) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_2) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_2) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_2) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_2) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_2) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_3) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_2) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_3) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_3) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_3) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_3) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_3) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_3) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_3) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_4) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_3) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_4) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_4) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_4) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_4) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_4) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_4) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_4) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_5) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_4) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_5) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_5) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_5) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_5) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_5) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_5) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_5) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_6) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_5) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_6) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_6) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_6) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_6) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_6) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_6) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_6) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_7) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_6) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_7) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_7) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_7) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_7) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_7) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_7) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_7) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_8) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_7) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_8) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_8) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_8) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_8) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_8) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_8) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_8) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_9) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_8) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_9) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_9) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_9) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_9) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_9) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_9) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_9) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_10) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_9) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_10) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_10) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_10) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_10) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_10) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_10) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_10) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_11) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_10) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_11) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_11) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_11) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_11) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_11) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_11) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_11) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_12) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_11) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_12) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_12) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_12) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_12) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_12) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_12) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_12) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_13) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_12) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_13) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_13) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_13) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_13) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_13) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_13) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_13) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_14) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_13) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_14) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_14) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_14) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_14) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_14) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_14) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_14) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_15) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_14) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_15) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_15) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_15) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_15) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_15) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_15) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_15) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_16) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_15) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_16) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_16) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_16) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_16) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_16) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_16) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_16) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_17) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_16) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_17) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_17) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_17) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_17) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_17) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_17) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_17) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_18) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_17) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_18) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_18) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_18) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_18) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_18) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_18) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_18) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_19) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_18) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_19) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_19) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_19) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_19) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_19) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_19) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_19) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_20) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_19) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_20) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_20) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_20) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_20) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_20) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_20) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_20) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_21) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_20) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_21) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_21) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_21) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_21) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_21) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_21) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_21) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_22) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_21) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_22) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_22) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_22) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_22) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_22) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_22) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_22) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_23) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_22) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_23) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_23) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_23) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_23) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_23) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_23) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_23) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_24) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_23) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_24) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_24) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_24) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_24) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_24) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_24) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_24) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_25) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_24) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_25) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_25) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_25) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_25) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_25) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_25) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_25) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_26) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_25) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_26) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_26) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_26) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_26) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_26) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_26) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_26) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_27) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_26) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_27) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_27) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_27) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_27) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_27) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_27) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_27) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_28) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_27) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_28) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_28) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_28) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_28) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_28) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_28) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_28) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_29) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_28) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_29) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_29) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_29) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_29) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_29) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_29) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_29) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_30) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_29) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_30) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_30) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_30) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_30) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_30) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_30) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_30) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_31) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_30) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_31) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_31) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_31) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_31) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_31) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_31) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_31) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_32) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_resample_savgol_31) {
    Interpreter interp;

    expect_ok(interp, "rs = signal_resample([1; 2; 3; 4], 2, 2)");
    ASSERT_GT(interp.state().matrices.count("rs"), 0u);
    EXPECT_EQ(interp.state().matrices.at("rs").rows(), 4u);

    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_savgol(x, 3, 1)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, ctrb_sosfilt_32) {
    Interpreter interp;

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    ASSERT_GT(interp.state().matrices.at("Co").rows(), 0u);

    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_ok(interp, "y = signal_sosfilt(sos, x)");
    ASSERT_GT(interp.state().matrices.count("y"), 0u);
    EXPECT_EQ(interp.state().matrices.at("y").rows(), 5u);
}

TEST(ReplCommandsTest, conv2_impulse_32) {
    Interpreter interp;

    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_ok(interp, "C = signal_conv2(A, K)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& C = interp.state().matrices.at("C");
    EXPECT_EQ(C.rows(), 3u);
    EXPECT_EQ(C.cols(), 3u);
    EXPECT_NEAR(C(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(C(1, 1), 5.0, 1e-12);

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);
    EXPECT_EQ(interp.state().matrices.at("imp").cols(), 2u);
}

TEST(ReplCommandsTest, deconv_firwin_32) {
    Interpreter interp;

    expect_ok(interp, "y = [1; 3; 5; 3]");
    expect_ok(interp, "b = [1; 1]");
    expect_ok(interp, "x = signal_deconv(y, b)");
    ASSERT_GT(interp.state().matrices.count("x"), 0u);
    const auto& x = interp.state().matrices.at("x");
    EXPECT_EQ(x.rows(), 3u);
    EXPECT_EQ(x.cols(), 1u);
    EXPECT_NEAR(x(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(x(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(x(2, 0), 3.0, 1e-12);

    expect_ok(interp, "fw = signal_firwin(11, 0.3)");
    ASSERT_GT(interp.state().matrices.count("fw"), 0u);
    EXPECT_EQ(interp.state().matrices.at("fw").rows(), 11u);

    expect_ok(interp, "bh = signal_firwin_highpass(11, 0.3, 1)");
    ASSERT_GT(interp.state().matrices.count("bh"), 0u);
    EXPECT_EQ(interp.state().matrices.at("bh").rows(), 11u);
}

TEST(ReplCommandsTest, xcorr_midpoint_32) {
    Interpreter interp;

    expect_ok(interp, "xc = signal_xcorr([1; 2; 3], [1; 0; 0], 2)");
    ASSERT_GT(interp.state().matrices.count("xc"), 0u);
    const auto& xc = interp.state().matrices.at("xc");
    EXPECT_EQ(xc.rows(), 5u);
    EXPECT_EQ(xc.cols(), 1u);
    EXPECT_NEAR(xc(2, 0), 1.0, 1e-12);

    expect_ok(interp, "cv = signal_xcov([1; 2; 3; 4], [2; 1; 4; 3], 2)");
    ASSERT_GT(interp.state().matrices.count("cv"), 0u);
    EXPECT_EQ(interp.state().matrices.at("cv").rows(), 5u);

    expect_ok(interp, "tr = ode_midpoint(\"y\", 0, 1, 1, 5)");
    ASSERT_GT(interp.state().matrices.count("tr"), 0u);
    ASSERT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, lms_envelope_32) {
    Interpreter interp;

    expect_ok(interp, "ye = signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("ye"), 0u);
    const auto& ye = interp.state().matrices.at("ye");
    EXPECT_EQ(ye.rows(), 4u);
    EXPECT_EQ(ye.cols(), 2u);
    EXPECT_NEAR(ye(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(ye(0, 1), 0.5, 1e-12);

    expect_ok(interp, "w = signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)");
    ASSERT_GT(interp.state().matrices.count("w"), 0u);
    const auto& w = interp.state().matrices.at("w");
    EXPECT_EQ(w.rows(), 2u);
    EXPECT_EQ(w.cols(), 1u);
    EXPECT_NEAR(w(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(w(1, 0), 0.0, 1e-12);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    ASSERT_GT(interp.state().matrices.count("env"), 0u);
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, hilbert_phase_32) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_ok(interp, "h = signal_hilbert(x)");
    ASSERT_GT(interp.state().matrices.count("h"), 0u);
    EXPECT_EQ(interp.state().matrices.at("h").rows(), 4u);

    expect_ok(interp, "x2 = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "ph = signal_instantaneous_phase(x2)");
    ASSERT_GT(interp.state().matrices.count("ph"), 0u);
    EXPECT_EQ(interp.state().matrices.at("ph").rows(), 8u);
}

TEST(ReplCommandsTest, unwrap_rfft_32) {
    Interpreter interp;

    expect_ok(interp, "pw = [0; 1.5707963267948966; 3.141592653589793; -1.5707963267948966; 0]");
    expect_ok(interp, "u = signal_unwrap(pw)");
    ASSERT_GT(interp.state().matrices.count("u"), 0u);
    EXPECT_EQ(interp.state().matrices.at("u").rows(), 5u);

    expect_ok(interp, "R = fft_rfft([1; 0; 0; 0])");
    ASSERT_GT(interp.state().matrices.count("R"), 0u);
    EXPECT_EQ(interp.state().matrices.at("R").rows(), 3u);
}

TEST(ReplCommandsTest, coherence_rosenbrock_33) {
    Interpreter interp;

    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_ok(interp, "c = signal_coherence(x, x, 8.0, 8)");
    EXPECT_GT(interp.state().matrices.at("c").rows(), 0u);

    expect_ok(interp, "tr = ode_rosenbrock23(\"-10*y\", 0, 1, 1, 20)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);
}

TEST(ReplCommandsTest, signal_firwin_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_firwin(11, 0.3)", "b =");
    expect_contains(interp, "signal_firwin(11, 0.3, 1)", "b =");
    expect_contains(interp, "signal_firwin(11, 0.3, 2)", "b =");

    expect_error_contains(interp, "signal_firwin(1.5, 0.3)", "expected integer n_taps >= 1");
    expect_error_contains(interp, "signal_firwin(0, 0.3)", "expected integer n_taps >= 1");
    expect_error_contains(interp, "signal_firwin(1.5, 0.3, 1)", "expected integer n_taps >= 1");
    expect_error_contains(interp, "signal_firwin(abc, 0.3)",
                          "expected signal_firwin(n_taps, cutoff[, window])");
    expect_error_contains(interp, "signal_firwin(11, xyz)",
                          "expected signal_firwin(n_taps, cutoff[, window])");
    expect_error_contains(interp, "signal_firwin(abc, 0.3, 1)",
                          "expected signal_firwin(n_taps, cutoff[, window])");
    expect_error_contains(interp, "h = signal_firwin(1.5, 0.3)", "expected integer n_taps >= 1");
    expect_error_contains(interp, "h = signal_firwin(not_a_number, 0.3)", "numeric");
}

TEST(ReplCommandsTest, signal_firwin_highpass_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_firwin_highpass(11, 0.3)", "b =");
    expect_contains(interp, "signal_firwin_highpass(11, 0.3, 1)", "b =");

    expect_error_contains(interp, "signal_firwin_highpass(1.5, 0.3)",
                          "expected integer n_taps >= 1");
    expect_error_contains(interp, "signal_firwin_highpass(0, 0.3, 1)",
                          "expected integer n_taps >= 1");
    expect_error_contains(interp, "signal_firwin_highpass(abc, 0.3)",
                          "expected signal_firwin_highpass(n_taps, cutoff[, window])");
    expect_error_contains(interp, "signal_firwin_highpass(11, xyz, 1)",
                          "expected signal_firwin_highpass(n_taps, cutoff[, window])");
}

TEST(ReplCommandsTest, signal_lms_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)", "lms =");

    expect_error_contains(interp, "signal_lms(missing, [0.5; -0.25; 1; 0], 2, 0)",
                          "unknown matrix");
    expect_error_contains(interp, "signal_lms([1; 0; -1; 2], missing, 2, 0)", "unknown matrix");
    expect_error_contains(interp, "signal_lms([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, notnum)",
                          "expected signal_lms(x, d, filter_length, mu)");
    expect_error_contains(interp, "ye = signal_lms(missing, [0.5; -0.25; 1; 0], 2, 0)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, signal_lms_weights_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, 0)",
                    "lms_weights =");

    expect_error_contains(interp, "signal_lms_weights(missing, [0.5; -0.25; 1; 0], 2, 0)",
                          "unknown matrix");
    expect_error_contains(interp, "signal_lms_weights([1; 0; -1; 2], [0.5; -0.25; 1; 0], 2, notnum)",
                          "expected signal_lms_weights(x, d, filter_length, mu)");
}

TEST(ReplCommandsTest, signal_coherence_execute_errors) {
    Interpreter interp;

    expect_contains(interp,
                    "signal_coherence([1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0], "
                    "[1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0], 8.0, 8)",
                    "coherence =");

    expect_error_contains(
        interp, "signal_coherence(missing, [1; 0; -1; 0; 1; 0; -1; 0], 8.0, 8)",
        "unknown matrix");
    expect_error_contains(
        interp,
        "signal_coherence([1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0], "
        "[1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0], notnum, 8)",
        "expected signal_coherence(x, y, fs, nperseg)");
    expect_error_contains(
        interp, "c = signal_coherence(missing, [1; 0; -1; 0; 1; 0; -1; 0], 8.0, 8)",
        "unknown matrix");
}

TEST(ReplCommandsTest, signal_czt_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_czt([1; 2; 3; 4], 4, 0, -1, 1, 0)", "czt =");

    expect_error_contains(interp, "signal_czt([1; 2; 3; 4], 0, 0, -1, 1, 0)",
                          "expected positive integer m");
    expect_error_contains(interp, "signal_czt([1; 2; 3; 4], 1.5, 0, -1, 1, 0)",
                          "expected positive integer m");
    expect_error_contains(interp, "signal_czt([1; 2; 3; 4], notnum, 0, -1, 1, 0)",
                          "expected signal_czt(x, m, w_re, w_im, a_re, a_im)");
    expect_error_contains(interp, "Z = signal_czt(missing, 4, 0, -1, 1, 0)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_czt_zoom_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_czt_zoom([1; 2; 3; 4], 0.0, 0.5, 8, 4.0)", "czt_zoom =");

    expect_error_contains(interp, "signal_czt_zoom([1; 2; 3; 4], 0.0, 0.5, 0, 4.0)",
                          "expected positive integer m");
    expect_error_contains(interp, "signal_czt_zoom([1; 2; 3; 4], 0.0, 0.5, 1.5, 4.0)",
                          "expected positive integer m");
    expect_error_contains(interp, "signal_czt_zoom([1; 2; 3; 4], 0.0, 0.5, notnum, 4.0)",
                          "expected signal_czt_zoom(x, f_start, f_stop, m, fs)");
    expect_error_contains(interp, "zoom = signal_czt_zoom(missing, 0.0, 0.5, 8, 4.0)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, signal_cheby1_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_cheby1(2, 1.0, 0.25, 2.0)", "ba =");
    expect_contains(interp, "signal_cheby1(2, 1.0, 0.25, 2.0, 1)", "ba =");

    expect_error_contains(interp, "signal_cheby1(0, 1.0, 0.25, 2.0)",
                          "expected integer order >= 1");
    expect_error_contains(interp, "signal_cheby1(1.5, 1.0, 0.25, 2.0)",
                          "expected integer order >= 1");
    expect_error_contains(interp, "signal_cheby1(notnum, 1.0, 0.25, 2.0)",
                          "expected signal_cheby1(order, rp_db, cutoff, fs[, type])");
    expect_error_contains(interp, "ba = signal_cheby1(0, 1.0, 0.25, 2.0)",
                          "expected integer order >= 1");
    expect_error_contains(interp, "ba = signal_cheby1(abc, 1.0, 0.25, 2.0)",
                          "expected signal_cheby1(order, rp_db, cutoff, fs[, type])");
}

TEST(ReplCommandsTest, signal_bandpass_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_bandpass([1; 0; -1; 0], 0.1, 0.3, 1.0)", "filtered =");

    expect_error_contains(interp, "signal_bandpass(missing, 0.1, 0.3, 1.0)", "unknown matrix");
    expect_error_contains(interp, "signal_bandpass([1; 0; -1; 0], notnum, 0.3, 1.0)",
                          "expected signal_bandpass(x, low, high, fs)");
    expect_error_contains(interp, "bp = signal_bandpass(missing, 0.1, 0.3, 1.0)",
                          "unknown matrix");
}

TEST(ReplCommandsTest, signal_resample_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_resample([1; 2; 3; 4], 2, 2)", "resampled =");

    expect_error_contains(interp, "signal_resample(missing, 2, 2)", "unknown matrix");
    expect_error_contains(interp, "signal_resample([1; 2; 3; 4], notnum, 2)",
                          "expected signal_resample(x, p, q)");
    expect_error_contains(interp, "rs = signal_resample(missing, 2, 2)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_periodogram_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_periodogram([1; 0; -1; 0], 1.0)", "psd =");

    expect_error_contains(interp, "signal_periodogram(missing, 1.0)", "unknown matrix");
    expect_error_contains(interp, "signal_periodogram([1; 0; -1; 0], notnum)",
                          "expected signal_periodogram(x, fs)");
    expect_error_contains(interp, "p = signal_periodogram(missing, 1.0)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_moving_average_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_moving_average([5; 5; 5; 5], 3)", "smoothed =");

    expect_error_contains(interp, "signal_moving_average(missing, 3)", "unknown matrix");
    expect_error_contains(interp, "signal_moving_average([5; 5; 5; 5], 0)",
                          "expected positive integer window");
    expect_error_contains(interp, "signal_moving_average([5; 5; 5; 5], 1.5)",
                          "expected positive integer window");
    expect_error_contains(interp, "ma = signal_moving_average(missing, 3)", "unknown matrix");
    expect_error_contains(interp, "ma = signal_moving_average([5; 5; 5; 5], 0)",
                          "expected positive integer window");
}

TEST(ReplCommandsTest, signal_lowpass_execute_errors) {
    Interpreter interp;

    expect_contains(interp, "signal_lowpass([1; 0; -1; 0], 0.25, 1.0)", "filtered =");

    expect_error_contains(interp, "signal_lowpass(missing, 0.25, 1.0)", "unknown matrix");
    expect_error_contains(interp, "signal_lowpass([1; 0; -1; 0], notnum, 1.0)",
                          "expected signal_lowpass(x, cutoff, fs)");
    expect_error_contains(interp, "y = signal_lowpass(missing, 0.25, 1.0)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_envelope_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_envelope([1; 2; 3; 4])", "envelope");
    expect_error_contains(interp, "signal_envelope([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, signal_hilbert_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_hilbert([1; 2; 3; 4])", "hilbert");
    expect_error_contains(interp, "signal_hilbert([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, signal_instantaneous_phase_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_instantaneous_phase([1; 0; -1; 0; 1; 0; -1; 0])", "phase");
    expect_error_contains(interp, "signal_instantaneous_phase([1, 2; 3, 4])",
                          "coefficient vector");
}

TEST(ReplCommandsTest, signal_unwrap_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_unwrap([0; 1.5708; 3.1416; -1.5708; 0])", "unwrap");
    expect_error_contains(interp, "signal_unwrap([1, 2; 3, 4])", "coefficient vector");
}

TEST(ReplCommandsTest, signal_firwin_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_firwin(11, 0.3, 0)", "b =");
    expect_error_contains(interp, "signal_firwin(1.5, 0.2, 0)", "integer n_taps");
}

TEST(ReplCommandsTest, signal_firwin_highpass_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_firwin_highpass(11, 0.3, 1)", "b =");
    expect_error_contains(interp, "signal_firwin_highpass(1.5, 0.3, 1)", "integer n_taps");
}

TEST(ReplCommandsTest, signal_cheby2_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_cheby2(2, 40.0, 0.25, 2.0)", "ba =");
    expect_error_contains(interp, "signal_cheby2(1.5, 40.0, 0.25, 2.0)", "integer order >= 1");
}

TEST(ReplCommandsTest, signal_cheby2_type_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_cheby2(2, 40.0, 0.25, 2.0, 1)", "ba =");
    expect_error_contains(interp, "signal_cheby2(2, 40.0, 0.25, 2.0, 2)",
                          "type 0 (lowpass) or 1 (highpass)");
}

TEST(ReplCommandsTest, signal_welch_psd_noassign) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0; 1; 0; -1; 0]");
    expect_contains(interp, "signal_welch_psd(x, 8.0, 8)", "psd =");
    expect_error_contains(interp, "signal_welch_psd(missing, 8.0, 8)", "unknown matrix");
    expect_error_contains(interp, "signal_welch_psd(x, 8.0, 1.5)", "positive integer nperseg");
}

TEST(ReplCommandsTest, signal_savgol_noassign) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 2; 3; 4]");
    expect_contains(interp, "signal_savgol(x, 3, 2)", "savgol =");
    expect_error_contains(interp, "signal_savgol(missing, 3, 2)", "unknown matrix");
    expect_error_contains(interp, "signal_savgol(x, 4, 2)", "odd positive window_length");
}

TEST(ReplCommandsTest, signal_median_filter_noassign) {
    Interpreter interp;
    expect_ok(interp, "x = [5; 1; 3; 2; 4]");
    expect_contains(interp, "signal_median_filter(x, 3)", "filtered =");
    expect_error_contains(interp, "signal_median_filter(missing, 3)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_upsample_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_upsample([1; 2], 3)", "upsampled =");
    expect_error_contains(interp, "signal_upsample(missing, 3)", "unknown matrix");
    expect_error_contains(interp, "signal_upsample([1; 2], 0)", "positive integer n");
}

TEST(ReplCommandsTest, signal_downsample_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_downsample([1; 2; 3; 4; 5; 6], 2)", "downsampled =");
    expect_error_contains(interp, "signal_downsample(missing, 2)", "unknown matrix");
    expect_error_contains(interp, "signal_downsample([1; 2; 3; 4], 1.5)", "positive integer n");
}

TEST(ReplCommandsTest, signal_decimate_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_decimate([1; 2; 3; 4; 5; 6], 2)", "decimated =");
    expect_error_contains(interp, "signal_decimate(missing, 2)", "unknown matrix");
    expect_error_contains(interp, "signal_decimate([1; 2; 3; 4], 0)", "positive integer q");
}

TEST(ReplCommandsTest, signal_interpolate_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_interpolate([1; 2], 2)", "interpolated =");
    expect_error_contains(interp, "signal_interpolate(missing, 2)", "unknown matrix");
    expect_error_contains(interp, "signal_interpolate([1; 2], 0)", "positive integer p");
}

TEST(ReplCommandsTest, signal_convolve_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_convolve([1;2],[0;1])", "conv =");
    expect_error_contains(interp, "signal_convolve(missing, [0;1])", "unknown matrix");
}

TEST(ReplCommandsTest, signal_correlate_noassign) {
    Interpreter interp;
    expect_contains(interp, "signal_correlate([1;2;3],[1;0;0])", "corr =");
    expect_error_contains(interp, "signal_correlate(missing, [1;0;0])", "unknown matrix");
}

TEST(ReplCommandsTest, signal_sosfilt_noassign) {
    Interpreter interp;
    expect_ok(interp, "sos = [2, -2, 0, 2, -1, 0]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_contains(interp, "signal_sosfilt(sos, x)", "y =");
    expect_error_contains(interp, "signal_sosfilt(missing, x)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_conv2_noassign) {
    Interpreter interp;
    expect_ok(interp, "A = [1, 2; 3, 4]");
    expect_ok(interp, "K = [1, 0; 0, 1]");
    expect_contains(interp, "signal_conv2(A, K)", "C =");
    expect_error_contains(interp, "signal_conv2(missing, K)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_filtfilt_noassign) {
    Interpreter interp;
    expect_ok(interp, "b = [1, -1]");
    expect_ok(interp, "a = [1, -0.5]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_contains(interp, "signal_filtfilt(b, a, x)", "y =");
    expect_error_contains(interp, "signal_filtfilt(missing, a, x)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_filter_noassign) {
    Interpreter interp;
    expect_ok(interp, "b = [1, -1]");
    expect_ok(interp, "a = [1, -0.5]");
    expect_ok(interp, "x = [1; 2; 3; 4; 5]");
    expect_contains(interp, "signal_filter(b, a, x)", "y =");
    expect_error_contains(interp, "signal_filter(missing, a, x)", "unknown matrix");
}

TEST(ReplCommandsTest, signal_spectrogram_noassign) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_contains(interp, "signal_spectrogram(x, 8.0)", "S =");
    expect_error_contains(interp, "signal_spectrogram(missing, 8.0)", "unknown matrix");
    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_error_contains(interp, "signal_spectrogram(M, 8.0)", "coefficient vector");
}

TEST(ReplCommandsTest, signal_instantaneous_freq_noassign) {
    Interpreter interp;
    expect_ok(interp, "x = [1; 0; -1; 0; 1; 0; -1; 0]");
    expect_contains(interp, "signal_instantaneous_freq(x, 8.0)", "freq =");
    expect_error_contains(interp, "signal_instantaneous_freq(missing, 8.0)", "unknown matrix");
    expect_ok(interp, "M = [1, 2; 3, 4]");
    expect_error_contains(interp, "signal_instantaneous_freq(M, 8.0)", "coefficient vector");
}
