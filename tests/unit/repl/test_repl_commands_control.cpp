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

TEST(ReplCommandsTest, control_dcgain) {
    Interpreter interp;
    expect_contains(interp, "help", "control_dcgain(num,den)");

    expect_contains(interp, "control_dcgain([1], [1, 1])", "1");

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "g = control_dcgain(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("g"), 1.0, 1e-6);
}

TEST(ReplCommandsTest, control_is_stable) {
    Interpreter interp;
    expect_contains(interp, "help", "control_is_stable(num,den)");

    expect_contains(interp, "control_is_stable([1], [1, 1])", "1");

    expect_ok(interp, "num = [1]");
    expect_ok(interp, "den = [1, 1]");
    expect_ok(interp, "s = control_is_stable(num, den)");
    EXPECT_NEAR(interp.state().scalars.at("s"), 1.0, 1e-6);
}

TEST(ReplCommandsTest, control_is_controllable) {
    Interpreter interp;
    expect_contains(interp, "help", "control_is_controllable(A,B)");

    expect_ok(interp, "A = [0, 1; 0, 0]");
    expect_ok(interp, "B = [0; 1]");
    expect_ok(interp, "ctrl = control_is_controllable(A, B)");
    EXPECT_NEAR(interp.state().scalars.at("ctrl"), 1.0, 1e-9);

    expect_contains(interp, "control_is_controllable([0, 1; 0, 0], [0; 1])", "1");
}

TEST(ReplCommandsTest, control_is_observable) {
    Interpreter interp;
    expect_contains(interp, "help", "control_is_observable(A,C)");

    expect_ok(interp, "A = [0, 1; 0, 0]");
    expect_ok(interp, "C = [1, 0]");
    expect_ok(interp, "obs = control_is_observable(A, C)");
    EXPECT_NEAR(interp.state().scalars.at("obs"), 1.0, 1e-9);

    expect_contains(interp, "control_is_observable([0, 1; 0, 0], [1, 0])", "1");
}

TEST(ReplCommandsTest, control_impulse_final) {
    Interpreter interp;
    expect_contains(interp, "help", "control_impulse_final(num,den)");

    expect_ok(interp, "imp = control_impulse_final([1], [1, 1])");
    EXPECT_NEAR(interp.state().scalars.at("imp"), std::exp(-10.0), 1e-4);

    expect_contains(interp, "control_impulse_final([1], [1, 1])", "0.00004");
}

TEST(ReplCommandsTest, control_lyap) {
    Interpreter interp;
    expect_contains(interp, "help", "control_lyap(A,Q)");

    expect_ok(interp, "X = control_lyap([-1], [1])");
    ASSERT_GT(interp.state().matrices.count("X"), 0u);
    const auto& X = interp.state().matrices.at("X");
    EXPECT_EQ(X.rows(), 1u);
    EXPECT_EQ(X.cols(), 1u);
    EXPECT_NEAR(X(0, 0), 0.5, 1e-4);
}

TEST(ReplCommandsTest, control_lqr) {
    Interpreter interp;
    expect_contains(interp, "help", "control_lqr(A,B,Q,R)");

    expect_ok(interp, "Q = eye(2)");
    expect_ok(interp, "K = control_lqr([-2, 1; 0, -3], [1; 1], Q, [1])");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    const auto& Klqr = interp.state().matrices.at("K");
    EXPECT_EQ(Klqr.rows(), 1u);
    EXPECT_EQ(Klqr.cols(), 2u);
}

TEST(ReplCommandsTest, control_pidtune_kp) {
    Interpreter interp;
    expect_contains(interp, "help", "control_pidtune_kp(num,den)");

    expect_ok(interp, "kp = control_pidtune_kp([1], [1, 1])");
    EXPECT_GT(interp.state().scalars.at("kp"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("kp"), 1.414, 0.1);

    expect_contains(interp, "control_pidtune_kp([1], [1, 1])", "1.");
}

TEST(ReplCommandsTest, control_place) {
    Interpreter interp;
    expect_contains(interp, "help", "control_place(A,B,poles)");

    expect_ok(interp, "K = control_place([0, 1; 0, 0], [0; 1], [-2; -3])");
    ASSERT_GT(interp.state().matrices.count("K"), 0u);
    EXPECT_EQ(interp.state().matrices.at("K").rows(), 2u);
}

TEST(ReplCommandsTest, control_pidtune_ki) {
    Interpreter interp;
    expect_contains(interp, "help", "control_pidtune_ki(num,den)");

    expect_ok(interp, "ki = control_pidtune_ki([1], [1, 1])");
    EXPECT_GT(interp.state().scalars.at("ki"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("ki"), 0.141, 0.05);

    expect_contains(interp, "control_pidtune_ki([1], [1, 1])", "0.");
}

TEST(ReplCommandsTest, control_pidtune_kd) {
    Interpreter interp;
    expect_contains(interp, "help", "control_pidtune_kd(num,den)");

    expect_ok(interp, "kd = control_pidtune_kd([1], [1, 1])");
    EXPECT_GT(interp.state().scalars.at("kd"), 0.0);
    EXPECT_NEAR(interp.state().scalars.at("kd"), 0.141, 0.05);

    expect_contains(interp, "control_pidtune_kd([1], [1, 1])", "0.");
}

TEST(ReplCommandsTest, control_dlyap) {
    Interpreter interp;
    expect_contains(interp, "help", "control_dlyap(A,Q)");

    expect_ok(interp, "Xd = control_dlyap([0.5], [1])");
    ASSERT_GT(interp.state().matrices.count("Xd"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Xd")(0, 0), 4.0 / 3.0, 1e-3);

    expect_contains(interp, "control_dlyap([0.5], [1])", "1.333");
}

TEST(ReplCommandsTest, control_riccati) {
    Interpreter interp;
    expect_contains(interp, "help", "control_riccati(A,B,Q,R)");

    expect_ok(interp, "Q = eye(2)");
    expect_ok(interp, "Xr = control_riccati([-2, 1; 0, -3], [1; 1], Q, [1])");
    ASSERT_GT(interp.state().matrices.count("Xr"), 0u);
    EXPECT_GT(interp.state().matrices.at("Xr")(0, 0), 0.0);
}

TEST(ReplCommandsTest, control_dare) {
    Interpreter interp;
    expect_contains(interp, "help", "control_dare(A,B,Q,R)");

    expect_ok(interp, "Xdare = control_dare([0.5], [1], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("Xdare"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Xdare")(0, 0), 4.0 / 3.0, 0.25);
}

TEST(ReplCommandsTest, control_bode_mag_db) {
    Interpreter interp;
    expect_contains(interp, "help", "control_bode_mag_db(num,den,w)");

    expect_ok(interp, "bodeDb = control_bode_mag_db([1], [1, 1], 1)");
    EXPECT_NEAR(interp.state().scalars.at("bodeDb"), -3.01, 0.15);
}

TEST(ReplCommandsTest, control_bode_phase) {
    Interpreter interp;
    expect_contains(interp, "help", "control_bode_phase(num,den,w)");

    expect_ok(interp, "bodePh = control_bode_phase([1], [1, 1], 1)");
    EXPECT_NEAR(interp.state().scalars.at("bodePh"), -45.0, 2.0);
}

TEST(ReplCommandsTest, control_phase_margin) {
    Interpreter interp;
    expect_contains(interp, "help", "control_phase_margin(num,den)");

    expect_ok(interp, "pm = control_phase_margin([1], [1, 1])");
    EXPECT_NEAR(interp.state().scalars.at("pm"), 180.0, 5.0);
}

TEST(ReplCommandsTest, control_gain_margin) {
    Interpreter interp;
    expect_contains(interp, "help", "control_gain_margin(num,den)");

    expect_ok(interp, "gm = control_gain_margin([1], [1, 1])");
    expect_contains(interp, "control_gain_margin([1], [1, 1])", "inf");
}

TEST(ReplCommandsTest, control_bode) {
    Interpreter interp;
    expect_contains(interp, "help", "control_bode(num,den,w)");

    expect_ok(interp, "bode = control_bode([1], [1, 1], 1)");
    ASSERT_GT(interp.state().matrices.count("bode"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("bode")(0, 0), -3.01, 0.1);
    EXPECT_NEAR(interp.state().matrices.at("bode")(0, 1), -45.0, 2.0);
}

TEST(ReplCommandsTest, control_step_response) {
    Interpreter interp;
    expect_contains(interp, "help", "control_step_response(num,den[,t_end[,n_pts]])");

    // 1/(s+1): y(t) = 1 - e^{-t}; at t=5 should be near 1
    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_NEAR(step(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(step(step.rows() - 1, 0), 5.0, 1e-9);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);

    expect_ok(interp, "step_def = control_step_response([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("step_def"), 0u);
    EXPECT_EQ(interp.state().matrices.at("step_def").cols(), 2u);
    EXPECT_GT(interp.state().matrices.at("step_def").rows(), 0u);
}

TEST(ReplCommandsTest, control_impulse_response) {
    Interpreter interp;
    expect_contains(interp, "help", "control_impulse_response(num,den[,t_end[,n_pts]])");

    // 1/(s+1): impulse y(t) = e^{-t}; at t=5 should be near e^{-5}
    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("imp"), 0u);
    const auto& imp = interp.state().matrices.at("imp");
    EXPECT_EQ(imp.cols(), 2u);
    EXPECT_EQ(imp.rows(), 200u);
    EXPECT_NEAR(imp(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(imp(imp.rows() - 1, 0), 5.0, 1e-9);
    EXPECT_NEAR(imp(imp.rows() - 1, 1), std::exp(-5.0), 0.05);
}

TEST(ReplCommandsTest, control_tf2ss_c2d) {
    Interpreter interp;
    expect_contains(interp, "help", "control_tf2ss(num,den)");
    expect_contains(interp, "help", "control_c2d(A,B,C,D,Ts)");

    // 1/(s+1) -> A=[-1], B=[1], C=[1], D=[0] packed as [-1,1; 1,0]
    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    ASSERT_EQ(ss.rows(), 2u);
    ASSERT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    // Scalar ZOH: A=[-2], B=[3], C=[1], D=[0], Ts=0.1 -> Ad=exp(-0.2)
    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_B([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    const double Bd_exact = (std::exp(-0.2) - 1.0) / (-2.0) * 3.0;
    EXPECT_NEAR(interp.state().matrices.at("Bd")(0, 0), Bd_exact, 1e-6);
}

TEST(ReplCommandsTest, control_tf) {
    Interpreter interp;
    expect_contains(interp, "help", "control_series(num1,den1,num2,den2)");
    expect_contains(interp, "help", "control_parallel(num1,den1,num2,den2)");
    expect_contains(interp, "help", "control_feedback(numG,denG,numH,denH");
    expect_contains(interp, "help", "control_ss2tf(SS)");
    expect_contains(interp, "help", "control_d2c(A,B,C,D,Ts)");
    expect_contains(interp, "help", "control_c2d_tf(num,den,Ts)");
    expect_contains(interp, "help", "control_d2c_tf(num,den,Ts)");

    // series: (1/(s+1))*(2/(s+2)) -> 2/(s^2+3s+2)
    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    const auto& s = interp.state().matrices.at("S");
    ASSERT_EQ(s.rows(), 2u);
    ASSERT_EQ(s.cols(), 3u);
    EXPECT_NEAR(s(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(s(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(s(1, 1), 3.0, 1e-12);
    EXPECT_NEAR(s(1, 2), 2.0, 1e-12);

    // parallel: 1/(s+1) + 1/(s+1) -> DC gain 2
    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    const auto& p = interp.state().matrices.at("P");
    EXPECT_EQ(p.rows(), 2u);
    EXPECT_NEAR(packed_tf_dcgain(p), 2.0, 1e-9);

    // unity feedback around integrator 1/s -> 1/(s+1)
    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    const auto& f = interp.state().matrices.at("F");
    EXPECT_NEAR(packed_tf_dcgain(f), 1.0, 1e-3);

    // ss2tf roundtrip with control_tf2ss on 1/(s+1)
    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(packed_tf_dcgain(tf), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    // d2c inverse of scalar ZOH c2d: Ad=exp(-0.2) -> Ac=-2
    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);

    // TF discretization roundtrip on 1/(s+1)
    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    ASSERT_EQ(d.rows(), 2u);
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    const auto& c = interp.state().matrices.at("C");
    EXPECT_NEAR(packed_tf_dcgain(c), 1.0, 1e-2);
    EXPECT_NEAR(c(1, 0), 1.0, 1e-2);
    EXPECT_NEAR(c(1, 1), 1.0, 1e-2);
}

TEST(ReplCommandsTest, control_c2d_tustin) {
    Interpreter interp;
    expect_contains(interp, "help", "control_c2d_tustin(A,B,C,D,Ts)");
    expect_contains(interp, "help", "control_c2d_euler(A,B,C,D,Ts)");
    expect_contains(interp, "help", "control_d2c_tustin(A,B,C,D,Ts)");
    expect_contains(interp, "help", "control_d2c_euler(A,B,C,D,Ts)");
    expect_contains(interp, "help", "control_c2d_tf_tustin(num,den,Ts)");

    // Stable 1st-order plant G(s)=3/(s+2): A=-2, B=3, C=1, D=0, Ts=0.1
    // Tustin: Ad = (I+A*Ts/2)/(I-A*Ts/2), Bd = A^{-1}(Ad-I)B  (scalar: Ts/(1-A*Ts/2)*B)
    const double Ts = 0.1;
    const double A = -2.0;
    const double B = 3.0;
    const double tustin_denom = 1.0 - A * Ts * 0.5;
    const double Ad_tustin = (1.0 + A * Ts * 0.5) / tustin_denom;
    const double Bd_tustin = B * Ts / tustin_denom;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), Ad_tustin, 1e-9);

    expect_ok(interp, "Bd = [" + std::to_string(Bd_tustin) + "]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-6);

    const double Ad_euler = 1.0 + (-2.0) * Ts;
    const double Bd_euler = 3.0 * Ts;
    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), Ad_euler, 1e-12);
    expect_ok(interp, "Bd_e = [" + std::to_string(Bd_euler) + "]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac_e")(0, 0), -2.0, 1e-6);

    // TF Tustin roundtrip on 1/(s+1) should preserve DC gain ~1
    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    const auto& dt = interp.state().matrices.at("Dt");
    ASSERT_EQ(dt.rows(), 2u);
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
    EXPECT_NEAR(packed_tf_dcgain(interp.state().matrices.at("Ct")), 1.0, 1e-2);
}

TEST(ReplCommandsTest, control_kalman) {
    Interpreter interp;
    expect_contains(interp, "help", "control_kalman_predict(x,P,A,Q)");
    expect_contains(interp, "help", "control_kalman_update(x,P,z,H,R)");

    // Scalar closed-form predict+update from test_control.cpp (q=0.05, r=0.5, z=2).
    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-6);

    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    const double K = 1.05 / 1.55;
    EXPECT_NEAR(interp.state().matrices.at("Pu")(0, 0), (1.0 - K) * 1.05, 1e-6);
}

TEST(ReplCommandsTest, control_gram_ctrb_obsv) {
    Interpreter interp;
    expect_contains(interp, "help", "control_ctrb(A,B)");
    expect_contains(interp, "help", "control_obsv(A,C)");
    expect_contains(interp, "help", "control_ctrb_gram(A,B)");
    expect_contains(interp, "help", "control_obsv_gram(A,C)");

    // Controllability matrix of double integrator: [B AB] = [0 1; 1 0]
    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    ASSERT_GT(interp.state().matrices.count("Co"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Co").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Co").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("Co")(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Co")(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Co")(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Co")(1, 1), 0.0, 1e-12);

    // Observability matrix: [C; CA] = I for C=[1 0]
    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    ASSERT_GT(interp.state().matrices.count("Ob"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Ob").rows(), 2u);
    EXPECT_EQ(interp.state().matrices.at("Ob").cols(), 2u);
    EXPECT_NEAR(interp.state().matrices.at("Ob")(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Ob")(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Ob")(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(interp.state().matrices.at("Ob")(1, 1), 1.0, 1e-12);

    // Scalar closed form: A=-a, B=b => Wc = b^2/(2a)
    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Wc")(0, 0), 9.0 / 4.0, 1e-9);

    // Dual scalar: A=-a, C=c => Wo = c^2/(2a)
    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Wo")(0, 0), 9.0 / 4.0, 1e-9);
}

TEST(ReplCommandsTest, control_margins) {
    Interpreter interp;
    expect_contains(interp, "help", "control_margins(num,den)");

    expect_ok(interp, "marg = control_margins([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("marg"), 0u);
    expect_contains(interp, "control_margins([1], [1, 1])", "inf");
}

TEST(ReplCommandsTest, pde_control_2) {
    Interpreter interp;

    expect_ok(interp, "f0 = zeros(4, 4)");
    expect_ok(interp, "p2 = pde_poisson_2d(f0, 0.1, 0.1, 50, 1e-8)");
    EXPECT_EQ(interp.state().matrices.at("p2").rows(), 4u);

    expect_ok(interp, "ser = control_series([1], [1, 1], [2], [1, 2])");
    EXPECT_EQ(interp.state().matrices.at("ser").rows(), 2u);

    expect_ok(interp, "Wc = control_ctrb_gram([-1, 0; 0, -2], [1; 1])");
    EXPECT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, control_signal) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [2], [1, 2])");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "x = [1; 2; 3]");
    expect_ok(interp, "env = signal_envelope(x)");
    EXPECT_EQ(interp.state().matrices.at("env").rows(), 3u);
}

TEST(ReplCommandsTest, kalman_ctrb) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-12);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-6);

    expect_ok(interp, "Co = control_ctrb([0, 1; 0, 0], [0; 1])");
    EXPECT_GT(interp.state().matrices.at("Co").rows(), 0u);
}

TEST(ReplCommandsTest, euler_obsv_kalman_cov) {
    Interpreter interp;

    expect_ok(interp, "tr = ode_euler(\"y\", 0, 1, 1, 5)");
    EXPECT_GT(interp.state().matrices.at("tr").rows(), 0u);

    expect_ok(interp, "Ob = control_obsv([0, 1; 0, 0], [1, 0])");
    EXPECT_GT(interp.state().matrices.at("Ob").rows(), 0u);

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);
}

TEST(ReplCommandsTest, impulse_kalman_update_cov) {
    Interpreter interp;

    expect_ok(interp, "imp = control_impulse_response([1], [1, 1], 5, 200)");
    EXPECT_EQ(interp.state().matrices.at("imp").rows(), 200u);

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
}

TEST(ReplCommandsTest, gram_tf2ss_series) {
    Interpreter interp;

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    EXPECT_GT(interp.state().matrices.at("Wc").rows(), 0u);

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, c2d_parallel) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);
}

TEST(ReplCommandsTest, feedback_ss2tf) {
    Interpreter interp;

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
}

TEST(ReplCommandsTest, d2c_tf) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    EXPECT_EQ(interp.state().matrices.at("D").rows(), 2u);
    expect_ok(interp, "C = control_d2c_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
}

TEST(ReplCommandsTest, kalman) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_2) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_2) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_2) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_2) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_2) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_2) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_2) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_2) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_2) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_2) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_3) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_3) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_3) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_3) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_3) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_3) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_3) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_3) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_3) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_3) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_4) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_4) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_4) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_4) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_4) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_4) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_4) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_4) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_4) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_4) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_5) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_5) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_5) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_5) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_5) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_5) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_5) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_5) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_5) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_5) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_6) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_6) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_6) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_6) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_6) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_6) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_6) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_6) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_6) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_6) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_7) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_7) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_7) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_7) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_7) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_7) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_7) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_7) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_7) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_7) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_8) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_8) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_8) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_8) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_8) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_8) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_8) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_8) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_8) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_8) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_9) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_9) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_9) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_9) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_9) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_9) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_9) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_9) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_9) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_9) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_10) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_10) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_10) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_10) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_10) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_10) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_10) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_10) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_10) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_10) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_11) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_11) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_11) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_11) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_11) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_11) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_11) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_11) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_11) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_11) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_12) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_12) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_12) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_12) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_12) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_12) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_12) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_12) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_12) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_12) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_13) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_13) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_13) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_13) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_13) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_13) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_13) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_13) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_13) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_13) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_14) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_14) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_14) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_14) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_14) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_14) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_14) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_14) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_14) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_14) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_15) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_15) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_15) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_15) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_15) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_15) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_15) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_15) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_15) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_15) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_16) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_16) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_16) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_16) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_16) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_16) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_16) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_16) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_16) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_16) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_17) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_17) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_17) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_17) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_17) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_17) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_17) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_17) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_17) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_17) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_18) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_18) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_18) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_18) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_18) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_18) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_18) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_18) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_18) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_18) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_19) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_19) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_19) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_19) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_19) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_19) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_19) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_19) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_19) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_19) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_20) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_20) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_20) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_20) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_20) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_20) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_20) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_20) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_20) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_20) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_21) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_21) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_21) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_21) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_21) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_21) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_21) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_21) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_21) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_21) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_22) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_22) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_22) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_22) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_22) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_22) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_22) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_22) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_22) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_22) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_23) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_23) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_23) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_23) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_23) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_23) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_23) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_23) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_23) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_23) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_24) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_24) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_24) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_24) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_24) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_24) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_24) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_24) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_24) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_24) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_25) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_25) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_25) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_25) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_25) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_25) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_25) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_25) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_25) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_25) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_26) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_26) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_26) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_26) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_26) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_26) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_26) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_26) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_26) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_26) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_27) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_27) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_27) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_27) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_27) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_27) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_27) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_27) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_27) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_27) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_28) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_28) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_28) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_28) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_28) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_28) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_28) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_28) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_28) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_28) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_29) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_29) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_29) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_29) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_29) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_29) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_29) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_29) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_29) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_29) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_30) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_30) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_30) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_30) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_30) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_30) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_30) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_30) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_30) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_30) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_31) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_31) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_31) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_31) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_31) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_31) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_31) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_31) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_31) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_31) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}

TEST(ReplCommandsTest, kalman_32) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("xp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xp")(0, 0), 0.0, 1e-8);

    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);

    expect_ok(interp, "xu = control_kalman_update(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("xu"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("xu")(0, 0), 2.0 * 1.05 / 1.55, 1e-8);
}

TEST(ReplCommandsTest, control_c2d_32) {
    Interpreter interp;

    expect_ok(interp, "Ad = control_c2d_tustin([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), (1.0 - 0.1) / (1.0 + 0.1), 1e-8);

    expect_ok(interp, "Bd = [0.272727272727]");
    expect_ok(interp, "Ac = control_d2c_tustin(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);

    expect_ok(interp, "Ad_e = control_c2d_euler([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad_e"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad_e")(0, 0), 0.8, 1e-8);

    expect_ok(interp, "Bd_e = [0.3]");
    expect_ok(interp, "Ac_e = control_d2c_euler(Ad_e, Bd_e, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac_e"), 0u);
}

TEST(ReplCommandsTest, control_c2d_tf_32) {
    Interpreter interp;

    expect_ok(interp, "Dt = control_c2d_tf_tustin([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Dt"), 0u);
    EXPECT_EQ(interp.state().matrices.at("Dt").rows(), 2u);

    const auto& dt = interp.state().matrices.at("Dt");
    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < dt.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(dt(0, j));
        den_lit += std::to_string(dt(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "Ct = control_d2c_tf_tustin(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ct"), 0u);
}

TEST(ReplCommandsTest, kalman_cov_step_32) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    ASSERT_GT(interp.state().matrices.count("Pp"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Pp")(0, 0), 1.05, 1e-10);

    expect_ok(interp, "step = control_step_response([1], [1, 1], 5, 200)");
    ASSERT_GT(interp.state().matrices.count("step"), 0u);
    const auto& step = interp.state().matrices.at("step");
    EXPECT_EQ(step.rows(), 200u);
    EXPECT_EQ(step.cols(), 2u);
    EXPECT_NEAR(step(step.rows() - 1, 1), 1.0, 0.05);
}

TEST(ReplCommandsTest, update_cov_ctrb_gram_32) {
    Interpreter interp;

    expect_ok(interp, "x0 = [0]");
    expect_ok(interp, "P0 = [1]");
    expect_ok(interp, "xp = control_kalman_predict(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pp = control_kalman_predict_cov(x0, P0, [1], [0.05])");
    expect_ok(interp, "Pu = control_kalman_update_cov(xp, Pp, [2], [1], [0.5])");
    ASSERT_GT(interp.state().matrices.count("Pu"), 0u);
    ASSERT_GT(interp.state().matrices.at("Pu").rows(), 0u);

    expect_ok(interp, "Wc = control_ctrb_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wc"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wc").rows(), 0u);
}

TEST(ReplCommandsTest, tf2ss_series_32) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("SS"), 0u);
    const auto& ss = interp.state().matrices.at("SS");
    EXPECT_EQ(ss.rows(), 2u);
    EXPECT_EQ(ss.cols(), 2u);
    EXPECT_NEAR(ss(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(ss(0, 1), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 0), 1.0, 1e-12);
    EXPECT_NEAR(ss(1, 1), 0.0, 1e-12);

    expect_ok(interp, "S = control_series([1], [1, 1], [2], [1, 2])");
    ASSERT_GT(interp.state().matrices.count("S"), 0u);
    EXPECT_EQ(interp.state().matrices.at("S").rows(), 2u);
}

TEST(ReplCommandsTest, obsv_gram_c2d_32) {
    Interpreter interp;

    expect_ok(interp, "Wo = control_obsv_gram([-2], [3])");
    ASSERT_GT(interp.state().matrices.count("Wo"), 0u);
    ASSERT_GT(interp.state().matrices.at("Wo").rows(), 0u);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ad"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ad")(0, 0), std::exp(-0.2), 1e-6);

    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Bd"), 0u);
    ASSERT_GT(interp.state().matrices.at("Bd").rows(), 0u);
}

TEST(ReplCommandsTest, parallel_feedback_32) {
    Interpreter interp;

    expect_ok(interp, "P = control_parallel([1], [1, 1], [1], [1, 1])");
    ASSERT_GT(interp.state().matrices.count("P"), 0u);
    EXPECT_EQ(interp.state().matrices.at("P").rows(), 2u);

    expect_ok(interp, "F = control_feedback([1], [1, 0], [1], [1])");
    ASSERT_GT(interp.state().matrices.count("F"), 0u);
    ASSERT_GT(interp.state().matrices.at("F").rows(), 0u);
}

TEST(ReplCommandsTest, ss2tf_d2c_32) {
    Interpreter interp;

    expect_ok(interp, "SS = control_tf2ss([1], [1, 1])");
    expect_ok(interp, "TF = control_ss2tf(SS)");
    ASSERT_GT(interp.state().matrices.count("TF"), 0u);
    const auto& tf = interp.state().matrices.at("TF");
    EXPECT_NEAR(tf(1, 0), 1.0, 1e-9);
    EXPECT_NEAR(tf(1, 1), 1.0, 1e-9);

    expect_ok(interp, "Ad = control_c2d([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Bd = control_c2d_b([-2], [3], [1], [0], 0.1)");
    expect_ok(interp, "Ac = control_d2c(Ad, Bd, [1], [0], 0.1)");
    ASSERT_GT(interp.state().matrices.count("Ac"), 0u);
    EXPECT_NEAR(interp.state().matrices.at("Ac")(0, 0), -2.0, 1e-3);
}

TEST(ReplCommandsTest, c2d_tf_d2c_tf_32) {
    Interpreter interp;

    expect_ok(interp, "D = control_c2d_tf([1], [1, 1], 0.1)");
    ASSERT_GT(interp.state().matrices.count("D"), 0u);
    const auto& d = interp.state().matrices.at("D");
    EXPECT_EQ(d.rows(), 2u);

    std::string num_lit = "[";
    std::string den_lit = "[";
    for (size_t j = 0; j < d.cols(); ++j) {
        if (j > 0) {
            num_lit += ", ";
            den_lit += ", ";
        }
        num_lit += std::to_string(d(0, j));
        den_lit += std::to_string(d(1, j));
    }
    num_lit += "]";
    den_lit += "]";
    expect_ok(interp, "C = control_d2c_tf(" + num_lit + ", " + den_lit + ", 0.1)");
    ASSERT_GT(interp.state().matrices.count("C"), 0u);
    ASSERT_GT(interp.state().matrices.at("C").rows(), 0u);
}
