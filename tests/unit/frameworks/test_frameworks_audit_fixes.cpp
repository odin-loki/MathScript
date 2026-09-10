// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Regression tests for five audited framework defects.
//
//  * izaac::verify ignored the message entirely (its body ended with
//    `(void)msg;`) and checked only proof[i] - pub[i%32] == output[i], a
//    self-consistency identity anyone could satisfy: an attacker holding just
//    the public key could pick any output, set proof[i] = output[i%64] +
//    pub[i%32], and verification succeeded. izaac::prove also computed
//    output[i] = private_key[i%32] ^ msg[i%len], a one-time pad that handed the
//    private key to anyone who knew the message.
//  * izaac::crypto::encrypt built its nonce generator from scratch on every
//    call, seeded only by the key, so the nonce and the whole keystream were a
//    deterministic function of the key -- a two-time pad.
//  * gria::ca::alpha_ca passed compute_alpha an identity transform, and
//    gria::lfsr::alpha_lfsr passed a reversal, which cannot change a
//    permutation-invariant histogram entropy. Both were identically 0.
//  * cypha::nig_pdf was exp(-z)/(2*delta), which never reads beta and INCREASES
//    with |x - mu| toward a nonzero constant, so it had infinite mass.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/frameworks/izaac/izaac.hpp"

TEST(IzaacVrf, ProofBindsTheMessage) {
    const auto key = ms::izaac::keygen();
    const std::vector<std::uint8_t> m1{'h', 'e', 'l', 'l', 'o'};
    const std::vector<std::uint8_t> m2{'w', 'o', 'r', 'l', 'd'};
    const auto proof = ms::izaac::prove(key, m1);

    EXPECT_TRUE(ms::izaac::verify(key.public_key, m1, proof));
    EXPECT_FALSE(ms::izaac::verify(key.public_key, m2, proof))
        << "a proof for one message verified against another";
}

TEST(IzaacVrf, ForgedProofIsRejected) {
    // Exactly the forgery the old identity check accepted: choose any output,
    // then set proof[i] = output[i % 64] + pub[i % 32]. No private key involved.
    const auto key = ms::izaac::keygen();
    const std::vector<std::uint8_t> msg{'a', 't', 't', 'a', 'c', 'k'};

    ms::izaac::VRFProof forged{};
    for (std::size_t i = 0; i < forged.output.size(); ++i) {
        forged.output[i] = static_cast<std::uint8_t>(i);
    }
    for (std::size_t i = 0; i < forged.proof.size(); ++i) {
        forged.proof[i] = static_cast<std::uint8_t>(
            forged.output[i % forged.output.size()] + key.public_key[i % 32]);
    }
    EXPECT_FALSE(ms::izaac::verify(key.public_key, msg, forged));
}

TEST(IzaacVrf, WrongPublicKeyIsRejected) {
    const auto a = ms::izaac::keygen();
    const auto b = ms::izaac::keygen();
    const std::vector<std::uint8_t> msg{'x', 'y', 'z'};
    const auto proof = ms::izaac::prove(a, msg);
    EXPECT_TRUE(ms::izaac::verify(a.public_key, msg, proof));
    EXPECT_FALSE(ms::izaac::verify(b.public_key, msg, proof));
}

TEST(IzaacVrf, OutputIsUniquePerMessage) {
    // A VRF must be deterministic: the same key and message give the same
    // output every time, and different messages give different outputs.
    const auto key = ms::izaac::keygen();
    const std::vector<std::uint8_t> m1{1, 2, 3};
    const std::vector<std::uint8_t> m2{1, 2, 4};
    const auto p1 = ms::izaac::prove(key, m1);
    const auto p1_again = ms::izaac::prove(key, m1);
    const auto p2 = ms::izaac::prove(key, m2);
    EXPECT_EQ(p1.output, p1_again.output);
    EXPECT_NE(p1.output, p2.output);
}

TEST(IzaacVrf, KeygenProducesDistinctKeys) {
    const auto a = ms::izaac::keygen();
    const auto b = ms::izaac::keygen();
    EXPECT_NE(a.private_key, b.private_key);
    EXPECT_NE(a.public_key, b.public_key);
}

TEST(IzaacCipher, NonceIsFreshPerCall) {
    std::array<std::uint8_t, 32> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<std::uint8_t>(i);
    }
    const std::vector<std::uint8_t> pt{1, 2, 3, 4, 5, 6, 7, 8};
    const auto c1 = ms::izaac::crypto::encrypt(pt, key);
    const auto c2 = ms::izaac::crypto::encrypt(pt, key);
    ASSERT_GE(c1.data.size(), 16u);
    ASSERT_GE(c2.data.size(), 16u);
    EXPECT_NE(0, std::memcmp(c1.data.data(), c2.data.data(), 16))
        << "the nonce is still a function of the key alone";
    // Same plaintext, same key, different nonce => different ciphertext body.
    EXPECT_NE(0, std::memcmp(c1.data.data() + 16, c2.data.data() + 16, pt.size()));
}

TEST(IzaacCipher, IsNotATwoTimePad) {
    std::array<std::uint8_t, 32> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<std::uint8_t>(i * 7 + 1);
    }
    const std::vector<std::uint8_t> a{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::uint8_t> b{9, 10, 11, 12, 13, 14, 15, 16};
    const auto ca = ms::izaac::crypto::encrypt(a, key);
    const auto cb = ms::izaac::crypto::encrypt(b, key);
    ASSERT_GE(ca.data.size(), 16u + a.size());
    ASSERT_GE(cb.data.size(), 16u + b.size());

    bool pad = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if ((ca.data[16 + i] ^ cb.data[16 + i]) != (a[i] ^ b[i])) {
            pad = false;
            break;
        }
    }
    EXPECT_FALSE(pad) << "ct1 XOR ct2 still equals pt1 XOR pt2";
}

TEST(IzaacCipher, StillRoundTrips) {
    std::array<std::uint8_t, 32> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<std::uint8_t>(255 - i);
    }
    const std::vector<std::uint8_t> pt{0, 1, 0, 255, 42, 0, 7};
    const auto ct = ms::izaac::crypto::encrypt(pt, key);
    const auto back = ms::izaac::crypto::decrypt(ct, key);
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, pt);

    const std::vector<std::uint8_t> empty;
    const auto ct_empty = ms::izaac::crypto::encrypt(empty, key);
    const auto back_empty = ms::izaac::crypto::decrypt(ct_empty, key);
    ASSERT_TRUE(back_empty.has_value());
    EXPECT_TRUE(back_empty->empty());
}

TEST(GriaAlpha, CaDiscriminatesBetweenRules) {
    // alpha is the fraction of entropy the rule's own update destroys.
    // Rule 204 is the identity map and destroys none; rule 0 sends every
    // neighbourhood to 0 and destroys all of it. The old identity transform
    // returned exactly 0 for both.
    EXPECT_NEAR(ms::gria::ca::alpha_ca(204, 64, 65), 0.0, 1e-9);
    EXPECT_GT(ms::gria::ca::alpha_ca(0, 64, 65), 0.5);
    // Every result is a well-formed fraction.
    for (std::uint8_t rule : {std::uint8_t{30}, std::uint8_t{110}, std::uint8_t{90},
                              std::uint8_t{0}, std::uint8_t{204}}) {
        const double a = ms::gria::ca::alpha_ca(rule, 32, 33);
        EXPECT_GE(a, 0.0);
        EXPECT_LE(a, 1.0);
    }
}

TEST(GriaAlpha, LfsrDiscriminatesBetweenPolynomials) {
    // A maximal-length polynomial makes the state update a bijection, so it
    // loses (essentially) nothing; poly = 0 shifts a bit off the end every step
    // and collapses the state space. The old reversal transform could not tell
    // them apart because entropy here is a permutation-invariant histogram.
    const double good = ms::gria::lfsr::alpha_lfsr(0x1D, 256);
    const double degenerate = ms::gria::lfsr::alpha_lfsr(0, 256);
    EXPECT_LT(good, 0.1);
    EXPECT_GT(degenerate, 0.5);
    EXPECT_GT(degenerate, good);
}

TEST(CyphaNig, DensityDecaysAndIntegratesToOne) {
    ms::cypha::NIGParams p;
    p.mu = 0.0;
    p.alpha = 2.0;
    p.beta = 0.0;
    p.delta = 1.0;

    // Must DECAY away from the centre. The old form increased toward 1/(2 delta):
    // it gave f(0) = 0.0677 and f(50) = 0.4804.
    EXPECT_GT(ms::cypha::nig_pdf(0.0, p), ms::cypha::nig_pdf(5.0, p));
    EXPECT_GT(ms::cypha::nig_pdf(5.0, p), ms::cypha::nig_pdf(50.0, p));
    EXPECT_LT(ms::cypha::nig_pdf(50.0, p), 1e-20);

    double mass = 0.0;
    const double h = 0.002;
    for (double x = -60.0; x < 60.0; x += h) {
        mass += ms::cypha::nig_pdf(x, p) * h;
    }
    EXPECT_NEAR(mass, 1.0, 1e-3);
}

TEST(CyphaNig, DensityIsSkewedByBeta) {
    // beta was never read, so the density stayed symmetric however skewed the
    // parameters were.
    ms::cypha::NIGParams p;
    p.mu = 0.0;
    p.alpha = 2.0;
    p.beta = 1.0;
    p.delta = 1.0;
    EXPECT_GT(ms::cypha::nig_pdf(2.0, p), ms::cypha::nig_pdf(-2.0, p));

    p.beta = -1.0;
    EXPECT_LT(ms::cypha::nig_pdf(2.0, p), ms::cypha::nig_pdf(-2.0, p));
}

TEST(CyphaNig, OutOfDomainParametersAreRejected) {
    // NIG requires |beta| < alpha and delta > 0.
    ms::cypha::NIGParams p;
    p.mu = 0.0;
    p.alpha = 1.0;
    p.beta = 2.0;  // |beta| > alpha
    p.delta = 1.0;
    EXPECT_TRUE(std::isnan(ms::cypha::nig_pdf(0.0, p)));

    p.beta = 0.0;
    p.delta = -1.0;
    EXPECT_TRUE(std::isnan(ms::cypha::nig_pdf(0.0, p)));
}
