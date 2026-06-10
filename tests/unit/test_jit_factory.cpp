#include <gtest/gtest.h>
#include "ms/interp/jit_backend.hpp"

using namespace ms::interp;

// ---------------------------------------------------------------------------
// Factory: Repl backend
// ---------------------------------------------------------------------------

TEST(JitFactoryTest, repl_backend_name) {
    auto backend = create_backend(Backend::Repl);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->backend_name(), "repl");
}

TEST(JitFactoryTest, repl_backend_capabilities) {
    auto backend = create_backend(Backend::Repl);
    ASSERT_NE(backend, nullptr);
    const auto caps = backend->capabilities();
    EXPECT_FALSE(caps.llvm_linked);
    EXPECT_FALSE(caps.can_compile);
    EXPECT_FALSE(caps.notes.empty());
}

TEST(JitFactoryTest, repl_backend_executes_scalar_assignment) {
    auto backend = create_backend(Backend::Repl);
    const auto r = backend->execute("x = 3.14");
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(backend->state().scalars.at("x"), 3.14, 1e-12);
}

TEST(JitFactoryTest, repl_backend_reset_clears_state) {
    auto backend = create_backend(Backend::Repl);
    ASSERT_TRUE(backend->execute("a = 99").has_value());
    EXPECT_EQ(backend->state().scalars.count("a"), 1u);
    backend->reset();
    EXPECT_EQ(backend->state().scalars.count("a"), 0u);
}

// ---------------------------------------------------------------------------
// Factory: OrcJit backend
// ---------------------------------------------------------------------------

TEST(JitFactoryTest, orc_backend_not_null) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_NE(backend, nullptr);
}

TEST(JitFactoryTest, orc_backend_name_not_empty) {
    auto backend = create_backend(Backend::OrcJit);
    EXPECT_FALSE(backend->backend_name().empty());
}

TEST(JitFactoryTest, orc_backend_executes_assignment) {
    auto backend = create_backend(Backend::OrcJit);
    const auto r = backend->execute("y = 42");
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(backend->state().scalars.at("y"), 42.0);
}

TEST(JitFactoryTest, orc_backend_reset_clears_state) {
    auto backend = create_backend(Backend::OrcJit);
    ASSERT_TRUE(backend->execute("z = 7").has_value());
    EXPECT_EQ(backend->state().scalars.count("z"), 1u);
    backend->reset();
    EXPECT_TRUE(backend->state().scalars.empty());
    EXPECT_TRUE(backend->state().matrices.empty());
}

TEST(JitFactoryTest, orc_backend_capabilities_notes_not_empty) {
    auto backend = create_backend(Backend::OrcJit);
    const auto caps = backend->capabilities();
    EXPECT_FALSE(caps.notes.empty());
}

// ---------------------------------------------------------------------------
// Default backend
// ---------------------------------------------------------------------------

TEST(JitFactoryTest, default_is_repl) {
    auto backend = create_backend();
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->backend_name(), "repl");
}
