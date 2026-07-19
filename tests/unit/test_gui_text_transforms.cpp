#include <gtest/gtest.h>

#include "gui/TextTransforms.hpp"

using ms::gui::camel_case_text;
using ms::gui::screaming_snake_case_text;
using ms::gui::trim_leading_whitespace_line;

TEST(GuiTextTransformsTest, camel_case_selection) {
    EXPECT_EQ(camel_case_text("hello world"), "helloWorld");
    EXPECT_EQ(camel_case_text("hello_world"), "helloWorld");
    EXPECT_EQ(camel_case_text("hello-world"), "helloWorld");
    EXPECT_EQ(camel_case_text("HelloWorld"), "helloWorld");
    EXPECT_EQ(camel_case_text("HELLO_WORLD"), "helloWorld");
    EXPECT_EQ(camel_case_text("foo bar-baz"), "fooBarBaz");
}

TEST(GuiTextTransformsTest, screaming_snake_case_selection) {
    EXPECT_EQ(screaming_snake_case_text("hello world"), "HELLO_WORLD");
    EXPECT_EQ(screaming_snake_case_text("helloWorld"), "HELLO_WORLD");
    EXPECT_EQ(screaming_snake_case_text("foo-bar baz"), "FOO_BAR_BAZ");
    EXPECT_EQ(screaming_snake_case_text("already_snake"), "ALREADY_SNAKE");
}

TEST(GuiTextTransformsTest, trim_leading_whitespace_line) {
    EXPECT_EQ(trim_leading_whitespace_line("  hello"), "hello");
    EXPECT_EQ(trim_leading_whitespace_line("\t\t hello"), "hello");
    EXPECT_EQ(trim_leading_whitespace_line("hello"), "hello");
    EXPECT_EQ(trim_leading_whitespace_line("   "), "");
    EXPECT_EQ(trim_leading_whitespace_line(""), "");
}
