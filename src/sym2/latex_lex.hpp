// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
#pragma once

/// @file
/// @brief The TeX token layer under `parse_latex`.
///
/// This is a token lexer rather than a regular-expression sweep, and the difference is
/// not stylistic. `\varGamma` is one control word, not `\var` followed by `Gamma`;
/// `\infty x` is `\infty` and `x`, not `\inftyx`; `\int` is a control word while `\,` is
/// a control symbol whose one character happens to be a comma. A pattern that looks for
/// `\\[a-z]+` somewhere in the input finds all three inside `\mathrm{a\infty b}` too,
/// where they are the bytes of a name. Maximal munch over a backslash, once, is what
/// makes each of those exactly one answer.
///
/// Three decisions here that the parser above depends on:
///
///   - **Whitespace is dropped, `\,` is not.** Math mode discards source spacing, so
///     `2 x` and `2x` are the same product. `\,` survives because
///     `notation_latex.cpp:346` and `:444` both emit it as the only mark between two
///     things that would otherwise fuse, and dropping it turns `2\,10^{n}` into
///     `210^{n}` -- a different number. The spacing commands of `docs/LATEX_SUBSET.md`
///     H5 (`\;`, `\:`, `\!`, `\quad`, `\qquad`, `\thinspace`, `\ `, `~`) are dropped
///     here because the printer never emits one, so they can carry no meaning that a
///     later pass would need.
///   - **Tokens keep their byte offset.** Two digits are one numeral only when they are
///     adjacent in the source (`23`) and two factors when they are not (`2 3`,
///     `2\,3`), so the parser has to ask about bytes even though it reads tokens.
///     `\mathrm{a b}` needs the same thing for the opposite reason: the space inside a
///     name is part of the name, and the parser recovers it by slicing the original
///     text between the brace tokens rather than by re-joining what it read.
///   - **The column counts UTF-8 scalar values, not bytes.** A symbol name may carry raw
///     non-ASCII -- `escape` passes every byte above 0x7F through untouched
///     (notation_latex.cpp:70) -- and a byte column would point into the middle of a
///     character, which is worse than no column at all.
///
/// Nothing here can fail. A malformed byte sequence and a backslash at end of input are
/// both tokens; deciding that they are not part of the subset is the parser's job,
/// because only the parser knows what it was expecting and can say so.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ms::sym2::detail::latex {

enum class TokKind : std::uint8_t {
    End,           ///< Past the last token. Carries the position of the end of input.
    Char,          ///< One UTF-8 scalar value that was not preceded by a backslash.
    ControlWord,   ///< `\` and one or more letters. `text` is the letters.
    ControlSymbol, ///< `\` and one non-letter. `text` is that one character.
};

/// One token, pointing into the text it was lexed from.
///
/// `text` is a view rather than a string because every token in a document would
/// otherwise be an allocation, and the text outlives the parse by construction: it is
/// the `string_view` the caller passed to `parse_latex`.
struct Token {
    TokKind kind = TokKind::End;
    std::string_view text{};
    /// The scalar value, for a `Char`. Zero otherwise. The parser reads this rather
    /// than the bytes to recognise the Unicode spellings H10 allows -- U+2212 for a
    /// minus sign, U+221E for `\infty`, a literal Greek letter for its control word.
    char32_t code = 0;
    std::size_t offset = 0; ///< Byte offset of the token's first byte.
    std::size_t line = 1;   ///< 1-based.
    std::size_t col = 1;    ///< 1-based, in UTF-8 scalar values.
};

inline bool is_char(const Token& token, char expected) {
    return token.kind == TokKind::Char && token.text.size() == 1 &&
           token.text.front() == expected;
}

inline bool is_word(const Token& token, std::string_view name) {
    return token.kind == TokKind::ControlWord && token.text == name;
}

inline bool is_symbol(const Token& token, char expected) {
    return token.kind == TokKind::ControlSymbol && token.text.size() == 1 &&
           token.text.front() == expected;
}

inline bool is_ascii_letter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

inline bool is_ascii_digit(char c) { return c >= '0' && c <= '9'; }

inline bool is_digit_token(const Token& token) {
    return token.kind == TokKind::Char && token.text.size() == 1 &&
           is_ascii_digit(token.text.front());
}

inline bool is_letter_token(const Token& token) {
    return token.kind == TokKind::Char && token.text.size() == 1 &&
           is_ascii_letter(token.text.front());
}

/// Two tokens with no byte between them. A numeral's digits have to be adjacent --
/// `2 3` and `2\,3` are a product and `23` is a number -- and so do the pieces of
/// `1{,}5` and of `1e20`.
inline bool adjacent(const Token& left, const Token& right) {
    return left.offset + left.text.size() == right.offset;
}

/// How a token is named in a diagnostic. The text is quoted verbatim and never
/// normalised: a message about `\varGamma` has to say `\varGamma`, because saying
/// `\Gamma` would be reporting an input the author did not write.
inline std::string describe(const Token& token) {
    switch (token.kind) {
    case TokKind::End:
        return "end of input";
    case TokKind::Char:
        return "'" + std::string(token.text) + "'";
    case TokKind::ControlWord:
    case TokKind::ControlSymbol:
        return "'\\" + std::string(token.text) + "'";
    }
    return "end of input";
}

/// The number of bytes in the UTF-8 scalar value starting at `at`.
///
/// A lead byte that promises more continuation bytes than follow it is one byte, so the
/// cursor always advances and a truncated file cannot hang the lexer. The alternative,
/// trusting the lead byte, reads past the end of the buffer.
inline std::size_t scalar_length(std::string_view text, std::size_t at) {
    const auto lead = static_cast<unsigned char>(text[at]);
    std::size_t want = 1;
    if ((lead & 0xE0U) == 0xC0U) {
        want = 2;
    } else if ((lead & 0xF0U) == 0xE0U) {
        want = 3;
    } else if ((lead & 0xF8U) == 0xF0U) {
        want = 4;
    }
    for (std::size_t i = 1; i < want; ++i) {
        if (at + i >= text.size() ||
            (static_cast<unsigned char>(text[at + i]) & 0xC0U) != 0x80U) {
            return 1;
        }
    }
    return want;
}

inline char32_t decode_scalar(std::string_view text, std::size_t at, std::size_t length) {
    const auto lead = static_cast<unsigned char>(text[at]);
    if (length == 1) {
        return static_cast<char32_t>(lead);
    }
    constexpr unsigned char kLeadMask[5] = {0U, 0U, 0x1FU, 0x0FU, 0x07U};
    auto value = static_cast<char32_t>(lead & kLeadMask[length]);
    for (std::size_t i = 1; i < length; ++i) {
        value = (value << 6) |
                static_cast<char32_t>(static_cast<unsigned char>(text[at + i]) & 0x3FU);
    }
    return value;
}

/// The H5 spacing control words. They are discarded rather than tokenised: the printer
/// emits none of them, so one can only have come from a human, and to a human they mean
/// exactly the space the lexer is already throwing away.
inline bool is_spacing_word(std::string_view name) {
    return name == "quad" || name == "qquad" || name == "thinspace";
}

/// Every token in `text`, ending with a `TokKind::End` that carries the end position.
inline std::vector<Token> tokenize(std::string_view text) {
    std::vector<Token> out;
    std::size_t at = 0;
    std::size_t line = 1;
    std::size_t col = 1;
    const auto bump = [&](std::size_t length) {
        if (length == 1 && text[at] == '\n') {
            ++line;
            col = 1;
        } else {
            ++col;
        }
        at += length;
    };
    while (at < text.size()) {
        const char c = text[at];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '~') {
            bump(1);
            continue;
        }
        if (c == '%') {
            // H10. The newline itself is left for the whitespace arm above, which is
            // where it has to be counted.
            while (at < text.size() && text[at] != '\n') {
                bump(scalar_length(text, at));
            }
            continue;
        }
        Token token;
        token.offset = at;
        token.line = line;
        token.col = col;
        if (c != '\\') {
            const std::size_t length = scalar_length(text, at);
            token.kind = TokKind::Char;
            token.text = text.substr(at, length);
            token.code = decode_scalar(text, at, length);
            bump(length);
            out.push_back(token);
            continue;
        }
        bump(1);
        if (at < text.size() && is_ascii_letter(text[at])) {
            const std::size_t name_at = at;
            while (at < text.size() && is_ascii_letter(text[at])) {
                bump(1);
            }
            token.kind = TokKind::ControlWord;
            token.text = text.substr(name_at, at - name_at);
            // TeX absorbs the spaces after a control *word* into the word, which is why
            // `\alpha x` is two tokens and `\alpha{}x` is needed to butt them together.
            // Absorbing them here rather than leaving them to the whitespace arm costs
            // nothing and keeps the token's extent the one TeX would report.
            while (at < text.size() && (text[at] == ' ' || text[at] == '\t')) {
                bump(1);
            }
            if (is_spacing_word(token.text)) {
                continue;
            }
        } else if (at < text.size()) {
            const std::size_t length = scalar_length(text, at);
            token.kind = TokKind::ControlSymbol;
            token.text = text.substr(at, length);
            bump(length);
            const char symbol = token.text.size() == 1 ? token.text.front() : '\0';
            // `\,` is not in this list, and that is the whole point of §1.7: it is the
            // differential separator and the factor separator, so it carries meaning
            // where its neighbours carry only width.
            if (symbol == ';' || symbol == ':' || symbol == '!' || symbol == ' ') {
                continue;
            }
        } else {
            // A backslash with nothing after it. It is a control sequence with an empty
            // name rather than a lexer error, so that the parser can quote it and say
            // which subset it is not in.
            token.kind = TokKind::ControlSymbol;
            token.text = text.substr(at, 0);
        }
        out.push_back(token);
    }
    Token end;
    end.kind = TokKind::End;
    end.text = text.substr(text.size(), 0);
    end.offset = text.size();
    end.line = line;
    end.col = col;
    out.push_back(end);
    return out;
}

} // namespace ms::sym2::detail::latex
