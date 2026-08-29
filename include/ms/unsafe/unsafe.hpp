// MathScript [[ms::unsafe]] attribute macro (master plan §7.4)
//
// Usage: [[MS_UNSAFE("justification")]] on declarations or statements.
// Full profile enforcement and unsafe-site audit reporting require MS_BUILD_PLUGIN
// (Clang plugin); without the plugin the attribute is accepted but not enforced.

#pragma once

#ifndef MS_UNSAFE

#if defined(__clang__) || defined(__GNUC__)
#define MS_UNSAFE(reason) ms::unsafe(reason)
#elif defined(_MSC_VER)
// MSVC warns C5030 on unknown attributes; the Clang plugin is the enforcement path.
#define MS_UNSAFE(reason)
#else
#define MS_UNSAFE(reason)
#endif

#endif // MS_UNSAFE
