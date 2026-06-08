// Compile-only smoke: a translation unit annotated with [[ms::unsafe]] must build
// when MS_UNSAFE is defined in include/ms (see ms/error/error_types.hpp).

#include "ms/error/error_types.hpp"

#ifndef MS_UNSAFE
#error "MS_UNSAFE attribute macro not defined — skip compliance_unsafe_annotation"
#endif

[[MS_UNSAFE("compliance: unsafe annotation compile smoke")]]
static int annotated_fn() {
    return 0;
}

int main() {
    return annotated_fn();
}
