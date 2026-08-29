// Anchor TU for libms_bundle.so (MS_LINK_TESTS_SHARED). The archive members
// come from the static ms_* libraries via WHOLE_ARCHIVE.
namespace ms {
int ms_bundle_anchor() { return 0; }
}
