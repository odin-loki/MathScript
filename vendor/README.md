# Vendored Dependencies

This directory holds third-party sources copied into the MathScript monorepo at
pinned commits. There are no git submodules and no network access during the
CMake build.

## Current status

The directory is **empty** — no vendored libraries are checked in yet. Planned
vendored dependencies (see `mathscript-master-plan.md` §3.6) include mimalloc,
oneTBB, highway, symengine, and optionally googletest.

## Checksum policy

Every file under `vendor/` (excluding this README and `CHECKSUMS.sha256`
itself) must have a SHA-256 entry in `vendor/CHECKSUMS.sha256`.

Format (GNU `sha256sum` style, paths relative to the repository root):

```
<64-char-hex-hash>  vendor/<package>/<path/to/file>
```

When adding or updating a vendored dependency:

1. Copy the source tree into `vendor/<name>/`.
2. Record the upstream URL and commit in `vendor/VERSIONS.txt` (when that file
   is introduced).
3. Regenerate checksums, e.g. on Linux:

   ```bash
   cd /path/to/MathScript
   find vendor -type f ! -name 'CHECKSUMS.sha256' ! -name 'README.md' \
     -exec sha256sum {} \; | sed 's| vendor/| vendor/|' > vendor/CHECKSUMS.sha256
   ```

   Or append hashes manually for the files you add.
4. Run verification:

   ```bash
   ./scripts/verify_vendor.sh
   # or
   cmake --build <build-dir> --target verify_vendor
   ```

Builds with `MS_VERIFY_VENDOR=ON` (the default when `CHECKSUMS.sha256` contains
entries) run this check automatically before compiling.
