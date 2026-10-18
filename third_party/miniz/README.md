# miniz 2.2.0

Vendored compression library used by `src/api/reqlog.cpp` to write one ZIP per
logged API request.

- Upstream: https://github.com/richgel999/miniz (tag `2.2.0`)
- Files: the split 2.2.0 sources (`miniz.c`, `miniz.h`, `miniz_common.h`,
  `miniz_tdef.*`, `miniz_tinfl.*`, `miniz_zip.*`) plus a local
  `miniz_export.h` shim for static linking.
- License: public domain / Unlicense (see the header of `miniz.c`).
- Build: the four `.c` files are compiled as C (`-x c`) and linked into the
  API. No system zlib is required.
