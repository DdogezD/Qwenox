/* miniz_export.h — static-build shim for the split miniz 2.2.0 sources.
 *
 * Upstream generates this header from CMake for shared-library builds. We
 * compile miniz into qwenox-api, so every symbol stays private and the export
 * markers expand to nothing.
 */
#ifndef MINIZ_EXPORT_H
#define MINIZ_EXPORT_H

#define MINIZ_EXPORT
#define MINIZ_NO_EXPORT

#endif /* MINIZ_EXPORT_H */
