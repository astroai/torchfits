#pragma once

#include <string>
#include <stdexcept>

namespace torchfits {

// Detect whether `filename` carries a CFITSIO extended filename syntax
// section, e.g. "file.fits[1]" or "file.fits[1:10,1:10]". CFITSIO only
// parses bracket syntax when it terminates the final path component, so a
// naive `filename.find('[') != npos` check produces false positives for
// paths with a literal '[' in a directory component, e.g.
// "/home/user/[data]/file.fits". Require the string to end in ']' *and*
// contain a '[' after the last '/' before treating it as an extended
// filename section.
inline bool has_cfitsio_extended_filename_syntax(const std::string& filename) {
    if (filename.empty() || filename.back() != ']') return false;
    const size_t last_slash = filename.find_last_of('/');
    const size_t search_start = (last_slash == std::string::npos) ? 0 : last_slash + 1;
    return filename.find('[', search_start) != std::string::npos;
}

inline void check_fits_filename_security(const std::string& filename) {
    if (!filename.empty()) {
        // Strip all standard whitespace characters since cfitsio ignores them,
        // and they could be used to bypass prefix/suffix checks.
        size_t first = filename.find_first_not_of(" \t\n\r\v\f");
        size_t last = filename.find_last_not_of(" \t\n\r\v\f");

        if (first != std::string::npos) {
            size_t start_idx = first;

            // Allow multiple leading '!' because CFITSIO uses them for forced overwrite,
            // and skip whitespace between '!' if any.
            while (start_idx != std::string::npos && filename[start_idx] == '!') {
                start_idx = filename.find_first_not_of(" \t\n\r\v\f", start_idx + 1);
            }

            if (start_idx != std::string::npos) {
                if (filename[start_idx] == '|') {
                    throw std::runtime_error("Security Error: Filenames starting with '|' are not allowed to prevent command execution.");
                }

                // Check for sh:// prefix
                if (filename.compare(start_idx, 5, "sh://") == 0) {
                    throw std::runtime_error("Security Error: Filenames starting with 'sh://' are not allowed to prevent command execution.");
                }
            }

            if (filename[last] == '|') {
                throw std::runtime_error("Security Error: Filenames ending with '|' are not allowed to prevent command execution.");
            }
        }
    }
}

} // namespace torchfits
