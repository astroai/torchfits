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

// The local file whose stat identity stands for `filename`, or "" when there
// is none. A URL (any "scheme://") is fetched by a CFITSIO driver and has no
// local identity. An extended-syntax name -- "f.fits[1]", or with an output
// name, "f.fits(out.fits)[1]" -- reads the base file before the first bracket
// section that has_cfitsio_extended_filename_syntax() detects.
inline std::string cfitsio_stat_path(const std::string& filename) {
    if (filename.find("://") != std::string::npos) return std::string();
    if (!has_cfitsio_extended_filename_syntax(filename)) return filename;
    const size_t last_slash = filename.find_last_of('/');
    const size_t search_start = (last_slash == std::string::npos) ? 0 : last_slash + 1;
    std::string root = filename.substr(0, filename.find('[', search_start));
    if (!root.empty() && root.back() == ')') {
        const size_t open = root.rfind('(');
        if (open != std::string::npos && open >= search_start) root.resize(open);
    }
    return root;
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

                // Check for sh:// prefix (case-insensitive; CFITSIO may fold schemes).
                if (start_idx + 5 <= filename.size()) {
                    bool is_sh = true;
                    const char sh[] = {'s', 'h', ':', '/', '/'};
                    for (size_t i = 0; i < 5; ++i) {
                        char c = filename[start_idx + i];
                        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
                        if (c != sh[i]) { is_sh = false; break; }
                    }
                    if (is_sh) {
                        throw std::runtime_error("Security Error: Filenames starting with 'sh://' are not allowed to prevent command execution.");
                    }
                }
            }

            if (filename[last] == '|') {
                throw std::runtime_error("Security Error: Filenames ending with '|' are not allowed to prevent command execution.");
            }
        }
    }
}

} // namespace torchfits
