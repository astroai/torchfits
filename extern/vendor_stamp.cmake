# Freshness stamp for the vendored CFITSIO tree.
#
# extern/vendor.sh ends a pinned run by writing extern/cfitsio/.torchfits-vendor-stamp:
# one "<sha256>  <label>" line for extern/VERSIONS.txt, one for extern/vendor.sh,
# and one per extern/patches/<tag>-*.patch it applied, in byte order of the file
# name. This file computes the same lines from the checked-in inputs so the
# build can tell a tree vendored from different inputs (an older pin, a patch
# added or edited since) from a current one. Without it a stale extern/cfitsio
# was reused as-is and silently built without newer (security) patches.
#
# Keep the definition identical to vendor_stamp() in extern/vendor.sh;
# tests/test_vendor_stamp.py runs both and compares them.
#
# Script mode (used by that test and handy by hand):
#   cmake -DTORCHFITS_REPO_ROOT=<repo> [-DOUT=<file>] -P extern/vendor_stamp.cmake
# writes the expected stamp to OUT (or prints it) and the status of the
# vendored tree ("fresh", "missing" or "stale") to OUT.status.

set(TORCHFITS_VENDOR_STAMP_NAME ".torchfits-vendor-stamp")

# Expected stamp text for <root>, and the input files it hashes (for
# CMAKE_CONFIGURE_DEPENDS).
function(torchfits_vendor_stamp_expected root out_text out_inputs)
    set(_extern "${root}/extern")
    set(_versions "${_extern}/VERSIONS.txt")
    # Same tag resolution as vendor.sh's resolve_cfitsio_version for the
    # versions-file form: the cfitsio_tag= line, else the first line that is
    # not cfitsio_repo=.
    file(STRINGS "${_versions}" _lines)
    set(_tag "")
    foreach(_line IN LISTS _lines)
        if(_line MATCHES "^cfitsio_tag=(.*)$")
            set(_tag "${CMAKE_MATCH_1}")
            break()
        endif()
    endforeach()
    if(_tag STREQUAL "")
        foreach(_line IN LISTS _lines)
            if(NOT _line MATCHES "^cfitsio_repo=")
                string(REGEX REPLACE "[ \t\r\n]" "" _tag "${_line}")
                break()
            endif()
        endforeach()
    endif()

    file(SHA256 "${_versions}" _hash)
    set(_text "${_hash}  VERSIONS.txt\n")
    file(SHA256 "${_extern}/vendor.sh" _hash)
    string(APPEND _text "${_hash}  vendor.sh\n")
    set(_patches "")
    if(NOT _tag STREQUAL "")
        file(GLOB _patches LIST_DIRECTORIES false "${_extern}/patches/${_tag}-*.patch")
        list(SORT _patches COMPARE STRING)
    endif()
    foreach(_patch IN LISTS _patches)
        file(SHA256 "${_patch}" _hash)
        get_filename_component(_name "${_patch}" NAME)
        string(APPEND _text "${_hash}  patches/${_name}\n")
    endforeach()

    set(${out_text} "${_text}" PARENT_SCOPE)
    set(${out_inputs} "${_versions};${_extern}/vendor.sh;${_patches}" PARENT_SCOPE)
endfunction()

# <out_status>: "fresh" (stamp matches), "missing" (no stamp) or "stale".
# <out_detail>: the inputs that differ, for the error message.
function(torchfits_vendor_stamp_check root out_status out_detail)
    torchfits_vendor_stamp_expected("${root}" _expected _inputs)
    set(_stamp "${root}/extern/cfitsio/${TORCHFITS_VENDOR_STAMP_NAME}")
    if(NOT EXISTS "${_stamp}")
        set(${out_status} "missing" PARENT_SCOPE)
        set(${out_detail} "no ${TORCHFITS_VENDOR_STAMP_NAME}: vendored by an older vendor.sh or by hand" PARENT_SCOPE)
        return()
    endif()
    file(READ "${_stamp}" _actual)
    if(_actual STREQUAL _expected)
        set(${out_status} "fresh" PARENT_SCOPE)
        set(${out_detail} "" PARENT_SCOPE)
        return()
    endif()
    string(REPLACE "\n" ";" _want "${_expected}")
    string(REPLACE "\n" ";" _have "${_actual}")
    set(_changed "")
    foreach(_line IN LISTS _want _have)
        if(_line STREQUAL "")
            continue()
        endif()
        if(NOT _line IN_LIST _want OR NOT _line IN_LIST _have)
            string(REGEX REPLACE "^[0-9a-f]*  " "" _label "${_line}")
            list(APPEND _changed "${_label}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _changed)
    list(JOIN _changed ", " _changed)
    set(${out_status} "stale" PARENT_SCOPE)
    set(${out_detail} "changed since it was vendored: ${_changed}" PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT TORCHFITS_REPO_ROOT)
        get_filename_component(TORCHFITS_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    endif()
    torchfits_vendor_stamp_expected("${TORCHFITS_REPO_ROOT}" _text _inputs)
    torchfits_vendor_stamp_check("${TORCHFITS_REPO_ROOT}" _status _detail)
    if(OUT)
        file(WRITE "${OUT}" "${_text}")
        file(WRITE "${OUT}.status" "${_status}\n")
    else()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo_append "${_text}")
        message(STATUS "vendored tree: ${_status} ${_detail}")
    endif()
endif()
