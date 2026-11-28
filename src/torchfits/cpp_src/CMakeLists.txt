cmake_minimum_required(VERSION 3.21)
project(torchfits_cpp)

# The native sources include no torch/ATen header (tensors cross the Python
# boundary as DLPack capsules), so the language level is the project's own.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Detect sanitizer instrumentation requested through any configured C/CXX flag
# set. The Python extension already receives CMAKE_CXX_FLAGS; vendored CFITSIO
# is C and must receive the same instrumentation explicitly.
set(_TORCHFITS_SANITIZER_COMPILE_OPTIONS "")
set(_TORCHFITS_SANITIZER_FLAG_VARS
    CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
if(CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _TORCHFITS_BUILD_TYPE_UPPER)
    list(APPEND _TORCHFITS_SANITIZER_FLAG_VARS
         "CMAKE_C_FLAGS_${_TORCHFITS_BUILD_TYPE_UPPER}"
         "CMAKE_CXX_FLAGS_${_TORCHFITS_BUILD_TYPE_UPPER}")
endif()
foreach(_flag_var IN LISTS _TORCHFITS_SANITIZER_FLAG_VARS)
    separate_arguments(_configured_flags NATIVE_COMMAND "${${_flag_var}}")
    foreach(_configured_flag IN LISTS _configured_flags)
        if(_configured_flag MATCHES "^-fsanitize=")
            list(APPEND _TORCHFITS_SANITIZER_COMPILE_OPTIONS "${_configured_flag}")
        endif()
    endforeach()
endforeach()
list(REMOVE_DUPLICATES _TORCHFITS_SANITIZER_COMPILE_OPTIONS)
if(_TORCHFITS_SANITIZER_COMPILE_OPTIONS)
    message(STATUS
        "Applying sanitizer instrumentation to vendored CFITSIO: "
        "${_TORCHFITS_SANITIZER_COMPILE_OPTIONS}")
endif()

# ---------------------------------------------------------------------------
# Source coverage — opt-in via CMake option.
#   CMAKE_ARGS="-DTORCHFITS_COVERAGE=ON" pip install --no-build-isolation .
# Instruments the extension, the torch-free core and (because the flags are
# directory-scoped, and vendored CFITSIO is added as a subdirectory below) the
# vendored C sources too. -O0 keeps line counts meaningful: gcov attributes
# every statement the optimiser kept, and inlining merges lines silently.
# ---------------------------------------------------------------------------
option(TORCHFITS_COVERAGE "Instrument C/C++ sources and link the gcov runtime" OFF)
set(_TORCHFITS_COVERAGE_COMPILE_OPTIONS "")
set(_TORCHFITS_COVERAGE_LINK_OPTIONS "")
if(TORCHFITS_COVERAGE)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR
            "TORCHFITS_COVERAGE needs GCC or Clang (gcov/llvm-cov); "
            "found ${CMAKE_C_COMPILER_ID}")
    endif()
    # Applied per target below, not directory-wide: the bundled nanobind-static
    # is third-party code and would otherwise flood every gcov report.
    set(_TORCHFITS_COVERAGE_COMPILE_OPTIONS --coverage -O0 -g)
    set(_TORCHFITS_COVERAGE_LINK_OPTIONS --coverage)
    message(STATUS
        "Coverage: C/C++ instrumented with --coverage (optimisation pinned to -O0)")
endif()

# Enable IPO/LTO when available (portable opt-in)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
if(IPO_SUPPORTED AND NOT TORCHFITS_COVERAGE)
    # LTO runs after instrumentation and folds away counters; gcov data from an
    # LTO build under-reports badly, so coverage always builds without it.
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
elseif(IPO_SUPPORTED)
    message(STATUS "IPO/LTO disabled for coverage instrumentation")
else()
    message(STATUS "IPO/LTO not supported: ${IPO_ERROR}")
endif()

# ---------------------------------------------------------------------------
# Profile-Guided Optimization (PGO) — opt-in via CMake option.
# Two-step build:
#   1. cmake -DTORCHFITS_PGO=generate ... && make  (run representative workload)
#   2. cmake -DTORCHFITS_PGO=use ...    && make  (rebuild with profile data)
# ---------------------------------------------------------------------------
set(TORCHFITS_PGO "off" CACHE STRING "PGO mode: off, generate, or use")
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if(TORCHFITS_PGO STREQUAL "generate")
        add_compile_options(-fprofile-generate)
        message(STATUS "PGO: generating profile data")
    elseif(TORCHFITS_PGO STREQUAL "use")
        add_compile_options(-fprofile-use -fprofile-correction)
        message(STATUS "PGO: using profile data")
    endif()
endif()

# ---------------------------------------------------------------------------
# Finite-math-only — opt-in via CMake option.
# Assumes no NaN/Inf, enabling more aggressive vectorization and FMA fusion.
# UNSAFE for code that relies on NaN sentinel semantics (e.g., FITS null values).
# Only enable for benchmarks or when you know your codepath is NaN-free.
# ---------------------------------------------------------------------------
option(TORCHFITS_FINITE_MATH_ONLY "Assume no NaN/Inf (UNSAFE with null sentinels)" OFF)
if(TORCHFITS_FINITE_MATH_ONLY AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(-ffinite-math-only)
    message(STATUS "WARNING: -ffinite-math-only enabled — NaN sentinel codepaths will break")
endif()

# Enable SIMD optimizations globally (propagates to cfitsio)
# Note: For wheels, avoid -march=native. Using reasonable baselines.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # SSSE3 (pshufb) backs the SIMD byte-swap of big-endian image and table
    # data (internal_utils.h). Every x86_64 build therefore requires an SSSE3
    # CPU, which is also what modern PyTorch wheels assume.
    add_compile_options(-mssse3)
    message(STATUS "Enabled x86_64 optimizations (SSSE3)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # ARM64 implies NEON, -O3 handles vectorization well.
    # On Apple Silicon, -mcpu=native ensures correct scheduling for M-series.
    # This assumes building on M1+ for M1+ target (standard for macOS wheels).
    if(APPLE)
       message(STATUS "Enabled Native Apple Silicon optimizations")
    else()
       message(STATUS "Enabled ARM64 optimizations")
    endif()
endif()


# Add environment prefix to search path (important for pixi/conda)
if(DEFINED ENV{PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{PREFIX}")
    message(STATUS "Added environment prefix to CMAKE_PREFIX_PATH: $ENV{PREFIX}")
endif()

# Find required packages
# In pixi/conda packaging builds, ensure Python comes from the host prefix.
if(DEFINED ENV{PREFIX} AND EXISTS "$ENV{PREFIX}/bin/python")
    set(Python_ROOT_DIR "$ENV{PREFIX}" CACHE PATH "Python root directory" FORCE)
    set(Python_EXECUTABLE "$ENV{PREFIX}/bin/python" CACHE FILEPATH "Python executable" FORCE)
endif()
set(Python_FIND_STRATEGY LOCATION)
# SKBUILD_SABI_COMPONENT is Development.SABIModule when wheel.py-api asks for
# the stable ABI (CPython >= 3.12): the extensions are then abi3 and one wheel
# per platform serves every later CPython.
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# Find nanobind
# Detect the installed nanobind package and import it into CMake
# For pixi builds, nanobind is in the build environment, not host environment
if(DEFINED ENV{BUILD_PREFIX})
    # In pixi build environment, use build prefix python for nanobind detection
    set(NANOBIND_PYTHON_EXECUTABLE "$ENV{BUILD_PREFIX}/bin/python")
    message(STATUS "Using build environment Python for nanobind: ${NANOBIND_PYTHON_EXECUTABLE}")
else()
    # Normal development environment
    set(NANOBIND_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
endif()

execute_process(
  COMMAND "${NANOBIND_PYTHON_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  ERROR_QUIET
)

if(nanobind_ROOT)
    message(STATUS "Found nanobind cmake directory: ${nanobind_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
else()
    # Fallback: try to find nanobind via Python import in build environment
    execute_process(
        COMMAND "${NANOBIND_PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
        ERROR_QUIET
    )
    if(nanobind_ROOT)
        message(STATUS "Found nanobind via Python import: ${nanobind_ROOT}")
        list(APPEND CMAKE_PREFIX_PATH ${nanobind_ROOT})
    endif()
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

message(DEBUG "torchfits: Python_EXECUTABLE=${Python_EXECUTABLE}")
message(DEBUG "torchfits: Python_VERSION=${Python_VERSION}")
message(DEBUG "torchfits: NANOBIND_PYTHON_EXECUTABLE=${NANOBIND_PYTHON_EXECUTABLE}")
message(DEBUG "torchfits: nanobind_DIR=${nanobind_DIR}")
message(DEBUG "torchfits: nanobind_ROOT=${nanobind_ROOT}")

# Identity of the torch-free core, stamped into the core, into _core, and into
# _C. The three must agree at runtime: a wheel built by a half-finished build
# can leave a stale libtorchfits_core beside a fresh _C, and the failure mode
# for that is a segfault deep inside CFITSIO rather than an import error.
string(TIMESTAMP _TORCHFITS_BUILD_STAMP "%Y-%m-%dT%H:%M:%SZ" UTC)
string(REPLACE ";" ":" _TORCHFITS_BUILD_DIR "${CMAKE_BINARY_DIR}")
set(TORCHFITS_CORE_BUILD_ID
    "torchfits-core/1 build=${_TORCHFITS_BUILD_STAMP} type=${CMAKE_BUILD_TYPE} dir=${_TORCHFITS_BUILD_DIR}")
message(STATUS "torchfits: core build id = ${TORCHFITS_CORE_BUILD_ID}")

# Find ZLIB (Required for compressed FITS support)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
set(_TORCHFITS_ZLIB_SELECTED "")
if(DEFINED ZLIB_LIBRARY_RELEASE AND ZLIB_LIBRARY_RELEASE)
    set(_TORCHFITS_ZLIB_SELECTED "${ZLIB_LIBRARY_RELEASE}")
elseif(DEFINED ZLIB_LIBRARY AND ZLIB_LIBRARY)
    set(_TORCHFITS_ZLIB_SELECTED "${ZLIB_LIBRARY}")
endif()

if(_TORCHFITS_ZLIB_SELECTED AND NOT EXISTS "${_TORCHFITS_ZLIB_SELECTED}")
    get_filename_component(_TORCHFITS_ZLIB_DIR "${_TORCHFITS_ZLIB_SELECTED}" DIRECTORY)
    set(_TORCHFITS_ZLIB_CANDIDATES
        "${_TORCHFITS_ZLIB_DIR}/libz.1.dylib"
        "${_TORCHFITS_ZLIB_DIR}/libzlib.dylib"
    )
    foreach(_zlib_candidate IN LISTS _TORCHFITS_ZLIB_CANDIDATES)
        if(EXISTS "${_zlib_candidate}")
            message(WARNING "ZLIB library path '${_TORCHFITS_ZLIB_SELECTED}' does not exist; using '${_zlib_candidate}'")
            if(NOT EXISTS "${_TORCHFITS_ZLIB_SELECTED}")
                file(CREATE_LINK "${_zlib_candidate}" "${_TORCHFITS_ZLIB_SELECTED}" SYMBOLIC RESULT _TORCHFITS_ZLIB_LINK_RESULT)
                if(_TORCHFITS_ZLIB_LINK_RESULT)
                    message(WARNING "Failed to create compatibility symlink '${_TORCHFITS_ZLIB_SELECTED}' -> '${_zlib_candidate}': ${_TORCHFITS_ZLIB_LINK_RESULT}")
                else()
                    message(STATUS "Created compatibility symlink '${_TORCHFITS_ZLIB_SELECTED}' -> '${_zlib_candidate}'")
                endif()
            endif()
            set(ZLIB_LIBRARY_RELEASE "${_zlib_candidate}" CACHE FILEPATH "Path to a library." FORCE)
            set(ZLIB_LIBRARY "${_zlib_candidate}" CACHE FILEPATH "Path to a library." FORCE)
            if(TARGET ZLIB::ZLIB)
                set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION "${_zlib_candidate}")
                set_property(TARGET ZLIB::ZLIB PROPERTY IMPORTED_LOCATION_RELEASE "${_zlib_candidate}")
            endif()
            break()
        endif()
    endforeach()
endif()

if(ZLIB_FOUND)
    message(STATUS "Found ZLIB: ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLIB not found - required for compressed FITS support")
endif()

get_filename_component(TORCHFITS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(CFITSIO_SOURCE_DIR "${TORCHFITS_REPO_ROOT}/extern/cfitsio")
set(TORCHFITS_VENDOR_SCRIPT "${TORCHFITS_REPO_ROOT}/extern/vendor.sh")
include("${TORCHFITS_REPO_ROOT}/extern/vendor_stamp.cmake")
# Git checkouts may fetch CFITSIO on first configure. PyPI sdists (and any
# tree without .git) must not: pip installs are wheels-only.
if(EXISTS "${TORCHFITS_REPO_ROOT}/.git")
    set(_torchfits_auto_vendor_default ON)
else()
    set(_torchfits_auto_vendor_default OFF)
endif()
option(TORCHFITS_AUTO_VENDOR_DEPS "Auto-run extern/vendor.sh when vendored deps are missing or stale" "${_torchfits_auto_vendor_default}")
set(TORCHFITS_VENDOR_RAN FALSE)

macro(torchfits_run_vendor_once)
    if(TORCHFITS_AUTO_VENDOR_DEPS AND NOT TORCHFITS_VENDOR_RAN)
        if(EXISTS "${TORCHFITS_VENDOR_SCRIPT}")
            message(STATUS "Running ${TORCHFITS_VENDOR_SCRIPT} (pinned via extern/VERSIONS.txt)")
            execute_process(
                COMMAND bash "${TORCHFITS_VENDOR_SCRIPT}"
                        --cfitsio-version "${TORCHFITS_REPO_ROOT}/extern/VERSIONS.txt"
                WORKING_DIRECTORY "${TORCHFITS_REPO_ROOT}"
                RESULT_VARIABLE TORCHFITS_VENDOR_RESULT
            )
            if(NOT TORCHFITS_VENDOR_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to run ${TORCHFITS_VENDOR_SCRIPT} (exit ${TORCHFITS_VENDOR_RESULT})")
            endif()
            set(TORCHFITS_VENDOR_RAN TRUE)
        endif()
    endif()
endmacro()

# Find CFITSIO
option(TORCHFITS_USE_VENDORED_CFITSIO "Prefer vendored CFITSIO over environment" ON)
set(TORCHFITS_NIOBUF "" CACHE STRING "Override CFITSIO NIOBUF (default 40)")
set(TORCHFITS_MINDIRECT "" CACHE STRING "Override CFITSIO MINDIRECT (default 8640)")

if(TORCHFITS_USE_VENDORED_CFITSIO)
    unset(CFITSIO_FOUND CACHE)
    unset(CFITSIO_INCLUDE_DIR CACHE)
    unset(CFITSIO_LIBRARY CACHE)
    unset(CFITSIO_INCLUDE_DIRS CACHE)
    unset(CFITSIO_LIBRARIES CACHE)
    unset(CFITSIO_LIBRARY_DIRS CACHE)
    set(CFITSIO_FOUND FALSE)
endif()

if(NOT TORCHFITS_USE_VENDORED_CFITSIO)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(CFITSIO QUIET cfitsio)
    endif()

    if(NOT CFITSIO_FOUND)
        find_path(CFITSIO_INCLUDE_DIR fitsio.h)
        find_library(CFITSIO_LIBRARY cfitsio)
        if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
            set(CFITSIO_FOUND TRUE)
        endif()
    endif()
endif()

if(CFITSIO_FOUND)
    message(STATUS "Found environment CFITSIO: ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_LIBRARIES}")
    # An environment CFITSIO does not say whether it links libcurl; assume it
    # does and let its open error (now surfaced) explain an https failure.
    set(_TORCHFITS_CFITSIO_HAS_CURL ON)
else()
    # Vendor CFITSIO as fallback
    if(NOT EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
        torchfits_run_vendor_once()
    endif()

    if(EXISTS "${CFITSIO_SOURCE_DIR}/CMakeLists.txt")
        # An existing tree is only reused when it was vendored from the
        # checked-in inputs (extern/vendor_stamp.cmake). A tree vendored before
        # a patch was added or edited otherwise built silently without it --
        # including the http-* security patches.
        torchfits_vendor_stamp_check("${TORCHFITS_REPO_ROOT}" _vendor_stamp_status _vendor_stamp_detail)
        if(NOT _vendor_stamp_status STREQUAL "fresh" AND TORCHFITS_AUTO_VENDOR_DEPS AND NOT TORCHFITS_VENDOR_RAN)
            message(STATUS "Vendored CFITSIO is ${_vendor_stamp_status} (${_vendor_stamp_detail}); re-vendoring")
            torchfits_run_vendor_once()
            torchfits_vendor_stamp_check("${TORCHFITS_REPO_ROOT}" _vendor_stamp_status _vendor_stamp_detail)
        endif()
        if(NOT _vendor_stamp_status STREQUAL "fresh")
            message(FATAL_ERROR
                "${CFITSIO_SOURCE_DIR} does not match extern/VERSIONS.txt, "
                "extern/vendor.sh and extern/patches (${_vendor_stamp_detail}), so it "
                "may lack patches the build relies on. Re-vendor it: "
                "./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt")
        endif()
        # Re-run configure (and so this check) when any input changes, also on
        # incremental and editable rebuilds; the patches directory catches a
        # patch being added or removed.
        torchfits_vendor_stamp_expected("${TORCHFITS_REPO_ROOT}" _vendor_stamp_text _vendor_stamp_inputs)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${_vendor_stamp_inputs}
            "${TORCHFITS_REPO_ROOT}/extern/patches"
            "${CFITSIO_SOURCE_DIR}/${TORCHFITS_VENDOR_STAMP_NAME}")
        message(STATUS "Building vendored CFITSIO from ${CFITSIO_SOURCE_DIR}")

        # CFITSIO Build Options
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
        set(TESTS OFF CACHE BOOL "Disable tests" FORCE)
        set(UTILS OFF CACHE BOOL "Disable utils" FORCE)
        set(USE_PTHREADS ON CACHE BOOL "Enable thread-safe build" FORCE)

        # Enable BZIP2 support.  Upstream CFITSIO ships it disabled; the
        # vendored patch re-enables the BZIP2_1 tile codec in imcompress.c,
        # and HAVE_BZIP2 additionally turns on transparent whole-file `.bz2`
        # reading in drvrmem.c (CFITSIO decompresses on open).  libbz2 is
        # taken from the pixi/conda prefix when present, else from the
        # system via FIND_PACKAGE(BZip2) so wheel builds get it too.
        option(TORCHFITS_USE_BZIP2 "Build vendored CFITSIO with bzip2 support" ON)
        if(TORCHFITS_USE_BZIP2)
            # Prefer the HOST prefix: in pixi/rattler builds CONDA_PREFIX can
            # point at the build prefix, whose include dir would leak a
            # source-tree path ($BUILD_PREFIX/include lives under .pixi/bld/)
            # into cfitsio's PUBLIC interface and fail CMake generate — and
            # link the wrong-prefix archive when cross-compiling.
            set(_TORCHFITS_BZIP2_ROOT "")
            foreach(_pfx IN ITEMS "$ENV{PREFIX}" "$ENV{CONDA_PREFIX}")
                if(_pfx AND EXISTS "${_pfx}/include/bzlib.h")
                    set(_TORCHFITS_BZIP2_ROOT "${_pfx}")
                    break()
                endif()
            endforeach()
            if(_TORCHFITS_BZIP2_ROOT AND EXISTS "${_TORCHFITS_BZIP2_ROOT}/lib/libbz2.a")
                set(USE_BZIP2 ON CACHE BOOL "Enable bzip2 support" FORCE)
                set(BZIP2_INCLUDE_DIR "${_TORCHFITS_BZIP2_ROOT}/include" CACHE FILEPATH "bzip2 include dir" FORCE)
                set(BZIP2_LIBRARY "${_TORCHFITS_BZIP2_ROOT}/lib/libbz2.a" CACHE FILEPATH "bzip2 library" FORCE)
            else()
                find_package(BZip2 QUIET)
                if(BZIP2_FOUND)
                    message(STATUS "Using system BZip2 for CFITSIO: ${BZIP2_LIBRARIES}")
                    set(USE_BZIP2 ON CACHE BOOL "Enable bzip2 support" FORCE)
                else()
                    message(STATUS "TORCHFITS_USE_BZIP2 is ON but no bzlib.h/libbz2 was found; leaving CFITSIO bzip2 support disabled")
                    set(USE_BZIP2 OFF CACHE BOOL "Enable bzip2 support" FORCE)
                endif()
            endif()
        else()
            # Reconfiguring with the option turned off must also clear a
            # USE_BZIP2=ON left in the cache by an earlier configure.
            set(USE_BZIP2 OFF CACHE BOOL "Enable bzip2 support" FORCE)
        endif()

        # Force position-independent code for static library to be linked into shared library
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)

        # Build and link CFITSIO, but do not install its headers, static archive,
        # pkg-config file, or CMake metadata into the Python wheel.
        add_subdirectory("${CFITSIO_SOURCE_DIR}" cfitsio_build EXCLUDE_FROM_ALL)
        # CFITSIO's own CMakeLists adds -DCFITSIO_HAVE_CURL only when it found
        # libcurl (https/ftps support). Mirror that result as _C.HAS_CURL.
        get_directory_property(_cfitsio_dir_defs
            DIRECTORY "${CFITSIO_SOURCE_DIR}" COMPILE_DEFINITIONS)
        if("CFITSIO_HAVE_CURL" IN_LIST _cfitsio_dir_defs)
            set(_TORCHFITS_CFITSIO_HAS_CURL ON)
        else()
            set(_TORCHFITS_CFITSIO_HAS_CURL OFF)
        endif()
        message(STATUS "Vendored CFITSIO https/ftps (libcurl): ${_TORCHFITS_CFITSIO_HAS_CURL}")

        # The vendored cfitsio CMakeLists publishes its dependency include
        # dirs (zlib/curl/bzip2) as PUBLIC usage requirements. A dep resolved
        # inside the source tree (e.g. pixi's .pixi/bld/ build prefix) fails
        # CMake generate ("INTERFACE_INCLUDE_DIRECTORIES ... prefixed in the
        # source directory"). cfitsio links statically into _C — a leaf
        # module whose sources only need fitsio.h itself (wired PRIVATE
        # below) — so demote those dirs to PRIVATE: cfitsio's own .c files
        # still see them, but they no longer propagate.
        get_target_property(_cfitsio_iface_inc cfitsio INTERFACE_INCLUDE_DIRECTORIES)
        if(_cfitsio_iface_inc)
            target_include_directories(cfitsio PRIVATE ${_cfitsio_iface_inc})
        endif()
        set_target_properties(cfitsio PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")

        # CFITSIO target is named 'cfitsio'
        set(CFITSIO_LIBRARY cfitsio)
        set(CFITSIO_INCLUDE_DIR "${CFITSIO_SOURCE_DIR}")
        set(CFITSIO_FOUND TRUE)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            # Compressed-float decode must be bit-exact across platforms.
            # Clang contracts FP expressions (FMA fusion) by default at
            # -O2/-O3, and aarch64 GCC does too, which shifts low-ULP bits
            # of unquantized / HCOMPRESS-renormalized tiles: an arm64 build
            # then reads e.g. 1.94e-16 where the lossless tile stores exact
            # 0.0, disagreeing with wheel builds that disable contraction
            # (fitsio/astropy). Pin contraction off for vendored CFITSIO.
            target_compile_options(cfitsio PRIVATE -ffp-contract=off)
        endif()
        if(_TORCHFITS_SANITIZER_COMPILE_OPTIONS)
            target_compile_options(
                cfitsio PRIVATE ${_TORCHFITS_SANITIZER_COMPILE_OPTIONS})
        endif()
        if(TORCHFITS_COVERAGE)
            # CFITSIO is C and built in a subdirectory; directory-scoped flags
            # are deliberately not used (see the coverage block at the top).
            target_compile_options(
                cfitsio PRIVATE ${_TORCHFITS_COVERAGE_COMPILE_OPTIONS})
        endif()
        # Clang IPO stores LLVM bitcode in this static archive. torchfits_core
        # links it with --whole-archive and is not itself built with LTO, so
        # GNU ld rejects every member ("is not an object").
        set_target_properties(cfitsio PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
        # NIOBUF (fitsio.h) sizes arrays inside CFITSIO's FITSfile struct and
        # MINDIRECT (fitsio2.h) is the direct-I/O threshold. The vendored
        # headers take both from the command line through #ifndef guards
        # (extern/patches/cfitsio-4.7.0-niobuf-guards.patch), and every target
        # that includes them must see the same values -- a NIOBUF that only
        # CFITSIO saw gave torchfits a different FITSfile layout from the one
        # CFITSIO built. Applied to the torchfits targets below as well.
        set(_TORCHFITS_CFITSIO_TUNING_DEFS "")
        if(TORCHFITS_NIOBUF)
            list(APPEND _TORCHFITS_CFITSIO_TUNING_DEFS NIOBUF=${TORCHFITS_NIOBUF})
            message(STATUS "Overriding CFITSIO NIOBUF=${TORCHFITS_NIOBUF}")
        endif()
        if(TORCHFITS_MINDIRECT)
            list(APPEND _TORCHFITS_CFITSIO_TUNING_DEFS MINDIRECT=${TORCHFITS_MINDIRECT})
            message(STATUS "Overriding CFITSIO MINDIRECT=${TORCHFITS_MINDIRECT}")
        endif()
        if(_TORCHFITS_CFITSIO_TUNING_DEFS)
            # Without the guards the header's own #define wins and the -D is
            # silently ignored, so refuse a tree vendored before the patch.
            file(STRINGS "${CFITSIO_SOURCE_DIR}/fitsio.h" _niobuf_guard REGEX "^#ifndef NIOBUF")
            file(STRINGS "${CFITSIO_SOURCE_DIR}/fitsio2.h" _mindirect_guard REGEX "^#ifndef MINDIRECT")
            if(NOT _niobuf_guard OR NOT _mindirect_guard)
                message(FATAL_ERROR
                    "${CFITSIO_SOURCE_DIR} lacks the NIOBUF/MINDIRECT #ifndef guards; "
                    "re-run ./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt")
            endif()
            target_compile_definitions(cfitsio PRIVATE ${_TORCHFITS_CFITSIO_TUNING_DEFS})
        endif()
    else()
        message(FATAL_ERROR
            "CFITSIO sources not found at ${CFITSIO_SOURCE_DIR}. "
            "PyPI installs torchfits from prebuilt wheels (CPython 3.10-3.14, "
            "Linux x86_64/aarch64, macOS arm64) — pip should not compile. "
            "From a git checkout run: ./extern/vendor.sh --cfitsio-version extern/VERSIONS.txt")
    endif()
endif()

# Source files
set(SOURCES
    bindings.cpp
    fits_bindings.cpp
    fits_file.cpp
    table_bindings.cpp
    table_ops.cpp
    table_reader_buffered.cpp
    table_reader_filter.cpp
    table_reader_layout.cpp
    table_reader_mmap.cpp
    table_reader_read.cpp
    table_reader_vla.cpp
)

# A dlopen'd Python extension is an MH_BUNDLE, and dyld does not expand
# $ORIGIN in a bundle's LC_RPATH. @loader_path is the token that resolves
# against the image carrying the dependency; on ELF the equivalent is $ORIGIN.
if(APPLE)
    set(TORCHFITS_SIBLING_RPATH "@loader_path")
else()
    set(TORCHFITS_SIBLING_RPATH "\$ORIGIN")
endif()

# ---------------------------------------------------------------------------
# libtorchfits_core -- CFITSIO and the metadata cache, shared by both modules.
#
# CFITSIO and the FITS inspection / metadata-cache code live here rather than
# in _C because there must be exactly one CFITSIO in the process. Its error
# stack and internal buffers are global, so two statically linked copies would
# let a status produced by one be resolved by the other. _C and _core therefore
# resolve every fits_* against this library (see check_core_link.cmake, which
# fails the build if any of them is left undefined).
#
# No native target links a Torch target or includes an ATen/torch header; the
# guard after the _C target turns that from a convention into a build error.
# ---------------------------------------------------------------------------
add_library(torchfits_core SHARED
    core/fits_core.cpp
    core/metadata_api.cpp
    core/parallel.cpp
    hardware.cpp
)

target_include_directories(torchfits_core
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CFITSIO_INCLUDE_DIR}"
)
if(CFITSIO_INCLUDE_DIRS)
    target_include_directories(torchfits_core PRIVATE ${CFITSIO_INCLUDE_DIRS})
endif()

# The vendored CFITSIO target publishes its dependency include dirs as PUBLIC
# usage requirements; demote them here for the same reason as the _C wiring
# below (a build-prefix path would fail CMake generate and leak into the wheel).
if(TARGET cfitsio)
    get_target_property(_core_cfitsio_iface_inc cfitsio INTERFACE_INCLUDE_DIRECTORIES)
    if(_core_cfitsio_iface_inc)
        target_include_directories(torchfits_core PRIVATE ${_core_cfitsio_iface_inc})
    endif()
    target_link_libraries(torchfits_core PRIVATE cfitsio)
    # fits_get_diskfile_fd() comes from extern/patches/cfitsio-4.7.0-diskfile-fd
    # .patch; an environment CFITSIO does not have it (fits_core.cpp then falls
    # back to stat()ing the path around each open).
    target_compile_definitions(torchfits_core PRIVATE TORCHFITS_CFITSIO_HAS_DISKFILE_FD=1)
    # A static archive only contributes the members something references, so
    # the core would export just the subset its own sources happen to call --
    # ffpcks/ffvcks (the checksum paths _C owns) would be missing and _C would
    # fail to link. Force the whole archive in so the core really is *the*
    # CFITSIO: one copy, fully exported, for both modules.
    if(APPLE)
        target_link_options(torchfits_core PRIVATE
            "-Wl,-force_load,$<TARGET_FILE:cfitsio>")
    else()
        target_link_options(torchfits_core PRIVATE
            "-Wl,--whole-archive" "$<TARGET_FILE:cfitsio>" "-Wl,--no-whole-archive")
    endif()
else()
    target_link_libraries(torchfits_core PRIVATE ${CFITSIO_LIBRARIES} ${CFITSIO_LIBRARY})
    if(CFITSIO_LIBRARY_DIRS)
        target_link_directories(torchfits_core PRIVATE ${CFITSIO_LIBRARY_DIRS})
    endif()
endif()
target_link_libraries(torchfits_core PRIVATE ZLIB::ZLIB Threads::Threads)

target_compile_definitions(torchfits_core PRIVATE
    HAS_CFITSIO
    TORCHFITS_CORE_BUILDING=1
    TORCHFITS_CORE_BUILD_ID="${TORCHFITS_CORE_BUILD_ID}"
)
if(USE_BZIP2)
    target_compile_definitions(torchfits_core PRIVATE TORCHFITS_HAVE_BZIP2=1)
endif()
target_compile_features(torchfits_core PRIVATE cxx_std_20)

# IPO/LTO is deliberately off here. LTO is free to internalize a symbol the
# extension still needs to bind against, and the failure would only show up as
# an import-time "undefined symbol". The core does almost no arithmetic, so
# there is nothing for it to lose.
set_target_properties(torchfits_core PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    # CFITSIO's C ABI relies on default visibility and interposition rules that
    # must not be disturbed; hide nothing.
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET default
    MACOSX_RPATH ON
    INSTALL_RPATH "${TORCHFITS_SIBLING_RPATH}"
    BUILD_WITH_INSTALL_RPATH ON
)

# The _core module: the same library with nanobind/Python on top. No ATen.
nanobind_add_module(_core STABLE_ABI NB_STATIC core/core_bindings.cpp)
target_link_libraries(_core PRIVATE torchfits_core)
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CFITSIO_INCLUDE_DIR}")
if(CFITSIO_INCLUDE_DIRS)
    target_include_directories(_core PRIVATE ${CFITSIO_INCLUDE_DIRS})
endif()
target_compile_features(_core PRIVATE cxx_std_20)
target_compile_definitions(_core PRIVATE TORCHFITS_CORE_BUILD_ID="${TORCHFITS_CORE_BUILD_ID}")
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(_core PRIVATE -O3 -fno-semantic-interposition)
endif()
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    MACOSX_RPATH ON
    INSTALL_RPATH "${TORCHFITS_SIBLING_RPATH}"
    BUILD_WITH_INSTALL_RPATH ON
)

# Create python binding module. Like the core, it includes no torch/ATen header
# and links no Torch target: decoded buffers leave as DLPack capsules that
# torch's Python API adopts (native_py.h), so the extension is independent of
# the installed PyTorch's C++ ABI.
nanobind_add_module(_C STABLE_ABI NB_STATIC ${SOURCES})

# These optimize and isolate the Python extension. They must not propagate to
# vendored CFITSIO, whose C ABI relies on its own visibility/interposition rules.
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # nanobind defaults its module sources to -Os. Keep these options after
    # nanobind_add_module so the final effective _C optimization remains -O3.
    target_compile_options(_C PRIVATE -O3 -fno-semantic-interposition -fvisibility=hidden)
endif()

set_target_properties(_C PROPERTIES
    MACOSX_RPATH ON
    INSTALL_RPATH "${TORCHFITS_SIBLING_RPATH}"
    BUILD_WITH_INSTALL_RPATH ON
)

# CFITSIO comes from the core (see the comment above the core target). Link the
# core first so fits_* resolves against it.
target_link_libraries(_C PRIVATE torchfits_core)
# The core's headers (core/fits_core.h, core/metadata_api.h) are included by the
# extension's own sources, so its include root has to be visible here too.
target_include_directories(_C PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(TARGET cfitsio)
    target_include_directories(_C PRIVATE "${CFITSIO_INCLUDE_DIR}")
else()
    target_include_directories(_C PRIVATE ${CFITSIO_INCLUDE_DIRS} ${CFITSIO_INCLUDE_DIR})
endif()

target_compile_definitions(_C PRIVATE HAS_CFITSIO)
# Feature flag surfaced to Python as _C.HAS_BZIP2 so guards can be
# capability-aware instead of blanket-rejecting .bz2 inputs.
if(USE_BZIP2)
    target_compile_definitions(_C PRIVATE TORCHFITS_HAVE_BZIP2=1)
endif()
# _C.HAS_CURL: whether CFITSIO can open https:// and ftps:// URLs.
if(_TORCHFITS_CFITSIO_HAS_CURL)
    target_compile_definitions(_C PRIVATE TORCHFITS_HAVE_CURL=1)
endif()


# No Torch target may appear in any native target's link interface. Without
# this check a future `target_link_libraries(... torch)` would silently tie the
# wheel to one PyTorch C++ ABI again (and make `import torchfits._core` load
# libtorch).
foreach(_torchfits_target IN ITEMS torchfits_core _core _C)
    get_target_property(_TARGET_LINK_LIBS ${_torchfits_target} LINK_LIBRARIES)
    foreach(_lib IN LISTS _TARGET_LINK_LIBS)
        if(_lib MATCHES "^(torch|torch_python|torch_cpu|c10|caffe2)$|^(Torch|torch|caffe2)::|(^|/)lib(torch|torch_python|torch_cpu|c10)\\.")
            message(FATAL_ERROR
                "${_torchfits_target} must stay torch-free, but it links '${_lib}'")
        endif()
    endforeach()
endforeach()

# Compiler flags
target_compile_definitions(_C PRIVATE VERSION_INFO="")
target_compile_features(_C PRIVATE cxx_std_20)

# Build identity. _C records the core's id so a mismatched pair fails loudly at
# import (torchfits/__init__.py) instead of crossing an ABI boundary quietly.
target_compile_definitions(_C PRIVATE TORCHFITS_CORE_BUILD_ID="${TORCHFITS_CORE_BUILD_ID}")

# Same NIOBUF/MINDIRECT as the vendored CFITSIO build (see where they are set).
if(_TORCHFITS_CFITSIO_TUNING_DEFS)
    foreach(_torchfits_target IN ITEMS torchfits_core _core _C)
        target_compile_definitions(${_torchfits_target} PRIVATE ${_TORCHFITS_CFITSIO_TUNING_DEFS})
    endforeach()
endif()

# Prove the CFITSIO symbols actually bound to the core (see check_core_link.cmake).
add_custom_command(TARGET _C POST_BUILD
    COMMAND "${CMAKE_COMMAND}" "-DCMAKE_TARGET=$<TARGET_FILE:_C>"
            "-DCORE_LIBRARY=$<TARGET_FILE:torchfits_core>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_link.cmake"
    VERBATIM
)
add_custom_command(TARGET _core POST_BUILD
    COMMAND "${CMAKE_COMMAND}" "-DCMAKE_TARGET=$<TARGET_FILE:_core>"
            "-DCORE_LIBRARY=$<TARGET_FILE:torchfits_core>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_link.cmake"
    VERBATIM
)


# Install the compiled module
if(SKBUILD)
    # scikit-build-core (wheels): install to the package directory
    install(TARGETS _C _core torchfits_core DESTINATION torchfits)
else()
    # Bare-cmake installs (conda packages via pixi-build-cmake / rattler-build)
    # skip scikit-build-core, so mirror its wheel layout: the compiled
    # extension and the pure-Python package land in the active Python's
    # site-packages.
    set(_TORCHFITS_PYEXE "")
    if(EXISTS "${CMAKE_INSTALL_PREFIX}/bin/python")
        set(_TORCHFITS_PYEXE "${CMAKE_INSTALL_PREFIX}/bin/python")
    elseif(NANOBIND_PYTHON_EXECUTABLE)
        set(_TORCHFITS_PYEXE "${NANOBIND_PYTHON_EXECUTABLE}")
    endif()
    if(_TORCHFITS_PYEXE)
        execute_process(
            COMMAND "${_TORCHFITS_PYEXE}" -c
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])"
            OUTPUT_VARIABLE _TORCHFITS_PURELIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_TORCHFITS_PURELIB)
            install(TARGETS _C _core torchfits_core DESTINATION "${_TORCHFITS_PURELIB}/torchfits")
            get_filename_component(_TORCHFITS_PKG_DIR
                "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
            install(
                DIRECTORY "${_TORCHFITS_PKG_DIR}"
                DESTINATION "${_TORCHFITS_PURELIB}"
                FILES_MATCHING
                PATTERN "*.py"
                PATTERN "py.typed"
                PATTERN "cpp_src" EXCLUDE
            )
            # Console script (wheels get it from scikit-build-core entry
            # points; bare-cmake conda installs need it written out).
            get_filename_component(_TORCHFITS_BINDIR
                "${_TORCHFITS_PURELIB}/../../../bin" ABSOLUTE)
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/torchfits-cli"
                "#!/bin/sh\nexec \"${_TORCHFITS_BINDIR}/python\" -c 'from torchfits.cli.main import main; main()' \"\$@\"\n")
            install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/torchfits-cli"
                DESTINATION "${_TORCHFITS_BINDIR}" RENAME torchfits)
            message(STATUS "torchfits: bare-cmake install into ${_TORCHFITS_PURELIB}")
        else()
            message(WARNING "torchfits: could not locate site-packages; extension only")
        endif()
    else()
        message(WARNING "torchfits: no Python executable found; extension only")
    endif()
endif()

# ---------------------------------------------------------------------------
# Coverage instrumentation for torchfits' own targets.
# Kept out of the directory scope so bundled third-party code (nanobind) stays
# uninstrumented; vendored CFITSIO is handled at its target, above.
# ---------------------------------------------------------------------------
if(TORCHFITS_COVERAGE)
    foreach(_torchfits_cov_target IN ITEMS torchfits_core _core _C)
        if(TARGET ${_torchfits_cov_target})
            target_compile_options(${_torchfits_cov_target} PRIVATE
                ${_TORCHFITS_COVERAGE_COMPILE_OPTIONS})
            target_link_options(${_torchfits_cov_target} PRIVATE
                ${_TORCHFITS_COVERAGE_LINK_OPTIONS})
        endif()
    endforeach()
endif()
