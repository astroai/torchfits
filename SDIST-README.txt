torchfits source distribution
=============================

This sdist is provided for source inspection and packager review only.

Building it from source requires a C++20 toolchain and network access to fetch
the pinned CFITSIO sources (extern/vendor.sh --cfitsio-version
extern/VERSIONS.txt). The build needs no PyTorch: the extension includes and
links no libtorch (decoded buffers reach PyTorch as DLPack capsules through its
Python API), so one build serves every PyTorch the runtime requirement admits.
That requirement is the torch entry under `project.dependencies` in
pyproject.toml; `build-system.requires` lists only the build backend. It is NOT
built or tested as an install path: PyPI installs torchfits from prebuilt
wheels.

If you need a source build, clone the git repository instead — the CMake layer
auto-vendors the pinned CFITSIO there.
