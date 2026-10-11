# scripts/platform/darwin.sh -- platform facts for scripts/setup.sh on macOS
# (developed on an Apple Silicon MacBook Air, macOS 15.6, Command Line Tools 16.3).
# Sourced by setup.sh, never run on its own. Selected by `uname -s`, or by
# INCNS_PLATFORM=darwin.
#
# The contract (every platform file sets these):
#   PLATFORM_GCC_SPEC    the GCC setup.sh builds with Spack (gfortran always comes
#                        from it: Apple's toolchain has no Fortran)
#   PLATFORM_C_COMPILER  `gcc` (the Spack GCC compiles C and C++ too) or the name of
#                        a compiler package declared by platform_compiler_externals
#   PLATFORM_JOBS        default for --jobs
# and may define a function:
#   platform_compiler_externals   print extra `packages:` entries (YAML, indented 4
#                                 spaces) for the generated environment
#
# Every line below was found by running setup.sh on that machine
# (docs/status.md, "macos-gcc").

# 1. gcc@14.3.0 stops at configure, "Configuration aarch64-apple-darwin24.6.0 not
#    supported": Spack's recipe carries Iain Sandoe's aarch64-darwin patches only up
#    to 14.2.0 (which Spack prefers on macOS). Untested on an Intel Mac, where the
#    14.2.0 patches are also meant to apply.
# 2. ~libsanitizer: linking libasan.8.dylib crashes Apple's ld (CLT 16.3.0,
#    ld-1167.4.1, "Assertion failed: (entry.toAddend < 0x400000)"). This only drops
#    GCC's own sanitizer runtime; bessemer is compiled by Apple clang (below), which
#    brings its own.
PLATFORM_GCC_SPEC=gcc@14.2.0~libsanitizer

# C and C++ are Apple's clang: Spack refuses CPython with GCC on macOS ("CPython
# does not compile with GCC on macOS yet, use clang"), and Python is in the stack.
# GCC is the Fortran compiler only (MUMPS, ScaLAPACK, OpenBLAS, gslib ...).
PLATFORM_C_COMPILER=apple-clang

# No nproc on macOS.
PLATFORM_JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

# The system clang, declared external so Spack uses it and never tries to build it.
platform_compiler_externals() {
  _clang_version=$(/usr/bin/clang --version 2>/dev/null \
    | sed -n 's/^Apple clang version \([0-9.]*\).*/\1/p')
  [ -n "$_clang_version" ] \
    || die "cannot read the version of /usr/bin/clang (are the Xcode command line tools installed? xcode-select --install)"
  cat <<EOF
    apple-clang:
      buildable: false
      externals:
      - spec: apple-clang@$_clang_version
        prefix: /usr
        extra_attributes:
          compilers:
            c: /usr/bin/clang
            cxx: /usr/bin/clang++
EOF
}
