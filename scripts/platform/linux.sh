# scripts/platform/linux.sh -- platform facts for scripts/setup.sh on Linux
# (workstations and clusters). Sourced by setup.sh, never run on its own. See
# scripts/platform/darwin.sh for the contract and docs/install/spack.md, "Platforms".

# The GCC that setup.sh builds with Spack on a workstation (the system compiler
# builds it). A cluster passes --compiler PREFIX instead. This is the one place the
# version is written: change it here for a new environment; an existing environment
# keeps the GCC in its spack.lock until it is re-resolved.
PLATFORM_GCC_SPEC=gcc@14.3.0

# Which compiler the environment requires for the c and cxx virtuals: the Spack
# GCC above (gfortran comes from the same one).
PLATFORM_C_COMPILER=gcc

# Default for --jobs.
PLATFORM_JOBS=$(nproc 2>/dev/null || echo 4)
