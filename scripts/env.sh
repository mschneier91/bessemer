#!/bin/sh
# scripts/env.sh -- the single place that resolves the machine, activates its
# spack environment, and ASSERTS that the whole toolchain resolves inside spack.
#
# Sourced (not executed) by build.sh / test.sh / style.sh and the git hooks, so
# activation never depends on anyone remembering `spack env activate`. It does not
# use `set -e` (that would leak into the caller and trip on spack's own scripts);
# instead every failure path returns non-zero, which the wrappers check.
#
# Overrides:
#   INCNS_MACHINE  -- force a machine name (default: hostname-resolved, else desktop)
#   SPACK_ROOT     -- spack checkout (default: $HOME/spack)

# --- repo root -------------------------------------------------------------
# Wrappers export INCNS_REPO_ROOT before sourcing; fall back to git, then pwd.
if [ -z "${INCNS_REPO_ROOT:-}" ]; then
  INCNS_REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
fi
export INCNS_REPO_ROOT

# --- machine resolution (single source of truth) ---------------------------
if [ -z "${INCNS_MACHINE:-}" ]; then
  _host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)
  case "$_host" in
    # Add hostname patterns here as machines are brought up, e.g.:
    #   frontier* ) INCNS_MACHINE=frontier ;;
    # Bridges-2: br0xx login nodes + w0xx H100 nodes. psc_gpu is pinned to
    # cuda_arch=90 (H100 only, see docs/install/bridges2.md) -- v0xx (V100)
    # and gl0xx (L40S) are deliberately NOT matched here, since a cuda_arch=90
    # binary is wrong-SM-arch on them.
    br0* | w0* ) INCNS_MACHINE=psc_gpu ;;
    * ) INCNS_MACHINE=desktop ;;
  esac
fi
export INCNS_MACHINE

_env_dir="$INCNS_REPO_ROOT/environments/$INCNS_MACHINE"
if [ ! -d "$_env_dir" ]; then
  echo "env.sh: no spack env for machine '$INCNS_MACHINE' at $_env_dir" >&2
  return 1 2>/dev/null || exit 1
fi

# --- spack -----------------------------------------------------------------
: "${SPACK_ROOT:=$HOME/spack}"
if [ ! -f "$SPACK_ROOT/share/spack/setup-env.sh" ]; then
  echo "env.sh: spack not found at SPACK_ROOT=$SPACK_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi
export SPACK_DISABLE_LOCAL_CONFIG=1
# shellcheck disable=SC1091
. "$SPACK_ROOT/share/spack/setup-env.sh"
spack env activate "$_env_dir" || {
  echo "env.sh: failed to activate $_env_dir" >&2
  return 1 2>/dev/null || exit 1
}

_spack_view="$_env_dir/.spack-env/view"

# --- pin bare gcc/g++ to the spack compiler ---------------------------------
# The spack compiler (gcc@14.3.0) is registered but NOT symlinked into the env
# view, so a bare `gcc` would fall through to the system compiler. Derive the
# exact compiler that MPI/MFEM were built with from the mpicc wrapper and put its
# bin dir on PATH -- self-consistent and rebuild-robust (no hardcoded version).
_mpicc_cc=$(mpicc -show 2>/dev/null | awk '{print $1; exit}')
if [ -n "$_mpicc_cc" ] && [ -x "$_mpicc_cc" ]; then
  _gcc_bin=$(dirname "$_mpicc_cc")
  case ":$PATH:" in
    *":$_gcc_bin:"*) : ;;
    *) PATH="$_gcc_bin:$PATH"; export PATH ;;
  esac
fi

# Export MFEM_DIR so FindMFEM.cmake need not re-query spack.
MFEM_DIR=$(spack location -i mfem 2>/dev/null) && export MFEM_DIR

# --- gather compiler/MPI/cmake prefixes THIS env's own spack.yaml declares
# as externals (buildable:false) -- e.g. a cluster's vetted per-node-class
# compiler module (environments/psc_gpu pins gcc to /opt/packages/.../b2gpu).
# These are legitimate, not a system-toolchain leak: the exception already
# carved out for vendor MPI (see CLAUDE.md, Environment & build) generalizes
# to any declared external for the packages that actually provide these
# tools. Scoped to THIS file (not the merged/site Spack config, which can
# carry its own unrelated system-compiler fallbacks) and to a fixed whitelist
# of provider package names, so an unrelated external (e.g. slurm) can never
# widen what's accepted here. No cluster-specific path is hardcoded below.
_spack_ext_prefixes=$(awk '
  /^    [a-zA-Z0-9_-]+:[[:space:]]*$/ { key=$1; sub(":", "", key); next }
  /^[[:space:]]+prefix:/ {
    if (key ~ /^(gcc|llvm|intel-oneapi-compilers|openmpi|mpich|cmake)$/) print $2
  }
' "$_env_dir/spack.yaml" 2>/dev/null)

# --- assert: no system-toolchain leak (CLAUDE.md guardrail) ----------------
# "Inside spack" == the install tree ($SPACK_ROOT/...), the activated view,
# or one of the declared-external prefixes gathered just above.
_incns_assert_spack() {
  _tool=$1
  _path=$(command -v "$_tool" 2>/dev/null || true)
  case "$_path" in
    "$SPACK_ROOT"/*|"$_spack_view"/*) return 0 ;;
  esac
  for _p in $_spack_ext_prefixes; do
    case "$_path" in
      "$_p"/*) return 0 ;;
    esac
  done
  echo "env.sh: '$_tool' -> '${_path:-MISSING}' is NOT inside spack" \
       "($SPACK_ROOT, the env view, or a declared external prefix in" \
       "$_env_dir/spack.yaml). System-toolchain leak -- aborting." >&2
  return 1
}

_leak=0
for _t in gcc g++ cmake mpicc mpicxx mpirun; do
  _incns_assert_spack "$_t" || _leak=1
done
if [ "$_leak" -ne 0 ]; then
  return 1 2>/dev/null || exit 1
fi

echo "[env] machine=$INCNS_MACHINE  gcc=$(gcc -dumpversion)  mfem=$MFEM_DIR" >&2
