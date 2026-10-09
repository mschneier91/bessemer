#!/bin/sh
# scripts/doctor.sh [--build] -- can this checkout build and run bessemer?
#
# Checks the setup step by step and says how to fix whatever fails, then runs
# a small deck (cases/cavity.yaml, ~5 s at 2 MPI ranks) and reads its run
# summary. Exit status 0 when nothing failed. Written for people and coding
# agents arriving cold (AGENTS.md).
#
#   scripts/doctor.sh           # check; build only if the build is missing
#   scripts/doctor.sh --build   # also (re)build the cpu preset first
#
# It never edits the Spack environment and never activates one whose
# spack.yaml changed since its last resolve (activation would re-resolve it;
# docs/install/desktop.md, trap 2).

REPO=$(cd "$(dirname "$0")/.." && pwd)
INCNS_REPO_ROOT=$REPO
export INCNS_REPO_ROOT
BUILD=0
[ "${1:-}" = "--build" ] && BUILD=1

fails=0
warns=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
hint() { printf '        -> %s\n' "$*"; }
finish()
{
  echo
  if [ "$fails" -eq 0 ]; then
    echo "doctor: all good ($warns warning(s)). Next: docs/using.md"
    exit 0
  fi
  echo "doctor: $fails problem(s) to fix (see the -> lines). Setup: docs/install/"
  exit 1
}

echo "bessemer doctor ($REPO)"

# --- 1. where we run ----------------------------------------------------------
if [ -f /.flatpak-info ]; then
  warn "running inside a Flatpak sandbox ($(grep -m1 '^name=' /.flatpak-info | cut -d= -f2))"
  hint "builds and runs work here, but do Spack environment work (install, refresh,"
  hint "edits to environments/*/spack.yaml) from a host terminal: docs/install/desktop.md"
else
  ok "not in a sandboxed editor"
fi

# --- 2. Spack -----------------------------------------------------------------
: "${SPACK_ROOT:=$HOME/spack}"
if [ -f "$SPACK_ROOT/share/spack/setup-env.sh" ]; then
  ok "Spack at $SPACK_ROOT"
else
  fail "no Spack checkout at SPACK_ROOT=$SPACK_ROOT"
  hint "clone Spack there (or set SPACK_ROOT), then build the environment: docs/install/"
  finish
fi

# --- 3. the machine's environment -----------------------------------------------
# Resolve the machine the way env.sh does, without activating anything yet.
if [ -z "${INCNS_MACHINE:-}" ]; then
  _host=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)
  case "$_host" in
    br0* | w0* ) INCNS_MACHINE=psc_gpu ;;
    * ) INCNS_MACHINE=desktop ;;
  esac
fi
envdir="$REPO/environments/$INCNS_MACHINE"
if [ ! -f "$envdir/spack.yaml" ] || [ ! -f "$envdir/spack.lock" ]; then
  fail "no Spack environment for machine '$INCNS_MACHINE' ($envdir)"
  hint "a new machine needs its own environments/<name>/ (spack.yaml + spack.lock);"
  hint "set INCNS_MACHINE to use an existing one: docs/install/"
  finish
fi
ok "machine '$INCNS_MACHINE', environment $envdir"
yaml_t=$(stat -c %Y "$envdir/spack.yaml")
lock_t=$(stat -c %Y "$envdir/spack.lock")
if [ "$yaml_t" -gt $((lock_t + 2)) ]; then
  fail "spack.yaml changed after the last resolve (spack.lock is older)"
  hint "activating now would re-resolve the environment and hide the installed stack."
  hint "From a host terminal: spack -e $envdir concretize --force && spack -e $envdir install"
  hint "(or, if the edit was unintended: git checkout -- $envdir/spack.yaml)"
  finish
fi
ok "spack.yaml and spack.lock agree"

# --- 4. activation + toolchain -------------------------------------------------
out=$(mktemp)
if . "$REPO/scripts/env.sh" >"$out" 2>&1; then
  ok "toolchain: $(grep -m1 '^\[env\]' "$out" | sed 's/^\[env\] //')"
else
  fail "activating the environment failed:"
  sed 's/^/          /' "$out" | tail -5
  hint "the environment is probably not installed: spack -e $envdir install (host terminal)"
  rm -f "$out"
  finish
fi
rm -f "$out"

# --- 5. MFEM ---------------------------------------------------------------------
cfg="${MFEM_DIR:-}/share/mfem/config.mk"
if [ -f "$cfg" ]; then
  git_str=$(awk -F'= *' '/^MFEM_GIT_STRING/ {print substr($2,1,10)}' "$cfg")
  mpi=$(awk -F'= *' '/^MFEM_USE_MPI /{print $2}' "$cfg")
  mumps=$(awk -F'= *' '/^MFEM_USE_MUMPS /{print $2}' "$cfg")
  if [ "$mpi" = "YES" ]; then
    ok "MFEM $git_str (MPI yes, MUMPS ${mumps:-NO})"
  else
    fail "MFEM at $MFEM_DIR was built without MPI"
    hint "the environment's mfem spec needs +mpi: environments/$INCNS_MACHINE/spack.yaml"
  fi
else
  fail "MFEM not found (MFEM_DIR='${MFEM_DIR:-}')"
  hint "spack -e $envdir install (host terminal)"
  finish
fi

# --- 6. the build ------------------------------------------------------------------
if [ "$BUILD" -eq 1 ] || [ ! -x "$REPO/build/cpu/apps/run_case" ]; then
  echo "  ..    building the cpu preset (scripts/build.sh cpu)"
  if "$REPO/scripts/build.sh" cpu >"$REPO/build-doctor.log" 2>&1; then
    ok "build/cpu built (log: build-doctor.log)"
    rm -f "$REPO/build-doctor.log"
  else
    fail "the build failed; the end of build-doctor.log:"
    tail -15 "$REPO/build-doctor.log" | sed 's/^/          /'
    finish
  fi
else
  ok "build/cpu present (rebuild with scripts/build.sh cpu)"
fi

# --- 7. a smoke run ------------------------------------------------------------------
run_dir=$(mktemp -d)
summary="$run_dir/summary.json"
start=$(date +%s)
if (cd "$run_dir" && mpirun -np 2 "$REPO/build/cpu/apps/run_case" \
      "$REPO/cases/cavity.yaml" --summary "$summary") >"$run_dir/run.log" 2>&1 \
   && grep -q '"status": "ok"' "$summary" 2>/dev/null; then
  steps=$(awk -F': *' '/"steps"/ {gsub(/,/,"",$2); print $2; exit}' "$summary")
  ok "smoke run: lid-driven cavity at 2 ranks, $steps steps, $(( $(date +%s) - start )) s"
else
  fail "the smoke run (cases/cavity.yaml at 2 MPI ranks) failed; its output:"
  tail -12 "$run_dir/run.log" | sed 's/^/          /'
  hint "if mpirun refuses to start: is another MPI job holding the cores? scripts/reap.sh --dry"
fi
rm -rf "$run_dir"

# --- 8. stray processes ----------------------------------------------------------------
n=$(ps -eo args | grep -cE 'build/[a-z-]+/(test|apps|bench)/' | tr -d ' ')
n=$((n - 1)) # the grep itself
if [ "$n" -gt 0 ]; then
  warn "$n bessemer process(es) running (yours, or orphans of an interrupted run)"
  hint "list: scripts/reap.sh --dry; kill orphans: scripts/reap.sh"
else
  ok "no bessemer processes running"
fi

finish
