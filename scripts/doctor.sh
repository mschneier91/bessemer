#!/bin/sh
# scripts/doctor.sh [--run | --build] -- can this checkout build and run bessemer?
#
# Checks the setup step by step and says how to fix whatever fails. By default
# it only looks: it starts no build and no MPI run, so it is safe anywhere, a
# shared login node included. With --run it also builds if the build is missing
# and runs a small deck (cases/cavity.yaml, ~5 s at 2 MPI ranks), reading its
# run summary. Exit status 0 when nothing failed. Written for people and coding
# agents arriving cold (AGENTS.md).
#
#   scripts/doctor.sh           # check only: nothing is built or run
#   scripts/doctor.sh --run     # also build if the build is missing, then smoke-run
#   scripts/doctor.sh --build   # also (re)build, then smoke-run
#
# The build is the environment's preset (CMakePresets.json): cpu, unless
# environments/<name>/preset names another one.
#
# Use --run and --build where builds and MPI runs are allowed: on a cluster that
# is a compute node, not a login node.
#
# It never edits the Spack environment and never activates one whose
# spack.yaml changed since its last resolve (activation would re-resolve it;
# docs/install/spack.md, trap 2).

REPO=$(cd "$(dirname "$0")/.." && pwd)
INCNS_REPO_ROOT=$REPO
export INCNS_REPO_ROOT
BUILD=0
RUN=0
case "${1:-}" in
  "") ;;
  --run) RUN=1 ;;
  --build) BUILD=1; RUN=1 ;;
  *) echo "usage: doctor.sh [--run | --build]" >&2; exit 2 ;;
esac

fails=0
warns=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
hint() { printf '        -> %s\n' "$*"; }
finish()
{
  echo
  if [ "$fails" -eq 0 ] && [ "$RUN" -eq 1 ]; then
    echo "doctor: all good ($warns warning(s)). Next: docs/using.md"
    exit 0
  fi
  if [ "$fails" -eq 0 ]; then
    echo "doctor: the setup checks pass ($warns warning(s)); nothing was built or run."
    echo "Smoke run: scripts/doctor.sh --run (on a cluster: on a compute node)"
    exit 0
  fi
  if [ "${wrong_machine:-0}" -eq 1 ]; then
    echo "doctor: $fails problem(s) (see the -> lines). The setup itself may be fine:"
    echo "run the doctor again on a machine the environment is built for."
  else
    echo "doctor: $fails problem(s) to fix (see the -> lines). Setup: scripts/setup.sh --plan"
  fi
  exit 1
}

echo "bessemer doctor ($REPO)"

# --- 1. where we run ----------------------------------------------------------
if [ -f /.flatpak-info ]; then
  warn "running inside a Flatpak sandbox ($(grep -m1 '^name=' /.flatpak-info | cut -d= -f2))"
  hint "builds and runs work here, but do Spack environment work (install, refresh,"
  hint "edits to environments/*/spack.yaml) from a host terminal: docs/install/spack.md"
else
  ok "not in a sandboxed editor"
fi

# --- 2. Spack -----------------------------------------------------------------
: "${SPACK_ROOT:=$HOME/spack}"
if [ -f "$SPACK_ROOT/share/spack/setup-env.sh" ]; then
  ok "Spack at $SPACK_ROOT"
else
  fail "no Spack checkout at SPACK_ROOT=$SPACK_ROOT"
  hint "scripts/setup.sh --plan sets it up (or set SPACK_ROOT to an existing checkout)"
  finish
fi

# --- 3. the machine's environment -----------------------------------------------
# The same resolution env.sh uses, without activating anything yet.
if ! INCNS_MACHINE=$("$REPO/scripts/machine.sh" 2>/dev/null) || [ -z "$INCNS_MACHINE" ]; then
  fail "no environment selected for this machine"
  hint "set this machine up: scripts/setup.sh --plan (docs/install/spack.md)"
  finish
fi
envdir="$REPO/environments/$INCNS_MACHINE"
if [ ! -f "$envdir/spack.yaml" ] || [ ! -f "$envdir/spack.lock" ]; then
  fail "machine '$INCNS_MACHINE' has no resolved environment ($envdir)"
  hint "scripts/setup.sh --plan --name $INCNS_MACHINE (docs/install/spack.md)"
  finish
fi
ok "machine '$INCNS_MACHINE', environment $envdir"
preset=$(sed -n '1{s/[[:space:]]//g;p;}' "$envdir/preset" 2>/dev/null)
[ -n "$preset" ] || preset=cpu
# Spec names in a spack.yaml's specs: list.
spec_names()
{
  awk '/^  specs:/ { f = 1; next } f && /^  [a-z]/ { f = 0 }
       f && /^[[:space:]]*- / { s = $2; sub(/[@+~%^ ].*/, "", s); print s }' "$1" | sort -u
}
names=$(mktemp)
drift=$(spec_names "$REPO/environments/stack.yaml" > "$names"
        spec_names "$envdir/spack.yaml" | comm -3 "$names" - \
          | awk -F'\t' '$1 != "" { m = m " " $1 } $2 != "" { e = e " " $2 }
                         END { if (m) printf "missing:%s", m; if (e) printf "%sextra:%s", (m ? "; " : ""), e }')
rm -f "$names"
if [ -n "$drift" ]; then
  warn "this machine's environment differs from environments/stack.yaml ($drift)"
  hint "fine until you need those packages; to update: docs/install/spack.md, \"Updating\""
fi
if ! "$REPO/scripts/envlock.sh" check "$envdir"; then
  fail "spack.yaml changed since spack.lock was resolved"
  hint "the lock, and whatever is installed from it, no longer matches spack.yaml;"
  hint "activating can re-resolve the environment and hide the installed stack."
  if [ -f "$envdir/INSTALL" ]; then
    hint "Resolve and install by this environment's own procedure:" \
         "$(grep -v '^[[:space:]]*#' "$envdir/INSTALL" | sed -n '1p')"
  else
    hint "From a host terminal: scripts/setup.sh --yes --name $INCNS_MACHINE (resolves and installs)"
  fi
  hint "(or, if the edit was unintended: git checkout -- $envdir/spack.yaml)"
  finish
fi
ok "spack.yaml and spack.lock agree"

# --- 4. activation + toolchain -------------------------------------------------
out=$(mktemp)
if . "$REPO/scripts/env.sh" >"$out" 2>&1; then
  # In a subshell with core dumps off: a compiler built for another CPU dies
  # with SIGILL, and neither the shell's report nor a core file belongs here.
  if (ulimit -c 0; gcc -dumpversion) >/dev/null 2>&1; then
    ok "toolchain: $(grep -m1 '^\[env\] ' "$out" | sed 's/^\[env\] //')"
  else
    fail "the environment's compiler does not run on this machine ($(command -v gcc))"
    hint "this machine is not one the environment was built for. On a cluster, that is"
    hint "usually a login node: build and run on the compute nodes it targets."
    wrong_machine=1
    rm -f "$out"
    finish
  fi
else
  fail "activating the environment failed:"
  sed 's/^/          /' "$out" | tail -5
  hint "the environment is probably not installed: scripts/setup.sh --plan --name $INCNS_MACHINE"
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
  hint "scripts/setup.sh --plan --name $INCNS_MACHINE (installs the environment)"
  finish
fi

# --- 6-7. the build and a smoke run: only when asked ----------------------------------
if [ "$RUN" -eq 0 ]; then
  if [ -x "$REPO/build/$preset/apps/run_case" ]; then
    ok "build/$preset present (not run; smoke-run it: scripts/doctor.sh --run)"
  else
    warn "no build yet (build/$preset)"
    hint "scripts/doctor.sh --run builds it and smoke-runs a case"
    hint "(on a cluster: on a compute node, not a login node)"
  fi
else
  # --- 6. the build ------------------------------------------------------------------
  if [ "$BUILD" -eq 1 ] || [ ! -x "$REPO/build/$preset/apps/run_case" ]; then
    echo "  ..    building the $preset preset (scripts/build.sh $preset)"
    if "$REPO/scripts/build.sh" "$preset" >"$REPO/build-doctor.log" 2>&1; then
      ok "build/$preset built (log: build-doctor.log)"
      rm -f "$REPO/build-doctor.log"
    else
      fail "the build failed; the end of build-doctor.log:"
      tail -15 "$REPO/build-doctor.log" | sed 's/^/          /'
      finish
    fi
  else
    ok "build/$preset present (rebuild with scripts/build.sh $preset)"
  fi

  # --- 7. a smoke run ------------------------------------------------------------------
  run_dir=$(mktemp -d)
  summary="$run_dir/summary.json"
  start=$(date +%s)
  # $INCNS_MPIEXEC is unquoted on purpose: a launcher with options splits into words.
  # shellcheck disable=SC2086
  if (cd "$run_dir" && ${INCNS_MPIEXEC:-mpirun} -n 2 "$REPO/build/$preset/apps/run_case" \
        "$REPO/cases/cavity.yaml" --summary "$summary") >"$run_dir/run.log" 2>&1 \
     && grep -q '"status": "ok"' "$summary" 2>/dev/null; then
    steps=$(awk -F': *' '/"steps"/ {gsub(/,/,"",$2); print $2; exit}' "$summary")
    ok "smoke run: lid-driven cavity at 2 ranks, $steps steps, $(( $(date +%s) - start )) s"
  else
    fail "the smoke run (cases/cavity.yaml at 2 MPI ranks) failed; its output:"
    tail -12 "$run_dir/run.log" | sed 's/^/          /'
    hint "if ${INCNS_MPIEXEC%% *} refuses to start: is another MPI job holding the cores? scripts/reap.sh --dry"
  fi
  rm -rf "$run_dir"
fi

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
