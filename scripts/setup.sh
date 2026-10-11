#!/bin/sh
# scripts/setup.sh -- set bessemer up on this machine: Spack, a compiler, the
# machine's Spack environment, the build, a check. Meant to be driven by a
# coding agent with the user's permission (AGENTS.md, "Working with the user";
# how it all fits together: docs/install/spack.md).
#
#   scripts/setup.sh [--plan]   print what it would do; changes nothing (default)
#   scripts/setup.sh --yes      do it (hours the first time)
#
# Options (give the same ones to --plan and --yes):
#   --name NAME        environment name: environments/NAME/ (default: the one
#                      scripts/machine.sh picks, else this machine's hostname)
#   --compiler PREFIX  use an existing GCC with gfortran installed at PREFIX
#                      (e.g. a cluster module's) instead of building gcc@14.3.0
#                      (gcc@14.2.0 on an Apple Silicon Mac)
#   --env-only         stop once environments/NAME/spack.yaml exists, to add
#                      machine settings (cluster MPI, CUDA) before resolving
#   --jobs N           parallel build jobs (default: all cores)
#   --on-login-node    let --yes install and build on a cluster's login node
#                      (only where the site allows it)
#
# Every step checks whether it's already done, so rerunning --yes resumes where
# a previous run stopped. The Spack steps refuse to run inside a sandboxed
# editor (a Flatpak VSCode): there they must go to a host terminal.

REPO=$(cd "$(dirname "$0")/.." && pwd)
INCNS_REPO_ROOT=$REPO
export INCNS_REPO_ROOT
GCC_SPEC=gcc@14.3.0
# Spack's gcc recipe has no aarch64-darwin patches for 14.3.0 (only up to 14.2.0):
# the build stops at configure with "Configuration aarch64-apple-darwin... not
# supported". Apple Silicon Macs get 14.2.0, which Spack prefers there.
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
  GCC_SPEC=gcc@14.2.0
fi
SPACK_URL=https://github.com/spack/spack.git
: "${SPACK_ROOT:=$HOME/spack}"

MODE=plan
NAME=
COMPILER=
ENV_ONLY=0
LOGIN_OK=0
JOBS=$(nproc 2>/dev/null || echo 4)
while [ $# -gt 0 ]; do
  case "$1" in
    --plan) MODE=plan ;;
    --yes) MODE=yes ;;
    --name) NAME=$2; shift ;;
    --compiler) COMPILER=$2; shift ;;
    --env-only) ENV_ONLY=1 ;;
    --jobs) JOBS=$2; shift ;;
    --on-login-node) LOGIN_OK=1 ;;
    -h | --help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "setup.sh: unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$NAME" ]; then
  # scripts/machine.sh is the single source of truth for an existing choice:
  # $INCNS_MACHINE, environments/.machine, a committed environment's hosts file.
  NAME=$("$REPO/scripts/machine.sh" 2>/dev/null) || NAME=
  if [ -z "$NAME" ]; then
    NAME=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo machine)
    NAME=$(echo "$NAME" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '_')
  fi
fi
ENVDIR=$REPO/environments/$NAME
SANDBOX=0
[ -f /.flatpak-info ] && SANDBOX=1

# --- what is already done --------------------------------------------------------
missing=
for t in python3 git make patch tar gzip unzip bzip2 xz file curl; do
  command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
done

have_spack=0
[ -f "$SPACK_ROOT/share/spack/setup-env.sh" ] && have_spack=1

have_env=0
[ -f "$ENVDIR/spack.yaml" ] && have_env=1
env_kind=local
if [ "$have_env" -eq 1 ] && git -C "$REPO" ls-files --error-unmatch \
     "environments/$NAME/spack.yaml" >/dev/null 2>&1; then
  env_kind=committed
fi

# The compiler: only needed to create a new environment.
GCC_PREFIX=
compiler_state=done
if [ "$have_env" -eq 1 ]; then
  compiler_msg="the environment already declares its compiler"
elif [ -n "$COMPILER" ]; then
  GCC_PREFIX=$COMPILER
  if [ -x "$COMPILER/bin/gcc" ] && [ -x "$COMPILER/bin/g++" ] \
     && [ -x "$COMPILER/bin/gfortran" ]; then
    compiler_msg="GCC $("$COMPILER/bin/gcc" -dumpfullversion) at $COMPILER"
  else
    compiler_state=FAIL
    compiler_msg="$COMPILER/bin needs gcc, g++ and gfortran"
  fi
else
  if [ "$have_spack" -eq 1 ]; then
    GCC_PREFIX=$("$SPACK_ROOT/bin/spack" location -i "$GCC_SPEC" 2>/dev/null | head -n 1)
  fi
  if [ -n "$GCC_PREFIX" ] && [ -x "$GCC_PREFIX/bin/gfortran" ]; then
    compiler_msg="$GCC_SPEC, built by Spack at $GCC_PREFIX"
  else
    GCC_PREFIX=
    compiler_state="to do"
    compiler_msg="build $GCC_SPEC with Spack (the system compiler builds it; about an hour)"
    command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 \
      || missing="$missing a-C-compiler"
    command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 \
      || missing="$missing a-C++-compiler"
  fi
fi

resolve_state="to do"
resolve_msg="spack concretize (minutes)"
if [ -f "$ENVDIR/spack.lock" ]; then
  if "$REPO/scripts/envlock.sh" check "$ENVDIR"; then
    resolve_state=done
    resolve_msg="spack.lock is current"
  else
    resolve_msg="re-resolve: spack.yaml changed since spack.lock was resolved (minutes)"
  fi
fi

install_state="to do"
install_msg="spack install -j $JOBS (hours; about 20 GB under $SPACK_ROOT)"
if [ -f "$ENVDIR/.spack-env/view/include/mfem.hpp" ] && [ "$resolve_state" = done ]; then
  install_state=done
  install_msg="installed (the view has MFEM)"
fi

if [ -f "$ENVDIR/INSTALL" ]; then
  if [ "$install_state" = done ]; then
    install_msg="Spack packages installed; its procedure may have further steps (below)"
  else
    install_msg="by this environment's own procedure (below)"
  fi
fi

# Selected already when scripts/machine.sh picks this environment by itself
# ($INCNS_MACHINE, environments/.machine, or a hosts file).
select_state="to do"
select_msg="environments/.machine = $NAME"
if [ "$("$REPO/scripts/machine.sh" 2>/dev/null)" = "$NAME" ]; then
  select_state=done
  select_msg="scripts/machine.sh picks $NAME on this machine"
fi

# The build preset this environment is for (CMakePresets.json): cpu, unless
# environments/NAME/preset names another one.
PRESET=$(sed -n '1{s/[[:space:]]//g;p;}' "$ENVDIR/preset" 2>/dev/null)
[ -n "$PRESET" ] || PRESET=cpu

build_state="to do"
build_msg="scripts/build.sh $PRESET (about 15 min)"
if [ -x "$REPO/build/$PRESET/apps/run_case" ]; then
  if [ "$install_state" = done ] && [ "$select_state" = done ]; then
    build_state=done
    build_msg="build/$PRESET exists (rebuild: scripts/build.sh $PRESET)"
  else
    build_msg="scripts/build.sh $PRESET: rebuild against the new environment"
  fi
fi
check_msg="scripts/doctor.sh --run"

# An environment with its own procedure: this script reports, and runs nothing.
if [ -f "$ENVDIR/INSTALL" ]; then
  [ "$build_state" = done ] || build_msg="scripts/build.sh $PRESET, where its procedure says (below)"
  check_msg="scripts/doctor.sh; --run where its procedure says (below)"
fi

# --- the plan --------------------------------------------------------------------
row() { printf '  %-2s %-13s %-7s %s\n' "$1" "$2" "$3" "$4"; }
echo "bessemer setup for machine '$NAME' (environments/$NAME/)"
echo
if [ -n "$missing" ]; then
  row 1 prerequisites FAIL "missing:$missing"
else
  row 1 prerequisites ok "git, python3, make, patch, tar and archivers, file, curl"
fi
if [ "$have_spack" -eq 1 ]; then
  row 2 Spack done "$SPACK_ROOT"
else
  row 2 Spack "to do" "git clone $SPACK_URL into $SPACK_ROOT"
fi
row 3 compiler "$compiler_state" "$compiler_msg"
if [ "$have_env" -eq 1 ]; then
  row 4 environment done "environments/$NAME/spack.yaml ($env_kind)"
else
  row 4 environment "to do" "create environments/$NAME/spack.yaml: environments/stack.yaml + the compiler"
fi
if [ "$ENV_ONLY" -eq 1 ]; then
  row 5 "" stop "--env-only: edit environments/$NAME/spack.yaml, then rerun without it"
else
  row 5 resolve "$resolve_state" "$resolve_msg"
  row 6 install "$install_state" "$install_msg"
  row 7 select "$select_state" "$select_msg"
  row 8 build "$build_state" "$build_msg"
  row 9 check always "$check_msg"
fi
echo

spack_todo=0
{ [ "$have_spack" -eq 0 ] || [ "$compiler_state" = "to do" ] || [ "$have_env" -eq 0 ] \
  || { [ "$ENV_ONLY" -eq 0 ] && { [ "$resolve_state" != done ] || [ "$install_state" != done ]; }; }; } \
  && spack_todo=1

blocked=0
if [ -n "$missing" ]; then
  echo "Blocked: install the missing tools with the system package manager first"
  echo "(apt, dnf, zypper, pacman, brew ...; that may need the machine's administrator)."
  echo "On Debian/Ubuntu, for example, the packages are:"
  echo "  build-essential git python3 patch unzip bzip2 xz-utils file curl"
  blocked=1
fi
if [ "$compiler_state" = FAIL ]; then
  echo "Blocked: --compiler $COMPILER is not a usable GCC."
  blocked=1
fi
own_procedure=0
if [ -f "$ENVDIR/INSTALL" ]; then
  echo "Environment '$NAME' has its own install procedure, which this script doesn't run:"
  grep -v '^[[:space:]]*#' "$ENVDIR/INSTALL" | sed -e '/^[[:space:]]*$/d' -e 's/^/  /'
  echo "Rows marked done above are done: skip those parts of the procedure. In particular,"
  echo "don't resolve again when row 5 says done: that rewrites a spack.lock that is current."
  own_procedure=1
elif [ "$SANDBOX" -eq 1 ] && [ "$spack_todo" -eq 1 ]; then
  echo "This is a sandboxed editor (Flatpak): the Spack steps must run in a host terminal."
  echo "Give the user this command to run there:"
  echo "  cd $REPO && scripts/setup.sh --yes$( [ -n "${COMPILER}" ] && printf ' --compiler %s' "$COMPILER")$( [ "$ENV_ONLY" -eq 1 ] && printf ' --env-only') --name $NAME"
  blocked=1
fi

# A batch scheduler on PATH and no job around this shell: a cluster's login node.
login_node=0
if [ -z "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}${FLUX_JOB_ID:-}" ]; then
  for _s in sbatch qsub bsub; do
    command -v "$_s" >/dev/null 2>&1 && login_node=1
  done
fi
if [ "$login_node" -eq 1 ] && [ "$own_procedure" -eq 0 ] && [ "$LOGIN_OK" -eq 0 ]; then
  echo "This looks like a cluster login node (a batch scheduler, and no job around this"
  echo "shell). --yes compiles for hours on all cores, then builds and runs a smoke case:"
  echo "run it inside an allocation, or add --on-login-node where the site allows that."
  echo "If compute nodes have no internet, the install needs its own procedure"
  echo "(docs/install/spack.md, \"Clusters\")."
  [ "$MODE" = yes ] && blocked=1
fi

if [ "$MODE" = plan ]; then
  if [ "$own_procedure" -eq 1 ]; then
    echo "Nothing has been changed."
  elif [ "$blocked" -eq 0 ]; then
    echo "Nothing has been changed. To do it: scripts/setup.sh --yes (same options)."
  fi
  exit "$blocked"
fi
[ "$blocked" -eq 0 ] || exit 1
if [ "$own_procedure" -eq 1 ]; then
  echo "setup.sh: --yes does nothing for '$NAME': follow the procedure named above." >&2
  exit 1
fi

# --- doing it ----------------------------------------------------------------------
die() { echo "setup.sh: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }
run() { echo "  \$ $*"; "$@"; }

if [ "$have_spack" -eq 0 ]; then
  step "2 Spack: cloning into $SPACK_ROOT"
  run git clone -c feature.manyFiles=true --depth=2 "$SPACK_URL" "$SPACK_ROOT" \
    || die "cloning Spack failed"
fi
# shellcheck disable=SC1091
. "$SPACK_ROOT/share/spack/setup-env.sh" || die "cannot load Spack from $SPACK_ROOT"

if [ "$compiler_state" = "to do" ]; then
  step "3 compiler: building $GCC_SPEC (the system compiler builds it)"
  run spack compiler find || die "spack compiler find failed"
  run spack install -j "$JOBS" "$GCC_SPEC" || die "building $GCC_SPEC failed"
  GCC_PREFIX=$(spack location -i "$GCC_SPEC" | head -n 1)
  [ -x "$GCC_PREFIX/bin/gfortran" ] || die "$GCC_SPEC at '$GCC_PREFIX' has no gfortran"
fi

if [ "$have_env" -eq 0 ]; then
  step "4 environment: creating environments/$NAME/spack.yaml"
  gcc_version=$("$GCC_PREFIX/bin/gcc" -dumpfullversion)
  mkdir -p "$ENVDIR" || die "cannot create $ENVDIR"
  {
    echo "# environments/$NAME/spack.yaml -- this machine's bessemer environment."
    echo "# Generated by scripts/setup.sh on $(date +%Y-%m-%d) from environments/stack.yaml;"
    echo "# the machine settings are at the end. Edit it only from a host terminal and"
    echo "# resolve + install right after (docs/install/spack.md)."
    sed -n '/^spack:/,$p' "$REPO/environments/stack.yaml"
    cat <<EOF
    # --- machine settings (scripts/setup.sh) ---------------------------------
    # The compiler. Spack 1.x treats compilers as dependencies: requiring the
    # c/cxx/fortran virtuals pins every package to this GCC (a %gcc under
    # all:require would break externals). Declared external, so the requirement
    # doesn't apply to building the compiler itself.
    c:
      require:
      - gcc@$gcc_version
    cxx:
      require:
      - gcc@$gcc_version
    fortran:
      require:
      - gcc@$gcc_version
    gcc:
      buildable: false
      externals:
      - spec: gcc@$gcc_version languages:='c,c++,fortran'
        prefix: $GCC_PREFIX
        extra_attributes:
          compilers:
            c: $GCC_PREFIX/bin/gcc
            cxx: $GCC_PREFIX/bin/g++
            fortran: $GCC_PREFIX/bin/gfortran
EOF
  } >"$ENVDIR/spack.yaml" || die "cannot write $ENVDIR/spack.yaml"
  echo "  wrote $ENVDIR/spack.yaml"
fi

if [ "$ENV_ONLY" -eq 1 ]; then
  echo
  echo "Stopped (--env-only). Add the machine settings to environments/$NAME/spack.yaml"
  echo "(docs/install/spack.md, \"Clusters\"), then: scripts/setup.sh --yes --name $NAME"
  exit 0
fi

# The environment resolves and installs from its own spack.yaml only, exactly as
# scripts/env.sh activates it (no ~/.spack settings leaking in).
SPACK_DISABLE_LOCAL_CONFIG=1
export SPACK_DISABLE_LOCAL_CONFIG

if [ "$resolve_state" != done ]; then
  step "5 resolve: spack concretize"
  run spack -e "$ENVDIR" concretize --force || die "resolving the environment failed"
  run "$REPO/scripts/envlock.sh" record "$ENVDIR" || die "cannot record the spack.yaml checksum"
  if grep -q freedesktop "$ENVDIR/spack.lock"; then
    die "spack.lock targets a Flatpak runtime OS: this ran inside a sandbox (docs/install/spack.md, trap 1)"
  fi
fi

if [ "$install_state" != done ]; then
  step "6 install: spack install -j $JOBS (hours the first time)"
  run spack -e "$ENVDIR" install -j "$JOBS" || die "spack install failed; rerun --yes to resume"
fi

if [ "$select_state" != done ]; then
  step "7 select: environments/.machine = $NAME"
  echo "$NAME" >"$REPO/environments/.machine" || die "cannot write environments/.machine"
fi

unset SPACK_DISABLE_LOCAL_CONFIG
INCNS_MACHINE=$NAME
export INCNS_MACHINE
if [ "$build_state" != done ]; then
  step "8 build: scripts/build.sh $PRESET"
  run "$REPO/scripts/build.sh" "$PRESET" || die "the build failed"
fi

step "9 check: scripts/doctor.sh --run"
"$REPO/scripts/doctor.sh" --run
