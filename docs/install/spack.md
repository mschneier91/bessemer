# How the Spack setup works

Everything bessemer builds with comes from [Spack](https://spack.io), compiler included:
GCC, MPI, CMake, MFEM and its solver libraries, Python with NumPy and numba. Nothing from
the system toolchain is used for project code, so a build behaves the same on every
machine. This page is the map: which file does what, how a machine is set up and selected,
how activation checks the toolchain, and how to change an environment without breaking it.

## The pieces

```
environments/
  stack.yaml        what bessemer needs, for any machine (the portable spec list)
  desktop/          the maintainer's workstation: spack.yaml + spack.lock (committed)
  psc_gpu/          PSC Bridges-2 H100 nodes: spack.yaml + spack.lock (committed)
  <name>/           any other machine: created by scripts/setup.sh (untracked)
  .machine          which environment this checkout uses (untracked, one line)
scripts/
  setup.sh          sets a machine up: shows a plan, then does it with --yes
  machine.sh        picks the environment
  env.sh            activates it and checks the toolchain (every script sources it)
  doctor.sh         checks the whole setup and smoke-runs a case
```

- **`stack.yaml`** lists the packages and variants bessemer needs, with the reasons for
  each pin in comments. It is not an environment: it carries no compiler, CPU target or
  OS.
- **A machine's environment** (`environments/<name>/spack.yaml`) is `stack.yaml` plus that
  machine's settings: its compiler, and on clusters the system MPI, CUDA and the scheduler.
  Its `spack.lock` records the exact resolution: every version, variant, compiler and CPU
  target.
- **Committed environments** are the maintainer's known-good machines. Their locks are the
  reference when something resolves differently elsewhere: `environments/desktop/spack.lock`
  says exactly which versions worked. `psc_gpu` also works for any Bridges-2 user, since
  its paths are the cluster's own; it installs by [bridges2.md](bridges2.md), not by
  `setup.sh`.
- **Other machines' environments stay in their checkout.** Commit one only when the project
  maintains that machine.

## Selecting the environment

`scripts/machine.sh` prints the environment name, from the first of:

1. `$INCNS_MACHINE`, if set;
2. `environments/.machine`, written by `setup.sh`;
3. hostname patterns of shared clusters with a committed environment (`br0*`/`w0*`:
   `psc_gpu`).

With none of these it fails and points to `scripts/setup.sh --plan`. There is no default
machine: running someone else's environment on your machine fails in confusing ways.

## Activation (`scripts/env.sh`)

Every script sources `env.sh`; never activate an environment by hand. It:

1. finds Spack at `$SPACK_ROOT` (default `~/spack`);
2. sets `SPACK_DISABLE_LOCAL_CONFIG=1`, so only the environment's own `spack.yaml` and
   Spack's built-in defaults apply, and nothing in `~/.spack` leaks in;
3. activates `environments/<name>/`, whose view puts the stack on `PATH`;
4. puts the compiler that MPI was built with on `PATH` (read from `mpicc -show`) and
   exports `MFEM_DIR`;
5. **checks the toolchain:** `gcc`, `g++`, `cmake`, `mpicc`, `mpicxx` and `mpirun` must
   resolve inside `$SPACK_ROOT`, the environment's view, or a compiler/MPI/CMake prefix
   the environment's `spack.yaml` declares as an external. Anything else aborts with
   "System-toolchain leak".

It prints `[env] machine=<name> gcc=<version> mfem=<path>` when everything is in place.

## The compiler

Spack 1.x treats compilers as dependencies. A machine's environment pins one GCC with
Fortran (MUMPS and ScaLAPACK need it) through requirements on the language virtuals, and
declares that GCC as an external:

```yaml
  packages:
    c:       {require: [gcc@14.3.0]}
    cxx:     {require: [gcc@14.3.0]}
    fortran: {require: [gcc@14.3.0]}
    gcc:
      buildable: false
      externals:
      - spec: gcc@14.3.0 languages:='c,c++,fortran'
        prefix: /path/to/gcc
```

Don't use `all: require: '%gcc'` instead: it breaks externals such as Slurm. On a
workstation, `setup.sh` first builds `gcc@14.3.0` with Spack (the system compiler builds
it), then points the environment at it. On a cluster, `--compiler <prefix>` uses the
site's GCC module instead; `psc_gpu` does this.

The `desktop` environment predates this pattern: there Spack chose to build gcc 14.3.0 as
an ordinary dependency, because the system GCC has no Fortran. The result is the same.

## Setting up a new machine

This is the agent's procedure. The user's permission comes before any install.

1. **Plan:** run `scripts/setup.sh --plan` and show the user the output. It lists each step
   as done or to do: prerequisites, Spack, compiler, environment, resolve, install,
   select, build, check. Tell them what it costs the first time: a few hours of
   compiling (llvm for numba is the longest part) and about 20 GB under `~/spack`. It
   changes nothing.
2. **Blocked on prerequisites:** the system tools (`git`, `python3`, `make`, a C/C++
   compiler…) come from the system package manager, which usually needs an administrator.
   Give the user the command; don't try to work around a missing tool.
3. **Sandboxed editor:** if the plan says so, the Spack steps must run in a host
   terminal (trap 1). Give the user the exact `setup.sh --yes …` line it prints.
4. **Do it:** with the user's go-ahead, run `scripts/setup.sh --yes` (same options) in the
   background and report progress. Every step checks whether it's already done, so
   rerunning `--yes` after a failure resumes. Report a failure verbatim. Don't "fix" it by
   pinning or swapping versions or by using a system tool: that's the user's decision.
5. **Check:** `setup.sh` ends with `scripts/doctor.sh`; a clean doctor means the machine is
   ready.

Options: `--name` (default: the hostname), `--compiler <prefix>`, `--jobs N`, and
`--env-only` to stop after writing `environments/<name>/spack.yaml`.

Status (2026-10-09): `--plan`, the generated `spack.yaml` (Spack parses it; `env.sh`'s
toolchain check accepts its compiler) and the resume logic are tested. The full `--yes`
path has not yet run on a fresh machine. If it fails there, the fix belongs in
`setup.sh`.

## Clusters

Use the site's compiler and MPI rather than building them: MPI must match the
interconnect and the scheduler.

1. `scripts/setup.sh --env-only --name <cluster> --compiler <the GCC module's prefix>`.
2. Add the site's packages to `environments/<cluster>/spack.yaml` under `packages:`, as
   externals with `buildable: false`: the MPI (or build OpenMPI against the site's UCX and
   Slurm, as `psc_gpu` does), `slurm`, and `cuda` for GPUs. Then add the GPU variants to
   the specs (`+cuda cuda_arch=<sm>` on mfem, hypre, libceed).
3. `scripts/setup.sh --yes --name <cluster>`.

[bridges2.md](bridges2.md) is the worked example (and on Bridges-2 itself, the procedure
to follow). Cluster rules still apply: ask the human
before any batch job or GPU use (AGENTS.md rule 3). Follow the site's policy on building
on login nodes.

## Updating a machine's environment

To add a package that `stack.yaml` gained, change a variant, or move a pin:

1. In a host terminal, edit `environments/<name>/spack.yaml`.
2. Right away, in the same session: `scripts/setup.sh --yes --name <name>`. It sees that
   the lock is older than `spack.yaml`, then resolves and installs. Or do it by hand:
   `spack -e environments/<name> concretize --force` followed by
   `spack -e environments/<name> install -j <N>`.
3. `scripts/build.sh cpu`, then `scripts/test.sh cpu -L fast`. Re-bless test baselines only
   with a stated reason ([developing.md](../developing.md) §1).

`scripts/doctor.sh` warns when a machine's spec list has drifted from `stack.yaml`. That's
fine until you need the missing package. A change to `stack.yaml` reaches each machine
only when someone updates that machine.

**A full refresh** moves everything, MFEM included, to current versions. The maintainer
does one every couple of months: [desktop.md](desktop.md).

**Pins** (currently MFEM, at `1c19aba`, because of an upstream regression) are commented
where they're set, in `stack.yaml` and every machine's `spack.yaml`. Drop a pin in all of
them once the upstream fix lands.

## Two traps

1. **Run Spack from a host terminal, never inside a sandboxed editor.** A Flatpak VSCode,
   and any agent running inside it, sees the Flatpak runtime's OS and glibc, not the
   host's. Spack then resolves packages for the wrong OS (`os=org.freedesktop.platform25`
   in the lock), and links fail where they mix with the host-built stack. `ls
   /.flatpak-info` tells you: if the file exists, you're in a sandbox. Builds and runs are
   fine there; Spack environment work is not.
2. **Spack re-resolves an environment on activation once `spack.yaml` is newer than its
   lock.** Editing `spack.yaml` and then running anything that sources `env.sh` writes a
   new lock and points the view at packages that aren't built yet. `mpirun` and MFEM
   vanish and `env.sh` aborts until the install finishes. So edit `spack.yaml` only
   immediately before resolving and installing. The doctor refuses to activate in that
   state and says what to run.

## When `env.sh` fails

| Message | Meaning | Fix |
|---|---|---|
| `no environment selected for this machine` | no `$INCNS_MACHINE`, no `environments/.machine`, unknown host | `scripts/setup.sh --plan` (or `echo <name> > environments/.machine` for an existing one) |
| `no spack env for machine '<name>'` | the name doesn't match a directory in `environments/` | `scripts/setup.sh --plan --name <name>` |
| `spack not found at SPACK_ROOT=…` | no Spack checkout there | `scripts/setup.sh --plan`, or set `SPACK_ROOT` |
| `failed to activate` / tools missing | the environment isn't installed, or trap 2 | `scripts/setup.sh --plan --name <name>` shows which step is missing |
| `System-toolchain leak` | a tool resolves outside Spack | install the environment; never fix this by editing `env.sh` |
