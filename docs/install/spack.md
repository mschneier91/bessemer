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
  psc_cpu/          PSC Bridges-2 without CUDA: spack.yaml + spack.lock
  <name>/           any other machine: created by scripts/setup.sh (untracked)
    spack.yaml.sha256   checksum of the spack.yaml its lock was resolved from
    hosts               optional: hostname patterns of the machines that use it
    INSTALL             optional: the document it installs by, instead of setup.sh
    preset              optional: its build preset, when that isn't cpu
    mpiexec             optional: its MPI launcher, when that isn't mpirun
  .machine          which environment this checkout uses (untracked, one line)
scripts/
  setup.sh          sets a machine up: shows a plan, then does it with --yes
  machine.sh        picks the environment
  env.sh            activates it and checks the toolchain (every script sources it)
  envlock.sh        checks that a lock was resolved from its spack.yaml
  doctor.sh         checks the whole setup; with --run, builds and smoke-runs a case
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
  its paths are the cluster's own.
- **An environment `setup.sh` can't install** (a cluster where builds and downloads
  happen on different nodes) carries an `INSTALL` file naming its own procedure.
  `setup.sh --plan` prints it and stops; the Bridges-2 environments have one.
- **`spack.yaml.sha256`** is written whenever an environment is resolved and committed
  with the lock. It is how the scripts know a lock is stale: file times say nothing
  after a clone, where every file is equally new.
- **Other machines' environments stay in their checkout.** Commit one only when the project
  maintains that machine.

## Selecting the environment

`scripts/machine.sh` prints the environment name, from the first of:

1. `$INCNS_MACHINE`, if set;
2. `environments/.machine`, written by `setup.sh`;
3. an environment's `hosts` file: patterns for the full hostname (`hostname -f`), one
   per line, for shared clusters with a committed environment.

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
    c:       {require: [gcc@<version>]}
    cxx:     {require: [gcc@<version>]}
    fortran: {require: [gcc@<version>]}
    gcc:
      buildable: false
      externals:
      - spec: gcc@<version> languages:='c,c++,fortran'
        prefix: /path/to/gcc
```

Don't use `all: require: '%gcc'` instead: it breaks externals such as Slurm. On a
workstation, `setup.sh` first builds the GCC its platform file names (`PLATFORM_GCC_SPEC`
in `scripts/platform/`: "Platforms" below) with Spack (the system compiler builds it),
then points the environment at it. On a cluster, `--compiler <prefix>` uses the
site's GCC module instead; `psc_gpu` does this.

### Platforms

What differs between machine types lives in `scripts/platform/<uname -s>.sh` (`linux.sh`,
`darwin.sh`), sourced by `setup.sh`; `INCNS_PLATFORM=linux scripts/setup.sh --plan` shows
another platform's plan. Each file sets the GCC to build (`PLATFORM_GCC_SPEC`), which
compiler the `c`/`cxx` virtuals require (`PLATFORM_C_COMPILER`), the default job count
(`PLATFORM_JOBS`), and may print extra `packages:` entries
(`platform_compiler_externals`). `darwin.sh` documents the contract and why each line is
there. To support a new platform, add a file there. The GCC version is set in exactly
that one place; `<version>` in the YAML above is the version `setup.sh` built, and an
existing environment keeps the GCC in its lock until it is re-resolved.

**macOS (Apple Silicon, found 2026-10-10/11):**
- **`gcc@14.2.0~libsanitizer`, not 14.3.0.** Spack's gcc recipe carries Iain Sandoe's
  aarch64-darwin patches only up to 14.2.0; 14.3.0 stops at configure with
  `Configuration aarch64-apple-darwin24.6.0 not supported`. Linking `libasan.8.dylib`
  crashes Apple's linker (CLT 16.3, `Assertion failed: (entry.toAddend < 0x400000)`).
- **C and C++ are Apple's clang; GCC is the Fortran compiler only** (OpenBLAS, MUMPS,
  ScaLAPACK, gslib, ...). Spack refuses CPython with GCC on macOS (`CPython does not
  compile with GCC on macOS yet, use clang`), and Python is in the stack. The environment
  requires `apple-clang` for `c` and `cxx` (declared external, prefix `/usr`), and
  `scripts/env.sh` accepts a declared `apple-clang` external. This is a deliberate
  exception to "never build with the system compiler" (AGENTS.md rule 1): on a Mac the
  system clang is the toolchain.
- **`ccache` is not installed**, so rebuilds are full rebuilds; `nproc` does not exist.
- **Sanitizers:** GCC's own `libasan` is not built. bessemer is compiled by clang, which
  brings its own runtime, so `scripts/asan.sh` should work there. Not yet run.

The `desktop` environment predates this pattern: there Spack chose to build its GCC as
an ordinary dependency, because the system GCC has no Fortran. The result is the same.

## Setting up a new machine

This is the agent's procedure. The user's permission comes before any install.

1. **Plan:** run `scripts/setup.sh --plan` and show the user the output. It lists each step
   as done or to do: prerequisites, Spack, compiler, environment, resolve, install,
   select, build, check. Tell them what it costs the first time: a few hours of
   compiling (llvm for numba is the longest part) and about 20 GB under `$SPACK_ROOT`; those are a
   workstation's numbers, starting from nothing. It
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
5. **Check:** `setup.sh` ends with `scripts/doctor.sh --run`; a clean doctor means the
   machine is ready.

Options: `--name` (default: what `machine.sh` picks, else the hostname), `--compiler <prefix>`, `--jobs N`, and
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
   the specs (`+cuda cuda_arch=<sm>` on mfem, hypre, libceed). If the MPI has no
   `mpirun` (Cray MPICH starts its ranks through Slurm), name the launcher in
   `environments/<cluster>/mpiexec`, e.g. `srun --overlap -c 1 --cpu-bind=none`; CTest and
   the doctor run it with `-n <ranks>`. `delta` is the worked example
   ([delta.md](delta.md)).
3. `scripts/setup.sh --yes --name <cluster>`, inside an allocation. On a login node
   (a batch scheduler on `PATH`, no job around the shell) `--yes` refuses unless you add
   `--on-login-node`, for sites that allow building there.

When part of the install can't run where `setup.sh` runs (no internet on compute nodes,
no building on login nodes), write the procedure down, name it in
`environments/<cluster>/INSTALL`, and list the cluster's hostname patterns in
`environments/<cluster>/hosts`. [bridges2.md](bridges2.md) is the worked example.
Cluster rules still apply: ask the human
before any batch job or GPU use (AGENTS.md rule 3). Follow the site's policy on building
on login nodes.

### An A100 cluster (checked 2026-10-25)

The plan is to move from Bridges-2's H100s (`psc_gpu`) to an A100 cluster. This is what changes, checked
against the code on 2026-10-25. Bridges-2's own A100 (GPU-dev, `a001`) refuses jobs from this
allocation under every QOS, so none of it has run on an A100 yet.

**The architecture is one number.** Put `cuda_arch=80` on mfem, hypre, libceed and ucx in the new
`spack.yaml`. `scripts/env.sh` reads it into `INCNS_CUDA_ARCH`, which sets `CMAKE_CUDA_ARCHITECTURES`
for our kernels.
- Nothing in `src/` uses H100-only features: no thread-block clusters, TMA or wgmma.
- Two kernels opt into more than 48 KB of dynamic shared memory. Both stay under the A100's 163 KB
  per block: the fused momentum apply needs at most 93 KB (Q8 at 9 points), and the hyperelastic
  by-lines kernels 81 KB (N = 8, 18 stacks; orders 1-7 are compiled).

**The rest of `spack.yaml` is site-specific.** The compiler external (Bridges-2's `b2gpu` GCC build),
`cuda`, `slurm`, and OpenMPI against the site's UCX (or the site's CUDA-aware MPI, as an external) all
change. Keep:
- MFEM v4.10 with mfem/mfem#5464 from a local tree (`scripts/mfem_source.sh`, the `develop:`
  section);
- hypre's `ldflags="-ldl"` on RHEL 8;
- `target=` the oldest ISA among the nodes that build and run.

If the compute nodes are offline, mirror the sources and pip wheels first ([bridges2.md](bridges2.md),
Step 1).

**GPU-aware MPI** is on when MPI is CUDA-aware. `src/util/device.cpp` leaves UCX's `cuda_ipc`
transport out unless the job sets `UCX_TLS`. That was change 42, which saves 1-2.5 GB per GPU near
capacity ([gpu_memory.md](../gpu_memory.md)). Measure the trade again on the new interconnect.

**Memory and speed.** The Taylor-Green defaults take about 1.95 kB per velocity dof
([using.md](../using.md)). An 80 GB A100 holds about what an 80 GB H100 does, a 40 GB one about half.
`memory.max_dofs` trades 6-7 % of the speed for about 90 B per dof at orders 6-7. The throughput and
scaling figures ([throughput.md](../throughput.md), [multi_gpu.md](../multi_gpu.md)) are H100
measurements: the A100 has about half the memory bandwidth, so measure again.

**Validation.** Run the GPU suite at np 1 and 2, one GPU per rank:
`INCNS_DEVICE=cuda ctest -L "smoke|fast" -R "_np1$" -E "^(nse_mms_test|tgv_nse_test)_np" --timeout 3600`,
then `_np2$`. The two excluded tests time out on GPUs. `fsi_test` takes about 32 minutes on one H100
and will take longer on an A100. Then run the CPU tiers at np 1, 2 and 4.

## Updating a machine's environment

To add a package that `stack.yaml` gained, change a variant, or move a pin:

1. In a host terminal, edit `environments/<name>/spack.yaml`.
2. Right away, in the same session: `scripts/setup.sh --yes --name <name>`. It sees that
   `spack.yaml` no longer matches the checksum recorded with the lock, then resolves and
   installs. Or do it by hand: `spack -e environments/<name> concretize --force`,
   `scripts/envlock.sh record environments/<name>`, then
   `spack -e environments/<name> install -j <N>`. An environment with an `INSTALL` file
   updates by the document it names.
3. `scripts/build.sh cpu`, then `scripts/test.sh cpu -L fast`. For a committed
   environment, commit `spack.yaml`, `spack.lock` and `spack.yaml.sha256` together. Re-bless test baselines only
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
