# Bessemer GPU Environment on PSC Bridges-2 (H100)

Spack environment build instructions for the bessemer incompressible Navier–Stokes
solver (MFEM + CUDA) on Bridges-2 GPU nodes. Written after debugging the full
install in July 2026. Read the **Gotchas** section before doing anything — every
item in it cost real time.

**Architecture of the workflow (memorize this):** compute nodes have no internet;
the login node has internet but an unreliable build environment. So *everything
network-facing runs on the login node* (spack mirror, pip wheel download) and
*everything that executes builds runs on a GPU compute node*. Every step below is
labeled with which side of that wall it runs on.

**Agents:** this page describes the procedure; it does not authorize it. Every
step that requests an allocation or submits a job (Steps 2, 3 and 5) spends the
human's allocation and needs their explicit go-ahead first (AGENTS.md rule 3).
`projects` shows what is left of it. Steps 0 and 1 download software and write
configuration (`~/.bashrc`, the Spack checkout, the mirror): show the human the
plan and get a go-ahead for those too, as for any install.

---

## System facts that drive the design

| Fact | Consequence |
|---|---|
| Login/RM nodes are AMD EPYC (zen2); GPU nodes are Intel Xeon | Pin `target=x86_64_v3` (AVX2 baseline both support) so builds are portable regardless of where they happen |
| H100 nodes: 10× HPE Cray 670, 8× H100-SXM5-80GB, 2TB RAM | `cuda_arch=90`. H100 = 2 SU/GPU-hr |
| V100 nodes: Xeon Gold 6248 (Cascade Lake), 40 cores, 8 GPUs | 1 SU/GPU-hr — cheap CPU cores for compiling. `cuda_arch=90` binaries will NOT run on V100 (sm_70) |
| PSC ships per-node-class gcc builds: `b2rm` (RM) and `b2gpu` (GPU) under `/opt/packages/gcc/<version>/` | Compiler must be pinned to the `b2gpu` variant. See Gotchas #1–#3 |
| Compute nodes have **no internet** | Source mirror for Spack; wheel mirror for pip |
| Slurm 22.05.11, prefix `/usr`; CUDA module `cuda/12.6.1` at `/opt/packages/cuda/v12.6.1` | Both registered as externals |
| RHEL8 base: system openssl is 1.1.x, glibc 2.28 | Spack builds its own openssl (curl needs ≥3); hypre needs explicit `-ldl` |

Charging: GPU-shared = 1–4 GPUs, pay per GPU-hour; GPU partition = whole nodes
(multiples of 8). Max walltime 48 h, default 1 h. Short requests backfill faster.

---

## The spack.yaml

Lives in the repo at `environments/psc_gpu/spack.yaml`. The compiler external is
**pinned in this file deliberately** — never run `spack compiler find` for this
environment (Gotchas #1, #6). Note numba is intentionally absent: it installs via
pip wheels (see step 4) because building it in Spack drags in a multi-hour LLVM
source build for zero solver benefit.

The file is the reference, so it isn't copied here:
[environments/psc_gpu/spack.yaml](../../environments/psc_gpu/spack.yaml). The reason
for each pin is in a comment beside it.

Design decisions in brief:

- **`hypre+cuda cuda_arch=90 ldflags="-ldl"`** — BoomerAMG on device for the
  pressure Poisson block. The `-ldl` is REQUIRED on RHEL8 (Gotcha #8). GPU
  BoomerAMG effectively requires PMIS coarsening + ext+i interpolation;
  CPU-tuned Cahouet–Chabard parameters do not carry over 1:1.
- **`suite-sparse~cuda`** — UMFPACK is host-side in MFEM anyway; the CHOLMOD
  CUDA path is a chronic build breaker.
- **`~gdrcopy` on ucx** — the Spack gdrcopy package fails to build (its test
  suite gets no CUDA arch, Gotcha #9), and it only matters if the gdrdrv kernel
  module is loaded on nodes (`lsmod | grep gdrdrv` — it wasn't). UCX keeps full
  CUDA-awareness and GPUDirect RDMA without it; negligible impact for
  halo-exchange / AMG traffic.
- **No openssl external** — system openssl is 1.1.x; curl requires ≥3, so Spack
  builds openssl 3 itself (Gotcha #10).
- **`+shared` on mfem** — the pybind11 module links the shared library.
- **`unify: when_possible`, `reuse: true`** — avoids the numpy/python chain
  dictating versions DAG-wide; keeps installed packages across re-concretizations.
- **`mfem@develop commit=...`** — develop is a moving branch, so the commit is
  pinned in `spack.yaml` (the comment there says why that one). The same pin is
  in `environments/stack.yaml`; move them together.
- **`+mumps`, `^mumps+metis+parmetis~openmp`** — the direct solver; see the
  comment in `spack.yaml`.

---

## Before you start: what is already done?

This page is written as a first install, but most visits are not one. Check
first, and skip what is done:

- `scripts/setup.sh --plan` (changes nothing). If row 5 says "resolve done", the
  `spack.lock` matches `spack.yaml`: do not run `spack concretize -f` again. If
  it says "re-resolve", follow "Updating after `spack.yaml` changes" below.
- `squeue -u $USER`: is an install job already queued or running, for this or
  the other Bridges-2 environment? Then wait for it: they share one Spack. A
  second job, or a re-resolve underneath it, breaks the first.
- The plan's "install done" covers the Spack packages only. Step 4 (numba) is
  separate: `ls environments/psc_gpu/.spack-env/view/lib/python3*/site-packages`
  shows whether numba is there.
- `spack mirror list` shows `local_sources`, and `ls $PROJECT/pip-wheels` shows
  the numba wheels: Step 1's downloads are done.
- `spack -e environments/psc_gpu find` lists what is installed. Only the rest
  gets built, so a top-up is minutes, not hours.

## Fresh-install procedure

### Step 0 — one-time Spack setup [LOGIN NODE]

Spack lives on Ocean, not `$HOME` (quota):

```bash
cd $PROJECT     # your Ocean directory: /ocean/projects/<group>/<user>
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. $PROJECT/spack/share/spack/setup-env.sh     # add to ~/.bashrc
```

Put that last line in `~/.bashrc`. Sourcing it sets `SPACK_ROOT`, which is how
the repository's scripts find this Spack; without it they look in `~/spack`,
their default, and `scripts/setup.sh --plan` offers to clone one there.

### Step 1 — fetch all sources into mirrors [LOGIN NODE]

```bash
export HOSTNAME=$(hostname)        # Gotcha #15
module load gcc/13.3.1-p20240614 cuda/12.6.1
. $PROJECT/spack/share/spack/setup-env.sh
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .

export SPACK_DISABLE_LOCAL_CONFIG=1      # as scripts/env.sh: nothing from ~/.spack
spack bootstrap now                      # clingo etc. while online

# ONLY when `scripts/setup.sh --plan` says resolve is to do (see "Before you
# start"). On a current lock these two lines must be skipped:
spack concretize -f
../../scripts/envlock.sh record .        # the lock now matches spack.yaml

spack mirror create -d $PROJECT/spack-mirror --all
spack mirror list | grep -q local_sources \
  || spack mirror add --scope site local_sources file://$PROJECT/spack-mirror
```

`--scope site` stores the mirror in your own Spack checkout
(`$PROJECT/spack/etc/spack/site/`), not in the tracked `spack.yaml`, which stays
the same for every Bridges-2 user. It is needed once per Spack checkout.
`spack mirror create --all` is needed again after every re-resolve: it downloads
only what the mirror lacks, and without it the install job fails at its first
fetch (no internet on compute nodes).

The warning `mirror ... cannot be used in concretization (no index found)` is
benign — it refers to a *binary* index; this is a source mirror.

**Also mirror the pip wheels now** (the pip analog of `spack mirror create` —
download on login, install later wherever you like, no network needed):

```bash
# Use the environment view's python3 (`which python3` -> .spack-env/view/bin).
# On a truly fresh system the view doesn't exist yet: do this after Step 2,
# on the login node. Wheel downloads need internet but not a working compiler.
python3 -m pip download --only-binary=:all: -d $PROJECT/pip-wheels numba
```

`--only-binary=:all:` guarantees pip fetches prebuilt manylinux wheels only —
it is incapable of triggering a source build. If it errors "no matching
distribution", the python/numpy pins need adjusting; nothing was harmed.

**Checklist on `spack concretize -f` output:**

- `mfem`, `hypre`, `libceed` all `+cuda cuda_arch=90`; hypre shows `ldflags=-ldl`
- `mfem` shows `+mumps` and the commit pinned in `spack.yaml`; `mumps` is in the DAG
- `py-pip` is a root (Step 4 needs it)
- `suite-sparse ~cuda`; `ucx ~gdrcopy`
- everything `target=x86_64_v3`, gcc 13.3.1
- `cuda@12.6.1`, `slurm@22.05.11`, `gcc@13.3.1` flagged `[e]`
- `grep -c b2rm spack.lock` → **0**; `grep -c b2gpu spack.lock` → nonzero
- no `llvm` anywhere in the DAG (if present, py-numba snuck back into specs)

### Step 2 — build [GPU COMPUTE NODE]

Preferred: the batch job in the environment's directory (it survives
disconnects; see Gotcha #7):

```bash
cd $PROJECT/bessemer/environments/psc_gpu
sbatch install.sbatch
```

[install.sbatch](../../environments/psc_gpu/install.sbatch) requests one H100
slice for two hours, loads this node's compiler build and stops at once if it
got the wrong one, installs from `spack.lock`, then runs Step 3's checks and
Step 4's numba install, so one allocation covers Steps 2 to 4. Its output is
`spack-install-<jobid>.out` in that directory.

Or interactively (run inside `tmux` on the login node so disconnects don't kill
it): `interact -p GPU-shared --gres=gpu:h100-80:1 -t 2:00:00`, then the
commands of `install.sbatch` by hand, in its order. The order of its first
lines matters (Gotcha #15).

Flag rationale (all three matter):
- `--dirty`: do NOT sanitize the build environment. Required so perl's direct
  gcc invocations see the module's LD_LIBRARY_PATH (Gotcha #4). Keep the shell
  clean: only the two modules + spack setup loaded.
- `--use-buildcache never`: skip public binary-cache probing that stalls on the
  airgapped node.
- `-j $(nproc)`: the cgroup-limited core share; `$nproc` without parens is an
  empty variable and means unlimited.

Cost: you are charged for the time the job actually runs, not for the walltime
limit, so the limit is only a ceiling. Rebuilding a few changed packages is far
shorter than a from-scratch install.

Walltime expiry mid-build is harmless: resubmit/re-request and rerun the same
install command; Spack resumes at the first uninstalled package. `gmake` is the
canary — if the first package configures past "checking whether the C compiler
works... yes", the toolchain is sound.

### Step 3 — verify the compiled stack [GPU COMPUTE NODE]

`install.sbatch` runs these at its end; by hand:

```bash
which mpicxx                  # -> .spack-env/view/bin
ompi_info | grep -i cuda      # mca cuda support lines present
ucx_info -v                   # cuda listed (gdrcopy absent, by design)
```

### Step 4 — install numba from the wheel mirror [ANY NODE]

`install.sbatch` does this at its end. By hand, it works on compute nodes (no
network needed — wheels are local) or the login node.
Compute node is the conservative choice since the view's python is guaranteed
happy there:

```bash
spack env activate .          # after module loads + setup-env as usual
which python3                 # MUST resolve into .spack-env/view/bin
python3 -c "import numpy; print(numpy.__version__)"
python3 -m pip install --no-index --find-links=$PROJECT/pip-wheels numba
python3 -c "import numba, numpy; print(numba.__version__, numpy.__version__)"
python3 -c "from numba import cfunc, types; print('cfunc OK')"
```

**Watch for:** pip attempting to install its own numpy (i.e., numpy not reported
as "requirement already satisfied"). Two numpys in one env is a subtle disaster.
If it happens: pin `py-numpy@<version-in-numba's-window>` in spack.yaml,
reconcretize, rebuild py-numpy, re-download wheels on login, retry.

**Maintenance note:** the pip layer lives inside the Spack python's
site-packages but outside Spack's bookkeeping. If python is ever uninstalled or
its spec rebuilt, re-run Step 4. This is the single deviation from "spack.yaml
fully describes the environment."

### Step 5 — build bessemer and smoke test [H100 NODE]

```bash
interact -p GPU-shared --gres=gpu:h100-80:1 -t 2:00:00
# module loads + setup-env + env activate, then:
nvidia-smi --query-gpu=name,compute_cap --format=csv   # must report 9.0
cd $PROJECT/bessemer
./scripts/build.sh cuda
./scripts/test.sh cuda -L 'smoke|fast'                 # CPU backend
```

`scripts/build.sh` and `scripts/test.sh` source `scripts/env.sh`, which resolves
the machine and activates the env — no manual `cmake -DMFEM_DIR=...` needed.

**Verified 2026-10-10** (H100 nodes, `./scripts/test.sh cuda -L 'smoke|fast'` with
`INCNS_DEVICE=cuda`): the fast tier passes on the CUDA backend at np 2 (60/60) and at
np 1 except for two tests that time out when they share a card. np 4 and the slow tier
have not been run with one GPU per rank. The details are in
[docs/status.md](../status.md).

How to run the tests here:

- **One GPU per rank.** Two ranks on a card are several times slower, four run out of
  memory. Every np 1 test uses GPU 0, so run at most two tests at a time (`-j 2`).
- **Turn Open MPI's core binding off** when tests run concurrently:
  `export PRTE_MCA_hwloc_default_binding_policy=none`. Otherwise every `mpirun` pins
  to the same first cores.
- **`INCNS_DEVICE=cpu` in the cuda build is not a CPU reference.** Eight fast-tier tests
  fail there inside the CUDA-built hypre and MFEM's cuSPARSE path (`amr_checkpoint_test`,
  `amr_event_test`, `unsteady_mms_test` at np 1 and 2; `point_block_jacobi_test` at np 2
  and 4); all pass on the CUDA backend.

First GPU validation target is a Stokes case (e.g. TGV-Stokes), **not** the
Re_tau=180 channel — that is gated Sprint-2 NSE work.

---

## Updating after `spack.yaml` changes

When `environments/psc_gpu/spack.yaml` changes (a new variant, a moved pin), the
committed `spack.lock` can be older than it. `scripts/setup.sh --plan` and
`scripts/doctor.sh` then report "spack.yaml changed since spack.lock was
resolved". To bring the install up to date:

1. **[LOGIN NODE]** the Step 1 block again: `spack concretize -f`,
   `envlock.sh record`, `spack mirror create --all`. With `reuse: true` only the
   changed packages and what depends on them are new, and the mirror step
   downloads just their sources. Go through the checklist.
2. **[GPU COMPUTE NODE]** the Step 2 job. Spack builds only what isn't installed
   yet, so this is much shorter than the first install.
3. Step 3's checks, then rebuild bessemer (Step 5).
4. Commit `spack.lock` and `spack.yaml.sha256` together (an agent: only when
   the human asks; AGENTS.md rule 8).

---

## The CPU-only environment (`psc_cpu`)

`environments/psc_cpu` is the same stack with no CUDA in it: MFEM, hypre,
libCEED, Open MPI, UCX and hwloc are all `~cuda`. It is for CPU reference runs
and the `cpu` build preset; `psc_gpu` stays the environment for the `cuda`
preset.

- **Same procedure, other directory.** Every step on this page applies with
  `environments/psc_cpu` in place of `environments/psc_gpu`, including its own
  `install.sbatch`. Load only the `gcc` module (no `cuda/12.6.1`); Step 3's CUDA
  checks don't apply. The two
  environments share one Spack and one source mirror, and most packages are the
  same builds, so the second install is short.
- **Selecting it.** Hostnames map Bridges-2 to `psc_gpu`
  (`environments/psc_gpu/hosts`). For the CPU environment set
  `INCNS_MACHINE=psc_cpu` in the shell or job, or make it this checkout's
  default with `echo psc_cpu > environments/.machine`.
- **Building bessemer with it** (Step 5): on a compute node, with
  `INCNS_MACHINE=psc_cpu` set, `./scripts/build.sh cpu` and
  `./scripts/test.sh cpu -L fast`.
- **One preset per environment.** Build trees are named by preset, not by
  environment: build `cpu` (and `cpu-python`) with `psc_cpu` and `cuda` with
  `psc_gpu`. Building the same preset with both in one checkout mixes two MFEMs
  in one build tree.
- **Nodes.** It uses the `b2gpu` compiler build, because GPU nodes are where an
  allocation with only GPU service units builds and runs. The code is plain
  x86_64_v3 with no GPU dependency. For the RM (CPU) partitions, which need a
  Regular Memory allocation, pin the `b2rm` compiler build instead and resolve
  again.
- **Status (2026-10-10):** resolved and mirrored, not installed.
- **Every `~cuda` in its `spack.yaml` is needed**: with `reuse: true` Spack
  would otherwise pick up `psc_gpu`'s CUDA builds from the shared store.

---

## Gotchas (the debugging war log)

### 1. PSC ships per-node-class compiler builds — `b2rm` vs `b2gpu`

`/opt/packages/gcc/v13.3.1-p20240614/` contains `b2rm/` and `b2gpu/` variants;
the module serves the right one per node class. **`spack compiler find` records
an absolute path to whichever variant it sees.** Our first registration (from a
login/RM-side shell) captured `.../b2rm/`, producing `C compiler cannot create
executables` everywhere. Rule: never run `spack compiler find` for this env —
the correct external is pinned in spack.yaml. (`/opt/packages` may resolve to
`/jet/packages` — same filesystem; the `b2gpu` component is what matters.)

### 2. The stale path fossilizes in `spack.lock`

Fixing compiler config is not enough — `spack install` builds from the
lockfile. After any compiler change: `spack clean -m && spack concretize -f`,
then `grep -c b2rm spack.lock` must be 0.

### 3. Spack sanitizes build environments → gcc can't find its own libisl

Even with the right gcc, builds failed with `cc1: error while loading shared
libraries: libisl.so.23`. The toolset gcc relies on the module's
LD_LIBRARY_PATH (`.../b2gpu/lib64`); Spack strips it in build shells.
Interactive tests pass (module loaded) while Spack builds fail — deeply
misleading. Fix (both, in the yaml): `modules:` on the gcc external + explicit
`environment: prepend_path: LD_LIBRARY_PATH`.

### 4. Perl module builds bypass the compiler wrapper → `--dirty` required

perl-data-dumper kept failing with the libisl error even after #3's fix,
because perl embeds the raw gcc path in its Config and invokes it directly —
the external's environment edits didn't reach that invocation. Workaround:
`spack install --dirty` with the gcc module loaded in the shell, which lets the
shell's LD_LIBRARY_PATH flow into all builds. Keep the shell minimal when using
it. (Untested whether a from-scratch rebuild needs --dirty for other perl
modules; assume yes.)

### 5. Don't put `%gcc@...` under `packages:all:require` (Spack v1.x)

Compilers are dependencies now; that constraint applies to *externals* too, and
slurm (buildable:false, no compiler dep) then can't be satisfied: `Cannot build
slurm ... no externals satisfy the request`. Require via the `c:`/`cxx:`/
`fortran:` language virtuals instead.

### 6. `spack compiler find` writes into the *active environment's* spack.yaml

Not `~/.spack/packages.yaml` (platform-scoped files also exist). Use
`spack config blame packages` to locate entries; `spack compiler list` /
`spack compiler rm` to manage. Ensure exactly ONE gcc external block exists.

### 7. Session mortality: disconnects and walltime kill interactive builds

A dropped VSCode/SSH connection can kill the install while the allocation keeps
burning SUs (check `squeue -u <user>`; ssh back into the node — you can ssh to
nodes where you hold a job). A dying Ocean mount or expiring walltime produces
`[Errno 108] ... transaction_lock` and a cascade of phantom package "failures"
that vanish on retry. Defenses: batch jobs for anything long; `tmux` on the
login node wrapping any interactive session; short walltimes + resume semantics.
Stale lock after a crash: `rm .spack-env/transaction_lock` (safe only when no
spack process is running anywhere: check `squeue -u $USER` for an install job
first). Also: only one interactive (`interact`) GPU session may be queued/running
at once (QOSMaxSubmitJobPerUserLimit) — a zombie session blocks new requests.
Batch jobs are not limited this way.

### 8. hypre+cuda needs explicit `-ldl` on RHEL8

Final link of libHYPRE.so dies with hundreds of `undefined reference to
dlopen/dlsym/dlclose` from NVTX. hypre links with `-Wl,-z,defs` (no unresolved
symbols allowed) and RHEL8's glibc 2.28 still keeps dl* in a separate libdl
that must be linked explicitly. Fix: `ldflags="-ldl"` on the hypre spec.
(glibc ≥2.34 merged libdl into libc, so this gotcha is RHEL8-era specific.)

### 9. Spack's gdrcopy package is broken here — and unnecessary

Its test suite invokes nvcc with `arch=compute_none` (no CUDA arch plumbed
through) and fails. gdrcopy only matters if the gdrdrv kernel module is loaded
on compute nodes; it wasn't (`lsmod | grep gdrdrv`). Dropped via `~gdrcopy` on
ucx with negligible performance impact for this workload.

### 10. System openssl is 1.1.x; curl needs ≥3

An `openssl` external pointing at `/usr` breaks curl's configure ("OpenSSL
3.0.0 or upper required"). Don't register an openssl external; let Spack build
openssl 3.

### 11. numba would drag in a multi-hour LLVM source build

py-numba → llvmlite → full LLVM from source (the system llvm@17 external
doesn't satisfy llvmlite's version pin). LLVM is the single longest build in
the DAG by a wide margin. Avoided entirely by pip-installing numba's manylinux
wheel (llvmlite ships LLVM statically embedded). See Step 4; the wheel-mirror
download/install split mirrors the Spack source-mirror philosophy: fetch on
login, install offline. `--only-binary=:all:` guarantees no compilation can
occur.

### 12. Scheduling economics

- Compiling doesn't need H100s: nvcc needs only the toolkit, and cuda_arch is a
  compile-time flag. `install.sbatch` still asks for one H100, because that is
  the configuration that has been run (2026-10-10: a five-package top-up took
  10 minutes); a V100 build of the current stack is untried. On paper V100-32
  slices (1 SU/GPU-hr, 24 nodes) are the cheaper cores;
  v100-16 is the same price but only 9 nodes (often the slower queue). CPU
  share scales with GPU count (~5 cores/GPU on V100 nodes; more on H100).
- Check availability before queueing: `sinfo -N -p GPU-shared -o "%N %G %t %C"`
  and `squeue -p GPU-shared -t PENDING -o "%u %b %l"`.
- Short walltimes backfill dramatically faster; resume semantics make repeated
  short sessions free.
- Running on V100s would require a separate `cuda_arch=70` environment
  (including hypre). Compiling on them produces identical x86_64_v3 binaries.

### 13. A CUDA-enabled MFEM emits nvcc-syntax linker flags

Every link fails with `g++: error: unrecognized command-line option
'-Xlinker=-rpath,...'`. MFEM built with CUDA records `MFEM_CXX = .../nvcc` in
`share/mfem/config.mk`, and therefore sets `MFEM_XLINKER = -Xlinker=`, spelling
every rpath flag in nvcc syntax. `cmake/FindMFEM.cmake` reads config.mk and
passed those tokens through verbatim onto a link line driven by `mpicxx` (g++),
which rejects that form. Fixed in `6f8fca6`: FindMFEM.cmake now translates
`-Xlinker=ARG` → `-Wl,ARG`. A CPU MFEM already emits `-Wl,`, so the old verbatim
passthrough was correct only by accident — which is why this never reproduced on
the desktop, and why it will resurface on the next CUDA cluster if that
translation is ever removed.

### 14. `ccache` is absent from the env; the build goes without it

The `cpu` preset (which `cuda` inherits) sets
`CMAKE_CXX_COMPILER_LAUNCHER=ccache`, and neither Bridges-2 environment includes
ccache. The first builds here died instantly, every compile line exiting
`code=127` with `/bin/sh: ccache: command not found`. `CMakeLists.txt` now drops
the launcher when it isn't on `PATH` (it prints "building without it"), so
`scripts/build.sh` works as it is. The cost is that every rebuild is a cold
build. To get warm rebuilds, add `ccache` to the environment's `spack.yaml` and
update it (it is a source fetch, so the login-node step comes first).

### 15. The gcc module needs `HOSTNAME` exported, and it must be this node's

The gcc modulefile chooses `b2rm` or `b2gpu` from the `HOSTNAME` environment
variable. Two ways that goes wrong:

- **Unset:** only `/etc/profile` exports it, so only login shells have it. A
  command sent over ssh (`ssh bridges2 '<cmd>'`, which is how an agent on
  another machine works) gets a non-login shell: `module load gcc/...` fails
  with `attempt to index a nil value (local 'hostname')` and every later
  command silently uses the system gcc 8.5.
- **Stale:** bash keeps an inherited `HOSTNAME` instead of recomputing it, and
  `sbatch` passes the submitting shell's environment to the job. A job
  submitted from a login node can therefore load the `b2rm` build on a GPU
  node (not yet confirmed on a compute node; same family as Gotcha #1).

Fix for both: `export HOSTNAME=$(hostname)` immediately before the
`module load` line, as every block in this document does. In a batch job the
submitting shell's loaded gcc module is inherited as well, so `install.sbatch`
first unloads it (while `HOSTNAME` still names the login node, so the right
paths are removed) and then checks that `gcc` is the `b2gpu` build. Check with
`which gcc`: it must be under `/opt/packages/gcc/`, not `/usr/bin`.

---

## Quick-reference: rebuild from absolute zero

For a machine with nothing installed. Anywhere else, start at "Before you
start": this block re-resolves the lock, which must not happen under a queued
or running install job.

```bash
# LOGIN NODE ---------------------------------------------------------------
cd $PROJECT && git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. $PROJECT/spack/share/spack/setup-env.sh
export HOSTNAME=$(hostname)        # Gotcha #15
module load gcc/13.3.1-p20240614 cuda/12.6.1
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .
export SPACK_DISABLE_LOCAL_CONFIG=1
spack bootstrap now
spack concretize -f                      # verify checklist; no b2rm; no llvm
../../scripts/envlock.sh record .
spack mirror create -d $PROJECT/spack-mirror --all
spack mirror add --scope site local_sources file://$PROJECT/spack-mirror
python3 -m pip download --only-binary=:all: -d $PROJECT/pip-wheels numba

# GPU COMPUTE NODE: the install job (Steps 2-4). Ask the human first. -------
sbatch install.sbatch                    # output: spack-install-<jobid>.out
```