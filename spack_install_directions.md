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

```yaml
spack:
  specs:
    - mfem@develop+cuda cuda_arch=90 +gslib+lapack+libceed+mpi+shared+suite-sparse
      ^gslib@1.0.9 cflags="-fPIC"
      ^hypre+cuda cuda_arch=90 ldflags="-ldl"
      ^libceed+cuda cuda_arch=90
      ^suite-sparse~cuda

    - cmake
    - ninja
    - openmpi+cuda fabrics=ucx schedulers=slurm
      ^ucx+cuda~gdrcopy+verbs+rc+ud+dc+mlx5_dv cuda_arch=90

    - googletest
    - astyle
    - yaml-cpp

    # Python interface: our own pybind11 bindings over the incns Case surface
    # (PyMFEM is never used). numba provides the @cfunc fast path for
    # user-defined ICs/BCs/forcing -- a raw C function pointer the bindings wrap
    # as an mfem::Coefficient. No mpi4py: Python is a pure SPMD driver.
    # numba is NOT built by Spack: it is pip-installed as a manylinux wheel
    # (llvmlite ships LLVM statically) to avoid a multi-hour LLVM source build.
    # Keep py-numpy pinned inside numba's supported window.
    - python@3.12
    - py-pybind11
    - py-numpy

  packages:
    all:
      require:
      - target=x86_64_v3
      providers:
        mpi: [openmpi]
    # Spack v1.x: compilers are dependencies. Do NOT put "%gcc" under
    # all:require (breaks externals like slurm). Use language virtuals:
    c:
      require:
      - gcc@13.3.1
    cxx:
      require:
      - gcc@13.3.1
    fortran:
      require:
      - gcc@13.3.1
    gcc:
      externals:
      - spec: gcc@13.3.1 languages:='c,c++,fortran'
        prefix: /opt/packages/gcc/v13.3.1-p20240614/b2gpu
        modules:
        - gcc/13.3.1-p20240614
        extra_attributes:
          compilers:
            c: /opt/packages/gcc/v13.3.1-p20240614/b2gpu/bin/gcc
            cxx: /opt/packages/gcc/v13.3.1-p20240614/b2gpu/bin/g++
            fortran: /opt/packages/gcc/v13.3.1-p20240614/b2gpu/bin/gfortran
          environment:
            prepend_path:
              LD_LIBRARY_PATH: /opt/packages/gcc/v13.3.1-p20240614/b2gpu/lib64
      buildable: false
    cuda:
      buildable: false
      externals:
      - spec: cuda@12.6.1
        prefix: /opt/packages/cuda/v12.6.1
    slurm:
      buildable: false
      externals:
      - spec: slurm@22.05.11
        prefix: /usr

  concretizer:
    unify: when_possible
    reuse: true

  view: true
```

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
- **`~static` on mfem** — pybind11 module links the shared lib only.
- **`unify: when_possible`, `reuse: true`** — avoids the numpy/python chain
  dictating versions DAG-wide; keeps installed packages across re-concretizations.
- **`mfem@develop`** — moving branch; the mirror freezes whatever commit was
  current at `spack mirror create` time. Consider pinning a commit.

---

## Fresh-install procedure

### Step 0 — one-time Spack setup [LOGIN NODE]

Spack lives on Ocean, not `$HOME` (quota):

```bash
cd $PROJECT     # /ocean/projects/mth260019p/schneier
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. $PROJECT/spack/share/spack/setup-env.sh     # add to ~/.bashrc
```

### Step 1 — fetch all sources into mirrors [LOGIN NODE]

```bash
module load gcc/13.3.1-p20240614 cuda/12.6.1
. $PROJECT/spack/share/spack/setup-env.sh
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .

spack bootstrap now                      # clingo etc. while online
spack concretize -f
spack mirror create -d $PROJECT/spack-mirror --all
spack mirror add local_sources file://$PROJECT/spack-mirror
```

The warning `mirror ... cannot be used in concretization (no index found)` is
benign — it refers to a *binary* index; this is a source mirror.

**Also mirror the pip wheels now** (the pip analog of `spack mirror create` —
download on login, install later wherever you like, no network needed):

```bash
$PROJECT/spack/opt/spack/.../python3 --version 2>/dev/null || true
# use the env view's python once it exists; on a truly fresh system where the
# view doesn't exist yet, defer this command to after Step 2 and run it on the
# login node then. Wheel downloads need internet but not a working compiler.
python3 -m pip download --only-binary=:all: -d $PROJECT/pip-wheels numba
```

`--only-binary=:all:` guarantees pip fetches prebuilt manylinux wheels only —
it is incapable of triggering a source build. If it errors "no matching
distribution", the python/numpy pins need adjusting; nothing was harmed.

**Checklist on `spack concretize -f` output:**

- `mfem`, `hypre`, `libceed` all `+cuda cuda_arch=90`; hypre shows `ldflags=-ldl`
- `suite-sparse ~cuda`; `ucx ~gdrcopy`
- everything `target=x86_64_v3`, gcc 13.3.1
- `cuda@12.6.1`, `slurm@22.05.11`, `gcc@13.3.1` flagged `[e]`
- `grep -c b2rm spack.lock` → **0**; `grep -c b2gpu spack.lock` → nonzero
- no `llvm` anywhere in the DAG (if present, py-numba snuck back into specs)

### Step 2 — build [GPU COMPUTE NODE]

Preferred: batch job (survives disconnects; see Gotcha #7). Script:

```bash
#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH -t 8:00:00
#SBATCH -o spack-install-%j.out

module load gcc/13.3.1-p20240614 cuda/12.6.1
. $PROJECT/spack/share/spack/setup-env.sh
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .
module list
spack install -j $(nproc) --dirty --use-buildcache never
```

Or interactively (run inside `tmux` on the login node so disconnects don't kill
it): `interact -p GPU-shared --gres=gpu:v100-32:2 -t 8:00:00`, then the same
module/activate/install block by hand.

Flag rationale (all three matter):
- `--dirty`: do NOT sanitize the build environment. Required so perl's direct
  gcc invocations see the module's LD_LIBRARY_PATH (Gotcha #4). Keep the shell
  clean: only the two modules + spack setup loaded.
- `--use-buildcache never`: skip public binary-cache probing that stalls on the
  airgapped node.
- `-j $(nproc)`: the cgroup-limited core share; `$nproc` without parens is an
  empty variable and means unlimited.

Walltime expiry mid-build is harmless: resubmit/re-request and rerun the same
install command; Spack resumes at the first uninstalled package. `gmake` is the
canary — if the first package configures past "checking whether the C compiler
works... yes", the toolchain is sound.

### Step 3 — verify the compiled stack [GPU COMPUTE NODE]

```bash
which mpicxx                  # -> .spack-env/view/bin
ompi_info | grep -i cuda      # mca cuda support lines present
ucx_info -v                   # cuda listed (gdrcopy absent, by design)
```

### Step 4 — install numba from the wheel mirror [ANY NODE]

Works on compute nodes (no network needed — wheels are local) or the login node.
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
interact -p GPU-shared --gres=gpu:h100-80:1 -t 1:00:00
# module loads + setup-env + env activate, then:
nvidia-smi                                   # confirm H100-SXM5-80GB
cmake -DMFEM_DIR=$SPACK_ENV/.spack-env/view ...
./bessemer <case> -d cuda
```

First validation: reduced Re_tau=180 channel; confirm the pressure Poisson
solve converges under GPU BoomerAMG before re-tuning Cahouet–Chabard.

### Production job template

```bash
#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:8
#SBATCH -t 8:00:00

module load gcc/13.3.1-p20240614 cuda/12.6.1
. $PROJECT/spack/share/spack/setup-env.sh
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .

mpirun -np 8 ./bessemer <case> -d cuda    # one rank per GPU; device by local rank
```

GPU partition = whole nodes, GPU counts in multiples of 8, 16 SU/node-hr on
H100. `srun --mpi=pmix -n 8` also works.

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
Stale lock after a crash: `rm .spack-env/transaction_lock` (safe when no spack
process is running). Also: only one GPU-shared job may be queued/running at
once (QOSMaxSubmitJobPerUserLimit) — a zombie session blocks new requests.

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
  compile-time flag. V100-32 slices (1 SU/GPU-hr, 24 nodes) are the sweet spot;
  v100-16 is the same price but only 9 nodes (often the slower queue). CPU
  share scales with GPU count (~5 cores/GPU on V100 nodes; more on H100).
- Check availability before queueing: `sinfo -N -p GPU-shared -o "%N %G %t %C"`
  and `squeue -p GPU-shared -t PENDING -o "%u %b %l"`.
- Short walltimes backfill dramatically faster; resume semantics make repeated
  short sessions free.
- Running on V100s would require a separate `cuda_arch=70` environment
  (including hypre). Compiling on them produces identical x86_64_v3 binaries.

---

## Quick-reference: rebuild from absolute zero

```bash
# LOGIN NODE ---------------------------------------------------------------
cd $PROJECT && git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. $PROJECT/spack/share/spack/setup-env.sh
module load gcc/13.3.1-p20240614 cuda/12.6.1
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .
spack bootstrap now
spack concretize -f                      # verify checklist; no b2rm; no llvm
spack mirror create -d $PROJECT/spack-mirror --all
spack mirror add local_sources file://$PROJECT/spack-mirror
python3 -m pip download --only-binary=:all: -d $PROJECT/pip-wheels numba

# GPU COMPUTE NODE (batch preferred; else interact inside tmux) -------------
module load gcc/13.3.1-p20240614 cuda/12.6.1
. $PROJECT/spack/share/spack/setup-env.sh
cd $PROJECT/bessemer/environments/psc_gpu
spack env activate .
spack install -j $(nproc) --dirty --use-buildcache never
ompi_info | grep -i cuda && ucx_info -v            # verify MPI stack
which python3                                      # -> .spack-env/view/bin
python3 -m pip install --no-index --find-links=$PROJECT/pip-wheels numba
python3 -c "from numba import cfunc; print('cfunc OK')"
```