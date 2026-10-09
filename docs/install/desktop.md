# Desktop environment (Linux workstation)

The reference development build. Everything, including GCC, comes from Spack; the
system compiler, CMake and MPI are never used for project code. The environment lives in
`environments/desktop/`: `spack.yaml`, the specs, and `spack.lock`, the exact
resolution, both committed.

| | |
|---|---|
| Spack | a checkout at `~/spack` (override with `SPACK_ROOT`) |
| Compiler | Spack-built `gcc@14.3.0`, reused as the compiler; the system GCC builds a few bootstrap tools only |
| Stack | `mfem@develop` (+MPI, +MUMPS, +SuiteSparse, +gslib, +libCEED, +LAPACK), hypre, OpenMPI, CMake, Ninja, googletest, astyle, yaml-cpp, Python 3.12 with pybind11, NumPy, numba |
| Activation | automatic: `scripts/*.sh` source `scripts/env.sh`, which activates the environment and aborts if any tool resolves outside Spack |

A different machine gets its own `environments/<name>/` directory; existing environments
are never edited to make another machine work. `scripts/env.sh` picks the machine from the
hostname (override with `INCNS_MACHINE`).

## Two traps (both hit on 2026-10-09)

1. **Run Spack from a host terminal, never inside a sandboxed editor.** A Flatpak VSCode
   (and any agent running inside it) sees the Flatpak runtime's OS and glibc, not the
   host's. Spack then resolves packages for a different OS
   (`os=org.freedesktop.platform25` instead of the host's), and links fail when they mix
   with the host-built stack. Check with `ls /.flatpak-info`: if that file exists, you are
   in a sandbox. Best: install the editor natively.
2. **Spack re-resolves an environment on activation after `spack.yaml` changes.** Editing
   `spack.yaml` and then running anything that sources `scripts/env.sh` writes a new
   `spack.lock` and points the environment's view at packages that aren't built yet. Until
   `spack install` finishes, `mpirun` and MFEM vanish and `env.sh` aborts. So edit
   `spack.yaml` only immediately before resolving and installing, in the same host session.

## Building or refreshing the environment

A refresh moves everything, MFEM included, to today's versions. Do it every couple of
months, from a host terminal:

```sh
cd ~/spack && git pull --ff-only
. ~/spack/share/spack/setup-env.sh
spack repo update

cd ~/bessemer
# (edit environments/desktop/spack.yaml now, if anything changes)
spack -e environments/desktop concretize --fresh --force 2>&1 | tee ~/concretize.txt
grep -c freedesktop environments/desktop/spack.lock     # must print 0 (trap 1)
spack -e environments/desktop install -j 16              # hours: llvm (for numba) is the long pole

scripts/build.sh cpu && scripts/build.sh cpu-python      # full rebuild against the new stack
scripts/test.sh cpu -L fast
```

Then commit `spack.yaml` and `spack.lock` together. The test baselines can shift with a
new MFEM or hypre: re-bless them only with a stated reason
([developing.md](../developing.md) §1).

**Changing one package** (for example pinning MFEM): edit `spack.yaml`, then resolve
*without* `--fresh` so the installed stack is reused and only that package rebuilds:
`spack -e environments/desktop concretize --force`, then `install`.

## MFEM tracks `develop`, so sometimes it needs a pin

`mfem@develop` resolves to the branch head at resolve time, and the lock records the
commit. When a new MFEM breaks something, pin the last good commit in `spack.yaml`
(`mfem@develop commit=<sha> +...`) with a comment naming the upstream issue, and drop the
pin once it's fixed. Any active pin is commented in `spack.yaml`.

## Verifying

```sh
scripts/doctor.sh             # environment, toolchain, MFEM (+MUMPS), build, smoke run
scripts/test.sh cpu -L fast   # the fast tier, ~3.5 min
```
