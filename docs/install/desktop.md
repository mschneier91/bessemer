# The `desktop` environment (the maintainer's workstation)

The maintainer's Linux workstation and the project's reference environment:
`environments/desktop/spack.yaml` and its `spack.lock`, both committed. The lock records
versions known to work together. Other machines get their own environment from
`scripts/setup.sh`: see [spack.md](spack.md), which also explains activation, the compiler
pattern and the two traps.

| | |
|---|---|
| Spack | `~/spack`, develop at `5ca6d8e` (1.3.0.dev0) at the 2026-10-09 refresh |
| Compiler | Spack-built `gcc@14.3.0`. The system GCC (11.4, no Fortran) builds a few bootstrap tools only |
| Stack (2026-10-09) | MFEM 4.10.1-dev at `1c19aba` (pinned) with MPI, MUMPS, SuiteSparse, gslib, libCEED, LAPACK; hypre 3.2.0; libCEED 1.0.0; SuiteSparse 7.14; OpenMPI; CMake; Ninja; googletest; astyle; yaml-cpp; Python 3.12 with pybind11, NumPy, numba |
| Selected by | `environments/.machine` containing `desktop` |

`desktop/spack.yaml` lacks `py-matplotlib`, which `stack.yaml` gained after the last
refresh; the doctor warns about it. It arrives with the next refresh.

## Refreshing

A refresh moves everything, MFEM included, to today's versions. The maintainer does one
every couple of months, from a host terminal (never the sandboxed editor: trap 1):

```sh
cd ~/spack && git pull --ff-only
. ~/spack/share/spack/setup-env.sh
spack repo update

cd ~/bessemer
# bring environments/desktop/spack.yaml's specs in line with environments/stack.yaml now
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
commit. When a new MFEM breaks something, pin the last good commit
(`mfem@develop commit=<sha> +...`) in `stack.yaml` and every machine's `spack.yaml`, with
a comment naming the upstream issue. Drop the pin once it's fixed.

## Verifying

```sh
scripts/doctor.sh             # environment, toolchain, MFEM (+MUMPS), build, smoke run
scripts/test.sh cpu -L fast   # the fast tier, ~3.5 min
```
