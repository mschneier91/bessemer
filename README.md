# bessemer

A **time-dependent, high-order finite element solver for the incompressible
Navier–Stokes equations**, built on [MFEM](https://mfem.org). It uses a
**coupled, monolithic mixed method**: velocity and pressure are solved together
as one saddle-point system per time step — not via operator splitting or a
pressure-correction scheme — so the divergence constraint is driven to the
solver tolerance with no splitting error.

The current, validated capability is **unsteady Stokes** (Sprint 1, complete and
hardened) plus its production-grade preconditioner stack. Navier–Stokes
(convection) is the next sprint and is deliberately gated.

- **Discretization:** continuous Taylor–Hood elements on tensor-product
  **quadrilaterals (2D) / hexahedra (3D)** only. Velocity order `k_u` and
  pressure order `k_p` are independent runtime options; the default is the
  inf-sup–stable **Q3/Q2** pairing.
- **Assembly:** partial assembly (matrix-free, sum-factorized) throughout —
  affordable high order, with a GPU path kept open by construction.
- **Time integration:** in-repo BDF2 semi-implicit stepper with a
  variable-step, PI-controlled **adaptive** mode (BDF2-vs-BDF3 error estimate).
- **Outer solver:** FGMRES on the full block system with a block preconditioner.
  The pressure Schur block is either a scaled pressure mass or the **Δt-robust
  Cahouet–Chabard** block (the case-level default); the velocity block is
  matrix-free Jacobi or LOR-BoomerAMG.

It builds as a **library** (`libincns`). A case is a thin driver — a YAML deck,
a C++ program, or a Python script — that supplies parameters, mesh, and
boundary/initial conditions. **The solver core is never edited to run a new
case.**

---

## Requirements

Everything — including the compiler — comes from [Spack](https://spack.io); the
system `gcc`/`cmake`/`MPI` are never used to build or run project code. You need:

- A Spack checkout (default location `$HOME/spack`; override with `SPACK_ROOT`).
- A per-machine environment under `environments/<machine>/`. One is committed for
  `desktop` (the reference build). The environment pins, via `spack.lock`:
  `mfem@develop` (+MPI, +LAPACK, +SuiteSparse, +gslib, +libCEED), a recent GCC,
  CMake, Ninja, OpenMPI, googletest, astyle, yaml-cpp, and — for the Python
  bindings — `python@3.12`, `py-pybind11`, `py-numpy`, `py-numba`.

Bringing up a **new machine** means adding a new `environments/<name>/` directory
(its own `spack.yaml` + committed `spack.lock`); existing environments are never
edited to make a different machine work. Machine resolution lives in exactly one
place, `scripts/env.sh` (hostname detection, overridable with `INCNS_MACHINE`,
defaulting to `desktop`).

---

## Install / build

The environment is built once, in a supervised session (a long compile —
`mfem@develop` from source). If `environments/<machine>/` already exists, you
skip straight to building:

```sh
git clone <this-repo> bessemer && cd bessemer
scripts/install-hooks.sh          # point git at the committed pre-commit hooks (once)

scripts/build.sh                  # configure + build the default `cpu` preset
```

`scripts/build.sh [preset]` activates the machine's Spack environment (asserting
the whole toolchain resolves inside it), then runs CMake + Ninja. Presets:

| Preset       | What it is                                                        |
|--------------|-------------------------------------------------------------------|
| `cpu`        | **The build that matters.** RelWithDebInfo, ccache, Ninja.        |
| `cpu-python` | `cpu` + the pybind11 `incns` module.                              |
| `cpu-asan`   | `cpu` + AddressSanitizer/UBSan (diagnostic sweep — `scripts/asan.sh`). |
| `cuda`       | Optional GPU build, off by default; enable only with a CUDA toolkit. |

```sh
scripts/build.sh cpu-python       # e.g. to build the Python bindings too
```

> **Never `rm -rf build/`** — it dumps the warm ccache and turns a 30 s rebuild
> into ~15 min. Build directories live under `build/<preset>/`.

You do **not** need to activate Spack manually — `build.sh`, `test.sh`,
`style.sh`, and the git hooks all source `scripts/env.sh` first.

---

## Running the tests

```sh
scripts/test.sh cpu -L fast        # the fast tier (< 3 min); the development loop
```

`scripts/test.sh` activates the environment and runs `ctest` in the requested
preset's build directory. **Every test runs under MPI at 1, 2, and 4 ranks** —
partition-boundary and reduction bugs are invisible at one rank, so a serial pass
is never sufficient. Tiers (by ctest label):

| Label   | What                                                                  | When                |
|---------|-----------------------------------------------------------------------|---------------------|
| `smoke` | A trivial MPI/MFEM hello — proves the harness.                        | —                   |
| `fast`  | ~85 registrations: every module's unit test + the TGV/MMS convergence oracles. | The inner loop. |
| `slow`  | The full Cahouet–Chabard robustness sweep (ν × σ × h × p, ~17 s at np=4). | Before blessing a preconditioner change. |

```sh
scripts/test.sh cpu -L fast                    # all fast tests
scripts/test.sh cpu -L fast -R stokes_solver   # one test, all rank counts
scripts/test.sh cpu -L slow                    # the CC sweep
```

Two extra diagnostic sweeps (not in the normal loop):

```sh
scripts/asan.sh            # rebuild + run the C++ tier under ASan/UBSan (np 1,2)
scripts/debug_device.sh    # run the tier on MFEM's mprotect "debug" device --
                           # faults on any silent host<->device fallback (GPU prep)
```

The unit tests favor **algebraic checks exact to machine precision** over
convergence studies, so a failure localizes to a module. The end-to-end oracle is
the 2D Taylor–Green vortex against its closed-form decaying solution (spatial +
temporal order), backed by polynomial manufactured-solution tests that pin the
mixed discretization, Dirichlet elimination, null-space handling, and the
time-dependent-BC path.

---

## Running a case

> **New here? Read [`examples/python/start_here.py`](examples/python/start_here.py)
> first.** It is a single, heavily-commented manufactured-solution case —
> every parameter, field, boundary condition, and API call is explained inline,
> and it ends by verifying the solver reproduced a known-exact answer to machine
> precision. It is the fastest way to understand both *how you drive the code*
> and *how the code is checked*.
>
> ```sh
> scripts/build.sh cpu-python
> . scripts/env.sh
> export PYTHONPATH=build/cpu-python/python
> mpirun -np 4 python examples/python/start_here.py
> # -> start_here: ||u - u_exact||_L2 = 2.7e-14 ... PASSED
> ```

### 1. YAML deck (no recompile)

```sh
scripts/build.sh cpu
. scripts/env.sh                                   # activate the toolchain
mpirun -np 4 build/cpu/apps/run_case cases/tgv2d_stokes.yaml
```

A deck supplies physics, mesh, time, solver, and output settings; omitted fields
take library defaults. Example (`cases/tgv2d_stokes.yaml`):

```yaml
physics:
  nu: 1.0
mesh:
  dim: 2
  elements: [8, 8]          # lengths default to 2*pi, fully periodic
time:
  dt: 0.02
  t_final: 0.2
initial_velocity: taylor_green_2d
output:
  enabled: true
  path: tgv2d_out
  name: tgv2d
  interval: 5
```

Solver knobs also live in the deck — e.g. `solver: { schur: cc, n_inner: 10 }`
selects the Cahouet–Chabard preconditioner, `discretization: { order_u: 4,
order_p: 3 }` bumps the order, `time: { adaptive: true, atol: 1e-6 }` turns on
adaptive stepping. Analytic initial/boundary fields beyond the named registry
(`taylor_green_2d`, `zero`) are supplied from Python or an in-code driver.

Output is written as a ParaView collection (`<path>/<name>/…`) with high-order
output enabled; open the `.pvd` in ParaView.

### 2. Python driver

Build the bindings (`scripts/build.sh cpu-python`), then:

```sh
. scripts/env.sh
export PYTHONPATH=build/cpu-python/python
mpirun -np 4 python examples/python/stokes_ex/run.py
```

Python is a **pure job driver**: configure a case, supply analytic
IC/BC/forcing, and run — *all numerics stay in C++*, no bulk field data crosses
the boundary, no `mpi4py` (you run `mpirun -np N python case.py`, N identical
interpreters each steering their rank). PyMFEM is not used; these are the
project's own bindings.

```python
import incns
import numpy as np

p = incns.Parameters()
p.nu = 1.0
p.mesh.box(dim=2, elements=(32, 32), periodic=(True, True))
p.dt, p.t_final = 1e-3, 1.0
case = incns.Case(p)

# Fields are f(x, t) -> components. The @incns.field decorator compiles them
# (numba, no GIL) for the hot path; a plain callable also works (slower).
@incns.field(dim=2)
def tgv(x, t, out):
    e = np.exp(-2.0 * p.nu * t)
    out[0] =  np.cos(x[0]) * np.sin(x[1]) * e
    out[1] = -np.sin(x[0]) * np.cos(x[1]) * e

case.set_initial_velocity(tgv)
case.run()
if incns.on_root():
    print(f"t={case.time:.3f}  steps={case.step_count}  KE={case.kinetic_energy():.4f}")
```

### 3. In-code C++ driver

`apps/taylor_green.cpp` is a worked example: it builds a `ParMesh`, sets fields
on a `Parameters` struct, constructs one `incns::Case`, and calls `Run()`. A YAML
case and an in-code case go through the *same* `Case` surface.

---

## What the code looks like

Modular, one concern per file, each with its own unit test (~6.7k lines of C++
across `src/`). The public surface a driver touches is small:
`incns::Parameters`, `incns::BoundaryConditions`, and `incns::Case`
(`SetInitialVelocity` / `SetForcing` / `Run` / `Step` / `Velocity` / `Pressure` /
diagnostics). Rough decomposition:

```
src/
  mesh/         periodic quad/hex box factory (uniform + wall-normal stretching)
  spaces/       velocity/pressure ParFESpaces, block offsets
  quadrature/   RuleBook -- owns the integration rules (GL default, GLL option)
  operators/    Stokes blocks [A Bᵀ; B 0] (PA), grad-div, block preconditioner,
                and the sum-factorized VectorDivDivIntegrator
  precond/      Cahouet-Chabard Schur preconditioner (BM⁻¹Bᵀ, L_p, block shapes)
  time/         BDF/AB coefficients, in-repo stepper, adaptive controller
  solver/       one implicit saddle-point solve; the unified Case
  bc/           per-attribute Dirichlet / outflow / periodic
  post/         mass-weighted mean-zero pressure, ParaView output, diagnostics,
                checkpoint/restart
  config/       Parameters struct + YAML loader, non-dimensionalization
  util/         nested scoped profiler, device configuration
apps/           run_case (generic YAML driver), taylor_green (in-code example)
cases/          input decks
python/         pybind11 module + the thin `incns` Python package
test/           unit tests + the TGV/MMS convergence oracles
bench/          micro-benchmarks (not in ctest)
docs/           preconditioner math + tuning notes
environments/   per-machine spack.yaml + committed spack.lock
```

Design invariants worth knowing before hacking on it:

- **Quad/hex only**, tensor-product; simplex meshes are rejected at load.
- **Pressure null space** (fully periodic / enclosed) is detected from the BC
  set + mesh and removed by **orthogonalization — never by pinning a DOF**.
- **Time integration is in-repo** — no `mfem::ODESolver`.
- **Quadrature comes from the `RuleBook`**, never hardcoded at a call site.
- **Parallel-correct by construction**: every reduction is global; diagnostics
  print on rank 0 only.
- Runtime options (orders, γ grad–div, time scheme, preconditioner choice) are
  set at runtime, not via `#ifdef`.

---

## Development

- **Formatting** is astyle (MFEM convention), pinned to a specific version:
  `scripts/style.sh`. Warnings are errors (`-Werror`).
- **Git workflow:** one branch per change; `main` is merged by a human. The
  committed pre-commit hook (installed via `scripts/install-hooks.sh`) runs the
  style check; run the fast tier yourself before pushing.
- **Docs:** `scripts/docs.sh` builds the Doxygen API docs; every public entity
  carries a Doxygen comment.
- `CLAUDE.md` is the authoritative design document — the full specification of the
  formulation, sprint plan, test tiers, and guardrails.

A task is **done** when the CPU build compiles clean (no warnings), the full fast
tier passes at np ∈ {1, 2, 4}, any new module carries its unit test, and stored
baselines are unchanged or explicitly re-blessed.

---

## Status & roadmap

- **Sprint 1 — unsteady Stokes:** complete and validated. Pre-GPU hardening
  (H1–H6: GPU-readiness plumbing, mesh stretching, sanitizers, physical
  diagnostics, LOR-AMG, Δt-refresh) done.
- **Cahouet–Chabard preconditioner:** implemented, validated across the (ν, σ, h,
  p) grid, and the case-level default.
- **Sum-factorized grad–div** (`VectorDivDivIntegrator`): in, ~5–25× over the
  elasticity-based path it replaces.
- **Next:** GPU port (at a DOE/leadership system), then Sprint 2 — Navier–Stokes
  (convection, dealiasing) behind an explicit sign-off, beginning with a
  re-baselining of every solver option under NSE stepping.

Production DNS/LES runs target DOE systems and are **not** run on the development
machine.
