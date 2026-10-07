# Spec: rotation-aware Schur complement (Cahouet-Chabard / Olshanskii switching) and the velocity-block escalation path

Target: the bessemer application (out of tree), built against MFEM `master` at commit `c362dea2` (2026-09-29). Every MFEM class, method and behavior cited here was checked against that commit. Every complete code block in Sections 4, 5 and 7 was compiled and run against it (serial build, host backend). The multigrid construction of Section 7.6 was validated with a stand-in that uses legacy-assembled level operators in place of Parts A and B. Re-verify Sections 5.7 and 7.8 if you build against a different MFEM.

Audience: an agent implementing this in C++. Read the whole document before writing code.

Prerequisites: Parts A and B of `rotational_convection_pa_spec.md` (`VectorRotationalConvectionIntegrator` with `GetRotationNumberStats`, and `PointBlockJacobi`). This document adds Parts C and D. Where it disagrees with Section 8 of that spec ("Schur complement: out of scope", "call `GetRotationNumberStats` every N steps"), this document wins.

Everything runs on GPUs with moderate-order hexahedral elements and partial assembly. Nothing in the production path assembles a high-order matrix. 3D only.

---

## 0. Summary

**Part C: Schur complement (Sections 4 to 6).** One `Solver`, `RotationalSchurPreconditioner`, that applies one of two pressure Schur complement approximations, chosen once per time step from the rotation-number diagnostic:

- **Cahouet-Chabard (current):** $\hat S^{-1} = \nu M_p^{-1} + \sigma L_p^{-1}$, applied exactly as bessemer does today.
- **Olshanskii tensor:** $\hat S^{-1} = \nu M_p^{-1} + L_T^{-1}$ with $L_T = -\nabla\cdot(T\nabla)$ and $T = (\sigma I + [\alpha\,\omega^*]_\times)^{-1}$. $L_T^{-1}$ is applied by a fixed small number of flexible GMRES iterations on a partially assembled $L_T$, preconditioned by bessemer's existing $\sigma L_p^{-1}$ (the Laplacian low-order-refined algebraic multigrid). Nothing new is built on the multigrid side.
- **Switch:** `Auto` mode uses hysteresis on $\mu_{\max}$ (default: tensor above 20, back to Cahouet-Chabard below 10). `CahouetChabard` and `Tensor` force a mode, for calibration runs.

**Part D: velocity block (Sections 7 and 8).** A two-rung ladder, selected at run time:

- **Level 0 (start here):** point-block Jacobi from Part B, configuration `pbj_krylov`. No new code.
- **Level 1 (escalation, optional):** p-multigrid with point-block Jacobi smoothing and the rotation term on every level (Olshanskii and Reusken's approach). On isotropic meshes, uniform or with nonconforming adaptive refinement, 7 to 15 iterations for viscous ratios from 0.01 to 300 and rotation numbers from 0 to 59. On stretched boundary-layer meshes it degrades (30 to 73) and costs more than Level 0. Its smoother damping must come from an eigenvalue estimate (Section 7.4).
- **Expected outcome for the adaptively refined bluff body at 100 times the Courant limit:** Level 0 converges, with roughly 3 to 8 times more inner iterations than at a Courant number near 1 (Table A5). Level 1 is an optimization, not a necessity.
- A viscous-ratio diagnostic $\hat v_{\max}$, logged every time step alongside $\mu_{\max}$, tells you when Level 0 is running out of steam.

**Files.** Two new source pairs and two test files (Section 3).

**Tests.** Three for Part C, three for Part D (Sections 6 and 8). Solver-level iteration studies on real bessemer cases are integration work and stay out of these files.

**Tried and rejected** (evidence in Section 1.5 and Appendix A): a scalar-coefficient version of the Olshanskii operator inside the low-order-refined multigrid; putting the tensor itself into the low-order-refined form; a tolerance-based inner solve for the tensor mode; point-block Jacobi followed by an algebraic multigrid correction (`ProductSolver`); an approximate factorization $H^{-1}D(D+N_L)^{-1}$; algebraic multigrid on $\sigma M + \nu K$ alone.

---

## 1. Background: what this design is based on

This section summarizes the analysis and prototype results behind the design. It is context for the implementer; nothing in it is code.

### 1.1 The system and the current preconditioner

BDF2 with EXT3 extrapolation, $\sigma = 3/(2\Delta t)$, lagged rotational convection (Part A):

$$\begin{pmatrix} A & B^T \\ B & 0\end{pmatrix}\begin{pmatrix} u \\ p\end{pmatrix} = \begin{pmatrix} f \\ g\end{pmatrix},\qquad A = \sigma M + \nu K + N(\omega^*),$$

where $M$ is the velocity mass matrix, $K$ the viscous (vector Laplacian) stiffness, $N$ the skew rotation operator built from the extrapolated vorticity $\omega^* = \nabla\times w^*$, and $p$ the Bernoulli pressure. The outer solver is FGMRES with a block preconditioner. Today the Schur complement $S = BA^{-1}B^T$ is approximated by Cahouet-Chabard, which assumes $A \approx \sigma M + \nu K$ and omits $N$ entirely.

### 1.2 Two dimensionless numbers

**Rotation number** $\mu = |\omega^*|/\sigma = \tfrac23|\omega^*|\Delta t$ (pointwise). The ratio of the rotation term to the time-derivative term in the zero-order part $\sigma I + \omega^*\times$ of $A$. Ways to read it:
- About $\tfrac43\times$ the angle (radians) a fluid parcel turns in one time step: $\mu = 1$ is about 45 degrees per step, $\mu = 18$ about two turns.
- A Courant number built on the velocity gradient: $\mu \approx \tfrac23\,\mathrm{CFL}\times\Delta U_{cell}/U$, where $\Delta U_{cell}$ is the velocity jump across one cell. In a shear layer resolved by $n$ nodes, $\mu \approx \mathrm{CFL}/n$.
- At a no-slip wall, $\mu_{wall} = \tfrac23\Delta t^+$ (the time step in viscous units). In bulk turbulence, $\mu \approx \Delta t/\tau_\eta$.
- Independent of the mesh. Large only where $\Delta t$ is large compared with the local shear time $1/|\omega|$.

**Viscous ratio** $v = \nu/(\sigma h^2) = \tfrac23\,\nu\Delta t/h^2$ at grid spacing $h$: the viscous term against the time-derivative term at the grid scale. $v = \mathrm{CFL}/Re_h$ with $Re_h = Uh/\nu$. At a wall, $v/\mu = \nu/(|\omega|h^2) = 1/(\Delta y^+)^2$: whether viscosity or rotation wins at the grid scale depends only on the mesh. Viscosity and rotation balance at the length $\ell_\nu = \sqrt{\nu/|\omega|}$ (one wall unit at a wall, the Kolmogorov scale in bulk turbulence).

### 1.3 The Olshanskii preconditioner

Olshanskii (Numer. Linear Algebra Appl. 6:353–378, 1999) proposed, for the rotation form,

$$Q^{-1} = \nu M_p^{-1} + L_T^{-1},\qquad L_T = -\nabla\cdot(T\nabla),\qquad T = (\sigma I + [o]_\times)^{-1} = \frac{\sigma^2 I + o\,o^T - \sigma[o]_\times}{\sigma(\sigma^2 + |o|^2)},\quad o = \alpha\,\omega^*.$$

Cahouet-Chabard is the special case $T = I/\sigma$ (since $\sigma L_p^{-1} = (-\nabla\cdot(\sigma^{-1}\nabla))^{-1}$). $T$ has conductivity $1/\sigma$ along $\omega$, $\sigma/(\sigma^2+|\omega|^2)$ across it, and a skew part of size $|\omega|/(\sigma^2+|\omega|^2)$ that acts like advection wherever $\omega$ varies. Benzi and Liu (SIAM J. Sci. Comput. 2007, Section 4.2) describe the same approximation as the block-triangular method for the rotation form.

Constant-coefficient analysis, with $\tilde v = \nu k^2/\sigma$ the viscous ratio at the scale of a pressure mode $k$:
- Cahouet-Chabard is off by a factor of about $1 + \mu^2/(1+\tilde v)^2$. Viscosity helps it.
- Olshanskii is off by about $1 + \tilde v\mu^2/((1+\tilde v)^2+\mu^2)$, exact as $\tilde v \to 0$ or $\infty$, worst at $\tilde v \approx \mu$ where it reaches $\mu^2/(2(\sqrt{1+\mu^2}+1)) \approx \mu/2$. This is Olshanskii's $\rho_{\max}$.

So the tensor version is nearly exact wherever rotation dominates viscosity at every resolved scale, and is at its worst where the grid resolves $\ell_\nu$ while $\mu$ is large: in practice the viscous sublayer when $\Delta y^+ \lesssim 1$, the start of separated shear layers, and resolved vortex cores. That regime has not been tested (Section 1.7).

### 1.4 Evidence (prototypes; full tables in Appendix A)

Small 3D problems: Taylor-Hood Q4/Q3 on a $3^3$ or $4^3$ hexahedral mesh, Dirichlet velocity everywhere, $\sigma = 1$, a nonuniform rotating velocity field scaled to set $\mu_{\max}$, exact velocity-block solves, outer FGMRES to $10^{-8}$. Laplacian solves were exact (a stand-in for the algebraic multigrid); a rerun with a V-cycle-quality stand-in (CG to relative tolerance 0.1) gave the same picture.

Outer iterations / pressure-Laplacian applications at $v = 0.014$ (inertia dominated at the grid scale):

| $\mu_{\max}$ | Cahouet-Chabard | tensor, 3 inner iterations |
|---|---|---|
| 1.8 | 65 / 65 | 40 / 120 |
| 18 | 307 / 307 | 158 / 474 |
| 59 | 783 / 783 | 264 / 792 |
| 183 | did not converge in 1500 | 455 / 1365 |

Cahouet-Chabard degrades steadily with $\mu$ and fails near 180. The tensor version needs about 3 times fewer outer iterations throughout, ties on Laplacian applications near $\mu = 60$, and wins outright above. With restart 50 the picture is the same (Appendix A). With viscosity significant at the grid scale ($v = 1.44$, $\mu = 18$) Cahouet-Chabard keeps an edge: 99/99 against 68/204. These numbers set the default switch thresholds (20 on, 10 off): below about 20 Cahouet-Chabard is cheaper, above about 50 the tensor version is.

### 1.5 What was tried and rejected

| Alternative | Result |
|---|---|
| Scalar $\kappa = \sigma/(\sigma^2+|\omega^*|^2)$ in the existing low-order-refined Laplacian | Worse than plain Cahouet-Chabard in every case: 75 vs 65, 42 vs 33, 334 vs 307, 150 vs 99 outer iterations. In 3D it discards the along-axis conductivity and is swamped by the neglected skew part. |
| The tensor inside the low-order-refined form | MFEM's batched low-order-refined assembly reads only the scalar coefficient pointer; with a `MatrixCoefficient` that pointer is null and it assembles with coefficient 1, silently (verified: difference exactly 0). |
| Tensor mode with an inner tolerance (0.1) instead of a fixed count | Fewest outer iterations, but 2 to 7 times more Laplacian applications than 3 fixed iterations at $\mu_{\max} \ge 18$. |
| Velocity block: point-block Jacobi, then an algebraic multigrid correction on $\sigma M + \nu K$ (MFEM `ProductSolver`) | Much worse than point-block Jacobi alone when $\mu \gg v$ (141 vs 30, 401 vs 41): the multigrid stage does not see $N$ and undoes the block solve. Good only when $v \gg \mu$. |
| Velocity block: approximate factorization $H^{-1}D(D+N_L)^{-1}$ | Excellent when $v$ is small (14 vs 30), fails when $v$ is large (158 to 349). |
| Velocity block: algebraic multigrid on $\sigma M + \nu K$ alone | Fails when $\mu$ is large (249 and 569 iterations at small $v$). |

### 1.6 Expected regime: bluff body at about 100 times the Courant limit

With adaptive refinement near the body and $\Delta t$ set by accuracy rather than stability, the smallest cells run at a local Courant number of about 100.
- $\mu \approx \mathrm{CFL}/n$ with $n \approx 10$ to 30 nodes across the boundary layer: a few at the boundary-layer edge, up to a few tens at the wall. Below 1 in the wake.
- $v = \mathrm{CFL}/Re_h$ with $Re_h$ of order 1 to 10 in the resolved boundary layer: about 10 to 100 in the smallest cells.

So the Schur side sits near the switch threshold (hence `Auto`), and the velocity block becomes viscous-dominated in the refined cells. Level 0 still converges there; on locally refined test meshes at 100 times the Courant limit its inner iteration count to $10^{-2}$ rose from 5 to 15 at a Courant number of 1 to 24 to 45 (Table A5).

### 1.7 Known gaps

- **Viscous sublayer.** With $\Delta y^+ < 1$ and large $\Delta t^+$, some resolved scale has $\tilde v \approx \mu$, so the tensor version's worst case (about $\mu/2$) occurs in a layer a few wall units thick. Expected to cost extra iterations, not convergence. Untested.
- **Corners.** At sharp corners of a bluff body the discrete vorticity is large and grows with refinement, so $\mu_{\max}$ may stay above the threshold regardless of $\Delta t$. If that happens, use the `VolumeFraction` criterion (Section 5.4).
- **All evidence is from small synthetic problems** with exact or near-exact inner solves on a CPU. The thresholds are starting values; calibrate them on bessemer runs (Section 10).

---

## 2. Scope and non-goals

In scope:
- 3D, hexahedral tensor-product elements, H1 velocity (vector, `vdim == 3`) and H1 pressure, partial assembly, serial and MPI, CPU and GPU (CUDA and HIP).
- `Ordering::byNODES` and `Ordering::byVDIM` for the velocity space.
- Pressure with Dirichlet dofs (outflow) or pure Neumann (constants in the null space).

Not in scope:
- 2D (abort with a clear message).
- Changing bessemer's Laplacian algebraic multigrid, its pressure mass solve, or the outer solver. Part C wraps them.
- Component-wise velocity boundary conditions in the p-multigrid (for example a symmetry plane that constrains only the normal component). `GeometricMultigrid` builds one essential list from boundary attributes for all components; abort if bessemer's velocity essential list differs (Section 7.6, step 1).
- Automatic switching of the velocity-block level. It is a run-time option; $\hat v_{\max}$ is logged to inform the choice.
- Simplices, mixed meshes, variable order.

---

## 3. Files

Exactly six new files in bessemer, in the application's namespace:

| File | Content |
|---|---|
| `rotational_schur.hpp` | `RotatingDarcyTensor` (a `MatrixCoefficient`) and `RotationalSchurPreconditioner`. |
| `rotational_schur.cpp` | Their implementation. The device kernel lives in `RotatingDarcyTensor::Project`. |
| `velocity_multigrid.hpp` | `ViscousRatioDiagnostic`, `OrderInterpolator`, `DampedSmoother`, `RotationalVelocityMultigrid`. |
| `velocity_multigrid.cpp` | Their implementation. |
| `test_rotational_schur.cpp` | Part C tests (Section 6). |
| `test_velocity_multigrid.cpp` | Part D tests (Section 8). |

Level 0 of the velocity ladder is `PointBlockJacobi` from Part B, unchanged.

---

## 4. Part C interface (`rotational_schur.hpp`)

```cpp
/** T(x) = (sigma I + [o]_x)^{-1}, o = alpha curl(w*). 3D only.
    Project() evaluates it on the device at the quadrature points of the form
    that uses it; Eval() is a host path used only by the legacy reference in
    the tests. Do not use this coefficient in a low-order-refined form
    (Section 5.7). */
class RotatingDarcyTensor : public MatrixCoefficient
{
public:
   RotatingDarcyTensor(const GridFunction &w_star, real_t sigma, real_t alpha);
   void SetSigma(real_t s) { sigma = s; }

   void Eval(DenseMatrix &T, ElementTransformation &Tr,
             const IntegrationPoint &ip) override;
   void Project(QuadratureFunction &qf, bool transpose = false) override;

   /// T = (s^2 I + o o^T - s [o]x) / (s (s^2 + |o|^2)), column-major T[r + 3c].
   /// Defined here: used by the device kernel in Project and by test S1.
   MFEM_HOST_DEVICE static inline void Fill(real_t s, real_t o0, real_t o1,
                                            real_t o2, real_t T[9])
   {
      const real_t c = 1.0 / (s*(s*s + o0*o0 + o1*o1 + o2*o2));
      const real_t o[3] = {o0, o1, o2};
      // [o]x column-major: col 0 = (0, o2, -o1), col 1 = (-o2, 0, o0), col 2 = (o1, -o0, 0)
      const real_t W[9] = {0, o2, -o1, -o2, 0, o0, o1, -o0, 0};
      for (int col = 0; col < 3; col++)
      {
         for (int row = 0; row < 3; row++)
         {
            T[row + 3*col] = c*((row == col ? s*s : 0.0) + o[row]*o[col]
                                - s*W[row + 3*col]);
         }
      }
   }

private:
   const GridFunction &w;       // not owned; bessemer's w* (an L-vector)
   real_t sigma, alpha;
   Vector w_e, der;             // device scratch, reused
};

/** Pressure Schur complement preconditioner with two modes:
      CahouetChabard: z = lap_inv(r) + nu * mass_inv(r)   (bessemer's existing Cahouet-Chabard)
      Tensor:         z = FGMRES_k(L_T, r; lap_inv) + nu * mass_inv(r)
    The returned z approximates +S^{-1} r; apply whatever sign bessemer's
    block preconditioner uses today. */
class RotationalSchurPreconditioner : public Solver
{
public:
   enum class Mode { CahouetChabard, Tensor, Auto };
   enum class Criterion { MaxMu, VolumeFraction };
   struct Options
   {
      Mode mode = Mode::Auto;
      Criterion criterion = Criterion::MaxMu;
      real_t mu_on = 20.0;      // Auto, MaxMu: switch to Tensor when max mu > mu_on
      real_t mu_off = 10.0;     // Auto, MaxMu: back to CahouetChabard when max mu < mu_off
      real_t vol_on = 1e-3;     // Auto, VolumeFraction: fraction of volume with mu > mu_on
      real_t vol_off = 2.5e-4;  //   (uncalibrated starting values)
      int inner_iterations = 3; // fixed FGMRES iterations on L_T
      bool remove_mean = false; // pressure is pure Neumann (constants in the null space)
   };

   /** pfes:          pressure space (ParFiniteElementSpace in parallel).
       p_ess_tdofs:   pressure essential true dofs, the same list bessemer uses
                      for L_p (not copied; must outlive this object).
       w_star, alpha: the rotation integrator's lagged velocity and scale.
       nu:            the coefficient bessemer's Cahouet-Chabard puts on M_p^{-1}
                      (nu, or nu + gamma with grad-div).
       mass_inv:      bessemer's existing M_p^{-1} (not owned).
       lap_inv_sigma: bessemer's existing sigma * L_p^{-1}, i.e. the inverse of
                      -div((1/sigma) grad), including its sigma scaling and its
                      null-space handling (not owned). */
   RotationalSchurPreconditioner(FiniteElementSpace &pfes,
                                 const Array<int> &p_ess_tdofs,
                                 const GridFunction &w_star, real_t alpha,
                                 real_t nu, Solver &mass_inv,
                                 Solver &lap_inv_sigma, const Options &opt);

   /** Once per time step, after w* is formed, the rotation integrator's
       UpdateVorticity() has run, and stats = GetRotationNumberStats(sigma,
       opt.mu_on). Decides the mode for this step; in Tensor mode, re-runs the
       partial assembly of L_T with the current sigma and w*. */
   void Update(real_t sigma, const RotationNumberStats &stats);

   void Mult(const Vector &r, Vector &z) const override;
   void SetOperator(const Operator &) override { }   // no-op by design

   bool TensorActive() const { return tensor_active; }
   int NumSwitches() const { return n_switches; }
   const Operator *GetTensorOperator() const { return T_op.Ptr(); }   // tests

private:
   /// Forwards Mult and ignores SetOperator (Section 5.7, item 3).
   class FixedPreconditioner : public Solver
   {
      const Solver &s;
   public:
      explicit FixedPreconditioner(const Solver &s_)
         : Solver(s_.Height(), s_.Width()), s(s_) { }
      void Mult(const Vector &x, Vector &y) const override { s.Mult(x, y); }
      void SetOperator(const Operator &) override { }
   };

   void AssembleTensor(real_t sigma);
   void RemoveMean(Vector &v) const;

   const Options opt;
   const Array<int> &ess;
   const real_t nu;
   Solver &mass_inv;
   Solver &lap_inv;
   FixedPreconditioner lap_fixed;
   RotatingDarcyTensor T_coeff;
   std::unique_ptr<BilinearForm> T_form;   // ParBilinearForm in parallel
   OperatorHandle T_op;
   std::unique_ptr<FGMRESSolver> inner;
   mutable Vector t, r0;                   // device scratch
   bool tensor_active = false;
   int n_switches = 0;
#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
   bool parallel = false;
#endif
};
```

`RotationNumberStats` is Part A's struct (`max_mu`, `vol_fraction`, `threshold`).

---

## 5. Part C implementation (`rotational_schur.cpp`)

### 5.1 `RotatingDarcyTensor`

Verified against legacy assembly through MFEM's own `DiffusionIntegrator` (Section 6, S1).

```cpp
RotatingDarcyTensor::RotatingDarcyTensor(const GridFunction &w_star,
                                         real_t sigma, real_t alpha)
   : MatrixCoefficient(3), w(w_star), sigma(sigma), alpha(alpha) { }

void RotatingDarcyTensor::Eval(DenseMatrix &T, ElementTransformation &Tr,
                               const IntegrationPoint &ip)
{
   Vector o(3);
   w.GetCurl(Tr, o);            // host; tests only
   o *= alpha;
   real_t K[9];
   Fill(sigma, o(0), o(1), o(2), K);
   T.SetSize(3);
   for (int k = 0; k < 9; k++) { T.GetData()[k] = K[k]; }
}

void RotatingDarcyTensor::Project(QuadratureFunction &qf, bool transpose)
{
   MFEM_VERIFY(qf.GetVDim() == 9, "RotatingDarcyTensor: expected vdim 9");
   const FiniteElementSpace &fes = *w.FESpace();
   const IntegrationRule &ir = qf.GetSpace()->GetIntRule(0);
   const int ne = fes.GetNE(), nq = ir.GetNPoints();
   const Operator *R = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   w_e.SetSize(R->Height()); w_e.UseDevice(true);
   der.SetSize(nq*9*ne);     der.UseDevice(true);
   R->Mult(w, w_e);
   const QuadratureInterpolator *qi = fes.GetQuadratureInterpolator(ir);
   qi->SetOutputLayout(QVectorLayout::byNODES);
   qi->PhysDerivatives(w_e, der);                 // D(q,c,d,e) = dw_c/dx_d
   const auto D = Reshape(der.Read(), nq, 3, 3, ne);
   auto Q = Reshape(qf.Write(), 3, 3, nq, ne);     // column-major 3x3 per point
   const real_t s = sigma, a = alpha;
   const bool tr = transpose;
   mfem::forall(nq*ne, [=] MFEM_HOST_DEVICE (int i)
   {
      const int q = i % nq, e = i / nq;
      const real_t o0 = a*(D(q,2,1,e) - D(q,1,2,e));
      const real_t o1 = a*(D(q,0,2,e) - D(q,2,0,e));
      const real_t o2 = a*(D(q,1,0,e) - D(q,0,1,e));
      real_t K[9];
      Fill(s, o0, o1, o2, K);
      for (int c = 0; c < 3; c++)
      {
         for (int r = 0; r < 3; r++)
         {
            Q(r,c,q,e) = tr ? K[c + 3*r] : K[r + 3*c];
         }
      }
   });
}
```

Notes:
- `Project` must honor `transpose`: partial assembly calls it through `CoefficientVector::ProjectTranspose`. Getting this wrong replaces $T$ by $T^T$, i.e. flips the sign of $\omega$; S1 catches it.
- The vorticity is the exact curl of the finite element $w^*$ at the pressure form's quadrature points. No projection of $\omega$ is stored.

### 5.2 Constructor

```cpp
RotationalSchurPreconditioner::RotationalSchurPreconditioner(
   FiniteElementSpace &pfes, const Array<int> &p_ess_tdofs,
   const GridFunction &w_star, real_t alpha, real_t nu, Solver &mass_inv,
   Solver &lap_inv_sigma, const Options &opt)
   : Solver(pfes.GetTrueVSize()), opt(opt), ess(p_ess_tdofs), nu(nu),
     mass_inv(mass_inv), lap_inv(lap_inv_sigma), lap_fixed(lap_inv_sigma),
     T_coeff(w_star, 1.0, alpha)
{
   MFEM_VERIFY(pfes.GetMesh()->Dimension() == 3, "3D only");
   MFEM_VERIFY(w_star.FESpace()->GetMesh() == pfes.GetMesh(),
               "w_star must live on the pressure space's mesh");
   MFEM_VERIFY(opt.mu_off <= opt.mu_on, "mu_off must not exceed mu_on");
   MFEM_VERIFY(opt.inner_iterations >= 1, "inner_iterations must be >= 1");
#ifdef MFEM_USE_MPI
   if (auto *pf = dynamic_cast<ParFiniteElementSpace*>(&pfes))
   {
      T_form.reset(new ParBilinearForm(pf));
      inner.reset(new FGMRESSolver(pf->GetComm()));
      comm = pf->GetComm();
      parallel = true;
   }
   else
#endif
   {
      T_form.reset(new BilinearForm(&pfes));
      inner.reset(new FGMRESSolver);
   }
   T_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   T_form->AddDomainIntegrator(new DiffusionIntegrator(T_coeff));
   inner->SetKDim(opt.inner_iterations);
   inner->SetMaxIter(opt.inner_iterations);
   inner->SetRelTol(1e-12);     // effectively a fixed count; guards against breakdown
   inner->SetAbsTol(0.0);
   inner->SetPrintLevel(IterativeSolver::PrintLevel().None());
   inner->iterative_mode = false;
   t.SetSize(height);  t.UseDevice(true);
   r0.SetSize(height); r0.UseDevice(true);
   tensor_active = (opt.mode == Mode::Tensor);
}
```

The tensor form is not assembled until the first `Update()` that activates the tensor mode, so runs that never switch pay nothing.

**Why flexible GMRES for the inner solve.** If bessemer's `lap_inv_sigma` is itself an inner Krylov solve to a tolerance (rather than a fixed number of V-cycles), it is a nonlinear operator and plain GMRES is invalid. FGMRES is correct in both cases and stores only `inner_iterations` extra vectors.

### 5.3 `Update` and `Mult`

```cpp
void RotationalSchurPreconditioner::Update(real_t sigma,
                                           const RotationNumberStats &stats)
{
   bool want = tensor_active;
   switch (opt.mode)
   {
      case Mode::CahouetChabard: want = false; break;
      case Mode::Tensor: want = true; break;
      case Mode::Auto:
         if (opt.criterion == Criterion::MaxMu)
         {
            if (!tensor_active && stats.max_mu > opt.mu_on) { want = true; }
            if (tensor_active && stats.max_mu < opt.mu_off) { want = false; }
         }
         else
         {
            MFEM_VERIFY(stats.threshold == opt.mu_on,
                        "VolumeFraction: compute the stats with threshold mu_on");
            if (!tensor_active && stats.vol_fraction > opt.vol_on) { want = true; }
            if (tensor_active && stats.vol_fraction < opt.vol_off) { want = false; }
         }
         break;
   }
   if (want != tensor_active) { n_switches++; tensor_active = want; }
   if (tensor_active) { AssembleTensor(sigma); }
}

void RotationalSchurPreconditioner::AssembleTensor(real_t sigma)
{
   T_coeff.SetSigma(sigma);
   T_form->Assemble();             // re-runs the partial assembly setup in place
   if (!T_op.Ptr())                // first activation only
   {
      T_form->FormSystemMatrix(ess, T_op);
      inner->SetOperator(*T_op);    // before SetPreconditioner: nothing to forward to
      inner->SetPreconditioner(lap_fixed);
   }
}

void RotationalSchurPreconditioner::Mult(const Vector &r, Vector &z) const
{
   if (!tensor_active)
   {
      lap_inv.Mult(r, z);                    // unchanged Cahouet-Chabard path
   }
   else
   {
      const Vector *rhs = &r;
      if (opt.remove_mean) { r0 = r; RemoveMean(r0); rhs = &r0; }
      inner->Mult(*rhs, z);                  // k FGMRES iterations on L_T
      if (opt.remove_mean) { RemoveMean(z); }
   }
   mass_inv.Mult(r, t);
   z.Add(nu, t);
}

void RotationalSchurPreconditioner::RemoveMean(Vector &v) const
{
   real_t loc[2] = { v.Sum(), (real_t) v.Size() };   // Sum() reduces on the device
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MPI_Allreduce(MPI_IN_PLACE, loc, 2, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, comm);
   }
#endif
   v -= loc[0] / loc[1];
}
```

Semantics to preserve:
- **Cahouet-Chabard mode is bitwise today's preconditioner.** It calls bessemer's own solvers and nothing else (S2 checks this).
- **The tensor operator is formed once.** Re-running `Assemble()` on a partially assembled form recomputes the quadrature data in place, and the constrained operator from the first `FormSystemMatrix` keeps referring to it (verified: identical action to a freshly built form after changing both $\sigma$ and $w^*$).
- **The mode is fixed within a time step.** It changes only in `Update()`. FGMRES would tolerate a change mid-solve, but logging and timing are clearer this way.
- **`remove_mean` only affects the tensor path.** In Cahouet-Chabard mode, bessemer's `lap_inv` already does whatever null-space handling it does. $L_T$ annihilates constants from both sides ($\nabla 1 = 0$), so removing the mean of the right-hand side and of the result is all the tensor path needs.

### 5.4 Switching criteria

- **`MaxMu` (default).** Uses `stats.max_mu`. Simple and predictive, but a few points can dominate it (sharp corners, Section 1.7).
- **`VolumeFraction`.** Uses `stats.vol_fraction`, the fraction of the domain volume with $\mu > \mu_{on}$, so compute the stats with `GetRotationNumberStats(sigma, opt.mu_on)`. Use this if the corner effect pins `max_mu` above the threshold. The default fractions are uncalibrated.
- **Hysteresis** (`mu_off < mu_on`, `vol_off < vol_on`) prevents flapping when the diagnostic hovers near the threshold.
- Both criteria rely on `GetRotationNumberStats` being called every step. Its cost is one pass over the rotation integrator's quadrature data, two device reductions and one `MPI_Allreduce`, negligible next to a solve.

### 5.5 Sign, scaling and coefficients

- **`lap_inv_sigma` must already include the factor $\sigma$**, i.e. invert $-\nabla\cdot(\sigma^{-1}\nabla)$, so that it is the exact inverse of $L_T$ when $\omega^* = 0$. If bessemer applies `sigma * amg(r)`, pass a small wrapper doing exactly that. S2 checks the scaling (tensor mode with $\omega^* = 0$ must reproduce Cahouet-Chabard).
- **`nu`** is whatever coefficient bessemer's Cahouet-Chabard puts on $M_p^{-1}$ ($\nu$, or $\nu + \gamma$ with a grad-div term of weight $\gamma$).
- **`alpha`** must match the rotation integrator's `alpha`, so that $T$ inverts the same zero-order operator $\sigma I + \alpha[\omega^*]_\times$ that is in $A$.
- **Sign.** `Mult` returns an approximation of $+S^{-1}r$ with $S = BA^{-1}B^T$. Keep bessemer's existing sign convention around it.

### 5.6 Cost and memory

- **Cahouet-Chabard mode:** unchanged.
- **Tensor mode, per time step:** one evaluation of $\nabla w^*$ at the pressure form's quadrature points plus the partial assembly setup of a nonsymmetric diffusion form (9 values per quadrature point, against 6 for a symmetric one).
- **Tensor mode, per Schur application:** `inner_iterations` applications of `lap_inv_sigma` (3 by default, against 1 for Cahouet-Chabard), the same number of partially assembled $L_T$ applications, one mass solve, and the FGMRES orthogonalization over 3 vectors.

### 5.7 MFEM facts and gotchas this part depends on (verified at `c362dea2`)

1. **Batched low-order-refined assembly ignores matrix coefficients.** `BatchedLORAssembly::FormIsSupported` accepts any form made of `DiffusionIntegrator` and `MassIntegrator`; `ProjectLORCoefficient` reads `GetCoefficient()`, which is null for a `MatrixCoefficient`, and then uses coefficient 1 (`fem/lor/lor_batched.hpp`). Never pass `RotatingDarcyTensor` to a form that feeds `LORSolver`.
2. **A general `MatrixCoefficient` is projected on the host** (`MatrixCoefficient::Project` loops over elements calling `Eval`). Overriding `Project(QuadratureFunction&, bool)` (virtual) keeps it on the device; `CoefficientVector::Project(MatrixCoefficient&, bool)` calls the override for non-constant coefficients (`fem/coefficient.cpp`).
3. **Krylov solvers forward `SetOperator` to their preconditioner** (`IterativeSolver::SetOperator` calls `prec->SetOperator`). Handing $L_T$ to bessemer's `HypreBoomerAMG` or `LORSolver` would abort or rebuild it. Hence `FixedPreconditioner`, and the call order in `AssembleTensor`. `SetPreconditioner` also sets the preconditioner's `iterative_mode = false`; the wrapper keeps that off bessemer's object.
4. **Partially assembled `DiffusionIntegrator` takes the nonsymmetric path automatically** when the projected coefficient has $3\times3 = 9$ components (`symmetric = (coeff_dim != dims*dims)` in `fem/integ/bilininteg_diffusion_pa.cpp`). Its `AddMultTransposePA` aborts in the nonsymmetric case, so never use a solver that calls `MultTranspose` on $L_T$ (FGMRES and GMRES do not).
5. **`QuadratureInterpolator`s are cached by `IntegrationRule` address** in the `FiniteElementSpace` (`GetQuadratureInterpolator(const IntegrationRule&)` compares `qi->IntRule == &ir`). `Project` passes the rule held by the form's `QuadratureSpace`, which for `DiffusionIntegrator::AssemblePA` is a rule from the global `IntRules` and lives for the whole run. A rule owned by a shorter-lived object would leave a dangling cache entry in $w^*$'s space (this crashed the Part D prototype; Section 7.4 uses persistent rules for that reason).
6. **`OrthoSolver` runs its projection on the host** (`HostRead`/`HostWrite`, with a "TODO: GPU" in `linalg/solvers.cpp`). Do not use it on the GPU; `RemoveMean` above stays on the device.
7. **`QuadratureSpace` stores pointers to rules, not copies**, so `qf.GetSpace()->GetIntRule(0)` returns the rule the form was assembled with.

---

## 6. Part C tests (`test_rotational_schur.cpp`)

Conventions as in the Part A/B tests: Catch2, `GENERATE` over devices (`cpu`, `debug`, and `cuda` or `hip` when available), `CAPTURE` every parameter, fixed random seeds. Mesh unless stated: $3\times3\times3$ hexahedra on the unit cube, curved (mesh order 4) with the perturbation of Appendix B, velocity order 4, pressure order 3, pressure Dirichlet on boundary attribute 1. Velocity field `vel_fn` of Appendix B with amplitude 5. $\sigma = 1.5$, $\alpha = 1$.

Stand-ins for bessemer's solvers in S1 and S2: `lap_inv_sigma` and `mass_inv` are CG to $10^{-14}$ with `OperatorJacobiSmoother` on partially assembled $-\nabla\cdot(\sigma^{-1}\nabla)$ and $M_p$. S3 uses bessemer's real Laplacian algebraic multigrid.

**S1. The tensor operator.**
- Partially assembled $L_T$ (device `Project`) against legacy assembly with the same `RotatingDarcyTensor` (host `Eval`): relative difference $\le 10^{-12}$ (calibrated $3.3\times10^{-15}$). This also catches a mishandled `transpose` flag.
- Null spaces: $\|L_T\mathbf{1}\|/\|L_T x\| \le 10^{-13}$ (calibrated $1.9\times10^{-16}$) and $|\mathbf{1}^T L_T x|/(\|\mathbf{1}\|\|L_T x\|) \le 10^{-13}$ (calibrated $1.7\times10^{-17}$), on a form without essential dofs.
- Nonsymmetry is present: $|x^TL_Tv - v^TL_Tx|/|x^TL_Tv| > 10^{-2}$ (calibrated 0.14).
- `Fill` is the inverse: $\|T(\sigma I + [o]_\times) - I\|_{\max} \le 10^{-14}$ for random $\sigma$, $o$ (calibrated $2.2\times10^{-16}$).
- $\omega^* = 0$: $L_T = L_p/\sigma$ to $10^{-14}$ (calibrated $1.8\times10^{-17}$).
- Update semantics: in `Tensor` mode call `Update(sigma1, ...)`, change $w^*$ in place and call `Update(sigma2, ...)`; `GetTensorOperator()` must act like a freshly built and constrained form with `sigma2` and the new $w^*$, to $10^{-14}$ (calibrated: bitwise equal).

**S2. Modes, switching and plumbing.**
- `Auto`/`MaxMu` with defaults and `max_mu` sequence 5, 15, 25, 15, 8, 25: modes Cahouet-Chabard, Cahouet-Chabard, tensor, tensor, Cahouet-Chabard, tensor; `NumSwitches() == 3`. `Auto`/`VolumeFraction` with `vol_fraction` sequence $10^{-4}$, $5\times10^{-4}$, $2\times10^{-3}$, $5\times10^{-4}$, $10^{-4}$, $2\times10^{-3}$: same modes.
- `CahouetChabard` mode stays in Cahouet-Chabard for `max_mu = 1e6`, and `Mult` equals `lap_inv(r) + nu * mass_inv(r)` bitwise.
- `Tensor` mode with $w^* = 0$ and the accurate stand-in `lap_inv_sigma`: `Mult` equals the Cahouet-Chabard `Mult` to $10^{-12}$ (calibrated $4.2\times10^{-15}$). This checks the $\sigma$ scaling contract of Section 5.5.
- `remove_mean = true` on a pure-Neumann pressure space: $|\mathrm{mean}(z)| \le 10^{-14}\|z\|_\infty$ (calibrated $1.8\times10^{-18}$), on 1 and 4 ranks.
- `SetOperator` is never forwarded: pass a `lap_inv_sigma` whose `SetOperator` calls `MFEM_ABORT`; two `Update` calls in `Tensor` mode and a `Mult` must succeed.

**S3. Iterations on a small saddle-point problem (`[Parallel]`, 1 rank; CPU).**
Taylor-Hood Q4/Q3 on an uncurved $3^3$ mesh, Dirichlet velocity on the whole boundary (pure-Neumann pressure, `remove_mean = true`), $\sigma = 1$, $\nu = 10^{-4}$. Velocity operator $A = \sigma M + \nu K + N$ with Part A's integrator; in the block lower-triangular preconditioner the velocity block is solved to $10^{-10}$ by GMRES with `PointBlockJacobi` (test only). Outer FGMRES, Krylov dimension 400, relative tolerance $10^{-8}$, random velocity right-hand side, zero pressure right-hand side. Schur: bessemer's real Cahouet-Chabard pieces, in `CahouetChabard` and in `Tensor` mode (3 inner iterations). Fields: `vel_fn` with amplitude 0.5 ($\mu_{\max} \approx 1.8$) and 5 ($\mu_{\max} \approx 18$).
- Every solve converges within 1000 iterations.
- At $\mu_{\max} \approx 18$: tensor outer count $\le 0.75\times$ the Cahouet-Chabard outer count. Prototypes: 158 vs 307 with exact Laplacian solves, 165 vs 366 with a V-cycle-quality stand-in.
- At $\mu_{\max} \approx 1.8$: tensor outer count $\le$ the Cahouet-Chabard outer count + 3. Prototypes: 40 vs 65, 42 vs 72.
- Record all counts in the test output. If the 0.75 bound fails narrowly with the real multigrid, investigate before relaxing it.

---

## 7. Part D: the velocity block

### 7.1 The ladder and when to escalate

| Level | $\hat A^{-1}$ | Good when | Cost per application |
|---|---|---|---|
| 0 (start) | Part B `pbj_krylov`: GMRES on the full $A$, `PointBlockJacobi` preconditioner, relative tolerance about $10^{-2}$, at most about 30 iterations | $\hat v_{\max} \lesssim 1$, any $\mu$ | per inner iteration: 1 operator application + 1 batched block solve |
| 1 | `RotationalVelocityMultigrid`, one V(2,2) cycle per application (`mg_vcycle`) | isotropic meshes, any $\hat v$, any $\mu$ tested; not stretched boundary-layer meshes | about 4 finest-level operator applications + 4 block solves, plus about 1/8 of that on the next level |

Measured (velocity block alone, FGMRES to $10^{-8}$, Appendix A, Table A3): Level 0 went from 20 to 46 iterations as $v$ went from 0.014 to 144 with no rotation, and up to 96 with rotation; Level 1 stayed at 7 to 15 across the whole sweep and on a finer mesh. Counting a V-cycle as about 6 finest-level operator applications against about 1 per Level 0 iteration, Level 1 pays off once Level 0 needs roughly 6 times more iterations than Level 1, and it removes the growth with $v$ and with mesh refinement that Level 0 has.

**Escalate from Level 0 to Level 1 when** any of these shows up in the per-step log (Section 9): the Level 0 inner solve regularly hits its iteration cap; outer FGMRES counts rise when the cap is hit; the (1,1) solve's share of step time becomes significant; or $\hat v_{\max}$ is well above 1 in the cells where the cap is hit. For the adaptively refined bluff body at 100 times the Courant limit, Level 0 is expected to be enough: on isotropic adaptive refinement Level 0 needed 24 to 45 inner iterations to $10^{-2}$ against about 3 to 4 V-cycles for Level 1, so Level 1 saves perhaps a factor of 1.5 to 2 in velocity-block work there, and on a stretched boundary-layer mesh it saved nothing (Table A5). Escalate only if the log shows the velocity block has become a large share of the step.

The selection is a run-time option (`--velocity-block pbj_krylov|mg_vcycle`), not automatic.

### 7.2 Level 0

Part B, Section 7.5, unchanged. Use `pbj_krylov` rather than `pbj_only`: with the Schur complement dominating the cost, a cheap (1,1) approximation that raises outer iteration counts is a bad trade.

### 7.3 Viscous-ratio diagnostic

$$\hat v_{\max} = \frac{\nu}{\sigma}\,\frac{\max_a K_{aa}/M_{aa}}{c_p},\qquad c_p = \frac{\max_a K_{aa}/M_{aa}\ \text{on one unit-cube element of order } p}{p^2},$$

with $K$, $M$ the scalar stiffness and mass matrices of the velocity order (default partial-assembly rules). The normalization makes $\hat v_{\max}$ equal $\nu/(\sigma(h_{el}/p)^2)$ exactly on a uniform hexahedral mesh, so it is in the same units as the $v$ of this document; on a locally refined mesh it reports the most refined region. Calibrated $c_p$: 20.22 ($p = 3$), 29.53 (4), 40.92 (5), 54.35 (6), 69.80 (7); independent of the number of elements.

Implementation: build a scalar H1 space of the velocity order, partially assembled `MassIntegrator` and `DiffusionIntegrator` forms, `AssembleDiagonal` both (no essential dofs), a `forall` for the ratio, `Vector::Max()`, `MPI_Allreduce` with `MPI_MAX`. Do this once per mesh (again after adaptive refinement). $c_p$ is computed once at setup the same way on a one-element unit-cube `Mesh`. Per step, $\hat v_{\max}$ is a scalar multiplication by $\nu/\sigma$.

```cpp
class ViscousRatioDiagnostic
{
public:
   /// vfes: the velocity space (any vdim; a scalar space of the same order is used).
   explicit ViscousRatioDiagnostic(const FiniteElementSpace &vfes);
   void UpdateMesh();                       // after adaptive refinement
   real_t VHatMax(real_t sigma, real_t nu) const { return nu / sigma * r_max / c_p; }
private:
   real_t r_max = 0, c_p = 0;
   ...
};
```

### 7.4 Level 1: p-multigrid with point-block Jacobi smoothing

**Hierarchy.** A `ParFiniteElementSpaceHierarchy` on bessemer's mesh, coarsest order 1, coarsening the order by about 2 per level, finest order = bessemer's velocity order: $p = 4$: {1, 2, 4}; $p = 5$: {1, 2, 5}; $p = 6$: {1, 3, 6}; $p = 8$: {1, 2, 4, 8}. Make the list an option. Every level must use bessemer's velocity `Ordering`: `AddOrderRefinedLevel(fec, dim, ordering)` defaults to `byVDIM`. The hierarchy does not own the collections passed to `AddOrderRefinedLevel`; keep them alive.

**Level operators.**
- **Finest level:** bessemer's existing constrained velocity operator (with $N$) and its existing `PointBlockJacobi`, not owned, not duplicated. The hierarchy's finest space has the same order, ordering and mesh, so the true-dof layouts coincide.
- **Coarser levels $l$:** a partially assembled form on the level space with the same integrators as bessemer's velocity form: `VectorMassIntegrator(sigma)`, `VectorDiffusionIntegrator(nu)`, any grad-div term, and a `VectorRotationalConvectionIntegrator(w_l, alpha)`. Here $w_l$ is $w^*$ interpolated to order $p_l$ (below). Constrain with `FormSystemMatrix` using the level's essential list from `GeometricMultigrid`. A `PointBlockJacobi` per level, with `SetDiagonal` from the level form's `AssembleDiagonal`.

**Transferring $w^*$ to coarser levels.** Part A requires $w$ in the integrator's own space, so each coarser level gets its own `ParGridFunction` $w_l$, filled every step by nodal interpolation on the device: evaluate the fine $w^*$ at the coarse element's GLL nodes with the fine space's `QuadratureInterpolator`, which yields exactly the coarse E-vector layout, then pick one copy per dof with `ElementRestriction::MultLeftInverse`. Verified bitwise equal to `GridFunction::ProjectGridFunction` (Section 8, V1). On nonconforming meshes $w_l$ is not exactly conforming at hanging faces, which is harmless for a preconditioner.

```cpp
/// GLL rules that live for the whole run. FiniteElementSpace caches its
/// QuadratureInterpolators by IntegrationRule address, so a rule owned by a
/// shorter-lived object leaves a dangling cache entry in the space.
inline IntegrationRules &GLLRules()
{
   static IntegrationRules gll(0, Quadrature1D::GaussLobatto);
   return gll;
}

/// Nodal interpolation of an H1 vector field onto a lower-order H1 space on
/// the same mesh (same vdim and ordering). Runs on the device.
class OrderInterpolator
{
public:
   OrderInterpolator(const FiniteElementSpace &fine, const FiniteElementSpace &coarse)
   {
      MFEM_VERIFY(fine.GetMesh() == coarse.GetMesh(), "same mesh required");
      MFEM_VERIFY(fine.GetVDim() == coarse.GetVDim(), "same vdim required");
      const int pc = coarse.GetMaxElementOrder();
      const Geometry::Type geom = coarse.GetMesh()->GetTypicalElementGeometry();
      const IntegrationRule &ir = GLLRules().Get(geom, 2*pc - 1);  // pc+1 GLL points per direction
      Rf = fine.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      Rc = dynamic_cast<const ElementRestriction*>(
              coarse.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
      MFEM_VERIFY(Rc, "coarse space must use an ElementRestriction");
      qi = fine.GetQuadratureInterpolator(ir);
      ef.SetSize(Rf->Height()); ef.UseDevice(true);
      ec.SetSize(Rc->Height()); ec.UseDevice(true);
   }
   void Mult(const Vector &w_fine, Vector &w_coarse) const
   {
      Rf->Mult(w_fine, ef);
      qi->SetOutputLayout(QVectorLayout::byNODES);   // (nq, vdim, ne) = coarse E-vector layout
      qi->Values(ef, ec);
      Rc->MultLeftInverse(ec, w_coarse);
   }
private:
   const Operator *Rf;
   const ElementRestriction *Rc;
   const QuadratureInterpolator *qi;
   mutable Vector ef, ec;
};
```

The coarse element's nodes are the GLL points (MFEM's default H1 basis), and both the tensor rule and the lexicographic E-vector order them x-fastest, which is why the interpolated values can be used directly as the coarse E-vector.

**Smoother.** Point-block Jacobi, damped, 2 pre- and 2 post-smoothing steps:

```cpp
/// y = theta * S(x). MultTranspose applies the same (non-transposed)
/// smoother: MFEM's MultigridBase post-smooths through MultTranspose.
class DampedSmoother : public Solver
{
public:
   DampedSmoother(const Solver &S, real_t theta)
      : Solver(S.Height(), S.Width()), S(S), theta(theta) { }
   void Mult(const Vector &x, Vector &y) const override { S.Mult(x, y); y *= theta; }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &) override { }
private:
   const Solver &S;
   const real_t theta;
};
```

**Damping from an eigenvalue estimate, not a constant.** Set $\theta_l = 1/\lambda_{\max}$ per level, where $\lambda_{\max}$ is the magnitude of the dominant eigenvalue of (point-block Jacobi) $\times A_l$, from about 20 to 60 power iterations at setup and again in `SetCoefficients`. A fixed $\theta = 0.7$ worked on uniform meshes but diverged on a stretched mesh, where $\lambda_{\max}$ reached 3.1, above $2/0.7$ (p-multigrid took 799 iterations; with the estimate, 73). This is the same reason MFEM's `examples/ex26p.cpp` smooths with `OperatorChebyshevSmoother` and an eigenvalue estimate. Without the `MultTranspose` override the cycle aborts at the first post-smoothing step (`Operator::MultTranspose() is not overridden`), since `PointBlockJacobi` has none.

**Coarse solver (order 1).** FGMRES with a fixed `coarse_iterations = 10` (relative tolerance $10^{-12}$ as a breakdown guard, print level `None`, `iterative_mode = false`), preconditioned by the order-1 `PointBlockJacobi`. In the prototypes this matched an exact coarse solve on $3^3$ and $8^3$ meshes; 4 damped block-Jacobi sweeps instead cost up to 4 extra outer iterations at large $v$. Because the coarse FGMRES is nonlinear, Level 1 must only be used inside flexible Krylov methods; bessemer's outer FGMRES qualifies. The order-1 level still spans the whole fine mesh, so on large production meshes its viscous part may eventually need more than block Jacobi. If coarse cost or Level 1 iteration counts grow with mesh size, revisit the coarse solver, measuring before changing it.

### 7.5 Interface (`velocity_multigrid.hpp`)

```cpp
class RotationalVelocityMultigrid : public GeometricMultigrid
{
public:
   struct Options
   {
      real_t theta = 0.0;   // 0: estimate 1/lambda_max(PBJ A) per level (recommended)
      int pre_smooth = 2, post_smooth = 2;
      int coarse_iterations = 10;
   };

   /** hierarchy:  velocity spaces, order 1 first; finest = bessemer's velocity
                   order and ordering (not owned).
       ess_bdr:    velocity essential boundary attributes (all components).
       w_star:     bessemer's extrapolated velocity on its velocity space.
       fine_op:    bessemer's constrained velocity operator, with N (not owned).
       fine_pbj:   bessemer's PointBlockJacobi for fine_op (not owned).
       vel_ess:    bessemer's velocity essential true dofs (checked against
                   the hierarchy's finest list). */
   RotationalVelocityMultigrid(ParFiniteElementSpaceHierarchy &hierarchy,
                               const Array<int> &ess_bdr,
                               const ParGridFunction &w_star, real_t alpha,
                               real_t sigma, real_t nu,
                               const Operator &fine_op,
                               const PointBlockJacobi &fine_pbj,
                               const Array<int> &vel_ess,
                               const Options &opt);

   /// Every step, after w* is formed: interpolate w* to the coarser levels,
   /// UpdateVorticity() and UpdateSkew() on each of them.
   void UpdateVorticity();

   /// When dt (sigma) or nu changes: re-assemble the coarser levels in place
   /// and call SetDiagonal on their PointBlockJacobi objects.
   void SetCoefficients(real_t sigma, real_t nu);

private:
   ConstantCoefficient sigma_coeff, nu_coeff;
   std::vector<std::unique_ptr<ParGridFunction>> w_l;          // levels < finest
   std::vector<std::unique_ptr<OrderInterpolator>> interp;     // levels < finest
   std::vector<VectorRotationalConvectionIntegrator*> rot_l;   // owned by the forms in bfs
   std::vector<OperatorHandle> op_l;                           // constrained level operators
   std::vector<std::unique_ptr<PointBlockJacobi>> pbj_l;
   const ParGridFunction &w_star;
};
```

### 7.6 Construction (pattern of MFEM's `examples/ex26p.cpp`, `DiffusionMultigrid`)

1. `GeometricMultigrid(hierarchy, ess_bdr)` builds the per-level essential lists and wraps each prolongation in a `RectangularConstrainedOperator` that keeps essential dofs at zero. Verify that `*essentialTrueDofs[finest]` equals `vel_ess` (sizes and entries); abort otherwise (component-wise boundary conditions are out of scope).
2. For each level `l` from 0 to finest − 1:
   - `w_l[l]` on the level space; `interp[l]` = `OrderInterpolator(*w_star.FESpace(), level space)`; `interp[l]->Mult(w_star, *w_l[l])`.
   - A partially assembled `ParBilinearForm` with the integrators listed in 7.4 (sharing `sigma_coeff` and `nu_coeff`); `Assemble()`; append it to the base class's `bfs` (it deletes them); keep the raw pointer to its rotation integrator in `rot_l`.
   - `FormSystemMatrix(*essentialTrueDofs[l], op_l[l])`.
   - `pbj_l[l] = PointBlockJacobi(level space, *rot_l[l], *essentialTrueDofs[l])`; `AssembleDiagonal` of the level form into a true-dof vector; `SetDiagonal`; `UpdateSkew()`.
   - Level 0: a new `FGMRESSolver` configured as in 7.4 (`SetOperator(*op_l[0])` first, then `SetPreconditioner(*pbj_l[0])`); `AddLevel(op_l[0].Ptr(), coarse_fgmres, false, true)`. Other levels: `AddLevel(op_l[l].Ptr(), new DampedSmoother(*pbj_l[l], theta_l), false, true)` with `theta_l` from the power-iteration estimate (or `opt.theta` if nonzero).
3. Finest level: `AddLevel(const_cast<Operator*>(&fine_op), new DampedSmoother(fine_pbj, theta_L), false, true)`, with `theta_L` estimated the same way on the finest operator.
4. `SetCycleType(Multigrid::CycleType::VCYCLE, pre_smooth, post_smooth)`.

`UpdateVorticity()`: for each level below the finest, `interp[l]->Mult(w_star, *w_l[l])`, `rot_l[l]->UpdateVorticity()`, `pbj_l[l]->UpdateSkew()`. The finest level is updated by bessemer as today.

`SetCoefficients(sigma, nu)`: set the two `ConstantCoefficient`s, re-`Assemble()` each coarser form (in place; the constrained operators stay valid, as verified for Part C), recompute each diagonal and call `SetDiagonal`. Bessemer updates the finest level as today.

**Memory:** coarser levels hold about $(p_l/p)^3$ of the finest level's partial assembly data each, about 14% extra in total for $p = 4$ with levels {1, 2, 4}.

### 7.7 Using it as the (1,1) block

Wire it as a third value of bessemer's existing velocity-block option: `mg_vcycle` applies one V-cycle (`RotationalVelocityMultigrid::Mult`) per outer iteration. Optionally `mg_krylov`: FGMRES (not GMRES, because of the nonlinear coarse solve) on the full $A$ preconditioned by one V-cycle, relative tolerance $10^{-2}$, at most 10 iterations. Start with `mg_vcycle`: reaching $10^{-8}$ in 7 to 15 FGMRES iterations corresponds to an error reduction of roughly 3 to 10 per V-cycle, already a good $\hat A^{-1}$ for an outer FGMRES. `Multigrid` ignores the initial guess, so `iterative_mode` is irrelevant here.

### 7.8 MFEM facts this part depends on (verified at `c362dea2`)

- `MultigridBase::Cycle` post-smooths with `SmoothingStep(level, false, true)`, i.e. the smoother's `MultTranspose` (`fem/multigrid.cpp`).
- `GeometricMultigrid(const FiniteElementSpaceHierarchy&, const Array<int>& ess_bdr)` computes `essentialTrueDofs` per level with `GetEssentialTrueDofs(ess_bdr, ...)` (all components) and wraps prolongations in `RectangularConstrainedOperator`. The single-argument constructor is deprecated and does not constrain the transfers.
- `FiniteElementSpaceHierarchy::AddOrderRefinedLevel(fec, dim = 1, ordering = Ordering::byVDIM)`.
- `ElementRestriction::MultLeftInverse` runs on the device and takes, for each L-dof, the value from one element (`fem/restriction.cpp`).
- `FiniteElementSpace::GetQuadratureInterpolator(const IntegrationRule&)` caches by rule address (Section 5.7, item 5).
- MFEM's `ProductSolver` allocates two temporaries per call without `UseDevice(true)`, so zeroing them runs on the host. Not used here (Section 1.5), but worth knowing if a two-stage preconditioner is ever revisited.

---

## 8. Part D tests (`test_velocity_multigrid.cpp`)

Velocity operator in V2 and V3: Part A's integrator plus `VectorMassIntegrator(sigma)` and `VectorDiffusionIntegrator(nu)`, partial assembly, Dirichlet on the whole boundary, uncurved $3^3$ hexahedral unit cube unless stated, $\sigma = 1$, $p = 4$, levels {1, 2, 4}, `byNODES`, default options. $v$ is set through $\nu = v\,\sigma\,(h_{el}/p)^2$ with $h_{el} = 1/3$. Field `vel_fn` with amplitude 0, 5, 16 ($\mu_{\max}$ = 0, 18.3, 58.6). FGMRES, Krylov dimension 100, relative tolerance $10^{-8}$, random right-hand side with zeros on essential dofs.

**V1. Building blocks (CPU and `debug`; GPU when available).**
- `OrderInterpolator` against `GridFunction::ProjectGridFunction` (host reference) for orders 4 to 2 and 4 to 1, both orderings, on the curved mesh of Appendix B: relative difference $\le 10^{-14}$ (calibrated: bitwise equal).
- Lifetime: construct an `OrderInterpolator`, destroy it, construct another for the same pair of spaces, and interpolate again; the result must be unchanged. With a rule owned by the interpolator this sequence crashed in the prototype (Section 5.7, item 5).
- `ViscousRatioDiagnostic` on uniform $n^3$ meshes, $p \in \{3, 4, 5\}$: `VHatMax(sigma, nu)` equals $\nu/(\sigma(1/(np))^2)$ to $10^{-12}$; going from $n = 3$ to $n = 6$ multiplies it by 4 to $10^{-12}$.
- `DampedSmoother::MultTranspose` equals `Mult` bitwise.

**V2. Robustness sweep (CPU; `[Parallel]` 1 rank).** For $v \in \{0.0144, 1.44, 14.4, 144\}$ and the three amplitudes, solve with one V-cycle of `RotationalVelocityMultigrid` per FGMRES iteration. Assert every case converges in at most 20 iterations (calibrated 7 to 15 with $\theta = 0.7$; Table A3; recalibrate with the eigenvalue-based damping, which gave 11 to 14 on the adaptively refined mesh of Table A5). Add one case on the stretched mesh of Table A5 ($\nu = 10^{-2}$, 100 times the Courant step): assert convergence within 120 iterations (calibrated 73). Also run Level 0 (FGMRES with `PointBlockJacobi`) on the same cases and print both counts; no assertion on Level 0.

**V3. Parallel, device and updates (`[Parallel]` 1 and 4 ranks; `debug`, and GPU when available).** One case, $v = 14.4$, amplitude 16:
- Iteration counts on 1 and 4 ranks within 2 of each other.
- Change $w^*$ in place (amplitude 8), call bessemer's finest-level updates (`UpdateVorticity` and `UpdateSkew`) and `RotationalVelocityMultigrid::UpdateVorticity()`; then `SetCoefficients(2 sigma, nu)` with the finest level updated too. After each change, the iteration count must be within 1 of a freshly constructed multigrid with the same data, and the first V-cycle's output must equal the fresh one's to $10^{-12}$. Catches stale $w_l$, stale skew data and stale diagonals on the coarse levels.

---

## 9. Per-step diagnostics and logging

Log one line per time step (or every N steps if output volume matters; compute every step regardless, because the Schur switch uses it):

| Field | Source |
|---|---|
| step, $\Delta t$, $\sigma$ | bessemer |
| $\mu_{\max}$, volume fraction with $\mu > \mu_{on}$ | `rot->GetRotationNumberStats(sigma, mu_on)` |
| $\hat v_{\max}$ | `ViscousRatioDiagnostic::VHatMax(sigma, nu)` |
| Schur mode this step, total switches | `RotationalSchurPreconditioner::TensorActive()`, `NumSwitches()` |
| velocity-block level | run-time option |
| outer FGMRES iterations | bessemer |
| total (1,1) inner iterations, and how many solves hit the cap | bessemer's (1,1) solver wrapper |
| wall time: Schur applications, (1,1) applications, whole step | `tic_toc` or the profiler |

These are what the thresholds in Sections 5.4 and 7.1 should be calibrated against.

---

## 10. Integration: order of operations per time step

```text
form w* = 3u^n - 3u^{n-1} + u^{n-2};  w*.SetFromTrueDofs(...)      (as today)
if dt or nu changed:
    update bessemer's velocity form and PointBlockJacobi::SetDiagonal   (as today)
    if Level 1: mg.SetCoefficients(sigma, nu)
rot->UpdateVorticity();  pbj.UpdateSkew();                           (as today)
if Level 1: mg.UpdateVorticity()
stats = rot->GetRotationNumberStats(sigma, opt.mu_on)
schur.Update(sigma, stats)
solve with outer FGMRES and the block preconditioner
log (Section 9)
```

Other notes:
- **Adaptive time stepping:** neither Schur mode needs an algebraic multigrid rebuild when $\Delta t$ changes. In Cahouet-Chabard $\sigma$ only scales the Laplacian solve; in the tensor mode $\sigma$ enters only the partially assembled $L_T$, re-assembled every step anyway. Level 1 re-assembles its coarser partially assembled levels (Section 7.6); no algebraic multigrid is involved anywhere in it.
- **Adaptive mesh refinement:** after a mesh change, rebuild the Schur preconditioner (its tensor form lives on the pressure space), the hierarchy and multigrid, and call `ViscousRatioDiagnostic::UpdateMesh()`. Bessemer already rebuilds its Laplacian algebraic multigrid then.
- **Calibration runs:** run one representative bluff-body case at the target $\Delta t$ three ways: `--schur cc`, `--schur tensor`, `--schur auto`. Compare Schur wall time per step against the logged $\mu_{\max}$ and volume fraction, and move `mu_on`/`mu_off` (or switch to `VolumeFraction`) accordingly.

---

## 11. Work order and acceptance

1. `RotatingDarcyTensor`; S1 on `cpu` and `debug`.
2. `RotationalSchurPreconditioner`; S2.
3. Wire Part C into bessemer behind `--schur cc|tensor|auto` (default `cc` until calibrated); S3.
4. Per-step logging (Section 9), including `ViscousRatioDiagnostic` (V1's diagnostic section).
5. Calibration run (Section 10). Set the `--schur` default.
6. Only if the log calls for it (Section 7.1): `OrderInterpolator`, `DampedSmoother`, `RotationalVelocityMultigrid`; V1 to V3; wire `--velocity-block mg_vcycle`.

Acceptance: all tests pass on `cpu` and `debug`, and on GPU when available; no new compiler warnings; only the files of Section 3.

---

## Appendix A: prototype results

Host-only prototypes at MFEM commit `c362dea2`, uncurved unit-cube hexahedral meshes, velocity $p = 4$, pressure $p = 3$, Dirichlet velocity on the whole boundary, $\sigma = 1$, field `vel_fn` (Appendix B). $v = \nu/(\sigma(h_{el}/p)^2)$. $N$ was assembled with `VectorMassIntegrator` and the skew `MatrixCoefficient` of Part A's Appendix B; point-block Jacobi was built from the assembled matrix. "Exact" inner solves are CG or GMRES to $10^{-12}$.

**Table A1. Schur approximations, exact inner solves (outer FGMRES iterations to $10^{-8}$, unrestarted).**

| $v$ | $\mu_{\max}$ | Cahouet-Chabard | scalar $\kappa$ | tensor, exact $L_T^{-1}$ |
|---|---|---|---|---|
| 0.014 | 1.8 | 65 | 75 | 35 |
| 2.56 ($4^3$ mesh) | 1.8 | 33 | 42 | 33 |
| 0.014 | 18 | 307 | 334 | 55 |
| 1.44 | 18 | 99 | 150 | 60 |

**Table A2. Practical variants: outer iterations / pressure-Laplacian applications.**

| $v$ | $\mu_{\max}$ | restart | Cahouet-Chabard | tensor, inner tolerance 0.1 | tensor, 3 inner iterations |
|---|---|---|---|---|---|
| 0.014 | 1.8 | none | 65 / 65 | 37 / 163 | 40 / 120 |
| 0.014 | 18 | none | 307 / 307 | 63 / 1541 | 158 / 474 |
| 0.014 | 59 | none | 783 / 783 | 100 / 3037 | 264 / 792 |
| 0.014 | 183 | 400 | not converged in 1500 | 161 / 5532 | 455 / 1365 |
| 0.014 | 18 | 50 | 531 / 531 | 65 / 1635 | 242 / 726 |
| 0.014 | 59 | 50 | 1443 / 1443 | 111 / 3550 | 533 / 1599 |
| 1.44 | 18 | none | 99 / 99 | 53 / 1485 | 68 / 204 |
| 0.014, V-cycle-quality stand-in | 1.8 | none | 72 / 72 | 38 / 173 | 42 / 126 |
| 0.014, V-cycle-quality stand-in | 18 | none | 366 / 366 | 64 / 1672 | 165 / 495 |

Inner solve alone ($4^3$ mesh, GMRES to $10^{-2}$ on $L_T$): preconditioned by the exact constant-coefficient Laplacian, 8, 24, 28, 28 iterations at $\mu_{\max}$ = 1.8, 18, 183, 1833; preconditioned by the exact scalar-$\kappa$ operator, 9, 46, 87, 84.

**Table A3. Velocity block alone (FGMRES, Krylov dimension 100, to $10^{-8}$; $3^3$ mesh).** "Multigrid stand-in" is the exact inverse of $\sigma M + \nu K$, standing in for algebraic multigrid on that operator and flattering it. "Product" is point-block Jacobi followed by the multigrid stand-in on the residual (`ProductSolver` semantics). "Factored" is $H^{-1}D(D+N_L)^{-1}$. "p-multigrid" is Level 1 (levels {1, 2, 4}, V(2,2), damping 0.7, coarse FGMRES(10)).

| $v$ | $\mu_{\max}$ | Jacobi | point-block Jacobi | multigrid stand-in | product | factored | p-multigrid |
|---|---|---|---|---|---|---|---|
| 0.0144 | 0 | 20 | 20 | 1 | 1 | 1 | 9 |
| 0.0144 | 18 | 797 | 30 | 249 | 141 | 14 | 13 |
| 0.0144 | 59 | > 2000 | 41 | 569 | 401 | 16 | 15 |
| 1.44 | 0 | 33 | 33 | 1 | 1 | 1 | 7 |
| 1.44 | 18 | 154 | 56 | 105 | 39 | 169 | 9 |
| 1.44 | 59 | 353 | 50 | 332 | 97 | 197 | 12 |
| 14.4 | 0 | 44 | 44 | 1 | 1 | 1 | 8 |
| 14.4 | 18 | 95 | 85 | 30 | 26 | 210 | 10 |
| 14.4 | 59 | 146 | 96 | 64 | 43 | 349 | 11 |
| 144 | 0 | 46 | 46 | 1 | 1 | 1 | 8 |
| 144 | 18 | 70 | 69 | 10 | 9 | 158 | 8 |
| 144 | 59 | 81 | 78 | 17 | 16 | 245 | 9 |

p-multigrid on an $8^3$ mesh: 11, 14, 14 iterations at $v = 0.0144$ and 8, 9, 10 at $v = 144$ ($\mu_{\max}$ = 0, 18, 59). With 4 damped point-block Jacobi sweeps as the coarse solver instead of FGMRES(10): 11, 14, 14 and 9, 13, 14. The production structure of Section 7.6 (hierarchy with constrained transfers, interpolated $w_l$ per level) reproduced the $3^3$ p-multigrid column exactly.

**Table A5. Locally refined meshes, $\Delta t$ = 1, 10, 100 times the Courant step of the smallest cell.** Shear layer near the wall $x = 0$ ($u_y = \tanh(x/\delta)$ plus a weaker 3D rotation), Dirichlet everywhere, $p = 4$, levels {1, 2, 4}. Level 0 is FGMRES with point-block Jacobi; Level 1 is p-multigrid with eigenvalue-based damping. $\hat v_{\max}$ and $\mu_{\max}$ are at the smallest cells.

*Isotropic nonconforming refinement* ($3^3$ base, two refinement levels toward the wall, $h$ from 1/12 to 1/3, 69,687 dofs):

| $\Delta t$ / Courant step | $\mu_{\max}$ | $\hat v_{\max}$ | Level 0 to $10^{-2}$ | Level 0 to $10^{-8}$ | Level 1 to $10^{-8}$ |
|---|---|---|---|---|---|
| 1 | 0.28 | 0.03 / 0.3 / 3 | 5 / 6 / 15 | 24 / 23 / 55 | 12 / 11 / 11 |
| 10 | 2.8 | 0.3 / 3 / 30 | 8 / 17 / 31 | 28 / 63 / 114 | 11 / 11 / 12 |
| 100 | 28 | 3 / 30 / 300 | 24 / 38 / 45 | 88 / 141 / 149 | 14 / 14 / 13 |

(The three entries per cell are three viscosities.)

*Stretched boundary-layer mesh* (10 elements in $x$ graded by a factor of 37, 4 in $y$ and $z$; aspect ratio up to about 27; 35,547 dofs):

| $\nu$ | $\Delta t$ / Courant step | $\mu_{\max}$ | $\hat v_{\max}$ | Level 0 to $10^{-2}$ | Level 0 to $10^{-8}$ | Level 1 to $10^{-8}$ |
|---|---|---|---|---|---|---|
| $10^{-4}$ | 1 / 10 / 100 | 0.08 / 0.8 / 7.8 | 0.03 / 0.3 / 2.9 | 6 / 7 / 15 | 26 / 33 / 96 | 13 / 14 / 33 (fixed $\theta$) |
| $10^{-3}$ | 1 / 10 / 100 | same | 0.3 / 2.9 / 29 | 6 / 13 / 34 | 29 / 74 / 197 | 13 / 28 / 86 (fixed $\theta$) |
| $10^{-2}$ | 1 / 10 / 100 | same | 2.9 / 29 / 290 | 11 / 22 / 42 | 58 / 110 / 150 | 30 / 55 / 73 (estimated $\theta$); 799 at 100 with fixed $\theta = 0.7$ |

**Table A4. Building-block checks (curved $3^3$ mesh, Appendix B).**

| Check | Result |
|---|---|
| $L_T$ partial assembly (device `Project`) vs legacy (host `Eval`) | $3.3\times10^{-15}$ |
| $\|L_T\mathbf 1\|/\|L_Tx\|$; $|\mathbf 1^TL_Tx|/(\|\mathbf 1\|\|L_Tx\|)$ | $1.9\times10^{-16}$; $1.7\times10^{-17}$ |
| $\|T(\sigma I+[o]_\times) - I\|_{\max}$ | $2.2\times10^{-16}$ |
| $\omega^* = 0$: $L_T$ vs $L_p/\sigma$ | $1.8\times10^{-17}$ |
| Re-`Assemble()` with new $\sigma$ and $w^*$ vs a fresh form | bitwise equal |
| Tensor mode ($\omega^* = 0$, exact Laplacian) vs Cahouet-Chabard | $4.2\times10^{-15}$ |
| Batched low-order-refined form with the tensor vs with coefficient 1 | identical (the tensor is ignored) |
| `OrderInterpolator` vs `ProjectGridFunction`, orders 4 to 2 and 4 to 1, both orderings | bitwise equal |
| $c_p$ for $p$ = 3, 4, 5, 6, 7 | 20.22, 29.53, 40.92, 54.35, 69.80 |

---

## Appendix B: test helpers

Velocity field (nonuniform, every vorticity component nonzero; `amp` sets $\mu_{\max}$):

```cpp
void vel_fn(const Vector &x, Vector &v)   // amp is a test parameter
{
   v(0) = amp*(sin(2*x(1)) + 0.3*cos(3*x(2)));
   v(1) = amp*(sin(2*x(2)) + 0.5*x(0)*x(0));
   v(2) = amp*(sin(2*x(0)) + 0.2*x(1));
}
```

With $\sigma = 1$ on the unit cube, amplitudes 0.5, 5, 16, 50 give $\mu_{\max}$ = 1.8, 18.3, 58.6, 183 (vorticity sampled at the quadrature points of a $2p$ rule).

Curved mesh: `Mesh::MakeCartesian3D(3, 3, 3, Element::HEXAHEDRON, 1.0, 1.0, 1.0)`, then `mesh.SetCurvature(4, false, -1, Ordering::byNODES)`, then for every node `x += 0.04 sin(pi y)`, `y += 0.04 sin(pi x)`. **Pass `Ordering::byNODES` explicitly**: `SetCurvature` defaults to `byVDIM`, and indexing the node vector as `(i, i + n)` with the default ordering scrambles coordinates and tangles the mesh. A Laplacian on a tangled mesh is indefinite and CG fails immediately, which is how this was found.
