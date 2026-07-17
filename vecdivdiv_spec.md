# Spec: `VectorDivDivIntegrator` — sum-factorized grad-div for H1 vector fields in MFEM

**Target:** MFEM master (verified against commit `951cf888`, 2026-07-14, v4.9.1-dev).
**Deliverable:** A new `BilinearFormIntegrator` for a(u,v) = (Q div u, div v), u,v in [H1(Ω)]^d,
with fused, sum-factorized partial-assembly (PA) GPU kernels, plus full-assembly fallback,
PA diagonal, tests, and a micro-benchmark.
**Audience:** Claude Code, working either in an MFEM checkout (in-tree) or against an installed
MFEM (out-of-tree, e.g. inside bessemer). Both paths are specified; ask the user which one
before starting if unclear.

---

## 1. Motivation and verified current state of MFEM

The grad-div bilinear form (γ ∇·u, ∇·v) is needed for grad-div stabilization / augmented-
Lagrangian formulations of Stokes and incompressible Navier–Stokes, and is the λ-term of
linear elasticity. Today the only way to get it in MFEM on H1 vector spaces is
`ElasticityIntegrator(lambda, mu)` with μ=0, and its PA path is slow on GPUs. Verified facts
from the current source (file:line references are to commit `951cf888`):

1. **`ElasticityIntegrator` PA does not use sum factorization.** Its kernels are explicitly
   documented as "does not use tensor basis" (`fem/integ/bilininteg_elasticity_kernels.hpp:53-55`).
   `AssemblePA` requests `DofToQuad::FULL` / `LEXICOGRAPHIC_FULL` maps
   (`fem/integ/bilininteg_elasticity_pa.cpp:55-58`), i.e. the dense (nq × ndof) basis matrices.
2. **It round-trips the full gradient field through global memory.** `ElasticityAddMultPA_`
   first calls `QuadratureInterpolator::PhysDerivatives(x, QVec)` into a scratch
   `QuadratureFunction` of size nq·d·d·NE, then a second kernel reads it back, applies the
   pointwise material law (recomputing `inv(J)` at every quadrature point on every apply),
   writes back into `QVec`, and finally a third kernel does the E-vector reduction as a
   **dense** contraction `sum_p Q(p,m,q,e) * G(p,m,i)` over all nq points for each of the
   ndof·d outputs (`fem/integ/bilininteg_elasticity_kernels.hpp:170-277`). That last stage is
   O(nq·ndof·d) ≈ O(p⁶) per element in 3D, versus O(p⁴) for sum-factorized kernels — roughly
   an order of magnitude more flops at p ≈ 6, on top of the extra global-memory traffic.
3. **No grad-div integrator exists for H1 vector spaces.** `DivDivIntegrator`
   (`fem/bilininteg.hpp:3116`) is for Raviart–Thomas / H(div) elements only (different
   pullback: div maps through 1/detJ, no J⁻¹ contraction). `VectorDivergenceIntegrator`
   (`fem/bilininteg.hpp:3060`) is the mixed (div u, q) block. Nothing assembles
   (Q div u, div v) on [H1]^d.
4. **`VectorDiffusionIntegrator` is the modern template to copy.** It has fused shared-memory
   sum-factorized PA kernels (`SmemPAVectorDiffusionApply2D/3D` in
   `fem/integ/bilininteg_vecdiffusion_pa.hpp`), built from reusable device helpers in
   `fem/kernels.hpp` (`LoadMatrix`, `LoadDofs3d`, `Grad3d`, `GradTranspose3d`, `WriteDofs3d`,
   register tensors `vd_regs3d_t<VDIM,DIM,N>`), and registers compile-time (D1D,Q1D)
   specializations through `MFEM_REGISTER_KERNELS` (`fem/kernel_dispatch.hpp`).

The new integrator is structurally a `VectorDiffusionIntegrator` whose pointwise operation
couples the vector components through a **rank-1** quadrature-point operator. It is *less*
work than vector diffusion at the quadrature points, not more.

Side note for the Navier–Stokes use case: with grad-div parameter γ added to the momentum
operator, the pressure Schur complement's mass-matrix part scales like (ν+γ) instead of ν,
so a Cahouet–Chabard preconditioner must be updated accordingly. Not part of this spec, but
the reason the coefficient is kept fully general.

---

## 2. Mathematics and quadrature-point factorization

### 2.1 Weak form and reference-space pullback

For u, v in [H1]^d with d = mesh dimension = space dimension (require `vdim == dim == sdim`):

    a(u, v) = ∫_Ω Q (∇·u)(∇·v) dx,   Q a scalar Coefficient (default 1).

With reference gradients Ĝ[i][k] = ∂û_i/∂ξ_k (component i, reference direction k), Jacobian
J with MFEM's `GeometricFactors` convention J(q, i, j, e) = ∂x_i/∂ξ_j, and A = adj(J) so that
J⁻¹ = A/detJ (row index k = reference direction, column index i = physical component):

    ∇·u (q) = (1/detJ) Σ_{i,k} Ĝ[i][k] A[k][i]        (contract each component with column i of A)

Quadrature contribution to the residual in reference-gradient space (what feeds the transpose
sum-factorized contraction):

    Ŷ[j][l](q) = α(q) · S(q) · A[l][j],  where
    S(q) = Σ_{i,k} Ĝ[i][k](q) A[k][i](q)              (unscaled divergence)
    α(q) = Q(q) · w_q / detJ(q)

Derivation: the physical-space integrand is Q w detJ (∇·u)(∇·v); substituting the pullbacks
for both divergences gives one factor 1/detJ from each, hence α = Q·w/detJ, exactly the same
scalar prefactor as `PADiffusionSetup` uses (`c_detJ = W/detJ` in
`fem/integ/bilininteg_vecdiffusion_pa.cpp:241`).

### 2.2 Rank-1 structure — the key design point

As a (d² × d²) quadrature-point operator acting on the flattened reference gradient,
D_q = α · vec(A) vec(A)ᵀ. It is **rank 1**. Consequences:

- **Storage:** only d² + 1 reals per quadrature point: A (unscaled adjugate, 9 in 3D / 4 in
  2D) plus α (1). Compare vector diffusion with scalar coefficient at vdim=3, which stores
  9 slots × 3 components = 27 reals/qpt (reading 18). The grad-div qdata traffic is ~2.7×
  smaller than vector diffusion's. Keeping α separate (rather than folding √α into A) keeps
  the integrator correct for sign-changing coefficients; a 9-real "folded" variant for Q ≥ 0
  is a possible later optimization, not part of this spec.
- **Pointwise flops:** ~2d² + 1 madds per qpt (d² for S, d² + 1 to distribute) — cheaper than
  vector diffusion's 3 symmetric 3×3 mat-vecs.
- **Same A used twice:** the contraction (building S) and the distribution (building Ŷ) read
  the identical d² values, so they are loaded once into registers per quadrature point.
- The tensor-contraction machinery (interpolate reference gradients of all d components,
  pointwise op, transpose-contract back) is *identical* to `SmemPAVectorDiffusionApply3D`.
  Only the pointwise block changes, and it must see all components at once (components couple),
  which just means: interpolate all components first, then one pointwise stage, then write all
  components back — instead of vector diffusion's per-component pipeline. The existing kernel
  already holds all VDIM×DIM gradient fields in its register tensors, so this is a small diff.

### 2.3 Adjugate formulas (copy verbatim from existing setup code)

Use exactly the expressions in `fem/integ/bilininteg_vecdiffusion_pa.cpp:244-252` (3D) and
`:190-193` (2D), which already implement A[k][i] in the required convention:

- 2D: A[0][0]=J22, A[0][1]=−J12, A[1][0]=−J21, A[1][1]=J11.
- 3D: A11=(J22·J33−J23·J32), A12=(J32·J13−J12·J33), A13=(J12·J23−J22·J13),
  A21=(J31·J23−J21·J33), A22=(J11·J33−J13·J31), A23=(J21·J13−J11·J23),
  A31=(J21·J32−J31·J22), A32=(J31·J12−J11·J32), A33=(J11·J22−J12·J21),
  with Jab = J(q, a−1, b−1, e) and detJ from the same file (line 240).

### 2.4 PA diagonal

diag(dof a, component c) = Σ_q α (Σ_k Ĝ_a[k] A[k][c])²
                         = Σ_q Σ_{k,l} Ĝ_a[k] · (α A[k][c] A[l][c]) · Ĝ_a[l].

Per component c this is exactly a scalar-diffusion diagonal with the symmetric d×d matrix
S^c = α A[:,c] A[:,c]ᵀ built **on the fly** from stored qdata. Reuse the sweep structure of
`PAVectorDiffusionDiagonal3D` / `2D` (`fem/integ/bilininteg_vecdiffusion_pa.cpp:329-514`),
replacing its D(q,·,c,e) reads with the computed S^c entries.

---

## 3. Public API

Name: **`VectorDivDivIntegrator`** — consistent with `DivDivIntegrator` (H(div) analogue) and
the `Vector*` prefix convention for vdim-H1 integrators. Doc comment should mention the
aliases people search for: "grad-div", "grad-div stabilization", λ-term of elasticity.

Declaration to add in `fem/bilininteg.hpp` immediately after `DivDivIntegrator` (ends at
line ~3155):

```cpp
/** @brief Integrator for the grad-div bilinear form
    $(Q \nabla \cdot u, \nabla \cdot v)$ where $u = (u_1, \ldots, u_d)$ and
    $v = (v_1, \ldots, v_d)$ are vector fields with components in the same
    scalar H1 space, and $Q$ is a scalar coefficient (default 1).

    This is the "grad-div" term used for grad-div stabilization and
    augmented-Lagrangian formulations of Stokes/Navier-Stokes, and equals the
    $\lambda$-part of the ElasticityIntegrator. Unlike ElasticityIntegrator,
    the PA kernels use fused sum-factorized (tensor-product) evaluation.

    Requires vdim == dim == sdim. PA requires tensor-product elements. */
class VectorDivDivIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q = nullptr;

private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, gshape;
   Vector divshape;
#endif
   // PA extension
   Vector pa_data;                // (nq, dim*dim + 1, ne): adj(J) then alpha
   const DofToQuad *maps = nullptr;         ///< Not owned
   const GeometricFactors *geom = nullptr;  ///< Not owned
   int dim, ne, dofs1D, quad1D;

public:
   VectorDivDivIntegrator(const IntegrationRule *ir = nullptr)
      : BilinearFormIntegrator(ir) { }
   VectorDivDivIntegrator(Coefficient &q, const IntegrationRule *ir = nullptr)
      : BilinearFormIntegrator(ir), Q(&q) { }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;  // = AddMultPA
   void AssembleDiagonalPA(Vector &diag) override;

   const Coefficient *GetCoefficient() const { return Q; }

   /// arguments: ne, B, G, pa_data, x, y, d1d, q1d
   using ApplyKernelType = void (*)(const int,
                                    const Array<real_t> &, const Array<real_t> &,
                                    const Vector &, const Vector &, Vector &,
                                    const int, const int);

   /// arguments: dim, d1d, q1d
   MFEM_REGISTER_KERNELS(ApplyPAKernels, ApplyKernelType, (int, int, int));

   template <int DIM, int D1D, int Q1D>
   static void AddSpecialization()
   { ApplyPAKernels::Specialization<DIM, D1D, Q1D>::Add(); }
};
```

Notes:
- Scalar `Coefficient` only (grad-div's coefficient is physically scalar). `CoefficientVector`
  projection makes `QuadratureFunctionCoefficient` / GridFunction coefficients work for free.
- No `VectorCoefficient`/`MatrixCoefficient` constructors — reject at compile time by omission.
- The operator is symmetric, so `AddMultTransposePA` simply forwards to `AddMultPA` (same
  pattern as `ElasticityIntegrator::AddMultTransposePA`,
  `fem/integ/bilininteg_elasticity_pa.cpp:77-80`).

---

## 4. Files to create / modify

Ask the user first: **in-tree (MFEM fork, upstreamable) or out-of-tree (inside bessemer)?**
The kernels are identical either way.

### Option A — in-tree (MFEM checkout)

| File | Action |
|---|---|
| `fem/bilininteg.hpp` | Add class declaration after `DivDivIntegrator` (~line 3155). |
| `fem/integ/bilininteg_vecdivdiv.cpp` | New: `AssembleElementMatrix` (full assembly). |
| `fem/integ/bilininteg_vecdivdiv_pa.cpp` | New: `AssemblePA` (setup kernels 2D/3D), `AddMultPA` dispatch + specialization registration, `AddMultTransposePA`, `AssembleDiagonalPA` + diagonal kernels. |
| `fem/integ/bilininteg_vecdivdiv_pa.hpp` | New: `SmemPAVectorDivDivApply2D/3D` kernel templates + `ApplyPAKernels::Kernel()/Fallback()` definitions (mirror `bilininteg_vecdiffusion_pa.hpp`). |
| `fem/CMakeLists.txt` | Add the two .cpp to SOURCES (near lines 40-41 where `bilininteg_vecdiffusion_*.cpp` sit) and the .hpp to HEADERS (near line 206). |
| `fem/makefile` (GNU make build) | Mirror every entry that mentions `bilininteg_vecdiffusion_pa` — grep and add the vecdivdiv analogues. |
| `tests/unit/fem/test_pa_kernels.cpp` | Add PA-vs-FA tests (see §7). |
| `tests/unit/fem/test_pa_diagonal.cpp` | Add diagonal test. |
| `CHANGELOG` | One line under "Discretization improvements". |

### Option B — out-of-tree (bessemer or any app linking installed MFEM)

Two files, e.g. `src/fem/vecdivdiv_integrator.hpp/.cpp`, containing the same class in
`namespace bessemer` (or app namespace), subclassing `mfem::BilinearFormIntegrator`. Everything
needed is in installed public headers: `mfem::GeometricFactors`, `DofToQuad`,
`CoefficientVector` (`fem/coefficient.hpp`), `mfem::forall*` (`general/forall.hpp`),
`Reshape`/`DeviceTensor` (`linalg/dtensor.hpp`), `MFEM_REGISTER_KERNELS`
(`fem/kernel_dispatch.hpp`), and the device helpers in `fem/kernels.hpp`.

**Version gate (do this first):** the register-tensor helpers used below
(`kernels::internal::vd_regs3d_t`, `LoadDofs3d`, `Grad3d`, `GradTranspose3d`, `WriteDofs3d`,
`LoadMatrix`, `SetMaxOf`) exist in current master's `fem/kernels.hpp` but are a relatively
recent refactor and live in an `internal` namespace (no API stability promise). Check the
user's MFEM version/headers:
- If the helpers exist (grep `vd_regs3d_t` in the installed `fem/kernels.hpp`): use **Path 1**
  kernels below (least code, matches upstream style).
- If not (older MFEM, e.g. 4.7/4.8 era): use **Path 2** — write self-contained shared-memory
  kernels with inline 1D contractions, using the older
  `PAVectorDiffusionApply3D`-style kernel from that MFEM version as the skeleton and swapping
  in the grad-div pointwise block from §5.3. The math, data layout, dispatch, and tests in
  this spec are unchanged.

---

## 5. Implementation detail

### 5.1 `AssembleElementMatrix` (full assembly — also the test oracle)

Mirror the λ-only part of `ElasticityIntegrator::AssembleElementMatrix`
(`fem/bilininteg.cpp`, search `ElasticityIntegrator::AssembleElementMatrix`) — it already uses
the exact helpers we need:

```cpp
void VectorDivDivIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   const int nd  = el.GetDof();
   const int dim = el.GetDim();
   MFEM_VERIFY(dim == Trans.GetSpaceDim(), "vdim == dim == sdim required");

   dshape.SetSize(nd, dim); gshape.SetSize(nd, dim); divshape.SetSize(dim*nd);
   elmat.SetSize(dim*nd);   elmat = 0.0;

   const IntegrationRule *ir = IntRule ? IntRule
                             : &DiffusionIntegrator::GetRule(el, el);
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      el.CalcDShape(ip, dshape);
      Trans.SetIntPoint(&ip);
      Mult(dshape, Trans.InverseJacobian(), gshape);   // physical gradients
      gshape.GradToDiv(divshape);                      // length dim*nd, byNODES blocks
      real_t w = ip.weight * Trans.Weight();
      if (Q) { w *= Q->Eval(Trans, ip); }
      AddMult_a_VVt(w, divshape, elmat);               // elmat += w * div divᵀ
   }
}
```

`GradToDiv` flattens the nd×dim gradient column-major into [i + c*nd], matching the byNODES
block layout of vdim element matrices — same as elasticity, so the two are directly comparable
in tests.

### 5.2 `AssemblePA` — setup

Mirror `VectorDiffusionIntegrator::AssemblePA` (`fem/integ/bilininteg_vecdiffusion_pa.cpp:71`)
minus the CEED branch and the vector/matrix-coefficient branches:

```cpp
void VectorDivDivIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetTypicalFE();
   const IntegrationRule *ir = IntRule ? IntRule
                             : &DiffusionIntegrator::GetRule(el, el);
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "dim must be 2 or 3");
   MFEM_VERIFY(mesh->SpaceDimension() == dim, "sdim must equal dim");
   MFEM_VERIFY(fes.GetVDim() == dim, "vdim must equal dim");

   const MemoryType mt = (pa_mt == MemoryType::DEFAULT)
                       ? Device::GetDeviceMemoryType() : pa_mt;
   ne   = fes.GetNE();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);   // tensor basis required
   dofs1D = maps->ndof;  quad1D = maps->nqpt;

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q) { coeff.Project(*Q); } else { coeff.SetConstant(1.0); }
   MFEM_VERIFY(coeff.GetVDim() == 1, "scalar coefficient required");

   const int nq = ir->GetNPoints();
   const int pa_size = dim*dim + 1;             // adj(J) entries + alpha
   pa_data.SetSize(nq * pa_size * ne, mt);
   // launch PAVectorDivDivSetup2D/3D (below)
}
```

Setup kernel, 3D (2D analogous with the 2×2 adjugate from §2.3). Layout:
`D = Reshape(pa_data.Write(), Q1D,Q1D,Q1D, dim*dim + 1, NE)`, slots `k + i*dim` = A[k][i]
(k = reference direction row of J⁻¹, i = physical component column), slot `dim*dim` = α.

```cpp
mfem::forall_3D(ne, q1d, q1d, q1d, [=] MFEM_HOST_DEVICE (int e)
{
   MFEM_FOREACH_THREAD(qz, z, q1d)
   MFEM_FOREACH_THREAD(qy, y, q1d)
   MFEM_FOREACH_THREAD(qx, x, q1d)
   {
      // J11..J33, detJ, A11..A33: copy verbatim from
      // fem/integ/bilininteg_vecdiffusion_pa.cpp:230-252
      const real_t alpha = C(0, qx,qy,qz, e) * W(qx,qy,qz) / detJ;
      D(qx,qy,qz, 0, e) = A11;  D(qx,qy,qz, 1, e) = A21;  D(qx,qy,qz, 2, e) = A31; // col i=0
      D(qx,qy,qz, 3, e) = A12;  D(qx,qy,qz, 4, e) = A22;  D(qx,qy,qz, 5, e) = A32; // col i=1
      D(qx,qy,qz, 6, e) = A13;  D(qx,qy,qz, 7, e) = A23;  D(qx,qy,qz, 8, e) = A33; // col i=2
      D(qx,qy,qz, 9, e) = alpha;
   }
});
```

(Aab above means adj(J) with row a = reference direction, col b = physical component, exactly
as the vecdiffusion source defines them — see §2.3.)

### 5.3 `AddMultPA` — the fused sum-factorized apply (Path 1, 3D)

New file `bilininteg_vecdivdiv_pa.hpp`, direct adaptation of `SmemPAVectorDiffusionApply3D`
(`fem/integ/bilininteg_vecdiffusion_pa.hpp:96-170`). Differences: (a) interpolate all three
components **before** the pointwise stage, (b) pointwise block is the rank-1 op, (c) then
transpose-contract all three components. E-vector layout is `(D1D,D1D,D1D, comp, NE)`, the
standard tensor-element restriction layout, valid for byNODES and byVDIM T-vector orderings.

```cpp
template<int T_D1D = 0, int T_Q1D = 0>
void SmemPAVectorDivDivApply3D(const int NE,
                               const Array<real_t> &b, const Array<real_t> &g,
                               const Vector &d, const Vector &x, Vector &y,
                               const int d1d = 0, const int q1d = 0)
{
   static constexpr int DIM = 3, VDIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const auto B = b.Read(), G = g.Read();
   const auto DE = Reshape(d.Read(), Q1D,Q1D,Q1D, DIM*DIM + 1, NE);
   const auto XE = Reshape(x.Read(), D1D,D1D,D1D, VDIM, NE);
   auto       YE = Reshape(y.ReadWrite(), D1D,D1D,D1D, VDIM, NE);

   mfem::forall_2D<T_Q1D*T_Q1D>(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD1 = T_D1D > 0 ? SetMaxOf(T_D1D) : DofQuadLimits::MAX_T1D;
      constexpr int MQ1 = T_Q1D > 0 ? SetMaxOf(T_Q1D) : DofQuadLimits::MAX_T1D;
      MFEM_SHARED real_t sB[MD1][MQ1], sG[MD1][MQ1], smem[MQ1][MQ1];
      kernels::internal::vd_regs3d_t<VDIM, DIM, MQ1> r0, r1;
      kernels::internal::LoadMatrix(D1D, Q1D, B, sB);
      kernels::internal::LoadMatrix(D1D, Q1D, G, sG);

      // (a) reference gradients of ALL components -> r1[c][k][qz](qy,qx)
      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadDofs3d(e, D1D, c, XE, r0);
         kernels::internal::Grad3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
      }
      // (b) pointwise rank-1 op: S = <Ghat, A>, r0 = alpha*S * A
      for (int qz = 0; qz < Q1D; qz++)
      {
         MFEM_FOREACH_THREAD_DIRECT(qy, y, Q1D)
         MFEM_FOREACH_THREAD_DIRECT(qx, x, Q1D)
         {
            real_t A[DIM][VDIM];                       // registers, used twice
            for (int i = 0; i < VDIM; i++)
               for (int k = 0; k < DIM; k++)
                  A[k][i] = DE(qx,qy,qz, k + i*DIM, e);
            const real_t alpha = DE(qx,qy,qz, DIM*DIM, e);
            real_t S = 0.0;
            for (int i = 0; i < VDIM; i++)
               for (int k = 0; k < DIM; k++)
                  S += r1[i][k][qz][qy][qx] * A[k][i];
            const real_t t = alpha * S;
            for (int i = 0; i < VDIM; i++)
               for (int k = 0; k < DIM; k++)
                  r0[i][k][qz][qy][qx] = t * A[k][i];
         }
      }
      MFEM_SYNC_THREAD;
      // (c) transpose gradients back, accumulate into y
      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::GradTranspose3d(D1D, Q1D, smem, sB, sG, r0, r1, c);
         kernels::internal::WriteDofs3d(e, D1D, c, c, r1, YE);
      }
   });
}
```

Implementation notes:
- On GPU the `[qy][qx]` indices of the register tensors are 0-extent thread dims (see
  `fem/kernels.hpp:34-55`), so this compiles to per-thread registers exactly like the
  vecdiffusion kernel; register pressure and shared-memory footprint match
  `SmemPAVectorDiffusionApply3D` (VDIM=3) — a known-good configuration.
- Verify against the vecdiffusion kernel whether `Grad3d` leaves data in a state requiring a
  `MFEM_SYNC_THREAD` between the interpolation loop and the pointwise stage; copy its
  synchronization placement exactly.
- 2D kernel: same structure with DIM=VDIM=2, `vd_regs2d_t`, `Grad2d`/`GradTranspose2d`,
  `LoadDofs2d`/`WriteDofs2d`, qdata slots 0..3 + α at 4.
- `AddMultTransposePA(x,y) { AddMultPA(x,y); }` — symmetric.

### 5.4 Dispatch and specializations

In `bilininteg_vecdivdiv_pa.hpp`, mirror `ApplyPAKernels::Kernel()/Fallback()` from
`bilininteg_vecdiffusion_pa.hpp:172-200` with the (DIM, D1D, Q1D) template signature. In
`AddMultPA`, register specializations exactly as vecdiffusion does
(`bilininteg_vecdiffusion_pa.cpp:291-320`), with two additions important for spectral-element
use (collocated GLL, Q1D == D1D):

- 3D: (2,2) (2,3) (3,3) (3,4) (4,4) (4,5) (4,6) (5,5) (5,6) (5,8) (6,6) (6,7) (7,7) (7,8)
  (8,8) (8,9) (9,9)
- 2D: (2,2) ... (9,9) as in vecdiffusion.

Then `ApplyPAKernels::Run(dim, dofs1D, quad1D, ne, maps->B, maps->G, pa_data, x, y, dofs1D,
quad1D);`. The runtime `Fallback` (untemplated D1D/Q1D) must also be provided — same two
functions with 0 template parameters.

### 5.5 `AssembleDiagonalPA`

Plain switch dispatch (no kernel registration needed — copy
`VectorDiffusionIntegrator::AssembleDiagonalPA`, `bilininteg_vecdiffusion_pa.cpp:522`).
Kernels `PAVectorDivDivDiagonal2D/3D`: copy the loop structure of
`PAVectorDiffusionDiagonal3D` (`bilininteg_vecdiffusion_pa.cpp:399-514`), which for each
component c and each (i,j) direction pair does the QQD/QDD partial sweeps. Replace its qdata
read with the on-the-fly symmetric matrix

```cpp
// inside the innermost quadrature loop, for component c and directions (i,j):
const real_t Sij = DE(q, DIM*DIM, e) * DE(q, i + c*DIM, e) * DE(q, j + c*DIM, e);
```

and note that unlike vector diffusion the diagonal differs per component (no shared `temp`
written to all components — compute per c).

---

## 6. Constraints, edge cases, error handling

- **Supported:** dim = sdim = vdim ∈ {2,3}; tensor-product elements (quads/hexes) for PA;
  any scalar `Coefficient` including `QuadratureFunctionCoefficient`/GridFunction-based
  (via `CoefficientVector`); byNODES and byVDIM orderings (tensor E-vector restriction
  normalizes layout); host and device execution via `mfem::forall`; `real_t` single/double.
- **Full assembly** (`AssembleElementMatrix`) works for any H1 element including simplices —
  PA is where tensor elements are required. `DofToQuad::TENSOR` on a non-tensor element will
  fail; add a friendly `MFEM_VERIFY` message telling users to use full assembly on simplices.
- **User-supplied IntegrationRule** must be honored (constructor or `SetIntRule`) — this is
  how collocated GLL (SEM) is selected. Default rule: `DiffusionIntegrator::GetRule(el, el)`
  (order 2p + dim − 1 for tensor elements, `fem/bilininteg.cpp:1347`).
- Negative or sign-changing coefficients are legal (α stored separately, never a √).
- Variable-order spaces and mixed-geometry meshes: out of scope, `MFEM_VERIFY` against them
  (same "typical FE" assumption as elasticity PA).
- Do **not** add a CEED branch.

## 7. Validation plan (write the tests before optimizing anything)

1. **PA vs FA operator action** (primary). Extend `tests/unit/fem/test_pa_kernels.cpp` using
   the existing harness `test_pa_vector_integrator<...>` (line ~410) as the model: curved
   `MakeCartesianNonaligned` meshes, `mesh.SetCurvature(p)`, random x, compare
   `y_pa = A_pa x` against `y_fa = A_fa x` from a fully assembled `BilinearForm`.
   Sweep: dim ∈ {2,3}, p ∈ {1,2,3,4}, coefficient ∈ {none, ConstantCoefficient,
   FunctionCoefficient}. Pass: relative L∞ error ≤ 1e-12 (double).
2. **Cross-check against elasticity.** Same mesh/space: full-assembly
   `ElasticityIntegrator(lambda=Qc, mu=zero)` matrix must equal full-assembly
   `VectorDivDivIntegrator(Qc)` matrix (`SparseMatrix` max-norm diff ≤ 1e-12, same IntRule
   forced on both). This pins down conventions (byNODES block layout, weights) exactly.
3. **Collocated GLL path.** Repeat test 1 with an explicit
   `IntRules.Get(geom, 2*p - 1)` Gauss–Lobatto rule (`IntegrationRules(0, Quadrature1D::GaussLobatto)`)
   so Q1D == D1D — the SEM configuration this integrator exists for.
4. **Diagonal.** In `tests/unit/fem/test_pa_diagonal.cpp`: `AssembleDiagonalPA` vs the
   diagonal of the fully assembled matrix, ≤ 1e-12 relative.
5. **Transpose.** `AddMultTransposePA` result equals `AddMultPA` (trivial, but keeps the
   symmetric-forwarding contract honest).
6. Run the unit tests on CPU and at least one GPU backend (`ceed` off): `-d cuda` or `-d hip`.

## 8. Benchmark and performance acceptance

Add a small standalone driver (out-of-tree: `bench/bench_graddiv.cpp`; in-tree: extend
`tests/benchmarks/`) measuring AddMult throughput (GDOF/s and time/apply, 50 warm iterations)
on a 3D hex mesh sized to ≥ 2M dofs, p ∈ {2,4,6,8}, comparing:

- (a) `VectorDivDivIntegrator` PA (this work);
- (b) `ElasticityIntegrator(Q, mu=0)` PA (current status quo);
- (c) `VectorDiffusionIntegrator` PA (structural upper bound: same contractions, denser
  pointwise/qdata).

Acceptance:
- Correctness gates in §7 all pass.
- (a) beats (b) by ≥ 5× at p ≥ 4 on GPU (expected ~10× at p = 6 from the O(p⁶) → O(p⁴)
  reduction plus removal of the global-memory gradient round trip);
- (a) runtime within ~1.3× of (c) at matching p/Q — it does the same 18 tensor contractions
  with a cheaper pointwise stage and ~2.7× less qdata traffic, so it should be ≤ vecdiffusion;
  if it's slower, something is wrong (likely a missing sync or spilled registers).

## 9. Ordered task list for Claude Code

1. Confirm in-tree vs out-of-tree with the user; confirm MFEM version and run the §4
   version gate (grep `vd_regs3d_t` in `fem/kernels.hpp`).
2. Add class declaration (+ build-system entries if in-tree).
3. Implement `AssembleElementMatrix`; add test 7.2 (elasticity cross-check) — get it green.
   This locks the math before any GPU work.
4. Implement `AssemblePA` setup (2D+3D) and `AddMultPA` with only the **Fallback**
   (untemplated) kernels; add test 7.1 on CPU — green.
5. Add kernel specializations list; verify test 7.1 + 7.3 on GPU backend.
6. Implement `AssembleDiagonalPA` (2D+3D); test 7.4.
7. `AddMultTransposePA` forwarding + test 7.5.
8. Benchmark driver; record numbers for (a)/(b)/(c) in the PR/commit message.
9. If in-tree: style pass (`make style`), CHANGELOG entry, doc comment check
   (`make docs` builds), then draft PR text summarizing §1-2.

Definition of done: all §7 tests pass on CPU + GPU; §8 acceptance numbers recorded.

## 10. Explicit non-goals (defer)

- `AssembleEA` / element assembly (dense (d·nd)² element matrices are impractical at high p;
  the elasticity EA path exists for p=1 LOR-AMG needs only).
- `AssembleMF` (true matrix-free without stored qdata).
- Simplex PA, NURBS/patch assembly, surface meshes (sdim > dim), libCEED path.
- √α-folded 9-real qdata for Q ≥ 0 (mild bandwidth win; do only if benchmarks demand).
- A fused "vector-diffusion + grad-div" kernel (single sweep applying ν-Laplacian and γ
  grad-div together). Natural follow-up for a Navier–Stokes momentum operator — the two
  share the interpolated gradients — but keep this integrator clean and orthogonal first.
- Refactoring `ElasticityIntegrator` PA onto sum-factorized kernels (the μ sym-grad term
  needs the full 9-field coupling and is a larger project; this integrator is a stepping
  stone and its kernel structure would be the template).

## 11. Verified reference map (commit 951cf888)

- `fem/integ/bilininteg_vecdiffusion_pa.hpp` — `SmemPAVectorDiffusionApply2D/3D`, kernel
  `Kernel()/Fallback()` pattern. **Primary template for §5.3-5.4.**
- `fem/integ/bilininteg_vecdiffusion_pa.cpp:71-283` — `AssemblePA` incl. 2D/3D adjugate/detJ
  code to copy; `:285-326` specialization registration + `Run`; `:329-540` diagonal kernels
  and dispatch. **Template for §5.2 and §5.5.**
- `fem/kernels.hpp:30-120, 206-330, 660-760` — register tensor types, `LoadMatrix`,
  `LoadDofs3d`, `Grad3d`, `GradTranspose3d`, `WriteDofs3d` semantics.
- `fem/kernel_dispatch.hpp` — `MFEM_REGISTER_KERNELS`.
- `fem/bilininteg.hpp:3060` (`VectorDivergenceIntegrator`), `:3116` (`DivDivIntegrator`),
  `:3161` (`VectorDiffusionIntegrator`), `:3272` (`ElasticityIntegrator`) — API conventions
  and the insertion point for the new class.
- `fem/bilininteg.cpp:1347` — `DiffusionIntegrator::GetRule` (default quadrature order).
- `fem/integ/bilininteg_elasticity_pa.cpp` + `bilininteg_elasticity_kernels.hpp` — the
  non-tensor status quo being replaced for the div-div term (documentation of its dense
  reduction and global scratch is at `bilininteg_elasticity_kernels.hpp:53-55, 170-277`).
- `tests/unit/fem/test_pa_kernels.cpp:410-470` — `test_pa_vector_integrator` harness to
  extend; `tests/unit/fem/test_pa_diagonal.cpp` — diagonal test patterns.
