# Spec: semi-implicit rotational convection with partial assembly, and point-block Jacobi for the velocity block

Target: the bessemer application (out of tree), built against MFEM `master` at commit `c362dea2` (2026-09-29). All MFEM file paths, class names and kernel helpers referenced below were checked against that commit. If you build against a different MFEM version, re-verify the items in Sections 5.8 and 7.7 before writing code.

Audience: an agent implementing this in C++. Read the whole document before writing code.

Everything runs on GPUs with moderate-order elements. Nothing in the production path assembles a matrix. An assembled matrix appears only as a small CPU reference inside one unit test (Section 7.6, test B2).

---

## 0. Summary

Two parts, delivered in order.

**Part A (Sections 1 to 6).** A partially assembled bilinear form integrator for the lagged-vorticity (semi-implicit) rotational form of the convection term,

$$a(u, v) = \alpha \int_\Omega (\omega \times u)\cdot v \, dx, \qquad \omega = \nabla \times w,$$

where `w` is a given lagged (extrapolated) velocity in the same H1 vector space as `u`, and `alpha` is a scalar. It must support:

1. Partial assembly (PA) on tensor-product quadrilaterals and hexahedra, on CPU and GPU (CUDA and HIP), serial and MPI parallel.
2. A cheap in-place update of the vorticity data every time step (`UpdateVorticity()`), without rebuilding the form or the constrained operator.
3. Two small extras: the nodal skew vector used by Part B (`AddNodalSkewPA`), and a rotation-number diagnostic (`GetRotationNumberStats`).

The operator is zero order (no derivatives act on `u`), so the apply is structurally a vector mass apply with a pointwise skew 3x3 (2D: 2x2) coupling between components. All derivative work happens once per time step, in the setup kernel that computes `curl w` at quadrature points.

**Part B (Section 7).** A matrix-free point-block Jacobi preconditioner for the velocity block $A = \sigma M + \nu K + N$. It inverts the `dim x dim` block of $A$ at each velocity node. It is identical to scalar Jacobi wherever the rotation term is negligible, and nearly exact wherever rotation dominates. It is the next step after Part A, for the (1,1) block only. The pressure Schur complement is out of scope for this document.

**File constraint.** One `.hpp` and one `.cpp` for the integrator, one `.hpp` and one `.cpp` for point-block Jacobi, one test file for each. No other files (Section 3).

**Tests.** Three unit tests for Part A and three for Part B (Sections 6 and 7.6). Each is a high-power check run over a parameter sweep. Convergence-rate, energy and time-stepping tests belong with the solver's integration tests, in separate files, and are not part of this document.

---

## 1. Mathematical definition

### 1.1 Continuous form and sign conventions

3D: $\omega = \nabla\times w = (\partial_y w_z - \partial_z w_y,\ \partial_z w_x - \partial_x w_z,\ \partial_x w_y - \partial_y w_x)$ and

$$\omega\times u = [\omega]_\times u,\qquad [\omega]_\times = \begin{pmatrix} 0 & -\omega_3 & \omega_2\\ \omega_3 & 0 & -\omega_1\\ -\omega_2 & \omega_1 & 0\end{pmatrix}.$$

2D: $\omega = \partial_x w_y - \partial_y w_x$ (scalar, out of plane) and $\omega\times u = (-\omega u_y,\ \omega u_x)$, so $[\omega]_\times = \begin{pmatrix} 0 & -\omega\\ \omega & 0\end{pmatrix}$.

This is the term in $(u\cdot\nabla)u = (\nabla\times u)\times u + \nabla(\tfrac12|u|^2)$, linearized by lagging the first factor. **Do not implement $u \times \omega$** (opposite sign). Test A1 exists specifically to catch this. Note: Benzi and Liu (SIAM J. Sci. Comput. 2007) print the 2D matrix with the opposite sign from their 3D one; the convention above is the correct one.

### 1.2 Discrete operator

For the H1 vector space with scalar basis $\{\varphi_a\}$ and a tensor quadrature rule $\{\hat\xi_q, \hat w_q\}$:

$$N_{(a,i),(b,j)} = \alpha \sum_e \sum_q \hat w_q \det J_q\, \varphi_a(\hat\xi_q)\, \varphi_b(\hat\xi_q)\, [\omega_q]_{\times,ij}$$

Row $(a, i)$ is test dof $a$, component $i$; column $(b, j)$ is trial dof $b$, component $j$. $\omega_q$ is the exact curl of the finite element function $w_h$ at the physical image of $\hat\xi_q$, from $\nabla_x w_h = \nabla_\xi w_h\, J^{-1}$.

Properties that hold **exactly for any mesh, any quadrature rule, any $w_h$**. The tests rely on them.

- **P1 (skew-symmetry).** $N^T = -N$, so $x^T N x = 0$ for every $x$.
- **P2 (zero diagonal).** $\mathrm{diag}(N) = 0$. The nodal `dim x dim` blocks (same dof $a$, all component pairs) are nonzero and skew. Part B is built on this.
- **P3 (linearity).** $N$ is linear in $\alpha$ and in $w$.
- **P4 (constant vorticity).** If $\omega$ is constant, $N = [\omega]_\times \otimes M_s$ with $M_s$ the scalar mass matrix for the same rule. A rigid rotation $w = \tfrac12\Omega\times x$ gives $\omega_h = \Omega$ exactly on any mesh of order at most $p$.
- **P5 (vector identity).** With $w_h = u_h$ and one rule for every term: $C(u_h)u_h - N(u_h)u_h - G(u_h) = 0$, where $C(u)u = \int (u\cdot\nabla)u\cdot v$ (MFEM's `VectorConvectionNLFIntegrator`) and $G(u) = \int (\nabla u)^T u \cdot v$. The identity is pointwise, so it holds to roundoff for any rule, including under-integrated ones.
- **P6 (transpose).** `AddMultTransposePA(x, y)` adds $-Nx$.
- **P7 (collocation).** With Gauss-Lobatto-Legendre (GLL) collocated quadrature ($p+1$ points per direction at the GLL nodes), both $M$ and $N$ are block diagonal over nodes: $\sigma M + N$ has exactly one `dim x dim` block per node. Part B's exactness test relies on this.

### 1.3 Partial assembly factorization

On each element, $y_e = B^T D_e B\, x_e$, where $B$ is the scalar tensor interpolation applied to each component and $D_e(q) = [d_q]_\times$ with

$$d_q = \alpha\, \hat w_q\, \det J_q\, \omega_q \quad (\text{3 numbers per point in 3D, 1 in 2D}).$$

Compute $d_q$ **without dividing by $\det J$**. Since $\det J\, J^{-1} = \mathrm{adj}(J)$:

$$H_{ck} := \det J\,\frac{\partial w_c}{\partial x_k} = \sum_d \frac{\partial w_c}{\partial \xi_d}\, \mathrm{adj}(J)_{dk},$$

$$d_q = \alpha\,\hat w_q\,(H_{21}-H_{12},\ H_{02}-H_{20},\ H_{10}-H_{01}) \ \ (3D), \qquad d_q = \alpha\,\hat w_q\,(H_{10}-H_{01})\ \ (2D).$$

Only the off-diagonal entries of $H$ are needed. `VectorConvectionNLFIntegrator::AssemblePA` uses the same trick (it stores $\hat w\, \mathrm{adj}(J)$).

Storing $d_q$ (3 values per point) is the right trade. A matrix-free variant would read $J$ (9 values per point) every apply, and `VectorMassIntegrator` with a matrix coefficient stores 9 values per point.

---

## 2. Scope and non-goals

In scope:
- Domain integrator only (the term has no boundary or face contributions).
- `dim == sdim == vdim`, `dim` in {2, 3}; H1 (map type `VALUE`) tensor-product elements; one element geometry; uniform order; `Ordering::byNODES` and `Ordering::byVDIM`.
- Any tensor-product rule: Gauss-Legendre, GLL collocated, over-integrated (3/2 rule).
- Affine, sheared, curved (high-order nodes), periodic, and nonconforming adaptively refined meshes.

Not in scope (verify and abort with a clear message where applicable):
- Simplices, mixed meshes, variable order, NURBS, `sdim > dim`.
- libCEED backends (Section 5.8, item 1).
- Legacy element assembly (`AssembleElementMatrix`), element assembly (`AssembleEA`), matrix-free (`AddMultMF`), `AddAbsMultPA`. Do not implement.
  **Update 2026-10-05 (bessemer):** `AssembleElementMatrix` and per-component `AssembleEA`
  (`VectorRotationalConvectionComponentIntegrator`) were added afterwards for LOR use; see
  CLAUDE.md "Convective form" for the design, tests, and the measured verdict on putting N
  into LOR-AMG (it does not pay off where rotation dominates). `AddMultMF` and
  `AddAbsMultPA` remain unimplemented.
- A coefficient mode that takes `omega` as a `VectorCoefficient`. The integrator always takes a `GridFunction` `w`; tests that need a known vorticity use a rigid rotation (P4).
- The pressure Schur complement and anything else in the Navier-Stokes solver beyond the (1,1) block (Section 8 has integration notes only).

---

## 3. Files

All code lives in bessemer, in the application's namespace. Exactly six files:

| File | Content |
|---|---|
| `rotational_convection.hpp` | `VectorRotationalConvectionIntegrator` declaration and `RotationNumberStats` struct. No kernels. |
| `rotational_convection.cpp` | Everything else for Part A: setup, apply, nodal-skew and diagnostic kernels (file-local, in an anonymous or `internal` namespace) and kernel specialization registration. |
| `point_block_jacobi.hpp` | `PointBlockJacobi` declaration. |
| `point_block_jacobi.cpp` | `PointBlockJacobi` implementation, plus the file-local `NodalSkewProxy` integrator (Section 7.4). |
| `test_rotational_convection.cpp` | Part A tests (Section 6). |
| `test_point_block_jacobi.cpp` | Part B tests (Section 7.6). |

Do not split kernels into separate headers.

MFEM headers used (paths relative to MFEM's include root; all are installed by `make install`): `mfem.hpp`; `fem/kernels.hpp` (`kernels::internal::LoadMatrix`, `LoadDofs{2,3}d`, `Eval{2,3}d`, `EvalTranspose{2,3}d`, `Grad{2,3}d`, `WriteDofs{2,3}d`); `fem/kernel_dispatch.hpp` (`MFEM_REGISTER_KERNELS`); `general/forall.hpp`. Part B uses MFEM's `BatchedDirectSolver` (`linalg/batched/solver.hpp`, included by `mfem.hpp`).

---

## 4. Part A interface

```cpp
/// Rotation-number diagnostic, mu_q = |omega_q| / sigma (Section 5.10).
struct RotationNumberStats
{
   real_t max_mu;        // max over quadrature points (global in parallel)
   real_t vol_fraction;  // volume fraction with mu_q > threshold (global)
   real_t threshold;
};

/** a(u,v) = alpha ((curl w) x u, v): lagged-vorticity rotational convection.

    u, v: H1 vector fields (vdim == dim, dim = 2 or 3).
    w:    lagged/extrapolated velocity, H1 vector GridFunction on the same mesh
          and same order. Not owned.

    Skew-symmetric: AddMultTransposePA adds -N x; the diagonal is zero. */
class VectorRotationalConvectionIntegrator : public BilinearFormIntegrator
{
public:
   explicit VectorRotationalConvectionIntegrator(const GridFunction &w,
                                                 real_t alpha = 1.0);

   void SetLaggedVelocity(const GridFunction &w);  // takes effect at UpdateVorticity()
   void SetAlpha(real_t a);                        // takes effect at UpdateVorticity()

   /** Recompute the quadrature data from the current w. Requires a previous
       AssemblePA(). No allocations after the first call. Runs on the device.
       w must be an up-to-date L-vector: for a ParGridFunction call
       SetFromTrueDofs()/Distribute() first. */
   void UpdateVorticity();

   /// Default rule: same as VectorConvectionNLFIntegrator::GetRule.
   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                         const ElementTransformation &T);

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;
   void AssembleDiagonalPA(Vector &diag) override;   // adds exact zeros (P2)

   /** For point-block Jacobi (Section 5.9). Adds into an E-vector of the
       velocity space, layout (D1D^dim, dim, NE), lexicographic:
       3D: component c += sum_q B_a(q)^2 d_q[c]
       2D: component 0 += sum_q B_a(q)^2 d_q;  component 1 untouched. */
   void AddNodalSkewPA(Vector &s_e) const;

   /// Section 5.10. sigma is the mass coefficient (BDF2: 3/(2 dt)).
   RotationNumberStats GetRotationNumberStats(real_t sigma,
                                              real_t threshold = 1.0) const;

   /** Quadrature data, for tests. Layout (column-major):
       3D: (Q1D, Q1D, Q1D, 3, NE), entry (qx,qy,qz,c,e) = alpha*w_q*detJ_q*omega_c
       2D: (Q1D, Q1D, NE),        entry (qx,qy,e)      = alpha*w_q*detJ_q*omega   */
   const Vector &GetQuadratureData() const { return pa_data; }

   // Kernel registries (Section 5.6). Kernel definitions and all
   // Specialization<...>::Add() calls live in rotational_convection.cpp.
   using ApplyType = void (*)(int ne, real_t sign, const Array<real_t> &B,
                              const Vector &d, const Vector &x, Vector &y,
                              int d1d, int q1d);
   MFEM_REGISTER_KERNELS(RotConvApplyPA, ApplyType, (int, int, int));
   using SetupType = void (*)(int ne, real_t alpha, const real_t *B,
                              const real_t *G, const real_t *W,
                              const real_t *J, const real_t *w_e,
                              real_t *d, int d1d, int q1d);
   MFEM_REGISTER_KERNELS(RotConvSetupPA, SetupType, (int, int, int));

private:
   const GridFunction *w;                    // not owned
   real_t alpha;
   int dim = 0, ne = 0, d1d = 0, q1d = 0;
   const DofToQuad *maps = nullptr;          // not owned
   const GeometricFactors *geom = nullptr;   // not owned
   const IntegrationRule *pa_ir = nullptr;   // rule used at AssemblePA
   const Mesh *pa_mesh = nullptr;            // for MPI reductions in diagnostics
   const Operator *w_restr = nullptr;        // lexicographic restriction of w's space
   Vector w_e;                               // E-vector buffer for w (reused)
   Vector pa_data;                           // d_q, layout above
   mutable Vector diag_tmp0, diag_tmp1;      // diagnostic scratch (reused)
};
```

---

## 5. Part A implementation

### 5.1 `AssemblePA(fes)`

1. `MFEM_VERIFY(!DeviceCanUseCeed(), ...)`: libCEED is unsupported (5.8, item 1).
2. Validate inputs (5.5).
3. `ir = IntRule ? IntRule : &GetRule(el, *mesh->GetTypicalElementTransformation())`. Store `pa_ir = ir`, `pa_mesh = mesh`.
4. `geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS, mt)`, with `mt` the `pa_mt` or device memory type, as in `VectorMassIntegrator::AssemblePA`.
5. `maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR)`; `d1d = maps->ndof`, `q1d = maps->nqpt`.
6. Allocate `pa_data` of size `(dim == 3 ? 3 : 1) * nq * ne` in memory type `mt`.
7. `w_restr = w->FESpace()->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)`, allocate `w_e` (size `w_restr->Height()`, device memory), then `UpdateVorticity()`.

`AssemblePA` may be called again (for example after the application calls `Assemble()` because `dt` changed). It must give the same result and must not leak.

### 5.2 `UpdateVorticity()`: fused setup kernel

1. `w_restr->Mult(*w, w_e)`. E-vector layout `(D1D^dim, VDIM, NE)`, lexicographic, independent of the global ordering (verified in `ElementRestriction::Mult`).
2. Setup kernel, mirroring `internal::SmemPAConvectionNLApply3D` in `fem/integ/nonlininteg_vecconvection_pa.hpp`:

```
forall_2D(ne, Q1D, Q1D) per element e:
  shared: sB[D1D][Q1D], sG[D1D][Q1D], smem
  LoadMatrix(B -> sB), LoadMatrix(G -> sG)
  vd_regs3d_t<3,3,MQ1> g0, g1
  LoadDofs3d(e, D1D, Wvec, g0)               // Wvec = Reshape(w_e, D1D,D1D,D1D,3,NE)
  Grad3d(D1D, Q1D, smem, sB, sG, g0, g1)     // g1[c][d][qz][qy][qx] = dw_c/dxi_d
  for qz in [0,Q1D): for (qy, qx) threads:
     Jm[i][j] = J(qx,qy,qz,i,j,e)            // GeometricFactors: J(q,i,j,e) = dx_i/dxi_j
     A = adj(Jm)                              // det(J) * inv(J), no division
     H[c][k] = sum_d g1[c][d] * A[d][k]       // det(J) dw_c/dx_k; off-diagonals only
     D(qx,qy,qz,0,e) = alpha * W(qx,qy,qz) * (H[2][1] - H[1][2])
     D(qx,qy,qz,1,e) = alpha * W(qx,qy,qz) * (H[0][2] - H[2][0])
     D(qx,qy,qz,2,e) = alpha * W(qx,qy,qz) * (H[1][0] - H[0][1])
```

2D: same with `vd_regs2d_t<2,2,MQ1>`, `Grad2d`, `adj(J) = [[J11, -J01], [-J10, J00]]` (0-indexed), single output `alpha * W * (H[1][0] - H[0][1])`.

Index conventions:
- `GeometricFactors::J` is `(NQ, SDIM, DIM, NE)` column-major, `J(q,i,j,e) = dx_i/dxi_j`.
- `kernels::internal::Grad3d` output `g1[c][d]` = derivative of component `c` with respect to reference coordinate `d`.
- Use the signed determinant implicitly through `adj(J)` and `ir->GetWeights()`, consistent with `VectorMassIntegrator::AssemblePA`.

Do not use `QuadratureInterpolator` in this path: it needs a transient `9 * nq * ne` gradient buffer, about 15 times a velocity vector at p = 7 with a 3/2 rule.

### 5.3 Apply kernel (`AddMultPA`)

Copy `internal::SmemPAVectorMassApply3D` and `...2D` from `fem/integ/bilininteg_vecmass_pa.hpp` and replace the coefficient block. With `D0, D1, D2` the stored values and `(Qx, Qy, Qz)` the interpolated `x`:

3D:
```
r0[0] = sign * (D1*Qz - D2*Qy);
r0[1] = sign * (D2*Qx - D0*Qz);
r0[2] = sign * (D0*Qy - D1*Qx);
```
2D:
```
r0[0] = -sign * D0*Qy;
r0[1] =  sign * D0*Qx;
```
then `EvalTranspose{2,3}d` and `WriteDofs{2,3}d` (they accumulate into `y`). `sign = +1` for `AddMultPA`, `-1` for `AddMultTransposePA`. Keep the thread layout, register tiles and fallback sizing (`DofQuadLimits::MAX_T1D`) identical to the vector mass kernel.

### 5.4 Transpose, diagonal

- `AddMultTransposePA`: apply kernel with `sign = -1`; equals `-AddMultPA` bitwise.
- `AssembleDiagonalPA`: adds nothing (P2) but **must exist**: the base class aborts, and `BilinearForm::AssembleDiagonal` calls every integrator. This keeps scalar Jacobi on the full velocity operator working unchanged.

### 5.5 Input validation (`MFEM_VERIFY`, explicit messages)

- `dim` in {2, 3}, `mesh->SpaceDimension() == dim`, `fes.GetVDim() == dim`.
- Typical element is a `TensorBasisElement`, map type `VALUE`; one geometry; not variable order; no NURBS.
- `w`: same mesh, same vdim, same order, tensor H1 (a different `FiniteElementSpace` object is fine).
- `d1d <= q1d`: the register tiles in `fem/kernels.hpp` are sized by `MQ1` and hold dof values before interpolation, so `q1d < d1d` overflows them. GLL collocation (`d1d == q1d`) is allowed and important.
- `d1d`, `q1d` within `DeviceDofQuadLimits::Get()` for fallback kernels; on failure name the values and suggest a specialization.
- `UpdateVorticity()`, `AddNodalSkewPA()`, `GetRotationNumberStats()` before `AssemblePA()` abort with a clear message.

### 5.6 Kernel specializations and device limits

Register specializations with `MFEM_REGISTER_KERNELS` (pattern: `VectorMassIntegrator::VectorMassAddMultPA` in `bilininteg_vecmass_pa.cpp`), for the apply and setup kernels, for every `(dim, d1d, q1d)` the application uses. All kernel templates and all `Specialization<...>::Add()` calls are in `rotational_convection.cpp`. At minimum for `p = 1..8` (`d1d = p + 1`):
- `q1d = p + 1` (GLL collocated, and Gauss with `p + 1` points),
- `q1d` of the default rule (table below),
- `q1d = ceil(3(p+1)/2)` (3/2 rule).

Default rule sizes (`VectorConvectionNLFIntegrator::GetRule`, measured at the target commit). The default depends on the **mesh** order through `OrderGrad`:

| dim, mesh order | p=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 2D, 1 | 2 | 4 | 5 | 7 | 8 | 10 | 11 | 13 |
| 3D, 1 | 3 | 4 | 6 | 7 | 9 | 10 | 12 | 13 |
| 3D, 2 | 4 | 5 | 7 | 8 | 10 | 11 | 13 | 14 |

Device limits: `DofQuadLimits_HIP::MAX_D1D = MAX_Q1D = 10`, `DofQuadLimits_CUDA` = 14. A 3/2 rule at `p >= 6` exceeds the HIP `MAX_Q1D`. Size fallback register tiles with `MAX_T1D` (32) as the vector mass kernel does, and register specializations for production sizes so the fallback is never hit in production.

The nodal-skew and diagnostic kernels (5.9, 5.10) run once per step, not per Krylov iteration, so they need no specializations.

### 5.7 Memory footprint (3D, double precision)

| Item | Size | Lifetime |
|---|---|---|
| `pa_data` | `3 * nq * ne` | persistent |
| `w_e` | `3 * d1d^3 * ne` | persistent (reused) |
| `diag_tmp0`, `diag_tmp1` | `nq * ne` each | allocated on first diagnostic call, reused |

For comparison at the same rule: scalar-coefficient `VectorMassIntegrator` PA stores `1 * nq * ne`; with a matrix coefficient, `9 * nq * ne`. MFEM's `VectorDiffusionIntegrator` PA stores `dim*dim*vdim = 27` values per point in 3D even for a scalar coefficient, so the rotation term is a small addition to the velocity operator.

### 5.8 Gotchas found while reading MFEM

1. **libCEED path receives L-vectors.** `PABilinearFormExtension::MultInternal` skips the element restriction when `DeviceCanUseCeed()` is true. Refuse to assemble on a libCEED device.
2. **Lexicographic ordering.** The PA form and the setup kernel both need `ElementDofOrdering::LEXICOGRAPHIC`. Native ordering differs even at p = 1.
3. **`w` must be distributed** in parallel before `UpdateVorticity()` (test A3).
4. **Moving meshes.** `GeometricFactors` are cached on the mesh; after node changes the caller must call `mesh->DeleteGeometricFactors()` and `AssemblePA` again. Document; do not handle.

### 5.9 `AddNodalSkewPA(s_e)`

Computes, per element and per lexicographic dof $a = (d_x, d_y, d_z)$:

$$s_e(a, c, e) \mathrel{+}= \sum_{q} B(q_x,d_x)^2\, B(q_y,d_y)^2\, B(q_z,d_z)^2\, D(q_x,q_y,q_z,c,e).$$

This is the diagonal of a scalar mass matrix weighted by $\alpha\,\omega_c$, which is exactly the off-diagonal part of the nodal block of $N$. It is the same sum-factorized contraction as `internal::SmemPAVectorMassAssembleDiagonal3D` in `fem/integ/bilininteg_vecmass_pa.hpp`, except that it reads a separate coefficient per component and writes component `c` only. 2D: one coefficient, written to component 0; component 1 is left untouched.

Implementation notes:
- **Do not copy the 3D thread layout of the MFEM diagonal kernel.** It uses a `Q1D^3` thread block and is limited to `q1d <= 10` (its own comment), which is too small for 3/2 rules at `p >= 6`. Use `forall_2D(ne, Q1D, Q1D)` with a loop over `qz`, like the apply kernels.
- Contraction order per component: z (threads over `(qx, qy)`, loop over `dz`, `qz`), sync, y (threads over `(qx, dy)`, loop `dz`, `qy`), sync, x (threads over `(dx, dy)`, loop `dz`, `qx`), add to output. Shared memory: two `Q1D^3` buffers, reused across components (q1d = 14 needs about 44 KB in double precision). Precompute `B^2` into shared memory.
- The method adds (does not overwrite), matching `AssembleDiagonalPA` semantics, because Part B calls it through MFEM's diagonal assembly path (Section 7.4).

### 5.10 `GetRotationNumberStats(sigma, threshold)`

Returns $\mu_{\max} = \max_q |\omega_q|/\sigma$ and the volume fraction where $\mu_q > $ `threshold`, both global over MPI ranks. This decides whether Part B is worth enabling in a given run (Section 7.1). It is not on the hot path; call it every N steps.

- Per point: `vol_q = W_q * detJ_q` (detJ from `geom->J`), `|omega_q| = |d_q| / (alpha * vol_q)` (3D: Euclidean norm of the three components), `mu_q = |omega_q| / sigma`.
- Write `mu_q` into `diag_tmp0` and `vol_q * (mu_q > threshold)` into `diag_tmp1`; also the total volume. Use `Vector::Max()` and `Vector::Sum()`, which reduce on the device.
- If `pa_mesh` is a `ParMesh`, `MPI_Allreduce` with `MPI_MAX` and `MPI_SUM`.
- Abort if `alpha == 0`.

---

## 6. Part A tests (`test_rotational_convection.cpp`)

Three test cases. Each runs over the sweep below using Catch2 `GENERATE`; `CAPTURE` every parameter so a failure names its case.

**Sweep (unless a test says otherwise).**

| Parameter | Values |
|---|---|
| dim | 2, 3 |
| p | 1, 2, 3, 4, plus the production orders if higher |
| rule | R1 default rule; R2 GLL collocated (`IntegrationRules(0, Quadrature1D::GaussLobatto).Get(geom, 2p-1)`, p+1 points); R3 Gauss 3/2 rule; R4 Gauss p+1 points |
| mesh | M2 curved (`MakeCurvedMesh`, Appendix B; sheared plus smooth curvature, mesh order p); M3 periodic in x (and z in 3D) via `Mesh::MakePeriodic` |
| device | `cpu` and `debug` always; `cuda` or `hip` when available |

The `debug` device (`Device device("debug")`) gives host and device separate memory on a CPU-only machine, so a missing `Read()`/`Write()` or a host access to device data fails loudly.

Fields: `vel_fn` from Appendix B (smooth, non-polynomial, every curl component nonzero). Random vectors: `Vector::Randomize(seed)` with fixed seeds. `alpha = 1.7` unless stated.

Tolerance for vector equality: `||a - b||_2 <= 1e-12 * max(||a||_2, ||b||_2)`. Calibration (Appendix A) lands at 1e-15 to 1e-17.

### A1. Vector identity against MFEM's convection integrator (the main test)

For `u_h` from `vel_fn`, `w = u_h`, `alpha = 1`, and one shared rule `ir` for all three terms:
- `C(u)u`: `NonlinearForm` with `VectorConvectionNLFIntegrator` (`SetIntRule(&ir)`), partial assembly.
- `N(u)u`: the new integrator, partial assembly.
- `G(u)`: `LinearForm` with `VectorDomainLFIntegrator(KEGrad(u_h))` (Appendix B), `SetIntRule(&ir)`.

Assert `||C - N - G|| / ||C|| <= 1e-12` for every case in the sweep, including R2 and R4, which under-integrate. Also assert the sign-flipped residual `||C + N - G|| / ||C|| > 0.1`, proving the test can detect `u x omega`.

Catches: sign and orientation, the curl, Jacobian handling (`adj(J)` versus its transpose), weights and determinants, component placement in setup and apply, lexicographic ordering, all against an operator that shares no code with the new kernels. Calibrated: residual 3e-16 to 1.2e-15; sign-flipped residual 1.4 to 1.6.

### A2. Operator properties

Partial assembly form with only the new integrator; random `x`, `y`. Sections:

- **Skew-symmetry and energy neutrality.** `|y.Nx + x.Ny| / (||x|| ||Ny||) <= 1e-12` and `|x.Nx| / (||x|| ||Nx||) <= 1e-12`. Calibrated about 1e-17. This also catches a kernel that reads `w` instead of `x`, which A1 cannot (there `x = w`).
- **Transpose.** `MultTranspose(x) == -Mult(x)` bitwise, or to 1e-15 if the form path reorders operations.
- **Diagonal.** For the form `sigma M + nu K + N`, `AssembleDiagonal` equals that of `sigma M + nu K` bitwise; for `N` alone, exactly zero.
- **Update semantics on a prebuilt operator.** Build the constrained operator once with `FormSystemMatrix`. Change `w` in place, call `UpdateVorticity()`, and compare its action with a freshly built form and operator. Repeat with `SetLaggedVelocity(w2)` and with `SetAlpha(2.0)`. Catches stale buffers and any copy of the quadrature data held elsewhere.
- **Rotation-number diagnostic.** With the rigid rotation `w = Omega x x` (for example `Omega = (0.3, -0.5, 0.8)`; 2D `w = Omega(-y, x)`), `omega_h = 2 Omega` exactly. With `sigma = 2`, `max_mu` equals `|Omega|` to 1e-12; `vol_fraction` is 1 for `threshold < |Omega|` and 0 for `threshold > |Omega|`.
- **Determinism (GPU only).** Apply 100 times to the same input; all results bitwise identical. A missing `MFEM_SYNC_THREAD` or a shared-memory race usually shows up only here.

### A3. Serial versus parallel (`[Parallel]`, 1 and 4 ranks)

Same global mesh partitioned so shared dofs exist. Set `w` as a `ParGridFunction` from true dofs (`SetFromTrueDofs`), then `UpdateVorticity()`. With `x`, `y` the interpolants of two smooth fields, compute `y^T N x` with global inner products on true-dof vectors through `FormSystemMatrix`, and the rotation-number statistics. Both must equal the serial values to 1e-12. Also assert `x^T N x = 0` in parallel (the assembled `P^T N P` stays skew).

Catches stale shared dofs in `w`, wrong reductions in the diagnostic, and anything that breaks skew-symmetry across ranks. This is the one bug class that would otherwise first appear as a wrong answer deep in an integration run.

---

## 7. Part B: point-block Jacobi for the velocity block

### 7.1 Why, and when it matters

The velocity block is $A = \sigma M + \nu K + N$ with $\sigma = 3/(2\Delta t)$ for BDF2. Scalar Jacobi sees nothing of $N$ (P2: its diagonal is zero). A low-order-refined algebraic multigrid built on $\sigma M + \nu K$ would not see it either. The rotation coupling lives entirely in the off-diagonal entries of the nodal `dim x dim` blocks, so inverting those blocks is the natural fix:

- **Where $N$ is negligible**, the blocks are diagonal and point-block Jacobi is exactly scalar Jacobi.
- **Where rotation dominates**, it is nearly exact. With GLL collocation and $\nu = 0$, $\sigma M + N$ is exactly block diagonal (P7), so point-block Jacobi is the exact inverse.
- **Where viscosity dominates**, it is no better than scalar Jacobi.

Measured on the velocity block alone (Appendix A, Table A2): with GLL collocation and the mass term dominant, Jacobi went from 9 to 10, 26, 202 and about 2000 GMRES iterations as $|\omega|\Delta t$ went from 0 to 0.1, 1, 10 and 100; point-block Jacobi stayed at 9.

The deciding quantity is the rotation number $\mu = |\omega^*|/\sigma$ ($= \tfrac23|\omega^*|\Delta t$ for BDF2), from `GetRotationNumberStats`. A GMRES model for a spectrum spread along $1 \pm i\mu$ gives about 1.6 extra iterations per digit at $\mu = 0.5$, 2.6 at $\mu = 1$, 5 at $\mu = 2$, 12 at $\mu = 5$ and 46 at $\mu = 20$, applied to the part of the residual that lives where $\mu$ is that large. Order-of-magnitude estimates for bessemer's cases:

| Case | Where $\mu$ is largest | Estimate |
|---|---|---|
| Channel, direct numerical simulation, $\Delta t$ from Courant number $C$ | viscous sublayer | band about $0.13C$, peaks about $0.33C$; noticeable from $C \approx 3$ |
| Channel, wall-resolved large eddy simulation | viscous sublayer | band about $0.6C$, peaks about $1.5C$; noticeable from $C \approx 1$ |
| Bluff body, $\Delta t$ from bulk accuracy, $\tau = U\Delta t/D$, locally refined near-body cells | front-face boundary layer, corners, separating shear layers | $\mu \approx \tau\sqrt{Re}$: 1 to 5 at $Re = 10^4$, 3 to 16 at $10^5$, 10 to 50 at $10^6$ (for $\tau$ = 0.01 to 0.05) |

The bluff-body case is the main motivation. Unconditional stability lets $\Delta t$ be set by the bulk flow, so the tiny near-body cells run at local Courant numbers of 10 to 100, exactly where the vorticity is largest. In those cells rotation dominates both other terms: $\mu \gg 1$, while with about 10 cells across the boundary layer $\nu\Delta t/h_{\min}^2 \approx 17\tau$, only 0.2 to 0.9. This is the regime where point-block Jacobi is nearly exact. Locality does not make the problem small for GMRES: the number of affected eigenvalues equals the number of degrees of freedom in the refined region.

Limits: point-block Jacobi fixes only the (1,1) part of the rotation term. The Cahouet-Chabard Schur approximation also omits $N$; that is out of scope here.

### 7.2 Mathematics

At velocity node $a$ (a true dof of the scalar space), the block of $A$ is

$$B_a = \mathrm{diag}(d_{a,0}, \dots, d_{a,\mathrm{dim}-1}) + [s_a]_\times,\qquad s_{a,c} = \sum_e\sum_q \alpha\,\hat w_q \det J_q\,\omega_c(q)\,\varphi_a(q)^2.$$

- $d$ is the diagonal of $A$, which equals the diagonal of $\sigma M + \nu K$ (+ grad-div if present) because $N$ adds zero (P2).
- $s$ is the off-diagonal part of the nodal block of $N$ (Section 5.9 computes its element contributions).
- 3D: $[s]_\times = \begin{pmatrix} 0 & -s_2 & s_1\\ s_2 & 0 & -s_0\\ -s_1 & s_0 & 0\end{pmatrix}$. 2D: $B_a = \begin{pmatrix} d_0 & -s\\ s & d_1\end{pmatrix}$.
- With the Laplacian-form viscous term, nothing else contributes off-diagonal nodal entries. A grad-div or stress-form term adds symmetric off-diagonal nodal entries that this ignores; the preconditioner stays valid, just less exact. The diagonal entries of such terms are included automatically through $d$, provided their integrators implement `AssembleDiagonalPA` (already required for the current scalar Jacobi).

**Essential dofs.** The constrained operator (`ConstrainedOperator`, `DIAG_ONE`) has identity rows and columns on essential true dofs. If component $c$ of node $a$ is essential: set $d_{a,c} = 1$ and, in 3D, zero $s_{a,k}$ for every $k \neq c$ (those are exactly the entries in row and column $c$). In 2D, zero $s_a$ if either component is essential. A no-slip node (all components essential) gets the identity block. Validated against constrained reference matrices for "all components essential" and "component 0 only" (Appendix A, Table A3).

**Nonconforming meshes.** MFEM's diagonal assembly sums element contributions to true dofs with $|P^T|$ when the prolongation is not the identity, which is approximate at hanging nodes. The same approximation already applies to the scalar Jacobi diagonal. Nothing extra is needed.

### 7.3 Interface (`point_block_jacobi.hpp`)

```cpp
/** Matrix-free point-block Jacobi for the velocity block
    A = sigma M + nu K (+ other symmetric terms) + N(omega).
    Inverts the dim x dim block at each velocity node:
    diag(d_a) + [s_a]_x. Runs entirely on the device. */
class PointBlockJacobi : public Solver
{
public:
   /** fes: velocity space (vdim == dim); a ParFiniteElementSpace in parallel.
       rot: the rotation integrator used in the velocity operator, after its
            AssemblePA() (not owned; must outlive this object).
       ess_tdofs: velocity essential true dofs (not copied; must outlive this). */
   PointBlockJacobi(FiniteElementSpace &fes,
                    const VectorRotationalConvectionIntegrator &rot,
                    const Array<int> &ess_tdofs);
   ~PointBlockJacobi() override;

   /** Set the diagonal of the velocity operator (true-dof vector), for example
       from velocity_form.AssembleDiagonal(d). Call at setup and whenever dt or
       nu changes. Applies the essential-dof rule. */
   void SetDiagonal(const Vector &d);

   /** Recompute s from rot's current quadrature data. Call every step after
       rot.UpdateVorticity(). Applies the essential-dof rule. */
   void UpdateSkew();

   void Mult(const Vector &r, Vector &z) const override;
   void SetOperator(const Operator &) override { }   // no-op by design

   // For tests: true-dof vectors after the essential-dof rule.
   const Vector &GetDiagonal() const { return d; }
   const Vector &GetSkew() const { return s; }

private:
   int dim, n;                     // n = true dofs per component
   bool by_vdim;                   // true-dof layout: byNODES c*n+a, byVDIM a*dim+c
   const Array<int> &ess;
   Vector d, s;                    // size dim*n each (2D: s uses component 0)
   std::unique_ptr<BilinearForm> skew_form;         // PA form holding the proxy integrator
   std::unique_ptr<BatchedDirectSolver> blocks_inv; // MFEM batched inverse of the nodal blocks
   mutable Vector r_node, z_node;  // node-contiguous buffers (byNODES only)
};
```

### 7.4 Implementation (`point_block_jacobi.cpp`)

**Proxy integrator (file-local).** MFEM's `BilinearForm::AssembleDiagonal` already does everything needed to turn element vectors into true-dof vectors on the device: it calls each integrator's `AssembleDiagonalPA` on an E-vector, sums into the L-vector with the element restriction's `AbsMultTranspose` (which ignores orientation signs but keeps the values' signs, verified in `fem/restriction.cpp`), and sums to true dofs with $P^T$, or $|P^T|$ on nonconforming meshes (`ParBilinearForm::AssembleDiagonal`, `BilinearForm::AssembleDiagonal`). A proxy integrator points that machinery at $s$:

```cpp
namespace
{
class NodalSkewProxy : public BilinearFormIntegrator
{
   const VectorRotationalConvectionIntegrator &rot;
public:
   explicit NodalSkewProxy(const VectorRotationalConvectionIntegrator &r) : rot(r) { }
   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &) override { }   // nothing to set up
   void AssembleDiagonalPA(Vector &s_e) override { rot.AddNodalSkewPA(s_e); }
};
}
```

A separate form is required because the rotation integrator's own `AssembleDiagonalPA` must keep returning zeros for the real operator.

**Constructor.**
1. Validate: `fes.GetVDim() == dim`, `dim` in {2, 3}, `rot` has been assembled on a space with the same mesh, order and element ordering (abort otherwise).
2. `n = fes.GetTrueVSize() / dim`; `by_vdim = (fes.GetOrdering() == Ordering::byVDIM)`.
3. Create `skew_form`: a `ParBilinearForm` if `fes` is a `ParFiniteElementSpace` (under `MFEM_USE_MPI`), otherwise a `BilinearForm`. `SetAssemblyLevel(AssemblyLevel::PARTIAL)`, add `new NodalSkewProxy(rot)`, `Assemble()`.
4. Allocate `d` and `s` (size `dim * n`) with `UseDevice(true)`; `d = 1`, `s = 0`.

**`SetDiagonal(d_in)`.** Copy into `d` (device), then a `forall` over `ess.Size()`: for essential true dof `k`, decode `(c, a)` from the layout, set `d[k] = 1`.

**`UpdateSkew()`.** `skew_form->AssembleDiagonal(s)`, then a `forall` over `ess.Size()` applying the rule of 7.2 (decode `(c, a)`; 3D: zero `s` at `(j, a)` for `j != c`; 2D: zero `s` at `(0, a)`). Concurrent writes of zero to the same entry from two essential components of one node are benign.

> **Implementation note (bessemer, 2026-10-05):** built this way first, then replaced by
> closed-form 2×2/3×3 inverses written in place plus one fused apply kernel on the true-dof
> layout (no gather/scatter, no per-step allocation). Measured on CPU: the apply went from
> ~10× to 2× a scalar Jacobi sweep, the rebuild from ~29 to ~2 ns/dof. Table A3's closed-form
> row is what this now matches. See CLAUDE.md "Convective form".

**Block inversion with MFEM's `BatchedDirectSolver`** (`linalg/batched/solver.hpp`). It takes a `DenseTensor` of shape `(m, m, n_mat)` holding `n_mat` square blocks, inverts all of them in one batched call at construction, and its `Mult` applies the block-diagonal inverse. The batched backend is chosen by `BatchedLinAlg`: MAGMA if MFEM was built with it, otherwise cuBLAS or hipBLAS on GPUs, otherwise a native `mfem::forall` implementation. Everything stays on the device (`DenseTensor` copies through `Memory::CopyFrom`).

At the end of `UpdateSkew()` (and of `SetDiagonal()`, so the inverse is valid after either call), rebuild the blocks and the solver:

```cpp
// blocks(i, j, a): column-major within each block, one block per node
DenseTensor blocks(dim, dim, n);
const auto D = d.Read(), S = s.Read();
auto Bt = Reshape(blocks.Write(), dim, dim, n);
mfem::forall(n, [=] MFEM_HOST_DEVICE (int a)
{
   // idx(c, a) = by_vdim ? a*dim + c : c*n + a
   if (dim == 3)
   {
      const real_t d0 = D[idx(0,a)], d1 = D[idx(1,a)], d2 = D[idx(2,a)];
      const real_t s0 = S[idx(0,a)], s1 = S[idx(1,a)], s2 = S[idx(2,a)];
      Bt(0,0,a) = d0;  Bt(0,1,a) = -s2; Bt(0,2,a) = s1;
      Bt(1,0,a) = s2;  Bt(1,1,a) = d1;  Bt(1,2,a) = -s0;
      Bt(2,0,a) = -s1; Bt(2,1,a) = s0;  Bt(2,2,a) = d2;
   }
   else
   {
      Bt(0,0,a) = D[idx(0,a)]; Bt(0,1,a) = -S[idx(0,a)];
      Bt(1,0,a) = S[idx(0,a)]; Bt(1,1,a) = D[idx(1,a)];
   }
});
blocks_inv.reset(new BatchedDirectSolver(blocks, BatchedDirectSolver::INVERSE));
```

Use `INVERSE` mode: each application is then a batched small matrix-vector product, which parallelizes better on GPUs than triangular solves. The blocks are always safely invertible: with positive diagonal entries, $\det B_a = d_0 d_1 d_2 + d_0 s_0^2 + d_1 s_1^2 + d_2 s_2^2 > 0$ for any vorticity, and essential nodes get identity rows and columns. `BatchedDirectSolver` has no `SetOperator` (it aborts), so a new one is constructed whenever the blocks change, which is once per time step.

**`Mult(r, z)`.** `BatchedDirectSolver::Mult` expects each node's components to be contiguous (`(dim, n)` layout).
- `Ordering::byVDIM`: the true-dof vector already has that layout. Call `blocks_inv->Mult(r, z)` directly.
- `Ordering::byNODES`: gather `r` into `r_node` (`r_node[dim*a + c] = r[c*n + a]`) with one `forall`, call `blocks_inv->Mult(r_node, z_node)`, and scatter back (`z[c*n + a] = z_node[dim*a + c]`). Allocate `r_node` and `z_node` once, on the device.

This path was validated in the prototype (Appendix A, Table A3).

**Cost and memory.** `UpdateSkew()`: about one diagonal assembly plus one batched inversion of `n` small blocks per time step. `Mult`: one batched small matrix-vector product (plus a gather and scatter for byNODES), comparable to a scalar Jacobi sweep. Persistent memory per velocity node in 3D: 9 values for the stored inverses, 6 for `d` and `s`, and 6 for the byNODES buffers. During the rebuild, the batched inversion briefly needs a few more copies of the blocks. All of this is small next to the velocity operator's partial assembly data. No global matrix is assembled.

If profiling ever shows the batched library slow for millions of tiny blocks, `BatchedLinAlg::SetActiveBackend(BatchedLinAlg::NATIVE)` switches to MFEM's own `forall` implementation without code changes.

### 7.5 Using it for the (1,1) block

Point-block Jacobi includes $N$ and is nonsymmetric, so **it cannot precondition PCG**. Provide three runtime-selectable configurations for $\hat A^{-1}$ in the block preconditioner, all with an outer FGMRES:

| Configuration | $\hat A^{-1}$ | Notes |
|---|---|---|
| `jacobi_pcg` (current default) | PCG on $\sigma M + \nu K$ (existing form, unchanged) with `OperatorJacobiSmoother` | $N$ enters only through the outer operator. Valid because PCG's operator stays symmetric. |
| `pbj_krylov` | GMRES (Krylov dimension 20) or BiCGStab on the full $A$, preconditioned by `PointBlockJacobi`, relative tolerance about $10^{-2}$, at most about 30 iterations | Recommended when `GetRotationNumberStats` shows a non-negligible volume with $\mu > 1$. |
| `pbj_only` | `PointBlockJacobi` applied once | Cheapest; the outer FGMRES does all the viscous work, so outer iterations (each with a Schur solve) may rise. |

Loose inner tolerances are deliberate: $\hat A$ is only an approximation inside the outer method, and solving it more accurately than the remaining Schur mismatch cannot lower outer counts.

**Building the full operator without duplicating partial assembly data.** Keep the existing symmetric form ($\sigma M + \nu K$, unchanged, used by `jacobi_pcg` and for the diagonal), and put the rotation integrator in its own form:

```cpp
ParBilinearForm n_form(&vfes);
auto *rot = new VectorRotationalConvectionIntegrator(w_star, 1.0);
n_form.AddDomainIntegrator(rot);
n_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
n_form.Assemble();

OperatorHandle N_rap;
n_form.FormSystemMatrix(Array<int>(), N_rap);   // P^T N P, no constraints
ConstrainedOperator N_c(N_rap.Ptr(), vel_ess_tdof, false, Operator::DIAG_ZERO);
SumOperator A_full(A_sym.Ptr(), 1.0, &N_c, 1.0, false, false);   // A_sym: existing constrained operator

PointBlockJacobi pbj(vfes, *rot, vel_ess_tdof);
Vector diag(vfes.GetTrueVSize());
sym_form.AssembleDiagonal(diag);
pbj.SetDiagonal(diag);                          // again whenever dt changes

// every step:
//   w_star = 3 u^n - 3 u^{n-1} + u^{n-2};  w_star.SetFromTrueDofs(...)
//   rot->UpdateVorticity();  pbj.UpdateSkew();
```

`DIAG_ZERO` matters: partially assembled `FormSystemMatrix` always constrains with `DIAG_ONE`, so summing two operators constrained that way puts 2 on the essential diagonal and halves inhomogeneous Dirichlet values (for example an inflow profile).

### 7.6 Tests (`test_point_block_jacobi.cpp`)

Three test cases, same conventions and sweep as Section 6 unless stated. `alpha = 1.7`, `sigma = 3`, fields from `vel_fn`, mesh M2.

The bar here is lower than for Part A. The outer solver always applies the true operator, so a bug in point-block Jacobi costs iterations, not correctness, and integration iteration counts are a second safety net.

**B1. Exactness under collocation, with essential dofs (matrix-free; runs on GPU).** GLL collocated rule for both the mass and rotation integrators, `nu = 0`, so the constrained partially assembled operator `A = sigma M + N` stays exactly block diagonal (P7). Assert `||pbj.Mult(A x) - x|| / ||x|| <= 1e-12` for:
- no essential dofs;
- all components essential on the boundary;
- only component 0 essential (`GetEssentialTrueDofs(bdr, ess, 0)`);
- both `Ordering::byNODES` and `Ordering::byVDIM`.

Catches a transposed block or a wrong fill of the `DenseTensor`, gather and scatter errors for byNODES, errors in the skew assembly path, and a wrong essential-dof rule. Calibrated 1.4e-16 (2D) and 1.6e-16 (3D) through `BatchedDirectSolver`.

**B2. Blocks against a reference with a Gauss rule and `nu > 0` (CPU, small mesh).** B1 cannot check the contraction in `AddNodalSkewPA`: with collocation the interpolation matrix is the identity, so a transposed index there is invisible. Build a legacy `BilinearForm` with `VectorMassIntegrator(sigma)`, `VectorDiffusionIntegrator(nu)` and `VectorMassIntegrator(SkewFromOmega(CurlGridFunctionCoefficient(&w), alpha))` (Appendix B; MFEM's own classes, not the new integrator), using the same Gauss rule R4 as the new integrator, `nu = 0.05`, 2 elements per side. For every node and component pair, the assembled entry must equal the block entry built from `GetDiagonal()` and `GetSkew()`, to 1e-12 relative to the largest entry. Calibrated 6e-17 to 3e-16.

**B3. Iterations at large rotation number (`[Parallel]`, 1 and 4 ranks).** Single solves, no time stepping. Operator `A = sigma M + nu K + N`, Dirichlet on the whole boundary, p = 4, rule R4, `nu dt / h^2` about 0.01 (mass dominated), random right-hand side, GMRES (Krylov dimension 50) to 1e-10 preconditioned by `PointBlockJacobi`. Scale `alpha` so `max|omega| dt` is 0 and then 100. Then change `w` (a different field), call `UpdateVorticity()` and `UpdateSkew()`, and solve again at `max|omega| dt = 100`. Assert:
- convergence in every solve;
- iterations at `max|omega| dt = 100` at most 3 times the count at 0, for both fields (Table A2: 54 versus 25 with this rule);
- the same iteration counts on 1 and 4 ranks (to within 2).

Catches stale skew data after `w` changes, wrong shared-dof sums in parallel, and a preconditioner that silently degrades to scalar Jacobi.

### 7.7 MFEM facts this part depends on (re-verify if MFEM changes)

- `PABilinearFormExtension::AssembleDiagonal` zeroes the local E-vector, calls each integrator's `AssembleDiagonalPA` (which add), then `ElementRestriction::AbsMultTranspose` (`fem/bilinearform_ext.cpp`, `fem/restriction.cpp`).
- `ParBilinearForm::AssembleDiagonal` applies $P^T$ (or $|P^T|$ for nonconforming meshes); `BilinearForm::AssembleDiagonal` uses $|cP^T|$ (`fem/pbilinearform.cpp`, `fem/bilinearform.cpp`).
- `BilinearFormIntegrator::AssemblePA(fes)` aborts by default, so the proxy must override it.
- `Operator::FormConstrainedSystemOperator` builds `ConstrainedOperator` with `DIAG_ONE` regardless of the form's diagonal policy (`linalg/operator.cpp`). A legacy `BilinearForm` keeps the original diagonal on essential rows by default (`DIAG_KEEP`), so any comparison of constrained legacy and partially assembled operators must set `DIAG_ONE` on the legacy form.
- `BatchedDirectSolver(const DenseTensor &A, Mode, Backend)` deep-copies `A` and inverts (or LU-factors) it in the constructor; `Mult(x, y)` applies the block-diagonal inverse with `x`, `y` laid out as `(m, n_mat)`; `SetOperator` aborts (`linalg/batched/solver.cpp`, `linalg/batched/native.cpp`). `DenseTensor` copies stay on the device (`Array` copy uses `Memory::CopyFrom`).
- `Vector::Max()` and `Vector::Sum()` reduce on the device.

---

## 8. Integration notes (velocity block and time stepping only)

- **The velocity operator is nonsymmetric.** MINRES on the coupled system is no longer valid; use FGMRES. PCG remains valid only on the symmetric part ($\sigma M + \nu K$) with a symmetric preconditioner (`jacobi_pcg` in 7.5).
- **Energy stability** of the semi-implicit step follows from P1 for any lagged vorticity and any divergence error.
- **Time stepping.** BDF2 with EXT3: $\sigma = 3/(2\Delta t)$, $w^* = 3u^n - 3u^{n-1} + u^{n-2}$. EXT3 is stable here (the term does no work for any $w^*$) but cannot raise the order above BDF2's 2; EXT2 ($2u^n - u^{n-1}$) gives the same order with one fewer stored field. Call `UpdateVorticity()` and `PointBlockJacobi::UpdateSkew()` once per step, after `SetFromTrueDofs` on $w^*$.
- **Variable time steps.** When $\Delta t$ changes, update the mass coefficient (reassemble the symmetric form) and call `PointBlockJacobi::SetDiagonal` again. Nothing else here depends on $\Delta t$.
- **Pressure is the Bernoulli pressure** $P = p + \tfrac12|u|^2$. This affects natural (do-nothing) outflow conditions and pressure boundary data. Recover static pressure for output with a separate Poisson solve if pressure statistics matter.
- **Dealiasing.** The rotational form aliases more than the skew-symmetric form when under-integrated (Zang 1991). Use a 3/2 rule for direct numerical simulation if the cost is acceptable.
- **Monitoring.** Log `GetRotationNumberStats(sigma)` and outer FGMRES counts every N steps. The volume fraction with $\mu > 1$ predicts when `pbj_krylov` pays off better than the maximum does.
- **Schur complement.** Out of scope. Cahouet-Chabard also omits $N$; with large local $\mu$ that may remain the bottleneck after Part B.
- **Integration tests (separate files, not part of this spec):** convergence rates (projection, expected rate p; manufactured steady problem, expected rate p + 1); energy and time stepping (implicit midpoint energy conservation, second-order convergence in time for a rigid rotation, BDF2-EXT3 energy never growing); a manufactured solution or Kovasznay flow in the full solver; Taylor-Green energy decay; iteration counts on the adaptively refined bluff-body mesh for each configuration in 7.5.

---

## 9. Work order and acceptance

1. Part A: header, setup kernel, apply kernel, transpose, diagonal. Pass A1 and A2 on `cpu` and `debug`.
2. Specializations for production sizes; A1 and A2 on GPU.
3. Parallel: A3.
4. `AddNodalSkewPA` and `GetRotationNumberStats` (A2's diagnostic section).
5. Part B: `PointBlockJacobi`; B1 and B2, then B3.
6. The three (1,1) configurations of 7.5 wired into bessemer behind a runtime option.

Acceptance: all six tests pass on `cpu` and `debug`, and on GPU when available; no new compiler warnings; exactly the six files of Section 3.

Optional, when profiling: time `AddMultPA` against scalar-coefficient `VectorMassIntegrator` (expect 1.2 to 1.8 times), `UpdateVorticity()` against one `VectorConvectionNLFIntegrator::AddMultPA`, and `PointBlockJacobi::Mult` against `OperatorJacobiSmoother::Mult` (expect within 2 times).

---

## Appendix A: calibration results

Obtained with host-only prototypes at MFEM commit `c362dea2`, building $N$ from existing MFEM classes (`VectorMassIntegrator` with the skew `MatrixCoefficient` of Appendix B). These are the numbers the new code must reproduce. Mesh M2 (sheared plus smooth curvature), mesh order = p.

**Table A1. Invariants (p = 3, 2 elements per side).**

| Quantity | 2D | 3D |
|---|---|---|
| `\|y.Nx + x.Ny\| / (\|x\| \|Ny\|)` | 4e-17 to 9e-17 | 8e-18 to 4e-17 |
| `\|x.Nx\| / (\|x\| \|Nx\|)` | 4e-17 to 5e-17 | 6e-17 |
| `max \|diag(N)\|` | 0 | 0 |
| A1 residual, rules R1, R2, Gauss 2p, affine and curved | 3e-16 to 8e-16 | 4e-16 to 1.2e-15 |
| A1 sign-flipped residual | 1.54 to 1.60 | 1.42 to 1.51 |
| Rigid rotation, `max \|curl w_h - 2 Omega\|` | | 1.4e-14 |

**Table A2. GMRES(50) iterations to 1e-10 on the velocity block alone, 3D, p = 4, 3x3x3 elements, 6591 dofs, Dirichlet, GLL basis. Entries: Jacobi / point-block Jacobi; "fail" = no convergence in 3000.**

| `nu dt/h^2` | `\|omega\| dt` = 0 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| Gauss p+1 rule: 0.01 | 25 / 25 | 27 / 26 | 54 / 27 | 559 / 37 | fail / 54 |
| 1 | 40 / 40 | 42 / 42 | 49 / 48 | 131 / 69 | 829 / 57 |
| 100 | 74 / 74 | 84 / 84 | 85 / 85 | 95 / 95 | 122 / 116 |
| GLL collocated: 0.01 | 9 / 9 | 10 / 9 | 26 / 9 | 202 / 9 | 1984 / 9 |
| 1 | 46 / 46 | 49 / 49 | 58 / 55 | 151 / 77 | 928 / 54 |
| 100 | 92 / 92 | 95 / 95 | 100 / 100 | 110 / 110 | 149 / 142 |

In this table the time step was applied as $\sigma = 1/\Delta t$. Reading: the skew term is harmless at `|omega| dt <= 0.1`, costs up to 2x with scalar Jacobi at `|omega| dt ~ 1` when the mass term dominates, and breaks scalar Jacobi beyond that; point-block Jacobi removes the dependence where mass and rotation dominate and changes little where viscosity dominates.

**Table A3. Point-block Jacobi design validation (p = 3, M2 with 2 elements per side, alpha = 1.7, sigma = 3).** $s$ computed by the tensor contraction of 5.9 from quadrature data, summed through MFEM's partially assembled `AssembleDiagonal` via the proxy integrator of 7.4; blocks inverted and applied both with a closed-form 3x3 inverse and with `BatchedDirectSolver` (`INVERSE` mode, byNODES gather and scatter) as in 7.4.

| Check | 2D GLL | 2D Gauss | 3D GLL | 3D Gauss |
|---|---|---|---|---|
| Nodal blocks versus legacy reference, `nu = 0` | 2.1e-16 | 3.1e-16 | 1.2e-16 | 3.3e-16 |
| Nodal blocks versus legacy reference, `nu = 0.05` (test B2) | 1.2e-16 | 1.3e-16 | 5.6e-17 | 9.4e-17 |
| Exactness `\|PBJ(Ax) - x\| / \|x\|`, `nu = 0` (test B1), closed form | 1.5e-16 | n/a | 1.8e-16 | n/a |
| Same, `BatchedDirectSolver` | 1.4e-16 | n/a | 1.6e-16 | n/a |
| Essential rule, all components and component 0 only | same as blocks | same as blocks | same as blocks | same as blocks |

## Appendix B: test-only code

```cpp
// alpha * [omega]_x from any VectorCoefficient omega (vdim 1 in 2D, 3 in 3D).
// Used with MFEM's VectorMassIntegrator(MatrixCoefficient&) as the reference in B2.
class SkewFromOmega : public MatrixCoefficient
{
   VectorCoefficient &om; real_t alpha; Vector o;
public:
   SkewFromOmega(int dim, VectorCoefficient &om_, real_t a = 1.0)
      : MatrixCoefficient(dim), om(om_), alpha(a), o(om_.GetVDim()) {}
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      om.Eval(o, T, ip);
      const int d = GetHeight();
      K.SetSize(d); K = 0.0;
      if (d == 2) { K(0,1) = -alpha*o(0); K(1,0) = alpha*o(0); }
      else
      {
         K(0,1) = -alpha*o(2); K(0,2) =  alpha*o(1);
         K(1,0) =  alpha*o(2); K(1,2) = -alpha*o(0);
         K(2,0) = -alpha*o(1); K(2,1) =  alpha*o(0);
      }
   }
};

// grad(|u|^2/2) = (grad u)^T u, evaluated from a GridFunction (test A1).
class KEGrad : public VectorCoefficient
{
   const GridFunction &u; DenseMatrix G; Vector U;
public:
   KEGrad(const GridFunction &u_, int d) : VectorCoefficient(d), u(u_) {}
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      u.GetVectorGradient(T, G);   // G(c,k) = du_c/dx_k
      u.GetVectorValue(T, ip, U);
      V.SetSize(vdim);
      G.MultTranspose(U, V);
   }
};

// Smooth test field (2D and 3D); every curl component is nonzero.
void vel_fn(const Vector &x, Vector &u)  // g_dim = mesh dimension
{
   const real_t X = x(0), Y = x(1), Z = (g_dim == 3) ? x(2) : 0.0;
   if (g_dim == 2)
   {
      u(0) = sin(M_PI*X)*cos(1.3*M_PI*Y) + 0.3*Y;
      u(1) = cos(0.7*M_PI*X)*sin(M_PI*Y) - 0.2*X*X;
   }
   else
   {
      u(0) = sin(M_PI*X)*cos(1.3*M_PI*Y)*cos(0.6*Z) + 0.3*Y;
      u(1) = cos(0.7*M_PI*X)*sin(M_PI*Y)*sin(0.9*Z) - 0.2*X*X;
      u(2) = sin(0.8*X + 1.1*Y)*cos(M_PI*Z) + 0.1*Z*Y;
   }
}

// Curved test mesh M2: shear (non-symmetric J) plus smooth curvature.
Mesh MakeCurvedMesh(int dim, int n, int mesh_order)
{
   Mesh mesh = (dim == 2)
      ? Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1.0, 1.0)
      : Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   mesh.SetCurvature(mesh_order, false, dim, Ordering::byNODES);
   mesh.Transform([=](const Vector &x, Vector &y)
   {
      y = x;
      y(1) += 0.2*x(0);
      if (dim == 3) { y(2) += 0.3*x(0); }
      y(0) += 0.04*sin(M_PI*x(1))*sin(M_PI*x(0));
      y(1) += 0.04*sin(M_PI*x(0))*sin(2*M_PI*x(1));
      if (dim == 3) { y(2) += 0.03*sin(M_PI*x(0))*sin(M_PI*x(2)); }
   });
   return mesh;
}
```
