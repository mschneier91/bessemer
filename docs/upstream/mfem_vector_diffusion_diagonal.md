# MFEM: `VectorDiffusionIntegrator::AssembleDiagonalPA` reserves ~11 GB of GPU memory

A finding from bessemer's GPU memory profiling (2026-10-13), written so that it can be
filed upstream. Status: **not reported to MFEM yet**; bessemer works around it
(`src/operators/lean_vector_diffusion.{hpp,cpp}`).

## Summary

On a CUDA device, the first call of `VectorDiffusionIntegrator::AssembleDiagonalPA` in 3D
leaves about **11 GB of device memory allocated for the rest of the process**, whatever the
mesh size, the polynomial order or the coefficient. Nothing is leaked in the usual sense:
it is thread stack that the CUDA runtime reserves for the kernel and keeps.

## Cause

`fem/integ/bilininteg_vecdiffusion_pa.cpp` (MFEM commit `1c19aba7`, develop, 2025):

```cpp
static void PAVectorDiffusionAssembleDiagonal(...)
{
   ...
   else if (dim == 3)
   {
      return PAVectorDiffusionDiagonal3D(NE, B, G, op, y, D1D, Q1D);   // T_D1D = T_Q1D = 0
   }
```

`PAVectorDiffusionDiagonal3D<T_D1D = 0, T_Q1D = 0>` is therefore always instantiated with
its run-time sizes, and inside its `mfem::forall(NE, ...)` body it declares

```cpp
constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;   // 14 under CUDA
constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;   // 14 under CUDA
real_t QQD[MQ1][MQ1][MD1];
real_t QDD[MQ1][MD1][MD1];
```

Two arrays of 14³ doubles are 43.9 kB of stack **per GPU thread**. The CUDA runtime sizes
its per-thread stack to the largest frame of any kernel launched and reserves it for every
thread the device can run concurrently. An H100 has 132 SMs × 2048 threads = 270,336;
270,336 × 43.9 kB = 11.9 GB, against 11,088 MiB (11.6 GB) measured. The reservation is
not released when the kernel returns.

The 2D kernel has the same pattern with three `[MQ1][MD1]` arrays (4.7 kB per thread,
about 1.3 GB by the same arithmetic; not measured).

## Measurements

Bridges-2, one NVIDIA H100 80 GB, CUDA 12.6.1, MFEM develop `1c19aba7` built with CUDA,
hypre 2.33.0. Device memory in use (`cudaMemGetInfo`, total − free) read before and after
`ParBilinearForm::AssembleDiagonal` on a form holding a `VectorDiffusionIntegrator`, a
`VectorMassIntegrator` and bessemer's grad-div integrator, H1 vector space, hexahedra:

| Mesh, order | First `AssembleDiagonal` | The same call on a second, identical form |
|---|---|---|
| 24³, Q3 | +11,088 MiB | +32 MiB |
| 32³, Q3 | ≈ +11,050 MiB | — |
| 32³, Q3, no grad-div term | ≈ +11,060 MiB | — |
| 32³, Q2 | ≈ +11,060 MiB | — |

(The last three are the difference between the first and second build of the whole
momentum form, 12,332 − 1,276, 11,986 − 930 and 11,700 − 644 MiB.) Across mesh sizes the
solver's peak memory fitted 12 GB + 0.36 MB per element before the workaround and
1.8 GB + 0.24 MB per element after it (the second coefficient also reflects a second,
unrelated fix).

With the diagonal computed by a kernel without local arrays, the same runs peak at:

| Mesh (Q3/Q2 Taylor–Green, 10 steps) | Before | After |
|---|---|---|
| 24³ | 17.2 GB | 5.3 GB |
| 32³ | 23.7 GB | 9.8 GB |
| 56³ | 74.5 GB | 44.8 GB |
| 64³ | out of memory | 63.9 GB |

## Reproducing it

Any 3D H1 vector space on a CUDA device: partially assemble a form with a
`VectorDiffusionIntegrator`, call `AssembleDiagonal`, and compare `nvidia-smi` (or
`cudaMemGetInfo`) before and after. A `ParBilinearForm` on a 4³ hex mesh is enough: the
size does not matter.

## Suggested fixes upstream

1. Dispatch the diagonal through a kernel table with `(D1D, Q1D)` specializations, as
   `AddMultPA` and the scalar `DiffusionIntegrator` diagonal already do, so the arrays are
   sized by the template arguments.
2. Or move the work arrays to `MFEM_SHARED` memory in a `forall_2D` / `forall_3D` body
   (the pattern of `SmemPADiffusionDiagonal3D`).
3. More generally: any `mfem::forall` body with local arrays sized by
   `DofQuadLimits::MAX_*` costs (array bytes) × (concurrent device threads) of device
   memory for the life of the process, once. A grep for `[MQ1]` / `[MD1]` inside plain
   `forall` bodies would find the others; this is the only one bessemer has hit.

## bessemer's workaround

`incns::LeanVectorDiffusionIntegrator` derives from `VectorDiffusionIntegrator` and
overrides only `AssembleDiagonalPA`: a direct sum over the quadrature points with no local
arrays (O(p⁶) per element in 3D; irrelevant for a setup operation). Scalar coefficient,
`sdim == dim`; anything else falls back to MFEM's kernel. `lean_vector_diffusion_test`
checks it against MFEM's diagonal (measured difference 1.5e-16 to 8.2e-16, the last digit)
and requires the operator's action to be bit-identical.
