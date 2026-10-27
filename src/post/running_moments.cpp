#include "post/running_moments.hpp"

#include "mfem/general/forall.hpp"

// This TU defines a device kernel (mfem::forall). A host compiler would build
// it as a host loop over device pointers, which segfaults on a GPU;
// src/CMakeLists.txt marks this file LANGUAGE CUDA, and this line turns a lost
// entry there into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "running_moments.cpp defines device kernels and must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Kernel in a NAMED namespace: nvcc rejects extended lambdas whose enclosing
// function has internal linkage.
namespace runmom
{

void Update(int n, int vdim, int sc, int sn, double f, double g,
            const Vector& x, Vector& m, Vector* normal, Vector* shear)
{
   const double* X = x.Read();
   double* M = m.ReadWrite();
   double* N = normal ? normal->ReadWrite() : nullptr;
   double* S = shear ? shear->ReadWrite() : nullptr;
   mfem::forall(n, [ = ] MFEM_HOST_DEVICE(int i)
   {
      double d[3] = {0.0, 0.0, 0.0};
      for (int a = 0; a < vdim; ++a)
      {
         const int j = a * sc + i * sn;
         d[a] = X[j] - M[j];
         M[j] += f * d[a];
      }
      if (N)
      {
         for (int a = 0; a < vdim; ++a)
         {
            const int j = a * sc + i * sn;
            N[j] = g * (N[j] + f * d[a] * d[a]);
         }
      }
      if (S)
      {
         // (0, 1); in 3D also (0, 2) and (1, 2).
         S[i * sn] = g * (S[i * sn] + f * d[0] * d[1]);
         if (vdim == 3)
         {
            S[sc + i * sn] = g * (S[sc + i * sn] + f * d[0] * d[2]);
            S[2 * sc + i * sn] = g * (S[2 * sc + i * sn] + f * d[1] * d[2]);
         }
      }
   });
}

} // namespace runmom

void UpdateRunningMoments(int n, int vdim, int stride_c, int stride_n,
                          double f, double g, const Vector& x, Vector& m,
                          Vector* normal, Vector* shear)
{
   MFEM_VERIFY(vdim >= 1 && vdim <= 3, "running_moments: vdim 1, 2 or 3");
   MFEM_VERIFY(!shear || (normal && vdim >= 2),
               "running_moments: covariances need the variances and vdim >= 2");
   if (n == 0) { return; }
   runmom::Update(n, vdim, stride_c, stride_n, f, g, x, m, normal, shear);
}

} // namespace incns
