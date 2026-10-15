#include "solver/persistent_fgmres.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace incns
{

using namespace mfem;

namespace
{

// MFEM's helpers for (F)GMRES (linalg/solvers.cpp), unchanged.
inline void GeneratePlaneRotation(real_t& dx, real_t& dy, real_t& cs,
                                  real_t& sn)
{
   if (dy == 0.0)
   {
      cs = 1.0;
      sn = 0.0;
   }
   else if (std::fabs(dy) > std::fabs(dx))
   {
      real_t temp = dx / dy;
      sn = 1.0 / std::sqrt(1.0 + temp * temp);
      cs = temp * sn;
   }
   else
   {
      real_t temp = dy / dx;
      cs = 1.0 / std::sqrt(1.0 + temp * temp);
      sn = temp * cs;
   }
}

inline void ApplyPlaneRotation(real_t& dx, real_t& dy, real_t& cs, real_t& sn)
{
   real_t temp = cs * dx + sn * dy;
   dy = -sn * dx + cs * dy;
   dx = temp;
}

inline void Update(Vector& x, int k, DenseMatrix& h, Vector& s,
                   Array<Vector*>& v)
{
   Vector y(s);
   // Backsolve:
   for (int i = k; i >= 0; i--)
   {
      y(i) /= h(i, i);
      for (int j = i - 1; j >= 0; j--)
      {
         y(j) -= h(j, i) * y(i);
      }
   }
   for (int j = 0; j <= k; j++)
   {
      x.Add(y(j), *v[j]);
   }
}

} // namespace

Vector* PersistentFGMRESSolver::Work(
   std::vector<std::unique_ptr<Vector>>& pool, int i, int n) const
{
   if (static_cast<int>(pool.size()) <= i) { pool.resize(i + 1); }
   if (!pool[i] || pool[i]->Size() != n)
   {
      pool[i] = std::make_unique<Vector>(n);
      pool[i]->UseDevice(true);
   }
   return pool[i].get();
}

void PersistentFGMRESSolver::Trim(std::vector<std::unique_ptr<Vector>>& pool,
                                  int keep)
{
   if (static_cast<int>(pool.size()) > keep) { pool.resize(keep); }
}

double PersistentFGMRESSolver::KeptBytes() const
{
   double n = r_.Size();
   for (const auto& p : v_) { if (p) { n += p->Size(); } }
   for (const auto& p : z_) { if (p) { n += p->Size(); } }
   return n * sizeof(real_t);
}

void PersistentFGMRESSolver::ReleaseWork() const
{
   r_.Destroy();
   v_.clear();
   z_.clear();
}

// mfem::FGMRESSolver::Mult (linalg/solvers.cpp, MFEM 1c19aba) with the
// residual, v[i] and z[i] taken from the kept pools instead of new / delete.
// Everything else, the arithmetic included, is MFEM's.
void PersistentFGMRESSolver::Mult(const Vector& b, Vector& x) const
{
   DenseMatrix H(m + 1, m);
   Vector s(m + 1), cs(m + 1), sn(m + 1);
   Vector x_monitor;
   if (r_.Size() != b.Size())
   {
      r_.Destroy();
      r_.SetSize(b.Size());
   }
   Vector& r = r_;
   int v_used = 0, z_used = 0;
   auto finish = [&]()
   {
      Trim(v_, v_used + 2);
      Trim(z_, z_used + 2);
   };

   b.UseDevice(true);
   x.UseDevice(true);
   r.UseDevice(true);

   if (ControllerRequiresUpdate())
   {
      x_monitor.SetSize(x.Size());
      x_monitor.UseDevice(true);
   }
   else
   {
      x_monitor.MakeRef(x, 0, x.Size());
   }

   int i, j, k;

   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r);
   }
   else
   {
      x = 0.;
      r = b;
   }
   real_t beta = initial_norm = Norm(r); // beta = ||r||
   MFEM_VERIFY(IsFinite(beta), "beta = " << beta);

   final_norm = std::max(rel_tol * beta, abs_tol);

   if (Monitor(0, beta, r, x) || beta <= final_norm)
   {
      converged = true;
      final_norm = beta;
      final_iter = 0;

      Monitor(0, beta, r, x, true);
      finish();
      return;
   }

   // initialize the first pass
   converged = false;

   if (print_options.iterations || print_options.first_and_last)
   {
      mfem::out << "   Pass : " << std::setw(2) << 1
                << "   Iteration : " << std::setw(3) << 0
                << "  || r || = " << beta
                << (print_options.first_and_last ? " ...\n" : "\n");
   }

   Array<Vector*> v(m + 1);
   Array<Vector*> z(m + 1);
   for (i = 0; i <= m; i++)
   {
      v[i] = NULL;
      z[i] = NULL;
   }

   j = 1;
   while (j <= max_iter)
   {
      if (v[0] == NULL)
      {
         v[0] = Work(v_, 0, b.Size());
         v_used = std::max(v_used, 1);
      }
      (*v[0]) = 0.0;
      v[0] -> Add(1.0 / beta, r); // v[0] = r / ||r||
      s = 0.0; s(0) = beta;

      for (i = 0; i < m && j <= max_iter; i++, j++)
      {

         if (z[i] == NULL)
         {
            z[i] = Work(z_, i, b.Size());
            z_used = std::max(z_used, i + 1);
         }
         (*z[i]) = 0.0;

         if (prec)
         {
            prec->Mult(*v[i], *z[i]);
         }
         else
         {
            (*z[i]) = (*v[i]);
         }
         oper->Mult(*z[i], r);

         for (k = 0; k <= i; k++)
         {
            H(k, i) = Dot(r, *v[k]); // H(k,i) = r * v[k]
            r.Add(-H(k, i), (*v[k])); // r -= H(k,i) * v[k]
         }

         H(i + 1, i) = Norm(r); // H(i+1,i) = ||r||
         if (v[i + 1] == NULL)
         {
            v[i + 1] = Work(v_, i + 1, b.Size());
            v_used = std::max(v_used, i + 2);
         }
         (*v[i + 1]) = 0.0;
         v[i + 1] -> Add(1.0 / H(i + 1, i), r); // v[i+1] = r / H(i+1,i)

         for (k = 0; k < i; k++)
         {
            ApplyPlaneRotation(H(k, i), H(k + 1, i), cs(k), sn(k));
         }

         GeneratePlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));
         ApplyPlaneRotation(H(i, i), H(i + 1, i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i + 1), cs(i), sn(i));

         const real_t resid = std::fabs(s(i + 1));
         MFEM_VERIFY(IsFinite(resid), "resid = " << resid);
         if (print_options.iterations || (print_options.first_and_last &&
                                          resid <= final_norm))
         {
            mfem::out << "   Pass : " << std::setw(2) << (j - 1) / m + 1
                      << "   Iteration : " << std::setw(3) << j
                      << "  || r || = " << resid << std::endl;
         }

         if (ControllerRequiresUpdate())
         {
            x_monitor = x;
            Update(x_monitor, i, H, s, z);
         }

         if (Monitor(j, resid, r, x_monitor) || resid <= final_norm)
         {
            Update(x, i, H, s, z);
            final_norm = resid;
            final_iter = j;
            converged = true;

            if (print_options.summary)
            {
               mfem::out << "FGMRES: Number of iterations: " << final_iter << '\n';
            }

            Monitor(j, resid, r, x, true);
            finish();
            return;
         }
      }

      if (print_options.iterations)
      {
         mfem::out << "Restarting..." << std::endl;
      }

      Update(x, i - 1, H, s, z);

      oper->Mult(x, r);
      subtract(b, r, r);
      beta = Norm(r);
      MFEM_VERIFY(IsFinite(beta), "beta = " << beta);
      if (beta <= final_norm)
      {
         converged = true;

         break;
      }
   }

   final_norm = beta;
   final_iter = converged ? j : max_iter;

   // Note: j is off by one when we arrive here
   if (!print_options.iterations && print_options.first_and_last)
   {
      mfem::out << "   Pass : " << std::setw(2) << (j - 1) / m + 1
                << "   Iteration : " << std::setw(3) << j - 1
                << "  || r || = " << final_norm << std::endl;
   }
   if (print_options.summary || (print_options.warnings && !converged))
   {
      mfem::out << "FGMRES: Number of iterations: " << final_iter << '\n';
   }
   if (print_options.warnings && !converged)
   {
      mfem::out << "FGMRES: No convergence!\n";
   }

   Monitor(final_iter, final_norm, r, x, true);
   finish();
}

} // namespace incns
