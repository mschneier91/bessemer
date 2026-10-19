// Shared helpers of the elasticity tests (linear_elasticity_test, the verify
// tier's verify_elasticity_test): exact solutions with their gradients and
// body forces, the exact traction sigma(u) n on any boundary, boundary
// attributes found by position, and displacement error norms.
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <utility>

namespace elasticity_test
{

using VecFn = std::function<void(const mfem::Vector&, mfem::Vector&)>;
using MatFn = std::function<void(const mfem::Vector&, mfem::DenseMatrix&)>;

/// sigma = lambda tr(G) I + mu (G + G^T) for the displacement gradient G.
inline void Stress(const mfem::DenseMatrix& G, double lambda, double mu,
                   mfem::DenseMatrix& S)
{
   const int dim = G.Height();
   S.SetSize(dim);
   const double tr = G.Trace();
   for (int i = 0; i < dim; ++i)
   {
      for (int j = 0; j < dim; ++j)
      {
         S(i, j) = mu * (G(i, j) + G(j, i)) + (i == j ? lambda* tr : 0.0);
      }
   }
}

/// The traction sigma(u) n of an exact solution on boundary elements, n the
/// unit outward normal (MFEM's boundary orientation, as for
/// BoundaryNormalLFIntegrator).
class TractionCoefficient : public mfem::VectorCoefficient
{
public:
   TractionCoefficient(int dim, MatFn grad, double lambda, double mu)
      : mfem::VectorCoefficient(dim), grad_(std::move(grad)), lambda_(lambda),
        mu_(mu) {}

   using mfem::VectorCoefficient::Eval;
   void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
             const mfem::IntegrationPoint& ip) override
   {
      const int dim = vdim;
      mfem::Vector x(dim), n(dim);
      T.SetIntPoint(&ip);
      T.Transform(ip, x);
      mfem::CalcOrtho(T.Jacobian(), n);
      n /= n.Norml2();
      mfem::DenseMatrix G(dim), S;
      grad_(x, G);
      Stress(G, lambda_, mu_, S);
      V.SetSize(dim);
      S.Mult(n, V);
   }

private:
   MatFn grad_;    ///< Exact displacement gradient du_i/dx_j.
   double lambda_; ///< First Lame parameter.
   double mu_;     ///< Shear modulus.
};

/// An exact displacement field with what a solve needs: u, grad u, the body
/// force -div sigma (zero when f is empty) and the traction on any face.
struct Exact
{
   std::unique_ptr<mfem::VectorFunctionCoefficient> u;    ///< Displacement.
   std::unique_ptr<mfem::MatrixFunctionCoefficient> grad; ///< Its gradient.
   std::unique_ptr<mfem::VectorFunctionCoefficient> f;    ///< Body force.
   std::unique_ptr<TractionCoefficient> traction;         ///< sigma n.

   Exact(int dim, VecFn uf, MatFn gf, VecFn ff, double lambda, double mu)
   {
      u = std::make_unique<mfem::VectorFunctionCoefficient>(dim, uf);
      grad = std::make_unique<mfem::MatrixFunctionCoefficient>(dim, gf);
      if (ff) { f = std::make_unique<mfem::VectorFunctionCoefficient>(dim, ff); }
      traction = std::make_unique<TractionCoefficient>(dim, gf, lambda, mu);
   }
};

/**
 * The boundary attributes whose faces all satisfy @p on (tested at each
 * boundary element's centre), as a sorted list, the same on every rank.
 */
inline std::vector<int> AttributesWhere(mfem::ParMesh& mesh,
                                        const std::function<bool(const mfem::Vector&)>& on)
{
   const int n = mesh.bdr_attributes.Size() ? mesh.bdr_attributes.Max() : 0;
   std::vector<int> yes(n, 0), no(n, 0);
   mfem::Vector c;
   for (int b = 0; b < mesh.GetNBE(); ++b)
   {
      mfem::ElementTransformation* T = mesh.GetBdrElementTransformation(b);
      T->Transform(mfem::Geometries.GetCenter(T->GetGeometryType()), c);
      const int a = mesh.GetBdrAttribute(b) - 1;
      if (on(c)) { yes[a] = 1; }
      else { no[a] = 1; }
   }
   MPI_Allreduce(MPI_IN_PLACE, yes.data(), n, MPI_INT, MPI_MAX, mesh.GetComm());
   MPI_Allreduce(MPI_IN_PLACE, no.data(), n, MPI_INT, MPI_MAX, mesh.GetComm());
   std::vector<int> out;
   for (int a = 0; a < n; ++a)
   {
      MFEM_VERIFY(!(yes[a] && no[a]), "elasticity_test: boundary attribute "
                  << a + 1 << " is only partly on the selected faces");
      if (yes[a]) { out.push_back(a + 1); }
   }
   return out;
}

/// Every boundary attribute of the mesh not in @p except.
inline std::vector<int> OtherAttributes(mfem::ParMesh& mesh,
                                        const std::vector<int>& except)
{
   std::vector<int> out;
   for (int i = 0; i < mesh.bdr_attributes.Size(); ++i)
   {
      const int a = mesh.bdr_attributes[i];
      bool skip = false;
      for (int e : except) { skip = skip || e == a; }
      if (!skip) { out.push_back(a); }
   }
   return out;
}

/// Displacement errors against an exact solution.
struct Errors
{
   double l2 = 0.0; ///< ||u_h - u||_L2.
   double h1 = 0.0; ///< |u_h - u|_H1 (seminorm).
   double u_norm = 0.0; ///< ||u||_L2 (for relative errors).
};

/// L2 and H1-seminorm errors by an element loop at the RuleBook rule of
/// order @p order (pass an elevated one, 2k + 4). Collective.
inline Errors ComputeErrors(const mfem::ParGridFunction& uh, Exact& ex,
                            const incns::RuleBook& rules, int order)
{
   mfem::ParMesh& mesh = *uh.ParFESpace()->GetParMesh();
   const int dim = mesh.Dimension();
   uh.HostRead();
   double loc[3] = {0.0, 0.0, 0.0};
   mfem::Vector a(dim), b(dim);
   mfem::DenseMatrix ga(dim), gb(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      const mfem::IntegrationRule& ir =
         rules.Get(mesh.GetElementBaseGeometry(e), order);
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const mfem::IntegrationPoint& ip = ir.IntPoint(i);
         T.SetIntPoint(&ip);
         const double w = ip.weight * T.Weight();
         uh.GetVectorValue(T, ip, a);
         ex.u->Eval(b, T, ip);
         loc[2] += w * (b * b);
         a -= b;
         loc[0] += w * (a * a);
         uh.GetVectorGradient(T, ga);
         ex.grad->Eval(gb, T, ip);
         ga -= gb;
         loc[1] += w * ga.FNorm2();
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, loc, 3, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   Errors r;
   r.l2 = std::sqrt(loc[0]);
   r.h1 = std::sqrt(loc[1]);
   r.u_norm = std::sqrt(loc[2]);
   return r;
}

// --- exact solutions ---------------------------------------------------------

/// The smooth manufactured solution u_i = prod_d sin(pi x_d) for every
/// component i, on the unit square / cube: div u != 0, so both Lame terms
/// enter; f = -mu lap u - (lambda + mu) grad div u.
inline std::unique_ptr<Exact> Trigonometric(int dim, double lambda, double mu)
{
   const double pi = M_PI;
   auto prod_sin = [dim, pi](const mfem::Vector & x, int skip1, int skip2)
   {
      double s = 1.0;
      for (int d = 0; d < dim; ++d)
      {
         if (d != skip1 && d != skip2) { s *= std::sin(pi * x(d)); }
      }
      return s;
   };
   VecFn uf = [dim, prod_sin](const mfem::Vector & x, mfem::Vector & v)
   {
      v.SetSize(dim);
      v = prod_sin(x, -1, -1);
   };
   MatFn gf = [dim, pi, prod_sin](const mfem::Vector & x, mfem::DenseMatrix & g)
   {
      g.SetSize(dim);
      for (int j = 0; j < dim; ++j)
      {
         const double dj = pi * std::cos(pi * x(j)) * prod_sin(x, j, -1);
         for (int i = 0; i < dim; ++i) { g(i, j) = dj; }
      }
   };
   VecFn ff = [dim, pi, prod_sin, lambda, mu](const mfem::Vector & x,
              mfem::Vector & v)
   {
      v.SetSize(dim);
      const double g = prod_sin(x, -1, -1);
      for (int i = 0; i < dim; ++i)
      {
         // d_i div u = pi^2 (-g + sum_{j != i} cos(pi x_i) cos(pi x_j) prod_{k
         // != i, j} sin(pi x_k)); lap u_i = -dim pi^2 g.
         double ddiv = -g;
         for (int j = 0; j < dim; ++j)
         {
            if (j == i) { continue; }
            ddiv += std::cos(pi * x(i)) * std::cos(pi * x(j)) * prod_sin(x, i, j);
         }
         ddiv *= pi * pi;
         v(i) = mu * dim * pi * pi * g - (lambda + mu) * ddiv;
      }
   };
   return std::make_unique<Exact>(dim, uf, gf, ff, lambda, mu);
}

/**
 * Timoshenko & Goodier's cantilever (Theory of Elasticity, 3rd ed., 1970,
 * section 21), plane stress: x in [0, L], y in [-D/2, D/2], loaded at x = 0
 * by a parabolic shear of total P, held at x = L. The displacement is a cubic
 * polynomial and the body force is zero, so Q_k with k >= 3 reproduces it.
 * The tip deflection u_y(0, 0) = P L^3 / (3 E I).
 */
inline std::unique_ptr<Exact> TimoshenkoCantilever(double E, double nu,
      double P, double L,
      double D)
{
   const double c = 0.5 * D, I = D * D * D / 12.0, G = E / (2.0 * (1.0 + nu));
   VecFn uf = [ = ](const mfem::Vector & x, mfem::Vector & v)
   {
      const double X = x(0), Y = x(1);
      v.SetSize(2);
      v(0) = -P * X * X * Y / (2 * E * I) - nu * P * Y * Y * Y / (6 * E * I) +
             P * Y * Y * Y / (6 * I * G) +
             (P * L * L / (2 * E * I) - P * c * c / (2 * I * G)) * Y;
      v(1) = nu * P * X * Y * Y / (2 * E * I) + P * X * X * X / (6 * E * I) -
             P * L * L * X / (2 * E * I) + P * L * L * L / (3 * E * I);
   };
   MatFn gf = [ = ](const mfem::Vector & x, mfem::DenseMatrix & g)
   {
      const double X = x(0), Y = x(1);
      g.SetSize(2);
      g(0, 0) = -P * X * Y / (E * I);
      g(0, 1) = -P * X * X / (2 * E * I) - nu * P * Y * Y / (2 * E * I) +
                P * Y * Y / (2 * I * G) + P * L * L / (2 * E * I) -
                P * c * c / (2 * I * G);
      g(1, 0) = nu * P * Y * Y / (2 * E * I) + P * X * X / (2 * E * I) -
                P * L * L / (2 * E * I);
      g(1, 1) = nu * P * X * Y / (E * I);
   };
   // Plane stress: lambda* = E nu / (1 - nu^2).
   const double lam = E * nu / (1.0 - nu * nu);
   return std::make_unique<Exact>(2, uf, gf, nullptr, lam, G);
}

/**
 * Lame's thick-walled cylinder under internal pressure p (Timoshenko &
 * Goodier 1970, section 28), plane strain, radii a < b: u = u_r(r) x / r with
 * u_r = (1 + nu) p a^2 / (E (b^2 - a^2)) ((1 - 2 nu) r + b^2 / r); zero body
 * force, sigma_rr(a) = -p, sigma_rr(b) = 0.
 */
inline std::unique_ptr<Exact> LameCylinder(double E, double nu, double p,
      double a, double b)
{
   const double C = (1.0 + nu) * p * a * a / (E * (b * b - a * a));
   VecFn uf = [ = ](const mfem::Vector & x, mfem::Vector & v)
   {
      const double r = std::hypot(x(0), x(1));
      const double ur = C * ((1.0 - 2.0 * nu) * r + b * b / r);
      v.SetSize(2);
      v(0) = ur * x(0) / r;
      v(1) = ur * x(1) / r;
   };
   MatFn gf = [ = ](const mfem::Vector & x, mfem::DenseMatrix & g)
   {
      // grad(u_r(r) x / r) = (u_r / r) I + (u_r' - u_r / r) x x^T / r^2.
      const double r = std::hypot(x(0), x(1));
      const double ur = C * ((1.0 - 2.0 * nu) * r + b * b / r);
      const double dur = C * ((1.0 - 2.0 * nu) - b * b / (r * r));
      g.SetSize(2);
      for (int i = 0; i < 2; ++i)
      {
         for (int j = 0; j < 2; ++j)
         {
            g(i, j) = (i == j ? ur / r : 0.0) +
                      (dur - ur / r) * x(i) * x(j) / (r * r);
         }
      }
   };
   const double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
   const double mu = E / (2.0 * (1.0 + nu));
   return std::make_unique<Exact>(2, uf, gf, nullptr, lam, mu);
}

/**
 * A quarter annulus a <= r <= b, 0 <= theta <= pi/2, as n x n quads with
 * curved nodes of order @p geom_order (exact up to the interpolation of the
 * circle). Boundary attributes as MakeCartesian2D's on the reference square.
 */
inline mfem::Mesh QuarterAnnulus(int n, double a, double b, int geom_order)
{
   mfem::Mesh m = mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL,
                  true, 1.0, 1.0);
   m.SetCurvature(geom_order);
   m.Transform([a, b](const mfem::Vector & xi, mfem::Vector & x)
   {
      const double r = a + (b - a) * xi(0);
      const double t = 0.5 * M_PI * xi(1);
      x.SetSize(2);
      x(0) = r * std::cos(t);
      x(1) = r * std::sin(t);
   });
   return m;
}

/**
 * Cook's membrane (Cook 1974): the quadrilateral (0, 0), (48, 44), (48, 60),
 * (0, 44) as n x n quads; clamped at x = 0, a uniform shear traction on
 * x = 48 of total 1. Boundary attributes as MakeCartesian2D's.
 */
inline mfem::Mesh CookMembrane(int n)
{
   mfem::Mesh m = mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL,
                  true, 1.0, 1.0);
   m.Transform([](const mfem::Vector & xi, mfem::Vector & x)
   {
      x.SetSize(2);
      x(0) = 48.0 * xi(0);
      const double yb = 44.0 * xi(0), yt = 44.0 + 16.0 * xi(0);
      x(1) = yb + xi(1) * (yt - yb);
   });
   return m;
}

// --- dynamics -----------------------------------------------------------------

/// A vector coefficient times a function of time (its own GetTime()).
class TimeScaled : public mfem::VectorCoefficient
{
public:
   TimeScaled(mfem::VectorCoefficient& c, std::function<double(double)> g)
      : mfem::VectorCoefficient(c.GetVDim()), c_(c), g_(std::move(g)) {}

   using mfem::VectorCoefficient::Eval;
   void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
             const mfem::IntegrationPoint& ip) override
   {
      c_.Eval(V, T, ip);
      V *= g_(GetTime());
   }

private:
   mfem::VectorCoefficient& c_;      ///< The spatial part.
   std::function<double(double)> g_; ///< The time factor.
};

/**
 * A separable motion u(x, t) = s(x) T(t) for rho u_tt - div sigma(u) = f:
 * s is quadratic and vanishes on x = 0 (so Q_k with k >= 2 holds it exactly
 * and x = 0 can be fixed), T(t) = cos(4 t) + sin(3 t) / 2. With the exact
 * body force and tractions the semi-discrete solution is exactly s T(t):
 * every error is the time integrator's.
 */
struct SeparableMotion
{
   int dim;       ///< Dimension.
   double lambda; ///< First Lame parameter.
   double mu;     ///< Shear modulus.
   double rho;    ///< Density.
   std::unique_ptr<mfem::VectorFunctionCoefficient> u;  ///< s T.
   std::unique_ptr<mfem::VectorFunctionCoefficient> v;  ///< s T'.
   std::unique_ptr<mfem::VectorFunctionCoefficient>
   f;  ///< rho s T'' - div sigma(s) T.
   std::unique_ptr<TractionCoefficient> traction_s;      ///< sigma(s) n.
   std::unique_ptr<TimeScaled> traction;                 ///< sigma(s) n T.

   static double T0(double t) { return std::cos(4 * t) + 0.5 * std::sin(3 * t); }
   static double T1(double t) { return -4 * std::sin(4 * t) + 1.5 * std::cos(3 * t); }
   static double T2(double t) { return -16 * std::cos(4 * t) - 4.5 * std::sin(3 * t); }

   /// s_i = x (a_i + b_i x + c_i y + d_i z).
   static constexpr double a[3] = {0.3, -0.4, 0.2};
   static constexpr double b[3] = {0.5, 0.1, -0.3}; ///< x coefficients.
   static constexpr double c[3] = {-0.2, 0.6, 0.1}; ///< y coefficients.
   static constexpr double d[3] = {0.1, -0.3, 0.4}; ///< z coefficients.

   static void S(int dim, const mfem::Vector& x, mfem::Vector& s)
   {
      s.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         double lin = a[i] + b[i] * x(0) + c[i] * x(1);
         if (dim == 3) { lin += d[i] * x(2); }
         s(i) = x(0) * lin;
      }
   }

   static void GradS(int dim, const mfem::Vector& x, mfem::DenseMatrix& g)
   {
      g.SetSize(dim);
      for (int i = 0; i < dim; ++i)
      {
         g(i, 0) = a[i] + 2 * b[i] * x(0) + c[i] * x(1) + (dim == 3 ? d[i] * x(2) : 0.0);
         g(i, 1) = c[i] * x(0);
         if (dim == 3) { g(i, 2) = d[i] * x(0); }
      }
   }

   /// div sigma(s): (lambda + mu) grad div s + mu lap s (constant).
   void DivSigma(mfem::Vector& r) const
   {
      // Hessian of s_i: H(0,0) = 2 b_i, H(0,1) = c_i, H(0,2) = d_i.
      auto H = [](int i, int j, int k)
      {
         if (j > k) { std::swap(j, k); }
         if (j != 0) { return 0.0; }
         return k == 0 ? 2 * b[i] : (k == 1 ? c[i] : d[i]);
      };
      r.SetSize(dim);
      for (int j = 0; j < dim; ++j)
      {
         double gdiv = 0.0;
         for (int i = 0; i < dim; ++i) { gdiv += H(i, j, i); }
         r(j) = (lambda + mu) * gdiv + mu * 2 * b[j];
      }
   }

   SeparableMotion(int dim_, double lambda_, double mu_, double rho_)
      : dim(dim_), lambda(lambda_), mu(mu_), rho(rho_)
   {
      const int n = dim;
      u = std::make_unique<mfem::VectorFunctionCoefficient>(n,
          [n](const mfem::Vector & x, double t, mfem::Vector & y)
      {
         S(n, x, y);
         y *= T0(t);
      });
      v = std::make_unique<mfem::VectorFunctionCoefficient>(n,
          [n](const mfem::Vector & x, double t, mfem::Vector & y)
      {
         S(n, x, y);
         y *= T1(t);
      });
      mfem::Vector ds;
      DivSigma(ds);
      const double r = rho;
      f = std::make_unique<mfem::VectorFunctionCoefficient>(n,
          [n, ds, r](const mfem::Vector & x, double t, mfem::Vector & y)
      {
         S(n, x, y);
         y *= r * T2(t);
         y.Add(-T0(t), ds);
      });
      traction_s = std::make_unique<TractionCoefficient>(n,
      [n](const mfem::Vector & x, mfem::DenseMatrix & g) { GradS(n, x, g); },
      lambda, mu);
      traction = std::make_unique<TimeScaled>(*traction_s, T0);
   }
};

} // namespace elasticity_test
