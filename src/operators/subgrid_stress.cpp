#include "operators/subgrid_stress.hpp"

#include "mesh/mesh_size_coefficient.hpp"
#include "operators/convection.hpp"
#include "util/profiler.hpp"

#include <mfem/general/forall.hpp>

// The kernels below must be compiled by nvcc in a CUDA build, or they run on
// the host while the data is on the device. See src/CMakeLists.txt's nvcc
// list: this line turns a lost entry into a compile error.
#if defined(MFEM_USE_CUDA) && !defined(__CUDACC__)
#error "subgrid_stress.cpp must be compiled by nvcc -- see the nvcc TU list in src/CMakeLists.txt"
#endif

namespace incns
{

using namespace mfem;

// Device kernels live in free functions in a NAMED namespace: nvcc rejects
// extended lambdas in functions with internal linkage or in private members.
namespace sgs
{

// inv(J)(m, i) = d xi_m / d x_i of a 2 x 2 Jacobian; returns det J.
MFEM_HOST_DEVICE inline double Inverse2(double j00, double j01, double j10,
                                        double j11, double iv[3][3])
{
   const double det = j00 * j11 - j01 * j10;
   iv[0][0] = j11 / det;
   iv[0][1] = -j01 / det;
   iv[1][0] = -j10 / det;
   iv[1][1] = j00 / det;
   iv[0][2] = iv[1][2] = iv[2][0] = iv[2][1] = iv[2][2] = 0.0;
   return det;
}

// inv(J)(m, i) = d xi_m / d x_i of a 3 x 3 Jacobian j[r][c]; returns det J.
MFEM_HOST_DEVICE inline double Inverse3(const double j[3][3], double iv[3][3])
{
   const double det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) -
                      j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) +
                      j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
   iv[0][0] = (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det;
   iv[0][1] = (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det;
   iv[0][2] = (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det;
   iv[1][0] = (j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det;
   iv[1][1] = (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det;
   iv[1][2] = (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det;
   iv[2][0] = (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det;
   iv[2][1] = (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det;
   iv[2][2] = (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det;
   return det;
}

// The stress at one point, weighted and pulled back:
// D[m][c] = wdet * sum_j tau_cj iv[m][j], tau = 2 nu_t S(g), so that the
// dof kernel's sum_q sum_m D[m][c](q) dphi/dxi_m(q) is int tau_cj dphi/dx_j.
// Returns nu_t; diss receives 2 nu_t S:S (the local dissipation density).
MFEM_HOST_DEVICE inline double PointStress(SubgridModel model, double cd2,
      double constant, const double g[3][3],
      const double iv[3][3], double wdet,
      double D[3][3], double& diss)
{
   const double nu_t = subgrid::Viscosity(model, g, cd2, constant);
   double tau[3][3];
   double ss = 0.0;
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         const double s = 0.5 * (g[i][j] + g[j][i]);
         tau[i][j] = 2.0 * nu_t * s;
         ss += s * s;
      }
   for (int m = 0; m < 3; ++m)
      for (int c = 0; c < 3; ++c)
      {
         D[m][c] = wdet * (tau[c][0] * iv[m][0] + tau[c][1] * iv[m][1] +
                           tau[c][2] * iv[m][2]);
      }
   diss = 2.0 * nu_t * ss;
   return nu_t;
}

// One thread per quadrature point: grad u, the model, the weighted stress
// Dq(m, c, q, e) = D[m][c] (m the reference direction, c the component).
void Points2D(int ne, int D1D, int Q1D, SubgridModel model, double constant,
              const Vector& cd2, const Array<real_t>& b, const Array<real_t>& g,
              const Array<real_t>& w, const Vector& jac, const Vector& xe,
              Vector& dq)
{
   const int nq = Q1D * Q1D;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto W = Reshape(w.Read(), Q1D, Q1D);
   const auto J = Reshape(jac.Read(), Q1D, Q1D, 2, 2, ne);
   const auto X = Reshape(xe.Read(), D1D, D1D, 2, ne);
   const auto CD2 = cd2.Read();
   auto Dq = Reshape(dq.Write(), 2, 2, nq, ne);
   mfem::forall(ne * nq, [ = ] MFEM_HOST_DEVICE(int idx)
   {
      const int e = idx / nq;
      const int q = idx - e * nq;
      const int qx = q % Q1D, qy = q / Q1D;
      double gr[2][2] = {{0.0, 0.0}, {0.0, 0.0}}; // d u_c / d xi_m
      for (int dy = 0; dy < D1D; ++dy)
         for (int dx = 0; dx < D1D; ++dx)
         {
            const double g0 = G(qx, dx) * B(qy, dy);
            const double g1 = B(qx, dx) * G(qy, dy);
            for (int c = 0; c < 2; ++c)
            {
               const double v = X(dx, dy, c, e);
               gr[c][0] += g0 * v;
               gr[c][1] += g1 * v;
            }
         }
      double iv[3][3];
      const double det = Inverse2(J(qx, qy, 0, 0, e), J(qx, qy, 0, 1, e),
                                  J(qx, qy, 1, 0, e), J(qx, qy, 1, 1, e), iv);
      double gu[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
      for (int c = 0; c < 2; ++c)
         for (int i = 0; i < 2; ++i)
         {
            gu[c][i] = gr[c][0] * iv[0][i] + gr[c][1] * iv[1][i];
         }
      const double wdet = W(qx, qy) * (det < 0.0 ? -det : det);
      double D[3][3], diss;
      PointStress(model, CD2[e], constant, gu, iv, wdet, D, diss);
      for (int m = 0; m < 2; ++m)
         for (int c = 0; c < 2; ++c) { Dq(m, c, q, e) = D[m][c]; }
   });
}

void Points3D(int ne, int D1D, int Q1D, SubgridModel model, double constant,
              const Vector& cd2, const Array<real_t>& b, const Array<real_t>& g,
              const Array<real_t>& w, const Vector& jac, const Vector& xe,
              Vector& dq)
{
   const int nq = Q1D * Q1D * Q1D;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto W = Reshape(w.Read(), Q1D, Q1D, Q1D);
   const auto J = Reshape(jac.Read(), Q1D, Q1D, Q1D, 3, 3, ne);
   const auto X = Reshape(xe.Read(), D1D, D1D, D1D, 3, ne);
   const auto CD2 = cd2.Read();
   auto Dq = Reshape(dq.Write(), 3, 3, nq, ne);
   mfem::forall(ne * nq, [ = ] MFEM_HOST_DEVICE(int idx)
   {
      const int e = idx / nq;
      const int q = idx - e * nq;
      const int qx = q % Q1D, qy = (q / Q1D) % Q1D, qz = q / (Q1D * Q1D);
      double gr[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
      for (int dz = 0; dz < D1D; ++dz)
         for (int dy = 0; dy < D1D; ++dy)
            for (int dx = 0; dx < D1D; ++dx)
            {
               const double g0 = G(qx, dx) * B(qy, dy) * B(qz, dz);
               const double g1 = B(qx, dx) * G(qy, dy) * B(qz, dz);
               const double g2 = B(qx, dx) * B(qy, dy) * G(qz, dz);
               for (int c = 0; c < 3; ++c)
               {
                  const double v = X(dx, dy, dz, c, e);
                  gr[c][0] += g0 * v;
                  gr[c][1] += g1 * v;
                  gr[c][2] += g2 * v;
               }
            }
      double j[3][3];
      for (int r = 0; r < 3; ++r)
         for (int c = 0; c < 3; ++c) { j[r][c] = J(qx, qy, qz, r, c, e); }
      double iv[3][3];
      const double det = Inverse3(j, iv);
      double gu[3][3];
      for (int c = 0; c < 3; ++c)
         for (int i = 0; i < 3; ++i)
         {
            gu[c][i] = gr[c][0] * iv[0][i] + gr[c][1] * iv[1][i] +
                       gr[c][2] * iv[2][i];
         }
      const double wdet = W(qx, qy, qz) * (det < 0.0 ? -det : det);
      double D[3][3], diss;
      PointStress(model, CD2[e], constant, gu, iv, wdet, D, diss);
      for (int m = 0; m < 3; ++m)
         for (int c = 0; c < 3; ++c) { Dq(m, c, q, e) = D[m][c]; }
   });
}

// One thread per element dof: Y(d, c, e) = sum_q sum_m Dq(m, c, q, e)
// dphi_d/dxi_m(q), the element vector in the lexicographic layout.
void Dofs2D(int ne, int D1D, int Q1D, const Array<real_t>& b,
            const Array<real_t>& g, const Vector& dq, Vector& ye)
{
   const int nq = Q1D * Q1D, nd = D1D * D1D;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Dq = Reshape(dq.Read(), 2, 2, nq, ne);
   auto Y = Reshape(ye.Write(), D1D, D1D, 2, ne);
   mfem::forall(ne * nd, [ = ] MFEM_HOST_DEVICE(int idx)
   {
      const int e = idx / nd;
      const int d = idx - e * nd;
      const int dx = d % D1D, dy = d / D1D;
      double y[2] = {0.0, 0.0};
      for (int qy = 0; qy < Q1D; ++qy)
         for (int qx = 0; qx < Q1D; ++qx)
         {
            const int q = qx + Q1D * qy;
            const double g0 = G(qx, dx) * B(qy, dy);
            const double g1 = B(qx, dx) * G(qy, dy);
            for (int c = 0; c < 2; ++c)
            {
               y[c] += Dq(0, c, q, e) * g0 + Dq(1, c, q, e) * g1;
            }
         }
      for (int c = 0; c < 2; ++c) { Y(dx, dy, c, e) = y[c]; }
   });
}

void Dofs3D(int ne, int D1D, int Q1D, const Array<real_t>& b,
            const Array<real_t>& g, const Vector& dq, Vector& ye)
{
   const int nq = Q1D * Q1D * Q1D, nd = D1D * D1D * D1D;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto Dq = Reshape(dq.Read(), 3, 3, nq, ne);
   auto Y = Reshape(ye.Write(), D1D, D1D, D1D, 3, ne);
   mfem::forall(ne * nd, [ = ] MFEM_HOST_DEVICE(int idx)
   {
      const int e = idx / nd;
      const int d = idx - e * nd;
      const int dx = d % D1D, dy = (d / D1D) % D1D, dz = d / (D1D * D1D);
      double y[3] = {0.0, 0.0, 0.0};
      for (int qz = 0; qz < Q1D; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + Q1D * (qy + Q1D * qz);
               const double g0 = G(qx, dx) * B(qy, dy) * B(qz, dz);
               const double g1 = B(qx, dx) * G(qy, dy) * B(qz, dz);
               const double g2 = B(qx, dx) * B(qy, dy) * G(qz, dz);
               for (int c = 0; c < 3; ++c)
               {
                  y[c] += Dq(0, c, q, e) * g0 + Dq(1, c, q, e) * g1 +
                          Dq(2, c, q, e) * g2;
               }
            }
      for (int c = 0; c < 3; ++c) { Y(dx, dy, dz, c, e) = y[c]; }
   });
}

// Statistics, one thread per element (diagnostics only, so the element-
// serial form is fine): out(0..3, e) = int nu_t, max nu_t, int 2 nu_t S:S,
// int 1 over the element.
void Stats(int dim, int ne, int D1D, int Q1D, SubgridModel model,
           double constant, const Vector& cd2, const Array<real_t>& b,
           const Array<real_t>& g, const Array<real_t>& w, const Vector& jac,
           const Vector& xe, Vector& out)
{
   const int nq = (dim == 3) ? Q1D * Q1D * Q1D : Q1D * Q1D;
   const int Q1Dz = (dim == 3) ? Q1D : 1, D1Dz = (dim == 3) ? D1D : 1;
   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto G = Reshape(g.Read(), Q1D, D1D);
   const auto W = w.Read();
   const auto J = Reshape(jac.Read(), nq, dim, dim, ne);
   const auto X = Reshape(xe.Read(), D1D, D1D, D1Dz, dim, ne);
   const auto CD2 = cd2.Read();
   auto Y = Reshape(out.Write(), 4, ne);
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e)
   {
      double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
      for (int qz = 0; qz < Q1Dz; ++qz)
         for (int qy = 0; qy < Q1D; ++qy)
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const int q = qx + Q1D * (qy + Q1D * qz);
               double gr[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
               for (int dz = 0; dz < D1Dz; ++dz)
                  for (int dy = 0; dy < D1D; ++dy)
                     for (int dx = 0; dx < D1D; ++dx)
                     {
                        const double bz = (dim == 3) ? B(qz, dz) : 1.0;
                        const double gz = (dim == 3) ? G(qz, dz) : 0.0;
                        const double g0 = G(qx, dx) * B(qy, dy) * bz;
                        const double g1 = B(qx, dx) * G(qy, dy) * bz;
                        const double g2 = B(qx, dx) * B(qy, dy) * gz;
                        for (int c = 0; c < dim; ++c)
                        {
                           const double v = X(dx, dy, dz, c, e);
                           gr[c][0] += g0 * v;
                           gr[c][1] += g1 * v;
                           gr[c][2] += g2 * v;
                        }
                     }
               double iv[3][3], det;
               if (dim == 3)
               {
                  double j[3][3];
                  for (int r = 0; r < 3; ++r)
                     for (int c = 0; c < 3; ++c) { j[r][c] = J(q, r, c, e); }
                  det = Inverse3(j, iv);
               }
               else
               {
                  det = Inverse2(J(q, 0, 0, e), J(q, 0, 1, e), J(q, 1, 0, e),
                                 J(q, 1, 1, e), iv);
               }
               double gu[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
               for (int c = 0; c < dim; ++c)
                  for (int i = 0; i < dim; ++i)
                  {
                     gu[c][i] = gr[c][0] * iv[0][i] + gr[c][1] * iv[1][i] +
                                gr[c][2] * iv[2][i];
                  }
               const double wdet = W[q] * (det < 0.0 ? -det : det);
               double D[3][3], diss;
               const double nu_t = PointStress(model, CD2[e], constant, gu, iv,
                                               wdet, D, diss);
               s0 += wdet * nu_t;
               s1 = nu_t > s1 ? nu_t : s1;
               s2 += wdet * diss;
               s3 += wdet;
            }
      Y(0, e) = s0;
      Y(1, e) = s1;
      Y(2, e) = s2;
      Y(3, e) = s3;
   });
}

// Column k of out(4, ne) into col (device).
void Column(int ne, int k, const Vector& out, Vector& col)
{
   col.SetSize(ne);
   col.UseDevice(true);
   const auto Y = Reshape(out.Read(), 4, ne);
   auto C = col.Write();
   mfem::forall(ne, [ = ] MFEM_HOST_DEVICE(int e) { C[e] = Y(k, e); });
}

} // namespace sgs

SubgridStress::SubgridStress(MixedSpaces& spaces, const RuleBook& rules,
                             const SubgridOptions& opts)
   : spaces_(spaces), rules_(rules), opts_(opts)
{
   MFEM_VERIFY(opts_.Enabled(), "subgrid_stress: no model selected");
   MFEM_VERIFY(opts_.filter_width > 0.0,
               "subgrid_stress: the filter width must be > 0");
   MFEM_VERIFY(opts_.Coefficient() >= 0.0,
               "subgrid_stress: the model coefficient must be >= 0");
   const int dim = spaces_.Dim();
   MFEM_VERIFY(dim == 2 || dim == 3, "subgrid_stress: 2D or 3D only");
   const Geometry::Type geom = (dim == 3) ? Geometry::CUBE : Geometry::SQUARE;
   // Convection's dealiasing rule: nu_t is not a polynomial in grad u, and
   // one rule for every nonlinear term keeps one set of geometric factors.
   rule_ = &rules_.Get(geom, Convection::DealiasedOrder(spaces_.OrderU()));
   ParFiniteElementSpace& fes = spaces_.Velocity();
   restrict_ = fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   MFEM_VERIFY(restrict_ || fes.GetNE() == 0,
               "subgrid_stress: no element restriction");
   // Delta_K = filter_width * h_K / k_u: the node spacing of the element,
   // the resolution a high-order element actually has.
   ParMesh& mesh = *fes.GetParMesh();
   MeshSizeCoefficient h(mesh, opts_.filter_width / spaces_.OrderU());
   h.ElementValues(delta_);
   const double c = opts_.Coefficient();
   cdelta2_.SetSize(mesh.GetNE());
   double* cd = cdelta2_.HostWrite();
   const double* d = delta_.HostRead();
   for (int e = 0; e < mesh.GetNE(); ++e) { cd[e] = (c * d[e]) * (c * d[e]); }
   cdelta2_.UseDevice(true);
}

double SubgridStress::FilterWidth(int e) const
{
   MFEM_VERIFY(e >= 0 && e < delta_.Size(), "subgrid_stress: element " << e
               << " is not local");
   return delta_.HostRead()[e];
}

void SubgridStress::LocalMult(const Vector& u, Vector& yl) const
{
   ParFiniteElementSpace& fes = spaces_.Velocity();
   ParMesh& mesh = *fes.GetParMesh();
   const int dim = spaces_.Dim();
   const int ne = fes.GetNE();
   const FiniteElement& el = *fes.GetTypicalFE();
   const DofToQuad& maps = el.GetDofToQuad(*rule_, DofToQuad::TENSOR);
   const GeometricFactors* geom =
      mesh.GetGeometricFactors(*rule_, GeometricFactors::JACOBIANS);
   const int D1D = maps.ndof, Q1D = maps.nqpt;
   const int nq = rule_->GetNPoints();
   // Scratch per apply, freed on return (no per-step state; see the header).
   Vector ul(fes.GetVSize());
   ul.UseDevice(true);
   const Operator* P = fes.GetProlongationMatrix();
   if (P) { P->Mult(u, ul); }
   else { ul = u; }
   Vector xe(restrict_->Height());
   xe.UseDevice(true);
   restrict_->Mult(ul, xe);
   Vector dq(dim * dim * nq * ne);
   dq.UseDevice(true);
   Vector ye(restrict_->Height());
   ye.UseDevice(true);
   const Array<real_t>& w = rule_->GetWeights();
   const SubgridModel model = opts_.model;
   const double constant = opts_.Coefficient();
   if (dim == 2)
   {
      sgs::Points2D(ne, D1D, Q1D, model, constant, cdelta2_, maps.B, maps.G, w,
                    geom->J, xe, dq);
      sgs::Dofs2D(ne, D1D, Q1D, maps.B, maps.G, dq, ye);
   }
   else
   {
      sgs::Points3D(ne, D1D, Q1D, model, constant, cdelta2_, maps.B, maps.G, w,
                    geom->J, xe, dq);
      sgs::Dofs3D(ne, D1D, Q1D, maps.B, maps.G, dq, ye);
   }
   restrict_->MultTranspose(ye, yl);
}

void SubgridStress::Mult(const Vector& u, Vector& y) const
{
   INCNS_PROFILE("subgrid::apply");
   ParFiniteElementSpace& fes = spaces_.Velocity();
   Vector yl(fes.GetVSize());
   yl.UseDevice(true);
   yl = 0.0;
   if (fes.GetNE() > 0) { LocalMult(u, yl); }
   y.SetSize(fes.GetTrueVSize());
   y.UseDevice(true);
   const Operator* P = fes.GetProlongationMatrix();
   if (P) { P->MultTranspose(yl, y); }
   else { y = yl; }
}

SubgridStress::Values SubgridStress::Statistics(const Vector& u) const
{
   INCNS_PROFILE("subgrid::statistics");
   ParFiniteElementSpace& fes = spaces_.Velocity();
   ParMesh& mesh = *fes.GetParMesh();
   const int dim = spaces_.Dim();
   const int ne = fes.GetNE();
   double sums[3] = {0.0, 0.0, 0.0}; // int nu_t, int 2 nu_t S:S, int 1
   double max_nu_t = 0.0;
   if (ne > 0)
   {
      const FiniteElement& el = *fes.GetTypicalFE();
      const DofToQuad& maps = el.GetDofToQuad(*rule_, DofToQuad::TENSOR);
      const GeometricFactors* geom =
         mesh.GetGeometricFactors(*rule_, GeometricFactors::JACOBIANS);
      Vector ul(fes.GetVSize());
      ul.UseDevice(true);
      const Operator* P = fes.GetProlongationMatrix();
      if (P) { P->Mult(u, ul); }
      else { ul = u; }
      Vector xe(restrict_->Height());
      xe.UseDevice(true);
      restrict_->Mult(ul, xe);
      Vector out(4 * ne);
      out.UseDevice(true);
      sgs::Stats(dim, ne, maps.ndof, maps.nqpt, opts_.model, opts_.Coefficient(),
                 cdelta2_, maps.B, maps.G, rule_->GetWeights(), geom->J, xe, out);
      Vector col;
      sgs::Column(ne, 0, out, col);
      sums[0] = col.Sum();
      sgs::Column(ne, 1, out, col);
      max_nu_t = col.Max();
      sgs::Column(ne, 2, out, col);
      sums[1] = col.Sum();
      sgs::Column(ne, 3, out, col);
      sums[2] = col.Sum();
   }
   MPI_Comm comm = fes.GetComm();
   MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(MPI_IN_PLACE, &max_nu_t, 1, MPI_DOUBLE, MPI_MAX, comm);
   Values v;
   v.nu_t_max = max_nu_t;
   v.volume = sums[2];
   v.nu_t_mean = (sums[2] > 0.0) ? sums[0] / sums[2] : 0.0;
   v.dissipation = sums[1];
   return v;
}

} // namespace incns
