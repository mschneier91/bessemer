#include "time/oifs.hpp"

#include "time/multistep_coeffs.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

OifsAdvector::OifsAdvector(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc, double sub_cfl)
   : spaces_(spaces), rules_(rules), bc_(bc), sub_cfl_(sub_cfl)
{
   MFEM_VERIFY(sub_cfl > 0.0, "oifs: the substep CFL number must be positive");
   ParFiniteElementSpace& V = spaces_.Velocity();
   MFEM_VERIFY(V.GetOrdering() == Ordering::byNODES, "oifs: the velocity "
               "space must be ordered byNODES (component blocks)");
   dim_ = V.GetMesh()->Dimension();
   sfes_ = std::make_unique<ParFiniteElementSpace>(V.GetParMesh(), V.FEColl());
   n_s_ = sfes_->GetTrueVSize();
   MFEM_VERIFY(dim_ * n_s_ == V.GetTrueVSize(), "oifs: the scalar companion "
               "space does not match the velocity space's components");

   // Row-sum-lumped GLL mass: the collocated mass applied to ones. On a
   // conforming mesh that is the (diagonal) GLL mass itself; on a
   // nonconforming one P^T D P 1 = P^T D 1, the row sums (P 1 = 1).
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   {
      ParBilinearForm m(sfes_.get());
      auto* mi = new MassIntegrator();
      mi->SetIntRule(&rules_.CollocatedMass(geom, spaces_.OrderU()));
      m.AddDomainIntegrator(mi);
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m.Assemble();
      OperatorPtr M;
      Array<int> none;
      m.FormSystemMatrix(none, M);
      Vector ones(n_s_);
      ones.UseDevice(true);
      ones = 1.0;
      inv_mass_.SetSize(n_s_);
      inv_mass_.UseDevice(true);
      M->Mult(ones, inv_mass_);
      double* d = inv_mass_.HostReadWrite(); // setup only
      for (int i = 0; i < n_s_; ++i) { d[i] = 1.0 / d[i]; }
   }
   cfl_ = std::make_unique<ConvectiveCfl>(V, rules_);
   bd_ = std::make_unique<ParGridFunction>(&V);
   bd_true_.SetSize(V.GetTrueVSize());
   tmp_s_.SetSize(n_s_);
   tmp_s_.UseDevice(true);
   for (Vector* v : {&k1_, &k2_, &k3_, &k4_, &y_})
   {
      v->SetSize(V.GetTrueVSize());
      v->UseDevice(true);
   }
}

void OifsAdvector::Rhs(double s, const Vector& phi, Vector& dphi)
{
   const std::vector<double> w = LagrangeWeights(s, times_ext_);
   dphi = 0.0;
   for (int c = 0; c < dim_; ++c)
   {
      Vector phi_c, dphi_c;
      phi_c.MakeRef(const_cast<Vector&>(phi), c * n_s_, n_s_);
      dphi_c.MakeRef(dphi, c * n_s_, n_s_);
      // MFEM's alias protocol (device memory): the aliases start from their
      // bases' valid location, and the base learns of the alias's writes.
      phi_c.SyncMemory(phi);
      dphi_c.SyncMemory(dphi);
      for (int l = 0; l < n_ext_; ++l)
      {
         conv_op_[l]->Mult(phi_c, tmp_s_);
         dphi_c.Add(-w[l], tmp_s_);
      }
      dphi_c *= inv_mass_; // element-wise: M_L^{-1}
      dphi_c.SyncAliasMemory(dphi);
   }
   dphi.SetSubVector(imposed_, 0.0);
}

void OifsAdvector::LumpedNodal(const Vector& g, Vector& nodal) const
{
   if (&nodal != &g) { nodal = g; }
   for (int c = 0; c < dim_; ++c)
   {
      Vector part;
      part.MakeRef(nodal, c * n_s_, n_s_);
      part.SyncMemory(nodal);
      part *= inv_mass_;
      part.SyncAliasMemory(nodal);
   }
}

void OifsAdvector::ClassifyBoundary(const ParGridFunction& w)
{
   ParFiniteElementSpace& V = spaces_.Velocity();
   ParMesh& mesh = *V.GetParMesh();
   if (!flag_) { flag_ = std::make_unique<ParGridFunction>(&V); }
   ParGridFunction& flag = *flag_;
   flag = 0.0; // device-aware; take the host pointer only afterwards
   double* fl = flag.HostReadWrite();
   w.HostRead();
   Array<int> vdofs;
   Vector wv(dim_), nor(dim_), x(dim_), c(dim_);
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      const FiniteElement* fe = V.GetBE(be);
      const IntegrationRule& nodes = fe->GetNodes();
      const int nd = fe->GetDof();
      V.GetBdrElementVDofs(be, vdofs);
      ElementTransformation* T = mesh.GetBdrElementTransformation(be);
      int el = -1, info = 0;
      mesh.GetBdrElementAdjacentElement(be, el, info);
      mesh.GetElementCenter(el, c);
      for (int k = 0; k < nd; ++k)
      {
         const IntegrationPoint& ip = nodes.IntPoint(k);
         T->SetIntPoint(&ip);
         T->Transform(ip, x);
         CalcOrtho(T->Jacobian(), nor);
         x -= c;
         if (nor * x < 0.0) { nor.Neg(); } // outward, whatever the orientation
         w.GetVectorValue(*T, ip, wv);
         if (wv * nor <= 1e-10 * wv.Norml2() * nor.Norml2())
         {
            // Inflow or tangential (walls: w = 0): impose here.
            for (int d = 0; d < dim_; ++d)
            {
               const int v = vdofs[k + d * nd];
               fl[v >= 0 ? v : -1 - v] = 1.0;
            }
         }
      }
   }
   Vector ft(V.GetTrueVSize());
   flag.ParallelAssemble(ft); // shared boundary dofs: any rank's inflow counts
   const double* f = ft.HostRead();
   std::vector<int> list;
   for (int i : bc_.EssentialTrueDofs())
   {
      if (f[i] > 0.0) { list.push_back(i); }
   }
   // A device-usable list: Get/SetSubVector pick host or device from the
   // index array, and a host one would drag phi back to the host at every
   // RK stage. Rebuilt through an explicit host write (never Append onto a
   // possibly device-valid buffer).
   imposed_.SetSize(static_cast<int>(list.size()));
   int* d = imposed_.HostWrite();
   for (std::size_t k = 0; k < list.size(); ++k) { d[k] = list[k]; }
   imposed_.GetMemory().UseDevice(true);
}

void OifsAdvector::SetDirichlet(double s, double weight, Vector& phi)
{
   // Every rank takes the same path (the projection may communicate).
   const Array<int>& ess = imposed_;
   bc_.SetTime(s);
   *bd_ = 0.0;
   bc_.ProjectDirichlet(*bd_);
   bd_->GetTrueDofs(bd_true_);
   bd_true_.GetSubVector(ess, bd_vals_);
   bd_vals_ *= weight;
   phi.SetSubVector(ess, bd_vals_);
}

void OifsAdvector::Advect(const std::vector<double>& c,
                          const std::deque<Vector>& hist,
                          const std::deque<double>& times, double t_new,
                          int ext, Vector& phi)
{
   // The advecting velocity: the newest `ext` history levels.
   const int n = std::max(1, std::min<int>(ext, static_cast<int>(hist.size())));
   std::deque<Vector> wind;
   for (int l = 0; l < n; ++l) { wind.emplace_back(hist[l]); }
   const std::vector<double> wind_times(times.begin(), times.begin() + n);
   AdvectWith(wind, wind_times, c, hist, times, t_new, phi);
}

void OifsAdvector::AdvectWith(const std::deque<Vector>& wind,
                              const std::vector<double>& wind_times,
                              const std::vector<double>& c,
                              const std::deque<Vector>& hist,
                              const std::deque<double>& times, double t_new,
                              Vector& phi)
{
   const int k = static_cast<int>(c.size()) - 1;
   MFEM_VERIFY(k >= 1 && static_cast<int>(hist.size()) >= k,
               "oifs: need at least as many fields as weights");
   MFEM_VERIFY(!wind.empty() && wind.size() == wind_times.size(),
               "oifs: wind and wind_times must match and be non-empty");
   ParFiniteElementSpace& V = spaces_.Velocity();
   const double t_bc = t_new;

   n_ext_ = static_cast<int>(wind.size());
   times_ext_ = wind_times;
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   const IntegrationRule& ir = rules_.Get(geom, 3 * spaces_.OrderU());
   double rate = 0.0;
   vel_.resize(n_ext_);
   vel_coef_.resize(n_ext_);
   conv_.resize(n_ext_);
   conv_op_.resize(n_ext_);
   Array<int> none;
   for (int l = 0; l < n_ext_; ++l)
   {
      if (!vel_[l]) { vel_[l] = std::make_unique<ParGridFunction>(&V); }
      vel_[l]->SetFromTrueDofs(wind[l]);
      rate = std::max(rate, cfl_->Rate(*vel_[l]));
      vel_coef_[l] = std::make_unique<VectorGridFunctionCoefficient>(vel_[l].get());
      conv_[l] = std::make_unique<ParBilinearForm>(sfes_.get());
      auto* ci = new ConvectionIntegrator(*vel_coef_[l], 1.0); // (w.grad u, v)
      ci->SetIntRule(&ir);
      conv_[l]->AddDomainIntegrator(ci);
      conv_[l]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      conv_[l]->Assemble();
      conv_[l]->FormSystemMatrix(none, conv_op_[l]);
   }

   ClassifyBoundary(*vel_[0]); // the newest wind decides in/outflow

   // One combined field, oldest level first.
   phi.SetSize(V.GetTrueVSize());
   phi.UseDevice(true);
   phi.Set(c[k], hist[k - 1]);
   double weight = c[k];
   double s = times[k - 1];
   SetDirichlet(s, weight, phi);
   last_substeps_ = 0;
   for (int j = k; j >= 1; --j)
   {
      const double s_end = (j > 1) ? times[j - 2] : t_new;
      const double span = s_end - s;
      const int n = std::max(1,
                             static_cast<int>(std::ceil(span * rate / sub_cfl_ - 1e-12)));
      const double h = span / n;
      for (int q = 0; q < n; ++q)
      {
         const double s0 = s + q * h;
         Rhs(s0, phi, k1_);
         add(phi, 0.5 * h, k1_, y_);
         SetDirichlet(s0 + 0.5 * h, weight, y_);
         Rhs(s0 + 0.5 * h, y_, k2_);
         add(phi, 0.5 * h, k2_, y_);
         SetDirichlet(s0 + 0.5 * h, weight, y_);
         Rhs(s0 + 0.5 * h, y_, k3_);
         add(phi, h, k3_, y_);
         SetDirichlet(s0 + h, weight, y_);
         Rhs(s0 + h, y_, k4_);
         phi.Add(h / 6.0, k1_);
         phi.Add(h / 3.0, k2_);
         phi.Add(h / 3.0, k3_);
         phi.Add(h / 6.0, k4_);
         SetDirichlet(s0 + h, weight, phi);
      }
      last_substeps_ += n;
      s = s_end;
      if (j > 1)
      {
         // The next level joins at its own time (it satisfies the boundary
         // data there, so the boundary values stay weight * u_D).
         phi.Add(c[j - 1], hist[j - 2]);
         weight += c[j - 1];
      }
   }
   bc_.SetTime(t_bc);
}

} // namespace incns
