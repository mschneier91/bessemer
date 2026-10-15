#include "time/oifs.hpp"

#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace incns
{

using namespace mfem;

OifsAdvector::OifsAdvector(MixedSpaces& spaces, const RuleBook& rules,
                           BoundaryConditions& bc, double sub_cfl,
                           Operator& mass, const Vector& mass_diag)
   : spaces_(spaces), rules_(rules), bc_(bc), sub_cfl_(sub_cfl), mass_(mass)
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

   SetupMass(mass_diag);
   cfl_ = std::make_unique<ConvectiveCfl>(V, rules_);
   if (dim_ == 3)
   {
      const IntegrationRule& ir = rules_.Get(Geometry::CUBE, 3 * spaces_.OrderU());
      if (OifsConvection::Supported(V, ir))
      {
         combo_ = std::make_unique<OifsConvection>(V, ir);
      }
   }
   bd_ = std::make_unique<ParGridFunction>(&V);
   bd_true_.SetSize(V.GetTrueVSize());
   bd_vals_.UseDevice(true);
   bd_static_.UseDevice(true);
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
   INCNS_PROFILE("oifs::rhs");
   const std::vector<double> w = LagrangeWeights(s, times_ext_);
   if (combo_)
   {
      // dphi = -sum_l w_l C(u_l) phi, every component, one kernel.
      std::vector<double> mw(w.size());
      for (std::size_t l = 0; l < w.size(); ++l) { mw[l] = -w[l]; }
      combo_->Mult(mw, phi, dphi);
   }
   else
   {
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
         dphi_c.SyncAliasMemory(dphi);
      }
   }
   // The FULL inverse, imposed rows included, then the imposed rates
   // dropped (SetDirichlet sets those values). The BDF step's right-hand
   // side M phi then reads M^{-1} N(u) on every row to first order -- the
   // boundary rows get it from OifsBoundaryValues -- which is IMEX's N(u).
   // Holding the imposed increments at zero inside a constrained solve
   // instead couples their (different) values into the interior rows through
   // M's off-diagonals: measured 3.6e-4 from IMEX with the consistent mass
   // (nse_mms_test OifsMatchesImexWithEitherMass2D). Identical for a
   // diagonal M.
   MassInverse(dphi, dphi);
   dphi.SetSubVector(imposed_, 0.0);
}

void OifsAdvector::SetupMass(const Vector& mass_diag)
{
   // The BDF step's mass M, inverted exactly (see the class comment). Whether
   // M is diagonal is measured here, never assumed from the options.
   ParFiniteElementSpace& V = spaces_.Velocity();
   const int n = V.GetTrueVSize();
   MFEM_VERIFY(mass_.Height() == n && mass_.Width() == n &&
               mass_diag.Size() == n, "oifs: the mass must act on the "
               "velocity true dofs");
   mass_diag_.SetSize(n);
   mass_diag_.UseDevice(true);
   mass_diag_ = mass_diag;
   inv_mass_diag_.SetSize(n);
   inv_mass_diag_.UseDevice(true);
   {
      const double* d = mass_diag_.HostRead(); // setup only
      double* id = inv_mass_diag_.HostWrite();
      for (int i = 0; i < n; ++i)
      {
         MFEM_VERIFY(d[i] > 0.0, "oifs: the mass diagonal must be positive");
         id[i] = 1.0 / d[i];
      }
   }
   mass_rhs_.SetSize(n);
   mass_rhs_.UseDevice(true);

   const MPI_Comm comm = V.GetComm();
   auto norm = [comm](const Vector & v)
   {
      return std::sqrt(InnerProduct(comm, v, v));
   };
   // A deterministic test vector without zeros.
   Vector z(n), mz(n), r(n);
   z.UseDevice(true);
   mz.UseDevice(true);
   r.UseDevice(true);
   {
      double* zh = z.HostWrite();
      const int rank = V.GetMyRank();
      for (int i = 0; i < n; ++i)
      {
         zh[i] = 1.0 + 0.5 * std::sin(0.7 * (i + 1) + 1.3 * rank);
      }
   }
   mass_.Mult(z, mz);
   r = z;
   r *= mass_diag_;
   r -= mz;
   mass_diagonal_ = norm(r) <= 1e-13 * norm(mz);

   if (!mass_diagonal_)
   {
      // SetOperator before SetPreconditioner: MFEM forwards SetOperator to
      // the PC, which would rebuild the Jacobi diagonal.
      mass_jacobi_ = std::make_unique<OperatorJacobiSmoother>(mass_diag_,
                     no_dofs_);
      mass_cg_ = std::make_unique<CGSolver>(comm);
      mass_cg_->SetOperator(mass_);
      mass_cg_->SetPreconditioner(*mass_jacobi_);
      mass_cg_->SetRelTol(1e-13);
      mass_cg_->SetAbsTol(0.0);
      mass_cg_->SetMaxIter(500);
      mass_cg_->SetPrintLevel(-1);
      mass_cg_->iterative_mode = false;
   }

   // The check (human 2026-10-10: error out if the substeps' mass and the
   // BDF step's are not the same): M^{-1} (M z) must give z back.
   MassInverse(mz, r);
   r -= z;
   const double mismatch = norm(r) / norm(z);
   MFEM_VERIFY(mismatch <= (mass_diagonal_ ? 1e-12 : 1e-9),
               "oifs: the substeps' mass inverse does not reproduce the BDF "
               "step's mass (relative mismatch " << mismatch << "); OIFS "
               "would converge to M M_sub^-1 N(u) instead of N(u)");
}

void OifsAdvector::MassInverse(const Vector& g, Vector& nodal)
{
   if (mass_diagonal_)
   {
      if (&nodal != &g) { nodal = g; }
      nodal *= inv_mass_diag_; // element-wise: M^{-1}
      return;
   }
   mass_rhs_ = g; // g and nodal may alias
   mass_cg_->Mult(mass_rhs_, nodal);
   MFEM_VERIFY(mass_cg_->GetConverged(), "oifs: the mass solve did not "
               "converge (" << mass_cg_->GetNumIterations() << " iterations)");
   mass_iterations_ += mass_cg_->GetNumIterations();
}

void OifsAdvector::CacheBoundaryNodes()
{
   ParFiniteElementSpace& V = spaces_.Velocity();
   ParMesh& mesh = *V.GetParMesh();
   bnode_vdofs_.clear();
   bnode_normals_.clear();
   bnode_be_.clear();
   bnode_k_.clear();
   bnode_nodal_ = true;
   Array<int> vdofs;
   Vector nor(dim_), x(dim_), c(dim_);
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      const FiniteElement* fe = V.GetBE(be);
      if (!dynamic_cast<const NodalFiniteElement*>(fe)) { bnode_nodal_ = false; }
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
         for (int d = 0; d < dim_; ++d)
         {
            bnode_vdofs_.push_back(vdofs[k + d * nd]);
            bnode_normals_.push_back(nor(d));
         }
         bnode_be_.push_back(be);
         bnode_k_.push_back(k);
      }
   }
   bnode_seq_ = V.GetSequence();
}

void OifsAdvector::ClassifyBoundary(const ParGridFunction& w)
{
   ParFiniteElementSpace& V = spaces_.Velocity();
   if (bnode_seq_ != V.GetSequence()) { CacheBoundaryNodes(); }
   if (!flag_) { flag_ = std::make_unique<ParGridFunction>(&V); }
   ParGridFunction& flag = *flag_;
   flag = 0.0; // device-aware; take the host pointer only afterwards
   double* fl = flag.HostReadWrite();
   const double* wh = w.HostRead();
   ParMesh& mesh = *V.GetParMesh();
   const int n_nodes = static_cast<int>(bnode_be_.size());
   // The wind at every boundary node: a nodal element's value at its node is
   // that node's dofs (exactly zero on a no-slip wall); otherwise evaluated
   // through the element.
   bnode_w_.resize(static_cast<std::size_t>(n_nodes) * dim_);
   double w_max = 0.0;
   {
      Vector wv(dim_);
      ElementTransformation* T = nullptr;
      for (int i = 0; i < n_nodes; ++i)
      {
         const int* v = &bnode_vdofs_[i * dim_];
         double* wi = &bnode_w_[i * dim_];
         if (bnode_nodal_)
         {
            for (int d = 0; d < dim_; ++d)
            {
               wi[d] = (v[d] >= 0) ? wh[v[d]] : -wh[-1 - v[d]];
            }
         }
         else
         {
            const int be = bnode_be_[i];
            if (T == nullptr || be != bnode_be_[i - 1])
            {
               T = mesh.GetBdrElementTransformation(be);
            }
            const IntegrationPoint& ip =
               V.GetBE(be)->GetNodes().IntPoint(bnode_k_[i]);
            T->SetIntPoint(&ip);
            w.GetVectorValue(*T, ip, wv);
            for (int d = 0; d < dim_; ++d) { wi[d] = wv(d); }
         }
         double ww = 0.0;
         for (int d = 0; d < dim_; ++d) { ww += wi[d] * wi[d]; }
         w_max = std::max(w_max, std::sqrt(ww));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &w_max, 1, MPI_DOUBLE, MPI_MAX, V.GetComm());
   // Free (outflow) only where the outward normal velocity is more than
   // roundoff of the boundary's largest wind; everything else, a no-slip wall
   // included, is imposed. (Until 2026-10-15 the test was relative to the
   // node's own |w|, with w evaluated through the element: on a wall that w
   // is roundoff, and its sign decided whether the node was imposed.)
   for (int i = 0; i < n_nodes; ++i)
   {
      const int* v = &bnode_vdofs_[i * dim_];
      const double* wi = &bnode_w_[i * dim_];
      const double* nor = &bnode_normals_[i * dim_];
      double wn = 0.0, nn = 0.0;
      for (int d = 0; d < dim_; ++d)
      {
         wn += wi[d] * nor[d];
         nn += nor[d] * nor[d];
      }
      if (wn <= 1e-10 * w_max * std::sqrt(nn))
      {
         for (int d = 0; d < dim_; ++d)
         {
            fl[v[d] >= 0 ? v[d] : -1 - v[d]] = 1.0;
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
   INCNS_PROFILE("oifs::dirichlet");
   if (bd_static_valid_)
   {
      // The data does not change over this window: no projection (a host
      // loop over the boundary and, on a GPU, a round trip of the whole
      // velocity, at every RK stage). bd_static_valid_ is the same on every
      // rank (StaticDirichlet reduces it).
      if (imposed_.Size() > 0 && bd_zero_) { phi.SetSubVector(imposed_, 0.0); }
      else if (imposed_.Size() > 0)
      {
         bd_vals_ = bd_static_;
         bd_vals_ *= weight;
         phi.SetSubVector(imposed_, bd_vals_);
      }
      return;
   }
   // Every rank takes the same path (the projection may communicate).
   DirichletValues(s, imposed_, bd_vals_);
   bd_vals_ *= weight;
   phi.SetSubVector(imposed_, bd_vals_);
}

void OifsAdvector::DirichletValues(double s, const Array<int>& dofs,
                                   Vector& vals)
{
   bc_.SetTime(s);
   *bd_ = 0.0;
   bc_.ProjectDirichlet(*bd_);
   bd_->GetTrueDofs(bd_true_);
   bd_true_.GetSubVector(dofs, vals);
}

bool OifsAdvector::StaticDirichlet(const Array<int>& dofs, double s0,
                                   double s1, Vector& vals)
{
   // Three times, the middle one irregular (a periodic signal is unlikely to
   // repeat there); equal values at all three mean time-independent data.
   Vector mid, end;
   DirichletValues(s0, dofs, vals);
   DirichletValues(s0 + 0.37 * (s1 - s0), dofs, mid);
   DirichletValues(s1, dofs, end);
   double diff = 0.0;
   if (vals.Size() > 0)
   {
      mid -= vals;
      end -= vals;
      diff = std::max(mid.Normlinf(), end.Normlinf());
   }
   MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX,
                 spaces_.Velocity().GetComm());
   return diff == 0.0;
}

void OifsAdvector::Advect(const std::vector<double>& c,
                          const std::deque<Vector>& hist,
                          const std::deque<double>& times, double t_new,
                          int ext, Vector& phi)
{
   // The advecting velocity: the newest `ext` history levels (borrowed).
   const int n = std::max(1, std::min<int>(ext, static_cast<int>(hist.size())));
   std::vector<const Vector*> wind(n);
   for (int l = 0; l < n; ++l) { wind[l] = &hist[l]; }
   const std::vector<double> wind_times(times.begin(), times.begin() + n);
   AdvectImpl(wind, wind_times, c, hist, times, t_new, phi);
}

void OifsAdvector::AdvectWith(const std::deque<Vector>& wind,
                              const std::vector<double>& wind_times,
                              const std::vector<double>& c,
                              const std::deque<Vector>& hist,
                              const std::deque<double>& times, double t_new,
                              Vector& phi)
{
   std::vector<const Vector*> w(wind.size());
   for (std::size_t l = 0; l < wind.size(); ++l) { w[l] = &wind[l]; }
   AdvectImpl(w, wind_times, c, hist, times, t_new, phi);
}

void OifsAdvector::AdvectImpl(const std::vector<const Vector*>& wind,
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
      vel_[l]->SetFromTrueDofs(*wind[l]);
   }
   {
      INCNS_PROFILE("oifs::rate");
      std::vector<const ParGridFunction*> us(n_ext_);
      for (int l = 0; l < n_ext_; ++l) { us[l] = vel_[l].get(); }
      rate = cfl_->MaxRate(us); // one evaluation of the Jacobians
   }
   for (int l = 0; l < n_ext_; ++l)
   {
      INCNS_PROFILE("oifs::setup");
      if (combo_)
      {
         combo_->SetWind(l, *vel_[l]);
         continue;
      }
      // New forms at every step: re-assembling them in place instead (same
      // coefficient, new wind) changed the OIFS MMS results and crashed
      // nondeterministically (2026-10-15; cause not found).
      vel_coef_[l] = std::make_unique<VectorGridFunctionCoefficient>(vel_[l].get());
      conv_[l] = std::make_unique<ParBilinearForm>(sfes_.get());
      auto* ci = new ConvectionIntegrator(*vel_coef_[l], 1.0); // (w.grad u, v)
      ci->SetIntRule(&ir);
      conv_[l]->AddDomainIntegrator(ci);
      conv_[l]->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      conv_[l]->Assemble();
      conv_[l]->FormSystemMatrix(none, conv_op_[l]);
   }

   {
      INCNS_PROFILE("oifs::classify");
      const Array<int>& ess = bc_.EssentialTrueDofs();
      const bool constant = bc_.DirichletIsConstant(); // same on every rank
      if (constant && bd_const_seq_ != V.GetSequence())
      {
         // Constant data (no-slip walls): projected once per space.
         *bd_ = 0.0;
         bc_.ProjectDirichlet(*bd_);
         bd_->GetTrueDofs(bd_const_true_);
         Vector e;
         bd_const_true_.GetSubVector(ess, e);
         double m = (e.Size() > 0) ? e.Normlinf() : 0.0;
         MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, V.GetComm());
         bd_const_zero_ = (m == 0.0);
         bd_const_seq_ = V.GetSequence();
      }
      // Zero data and a wind that is zero on every Dirichlet dof (the history
      // satisfies the data): ClassifyBoundary would impose them all, so skip
      // its host loop.
      bool all_imposed = false;
      if (constant && bd_const_zero_)
      {
         Vector& we = ess_scratch_;
         we.UseDevice(true);
         wind[0]->GetSubVector(ess, we);
         double m = (we.Size() > 0) ? we.Normlinf() : 0.0;
         MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, V.GetComm());
         all_imposed = (m == 0.0);
      }
      if (all_imposed)
      {
         imposed_.SetSize(ess.Size());
         const int* e = ess.HostRead();
         int* d = imposed_.HostWrite();
         for (int i = 0; i < ess.Size(); ++i) { d[i] = e[i]; }
         imposed_.GetMemory().UseDevice(true);
      }
      else
      {
         ClassifyBoundary(*vel_[0]); // the newest wind decides in/outflow
      }
      // Data that does not change over the window (no-slip walls, a
      // constant inflow): the substeps reuse it instead of projecting it at
      // every stage.
      if (constant)
      {
         bd_const_true_.GetSubVector(imposed_, bd_static_);
         bd_static_valid_ = true;
      }
      else
      {
         bd_static_valid_ = StaticDirichlet(imposed_, times[k - 1], t_new,
                                            bd_static_);
      }
      bd_zero_ = false;
      if (bd_static_valid_)
      {
         double m = (bd_static_.Size() > 0) ? bd_static_.Normlinf() : 0.0;
         MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, V.GetComm());
         bd_zero_ = (m == 0.0);
      }
   }

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
