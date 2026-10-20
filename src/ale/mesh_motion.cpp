#include "ale/mesh_motion.hpp"

#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <limits>

namespace incns
{

using namespace mfem;

MeshMotion::MeshMotion(ParMesh& mesh, const Options& opts,
                       const MeshMotionModelFactory& factory)
   : mesh_(mesh), opts_(opts), gll_rules_(0, Quadrature1D::GaussLobatto)
{
   INCNS_PROFILE("mesh_motion::setup");
   MFEM_VERIFY(opts_.history >= 2, "mesh_motion: history must be >= 2");
   MFEM_VERIFY(factory, "mesh_motion: no model factory");
   if (!mesh_.GetNodes())
   {
      const int g = opts_.geometry_order > 0 ? opts_.geometry_order :
                    std::max(opts_.order, 1);
      mesh_.SetCurvature(g, false, -1, Ordering::byNODES);
   }
   auto* nodes = dynamic_cast<ParGridFunction*>(mesh_.GetNodes());
   MFEM_VERIFY(nodes, "mesh_motion: the mesh's nodes are not a "
               "ParGridFunction");
   order_ = opts_.order > 0 ? opts_.order :
            nodes->ParFESpace()->FEColl()->GetOrder();
   sequence_ = mesh_.GetSequence();

   // The reference: a copy with its own nodes, which never move.
   ref_ = std::make_unique<ParMesh>(mesh_, true);
   model_ = factory(*ref_, order_);
   MFEM_VERIFY(model_, "mesh_motion: the factory built no model");
   MFEM_VERIFY(model_->Space().GetVDim() == mesh_.SpaceDimension(),
               "mesh_motion: the model's space has vdim "
               << model_->Space().GetVDim() << ", the mesh's space dimension "
               "is " << mesh_.SpaceDimension());

   x0_ = *nodes;
   disp_ = std::make_unique<ParGridFunction>(nodes->ParFESpace());
   *disp_ = 0.0;
   history_.assign(1, Vector(disp_->Size()));
   history_[0] = 0.0;
   times_.assign(1, opts_.t0);
}

void MeshMotion::AddFixed(int attr)
{
   MFEM_VERIFY(!finalized_, "mesh_motion: boundaries are fixed at the first "
               "move");
   fixed_.push_back(attr);
}

void MeshMotion::AddDisplacement(int attr, VectorCoefficient& g)
{
   MFEM_VERIFY(!finalized_, "mesh_motion: boundaries are fixed at the first "
               "move");
   moving_.push_back(attr);
   model_->AddDisplacement(attr, g);
}

void MeshMotion::Finalize()
{
   if (finalized_) { return; }
   const Array<int>& attrs = mesh_.bdr_attributes;
   const int n = attrs.Size() ? attrs.Max() : 0;
   std::vector<int> count(n + 1, 0);
   for (int a : fixed_)
   {
      MFEM_VERIFY(a >= 1 && a <= n, "mesh_motion: boundary attribute " << a
                  << " is not on the mesh (1.." << n << ")");
      ++count[a];
   }
   for (int a : moving_)
   {
      MFEM_VERIFY(a >= 1 && a <= n, "mesh_motion: boundary attribute " << a
                  << " is not on the mesh (1.." << n << ")");
      ++count[a];
   }
   for (int a = 1; a <= n; ++a)
   {
      MFEM_VERIFY(count[a] <= 1, "mesh_motion: boundary attribute " << a
                  << " is given more than once");
   }
   // Fixed: the named ones and every attribute of the mesh nobody named.
   for (int a : fixed_) { model_->AddFixed(a); }
   for (int i = 0; i < attrs.Size(); ++i)
   {
      if (count[attrs[i]] == 0) { model_->AddFixed(attrs[i]); }
   }
   finalized_ = true;
}

bool MeshMotion::Move(double t)
{
   INCNS_PROFILE("mesh_motion::move");
   MFEM_VERIFY(mesh_.GetSequence() == sequence_, "mesh_motion: the mesh was "
               "refined after the motion was set up (not supported)");
   MFEM_VERIFY(t > times_.front(), "mesh_motion: Move(" << t << ") is not "
               "after the current time " << times_.front());
   Finalize();
   bool ok = false;
   {
      INCNS_PROFILE("mesh_motion::model");
      ok = model_->Solve(t);
   }
   {
      INCNS_PROFILE("mesh_motion::nodes");
      // Interpolation at the nodes (the identity when the model's space is
      // the nodal space); element numbering is the reference copy's.
      disp_->ProjectGridFunction(model_->Displacement());
      GridFunction& nodes = *mesh_.GetNodes();
      nodes = x0_;
      nodes += *disp_;
      mesh_.NodesUpdated();
      if (mesh_.GetNFaceNeighbors() > 0) { mesh_.ExchangeFaceNbrNodes(); }
   }
   // The new state at the front; the oldest dropped beyond the history.
   if (static_cast<int>(history_.size()) < opts_.history)
   {
      history_.emplace_back(disp_->Size());
      times_.push_back(0.0);
   }
   std::rotate(history_.rbegin(), history_.rbegin() + 1, history_.rend());
   std::rotate(times_.rbegin(), times_.rbegin() + 1, times_.rend());
   history_[0] = *disp_;
   times_[0] = t;
   ++moves_;
   return ok;
}

void MeshMotion::MeshVelocity(int order, Vector& w) const
{
   MFEM_VERIFY(order >= 1 && order <= 3, "mesh_motion: BDF order " << order
               << " is not 1, 2 or 3");
   MFEM_VERIFY(order + 1 <= States(), "mesh_motion: BDF order " << order
               << " needs " << order + 1 << " states, " << States()
               << " are kept");
   const std::vector<double> times(times_.begin(), times_.begin() + order + 1);
   const std::vector<double> c = BdfWeights(times);
   // x = X + d and the weights sum to zero: w = sum_j c_j d_j.
   w.SetSize(disp_->Size());
   w = 0.0;
   for (int j = 0; j <= order; ++j) { w.Add(c[j], history_[j]); }
}

void MeshMotion::JacobianRange(double& min, double& max) const
{
   INCNS_PROFILE("mesh_motion::jacobian_range");
   const int k = mesh_.GetNodes()->FESpace()->FEColl()->GetOrder();
   // 2k - 1: the Gauss-Lobatto rule with k + 1 points per direction.
   const int rule_order = std::max(2 * k - 1, 1);
   double mm[2] = {std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()
                  };
   for (int e = 0; e < mesh_.GetNE(); ++e)
   {
      const IntegrationRule& ir =
         gll_rules_.Get(mesh_.GetElementBaseGeometry(e), rule_order);
      ElementTransformation* T = mesh_.GetElementTransformation(e);
      ElementTransformation* T0 = ref_->GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint& ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         T0->SetIntPoint(&ip);
         const double r = T->Weight() / T0->Weight();
         mm[0] = std::min(mm[0], r);
         mm[1] = std::min(mm[1], -r);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_DOUBLE, MPI_MIN, mesh_.GetComm());
   min = mm[0];
   max = -mm[1];
}

} // namespace incns
