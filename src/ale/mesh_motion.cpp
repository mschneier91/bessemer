#include "ale/mesh_motion.hpp"

#include "time/multistep_coeffs.hpp"
#include "util/profiler.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
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
   x0_.HostRead(); // valid on the host too: ReferenceNodes() is read there
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
      const int k = std::min(opts_.guess_order + 1, States());
      if (k >= 2)
      {
         // The kept states extrapolated to t: deterministic across a
         // restart (Save/Load carry them).
         std::vector<double> times(times_.begin(), times_.begin() + k);
         const std::vector<double> g = LagrangeWeights(t, times);
         if (!guess_nodes_)
         {
            guess_nodes_ = std::make_unique<ParGridFunction>(disp_->ParFESpace());
            guess_model_ = std::make_unique<ParGridFunction>(&model_->Space());
         }
         guess_nodes_->HostWrite();
         *guess_nodes_ = 0.0;
         for (int j = 0; j < k; ++j) { guess_nodes_->Add(g[j], history_[j]); }
         guess_model_->HostReadWrite();
         guess_model_->ProjectGridFunction(*guess_nodes_);
         ok = model_->Solve(t, *guess_model_);
      }
      else { ok = model_->Solve(t); }
      total_iterations_ += model_->Iterations();
   }
   {
      INCNS_PROFILE("mesh_motion::nodes");
      // Interpolation at the nodes (the identity when the model's space is
      // the nodal space); element numbering is the reference copy's.
      // ProjectGridFunction runs on the host: the model's displacement (a
      // device solve) must be current there, and disp_'s host copy becomes
      // the valid one.
      const ParGridFunction& d = model_->Displacement();
      d.HostRead();
      disp_->HostReadWrite();
      disp_->ProjectGridFunction(d);
      GridFunction& nodes = *mesh_.GetNodes();
      nodes = x0_;
      nodes += *disp_;
      // Both copies current: element transformations read the host one.
      nodes.HostRead();
      mesh_.NodesUpdated();
      if (mesh_.GetNFaceNeighbors() > 0) { mesh_.ExchangeFaceNbrNodes(); }
   }
   // The new state at the front; the oldest dropped beyond the history (and
   // kept aside, so that Revert() restores every level).
   if (static_cast<int>(history_.size()) < opts_.history)
   {
      history_.emplace_back(disp_->Size());
      times_.push_back(0.0);
      has_dropped_ = false;
   }
   else
   {
      dropped_ = history_.back();
      dropped_time_ = times_.back();
      has_dropped_ = true;
   }
   std::rotate(history_.rbegin(), history_.rbegin() + 1, history_.rend());
   std::rotate(times_.rbegin(), times_.rbegin() + 1, times_.rend());
   history_[0] = *disp_;
   times_[0] = t;
   ++moves_;
   return ok;
}

void MeshMotion::Revert()
{
   INCNS_PROFILE("mesh_motion::revert");
   MFEM_VERIFY(moves_ > 0 && history_.size() > 1, "mesh_motion: Revert() "
               "needs a move to undo");
   history_.erase(history_.begin());
   times_.erase(times_.begin());
   // The state the move dropped comes back (one level only: a second
   // Revert() in a row loses it).
   if (has_dropped_)
   {
      history_.push_back(dropped_);
      times_.push_back(dropped_time_);
      has_dropped_ = false;
   }
   --moves_;
   disp_->HostReadWrite();
   static_cast<Vector&>(*disp_) = history_[0];
   GridFunction& nodes = *mesh_.GetNodes();
   nodes = x0_;
   nodes += *disp_;
   nodes.HostRead();
   mesh_.NodesUpdated();
   if (mesh_.GetNFaceNeighbors() > 0) { mesh_.ExchangeFaceNbrNodes(); }
}

void MeshMotion::Save(const std::string& path) const
{
   const std::string file = path + ".r" + std::to_string(mesh_.GetMyRank());
   std::ofstream f(file, std::ios::binary);
   MFEM_VERIFY(f, "mesh_motion: cannot write " << file);
   const std::int64_t n = static_cast<std::int64_t>(history_.size());
   const std::int64_t size = disp_->Size();
   const std::int64_t moves = moves_;
   f.write(reinterpret_cast<const char*>(&n), sizeof(n));
   f.write(reinterpret_cast<const char*>(&size), sizeof(size));
   f.write(reinterpret_cast<const char*>(&moves), sizeof(moves));
   f.write(reinterpret_cast<const char*>(times_.data()),
           static_cast<std::streamsize>(n * sizeof(double)));
   for (const Vector& h : history_)
   {
      f.write(reinterpret_cast<const char*>(h.HostRead()),
              static_cast<std::streamsize>(size * sizeof(double)));
   }
   model_->SaveState(f);
   MFEM_VERIFY(f, "mesh_motion: writing " << file << " failed");
}

void MeshMotion::Load(const std::string& path)
{
   const std::string file = path + ".r" + std::to_string(mesh_.GetMyRank());
   std::ifstream f(file, std::ios::binary);
   MFEM_VERIFY(f, "mesh_motion: cannot read " << file
               << " (missing checkpoint or wrong rank count?)");
   std::int64_t n = 0, size = 0, moves = 0;
   f.read(reinterpret_cast<char*>(&n), sizeof(n));
   f.read(reinterpret_cast<char*>(&size), sizeof(size));
   f.read(reinterpret_cast<char*>(&moves), sizeof(moves));
   MFEM_VERIFY(f && n >= 1 && n <= opts_.history && size == disp_->Size(),
               "mesh_motion: " << file << " does not match this mesh (rank "
               "count, order or history length)");
   times_.resize(n);
   f.read(reinterpret_cast<char*>(times_.data()),
          static_cast<std::streamsize>(n * sizeof(double)));
   history_.assign(n, Vector(disp_->Size()));
   for (Vector& h : history_)
   {
      f.read(reinterpret_cast<char*>(h.HostWrite()),
             static_cast<std::streamsize>(size * sizeof(double)));
   }
   model_->LoadState(f);
   MFEM_VERIFY(f, "mesh_motion: reading " << file << " failed");
   moves_ = static_cast<int>(moves);
   has_dropped_ = false;
   Finalize();
   disp_->HostWrite();
   static_cast<Vector&>(*disp_) = history_.front();
   GridFunction& nodes = *mesh_.GetNodes();
   nodes = x0_;
   nodes += *disp_;
   nodes.HostRead();
   mesh_.NodesUpdated();
   if (mesh_.GetNFaceNeighbors() > 0) { mesh_.ExchangeFaceNbrNodes(); }
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
