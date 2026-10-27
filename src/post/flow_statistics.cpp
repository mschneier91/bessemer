#include "post/flow_statistics.hpp"

#include "post/checkpoint.hpp"
#include "post/running_moments.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace incns
{

using namespace mfem;

namespace
{

const char* const kMetaFile = "flow_statistics.txt";

// Component a of node i of a field on @p fes is at a * sc + i * sn.
void Strides(const ParFiniteElementSpace& fes, int& sc, int& sn)
{
   if (fes.GetOrdering() == Ordering::byNODES)
   {
      sc = fes.GetNDofs();
      sn = 1;
   }
   else
   {
      sc = 1;
      sn = fes.GetVDim();
   }
}

} // namespace

FlowStatistics::FlowStatistics(ParFiniteElementSpace& V,
                               ParFiniteElementSpace& Q, double start_time)
   : mean_u_(&V), mean_p_(&Q), normal_(&V), shear_(&V), start_(start_time)
{
   dim_ = V.GetParMesh()->Dimension();
   MFEM_VERIFY((dim_ == 2 || dim_ == 3) && V.GetVDim() == dim_,
               "flow_statistics: needs the velocity space (vdim = dim, 2D or 3D)");
   MFEM_VERIFY(Q.GetVDim() == 1, "flow_statistics: needs a scalar pressure space");
   for (ParGridFunction* f : {&mean_u_, &mean_p_, &normal_, &shear_})
   {
      f->UseDevice(true);
      *f = 0.0;
   }
}

bool FlowStatistics::Sample(const ParGridFunction& u, const ParGridFunction& p,
                            double t)
{
   if (t < start_ - 1e-12 * std::max(1.0, std::abs(start_))) { return false; }
   MFEM_VERIFY(u.Size() == mean_u_.Size() && p.Size() == mean_p_.Size(),
               "flow_statistics: the sample's fields are not on the statistics' "
               "spaces");
   // The first sample sets the means (f = 1) and zero stresses (g = 0); a
   // later one has the weight of the time since the previous.
   double f = 1.0, g = 0.0;
   if (samples_ > 0)
   {
      const double w = t - t_prev_;
      MFEM_VERIFY(w >= 0.0, "flow_statistics: samples must not go back in time");
      if (w == 0.0) { return false; }
      const double W = weight_ + w;
      f = w / W;
      g = weight_ / W;
      weight_ = W;
   }
   else
   {
      t_first_ = t;
   }
   int sc = 0, sn = 0;
   Strides(*mean_u_.ParFESpace(), sc, sn);
   UpdateRunningMoments(mean_u_.ParFESpace()->GetNDofs(), dim_, sc, sn, f, g, u,
                        mean_u_, &normal_, &shear_);
   Strides(*mean_p_.ParFESpace(), sc, sn);
   UpdateRunningMoments(mean_p_.ParFESpace()->GetNDofs(), 1, sc, sn, f, g, p,
                        mean_p_, nullptr, nullptr);
   t_prev_ = t;
   ++samples_;
   return true;
}

void FlowStatistics::Save(const std::string& dir) const
{
   const int rank = mean_u_.ParFESpace()->GetMyRank();
   auto file = [&](const char* base)
   {
      return Checkpoint::RankFile(dir, base, rank);
   };
   Checkpoint::WriteVector(file("flow_stats_u"), mean_u_);
   Checkpoint::WriteVector(file("flow_stats_p"), mean_p_);
   Checkpoint::WriteVector(file("flow_stats_rn"), normal_);
   Checkpoint::WriteVector(file("flow_stats_rs"), shear_);
   if (rank != 0) { return; }
   std::ofstream f(dir + "/" + kMetaFile);
   MFEM_VERIFY(f, "flow_statistics: cannot write " << dir << "/" << kMetaFile);
   f << std::setprecision(17); // round-trips every double exactly
   f << "incns_flow_statistics 1\n"
     << "dim " << dim_ << "\nstart_time " << start_ << "\nsamples " << samples_
     << "\nweight " << weight_ << "\nt_first " << t_first_ << "\nt_prev "
     << t_prev_ << "\n";
}

void FlowStatistics::Load(const std::string& dir)
{
   const std::string meta = dir + "/" + kMetaFile;
   std::ifstream f(meta);
   MFEM_VERIFY(f, "flow_statistics: cannot read " << meta);
   std::string tag;
   int version = 0, dim = 0;
   double start = 0.0;
   f >> tag >> version;
   MFEM_VERIFY(tag == "incns_flow_statistics" && version == 1,
               "flow_statistics: " << meta << " is not a statistics state");
   f >> tag >> dim >> tag >> start >> tag >> samples_ >> tag >> weight_ >> tag
     >> t_first_ >> tag >> t_prev_;
   MFEM_VERIFY(f, "flow_statistics: " << meta << " ends early");
   MFEM_VERIFY(dim == dim_, "flow_statistics: the state in " << dir << " is "
               << dim << "D, the case " << dim_ << "D");
   MFEM_VERIFY(std::abs(start - start_) <= 1e-12 * std::max(1.0, std::abs(start_)),
               "flow_statistics: the averages in " << dir << " started at "
               "start_time " << start << ", the deck says " << start_ << ": a "
               "restart continues them (to start new ones, remove "
               << kMetaFile << " from the checkpoint)");
   const int rank = mean_u_.ParFESpace()->GetMyRank();
   auto file = [&](const char* base)
   {
      return Checkpoint::RankFile(dir, base, rank);
   };
   Checkpoint::ReadVector(file("flow_stats_u"), mean_u_.Size(), mean_u_);
   Checkpoint::ReadVector(file("flow_stats_p"), mean_p_.Size(), mean_p_);
   Checkpoint::ReadVector(file("flow_stats_rn"), normal_.Size(), normal_);
   Checkpoint::ReadVector(file("flow_stats_rs"), shear_.Size(), shear_);
}

bool FlowStatistics::Saved(const std::string& dir)
{
   return std::filesystem::exists(dir + "/" + kMetaFile);
}

} // namespace incns
