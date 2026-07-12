#include "post/output.hpp"

#include <fstream>

namespace incns
{

using namespace mfem;

OutputWriter::OutputWriter(ParMesh& mesh, ParGridFunction& u,
                           ParGridFunction& p, const OutputParameters& out,
                           int order_u, const Nondimensionalization& nd)
   : out_(out), nd_(nd)
{
   MPI_Comm_rank(mesh.GetComm(), &rank_);
   pv_ = std::make_unique<ParaViewDataCollection>(out_.name, &mesh);
   pv_->SetPrefixPath(out_.path);
   // High-order output is non-negotiable (see class docs): the LOD refinement
   // must be at least k_u or high-order fields render as low-order mush.
   pv_->SetHighOrderOutput(true);
   pv_->SetLevelsOfDetail(std::max(order_u, 1));
   pv_->SetDataFormat(VTKFormat::BINARY);
   pv_->RegisterField("velocity", &u);
   pv_->RegisterField("pressure", &p);
}

void OutputWriter::MaybeSave(int cycle, double time)
{
   if (cycle % out_.interval != 0 && cycle != 0) { return; }
   pv_->SetCycle(cycle);
   pv_->SetTime(time);
   pv_->Save();

   // Record the reference scales once, next to the collection (fields are
   // nondimensional; this is how a consumer re-dimensionalizes them).
   if (!wrote_scales_ && rank_ == 0)
   {
      std::ofstream f(out_.path + "/" + out_.name + "/scales.yaml");
      f << "# reference scales; fields in this collection are nondimensional\n"
        << "mode: " << (nd_.mode == ScalingMode::Dimensional ? "dimensional"
                        : "dimensionless") << "\n"
        << "L_ref: " << nd_.L_ref << "\nU_ref: " << nd_.U_ref
        << "\nrho: " << nd_.rho << "\nT_ref: " << nd_.TRef()
        << "\nRe: " << nd_.Re << "\n";
   }
   wrote_scales_ = true;
}

} // namespace incns
