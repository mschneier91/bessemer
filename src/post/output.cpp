#include "post/output.hpp"

namespace incns
{

using namespace mfem;

OutputWriter::OutputWriter(ParMesh& mesh, ParGridFunction& u,
                           ParGridFunction& p, const OutputParameters& out,
                           int order_u)
   : out_(out)
{
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
}

} // namespace incns
