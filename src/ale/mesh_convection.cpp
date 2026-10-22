#include "ale/mesh_convection.hpp"

#include "operators/convection.hpp"
#include "util/profiler.hpp"

namespace incns
{

using namespace mfem;

MeshVelocityField::MeshVelocityField(ParFiniteElementSpace& vfes)
   : w_(&vfes), coeff_(&w_)
{
   w_ = 0.0;
   lead_ = 0.0;
}

void MeshVelocityField::Update(const MeshMotion& motion, int order)
{
   INCNS_PROFILE("ale::mesh_velocity");
   const ParGridFunction& d = motion.NodeDisplacement();
   Vector wn;
   motion.MeshVelocity(order, wn, &lead_);
   // The node velocity as a field on the nodal space, interpolated at the
   // velocity space's nodes (on the host, MFEM's element loop).
   ParGridFunction wnodes(const_cast<ParFiniteElementSpace*>(d.ParFESpace()));
   wnodes.HostWrite();
   static_cast<Vector&>(wnodes) = wn;
   wnodes.HostRead();
   w_.HostWrite();
   w_.ProjectGridFunction(wnodes);
}

void MeshVelocityField::Zero()
{
   w_ = 0.0;
}

MeshConvection::MeshConvection(MixedSpaces& spaces, const RuleBook& rules)
   : spaces_(spaces)
{
   ParFiniteElementSpace& V = spaces_.Velocity();
   MFEM_VERIFY(V.GetOrdering() == Ordering::byNODES, "mesh_convection: the "
               "velocity space must be ordered byNODES (component blocks)");
   dim_ = V.GetVDim();
   const Geometry::Type geom = (spaces_.Dim() == 3) ? Geometry::CUBE :
                               Geometry::SQUARE;
   rule_ = &rules.Get(geom, Convection::DealiasedOrder(spaces_.OrderU()));
   sfes_ = std::make_unique<ParFiniteElementSpace>(V.GetParMesh(), V.FEColl());
   n_s_ = sfes_->GetTrueVSize();
   MFEM_VERIFY(dim_ * n_s_ == V.GetTrueVSize(), "mesh_convection: the scalar "
               "companion space does not match the velocity components");
   wind_ = std::make_unique<ParGridFunction>(&V);
   *wind_ = 0.0;
   wind_coeff_ = std::make_unique<VectorGridFunctionCoefficient>(wind_.get());
}

void MeshConvection::SetWind(const ParGridFunction& w)
{
   INCNS_PROFILE("ale::mesh_convection_setup");
   *wind_ = w;
   wind_->HostRead();
   // A new form for the new geometry and wind (OifsAdvector builds its
   // per-step convection forms the same way).
   form_ = std::make_unique<ParBilinearForm>(sfes_.get());
   auto* ci = new ConvectionIntegrator(*wind_coeff_, 1.0); // (w.grad u, v)
   ci->SetIntRule(rule_);
   form_->AddDomainIntegrator(ci);
   form_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form_->Assemble();
   Array<int> none;
   form_->FormSystemMatrix(none, op_);
}

void MeshConvection::Mult(const Vector& u, Vector& y) const
{
   INCNS_PROFILE("ale::mesh_convection");
   MFEM_VERIFY(op_.Ptr(), "mesh_convection: SetWind() first");
   y.SetSize(u.Size());
   y.UseDevice(true);
   for (int c = 0; c < dim_; ++c)
   {
      Vector u_c, y_c;
      u_c.MakeRef(const_cast<Vector&>(u), c * n_s_, n_s_);
      y_c.MakeRef(y, c * n_s_, n_s_);
      // MFEM's alias protocol (device memory).
      u_c.SyncMemory(u);
      y_c.SyncMemory(y);
      op_->Mult(u_c, y_c);
      y_c.SyncAliasMemory(y);
   }
}

} // namespace incns
