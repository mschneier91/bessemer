#include "ale/mesh_motion_factory.hpp"

#include "ale/hyperelastic_motion.hpp"
#include "ale/linear_elastic_motion.hpp"

namespace incns
{

MeshMotionModelFactory DeckMeshMotionFactory(const MeshMotionParameters& mm)
{
   if (mm.model == MeshMotionModelType::NeoHookean)
   {
      MFEM_VERIFY(mm.preconditioner == ElasticityPC::LorAmg, "mesh_motion: the "
                  "neo_hookean model's Newton corrections use the LOR-AMG "
                  "(preconditioner lor_amg)");
      HyperelasticMotion::Options h;
      h.material = HyperelasticMaterial::NeoHookean;
      h.poisson_ratio = mm.poisson_ratio;
      h.stiffening = mm.stiffening;
      h.rtol = mm.rtol;
      h.atol = mm.atol;
      h.max_iter = mm.max_iter;
      return HyperelasticMotion::Factory(h);
   }
   LinearElasticMotion::Options e;
   e.poisson_ratio = mm.poisson_ratio;
   e.stiffening = mm.stiffening;
   e.preconditioner = mm.preconditioner;
   e.rtol = mm.rtol;
   e.atol = mm.atol;
   e.max_iter = mm.max_iter;
   e.basis_size = mm.basis_size;
   return LinearElasticMotion::Factory(e);
}

} // namespace incns
