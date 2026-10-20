/**
 * @file linear_elastic_motion.hpp
 * @brief The linear elastic mesh-motion model: LinearElasticity on the
 *        reference mesh with Jacobian-based stiffening (Stein, Tezduyar and
 *        Benney, J. Appl. Mech. 70 (2003) 58-63), docs/mesh_motion.md.
 */
#pragma once

#include "ale/mesh_motion_model.hpp"
#include "solid/linear_elasticity.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Mesh displacement from a pseudo-solid: small-strain isotropic
 *        elasticity (Young's modulus 1, which a displacement-driven problem
 *        does not depend on) with each element's stiffness multiplied by
 *        (V_mean / V_K)^chi, V_K its volume in the reference mesh.
 *
 * chi = 0 is a homogeneous solid; chi = 1 (the default) makes the small
 * elements, typically the boundary layer at a moving body, nearly rigid, so
 * the larger elements away from it absorb the deformation. The operator and
 * its LOR-AMG are built once (the reference mesh never moves); each Solve()
 * is one CG from a zero guess.
 */
class LinearElasticMotion : public MeshMotionModel
{
public:
   /// Settings.
   struct Options
   {
      double poisson_ratio = 0.3; ///< Poisson's ratio of the pseudo-solid.
      double stiffening = 1.0;    ///< Stiffening exponent chi (>= 0).
      /// CG preconditioner.
      ElasticityPC preconditioner = ElasticityPC::LorAmg;
      double rtol = 1e-10; ///< CG relative tolerance.
      double atol = 0.0;   ///< CG absolute tolerance.
      int max_iter = 2000; ///< CG iteration cap.
      int print_level = -1; ///< MFEM CG print level (-1 silent).
   };

   /**
    * @brief Build the solver on the reference mesh. Collective.
    * @param reference The reference mesh (borrowed; must outlive the model).
    * @param order     Displacement order (>= 1).
    * @param opts      Settings.
    */
   LinearElasticMotion(mfem::ParMesh& reference, int order,
                       const Options& opts);

   /**
    * @param opts Settings.
    * @return A factory for MeshMotion building this model.
    */
   static MeshMotionModelFactory Factory(const Options& opts);

   /**
    * @param mesh The mesh (its local element volumes are read).
    * @param chi  Stiffening exponent.
    * @return (V_mean / V_K)^chi on every local element K, V_mean the global
    *         mean element volume. Collective.
    */
   static mfem::Vector StiffeningFactors(mfem::ParMesh& mesh, double chi);

   const char* Name() const override { return "linear_elastic"; }
   mfem::ParFiniteElementSpace& Space() override { return solver_->Space(); }
   void AddFixed(int attr) override { solver_->AddFixed(attr); }
   void AddDisplacement(int attr, mfem::VectorCoefficient& g) override
   {
      solver_->AddDisplacement(attr, g);
   }
   bool Solve(double t) override { return solver_->Solve(t); }
   const mfem::ParGridFunction& Displacement() const override
   {
      return static_cast<const LinearElasticity&>(*solver_).Displacement();
   }
   int Iterations() const override { return solver_->Iterations(); }

   /// @return The elasticity solver (iterations, energies).
   LinearElasticity& Solver() { return *solver_; }
   /// @return The per-element stiffness factors in use (empty when chi = 0).
   const mfem::Vector& Factors() const { return factors_; }

private:
   mfem::Vector factors_; ///< Per-element stiffening (outlives solver_).
   std::unique_ptr<LinearElasticity> solver_; ///< The pseudo-solid.
};

} // namespace incns
