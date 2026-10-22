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
 * the larger elements away from it absorb the deformation. Stiffening does not
 * improve the mesh as a whole: it moves distortion from small elements into
 * large ones, which is the trade wanted for a boundary layer, and chi = 2
 * costs the far field more than it usually buys (docs/mesh_motion.md §1,
 * "Choosing the settings", with measurements). The operator and
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
      /// Solutions kept for reuse (0: none). The displacement is linear in
      /// the boundary data, so data in the span of earlier data (a rigid or
      /// modal motion: a few shapes times functions of t) are answered by
      /// the same combination of the earlier solutions, without a solve.
      int basis_size = 12;
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
   bool Solve(double t) override { return Solve(t, nullptr); }
   bool Solve(double t, const mfem::ParGridFunction& guess) override
   {
      return Solve(t, &guess);
   }
   const mfem::ParGridFunction& Displacement() const override
   {
      return static_cast<const LinearElasticity&>(*solver_).Displacement();
   }
   int Iterations() const override { return solver_->Iterations(); }
   void SaveState(std::ostream& out) const override;
   void LoadState(std::istream& in) override;

   /// @return Solves answered from the kept solutions (no CG).
   int ReusedSolves() const { return reused_; }
   /// @return Solutions kept.
   int BasisSize() const { return static_cast<int>(basis_x_.size()); }

   /// @return The elasticity solver (iterations, energies).
   LinearElasticity& Solver() { return *solver_; }
   /// @return The per-element stiffness factors in use (empty when chi = 0).
   const mfem::Vector& Factors() const { return factors_; }

private:
   /**
    * @brief The solve: from the kept solutions when the data lie in their
    *        span, else CG (from @p guess, or zero) and the solution kept.
    * @param t     Time of the boundary data.
    * @param guess Initial guess on Space(), or null.
    * @return Whether the solve converged.
    */
   bool Solve(double t, const mfem::ParGridFunction* guess);

   Options opts_;         ///< Settings.
   mfem::Vector factors_; ///< Per-element stiffening (outlives solver_).
   std::unique_ptr<LinearElasticity> solver_; ///< The pseudo-solid.
   mfem::Vector guess_; ///< A guess's true dofs (Solve with a guess).
   /// The kept data, orthonormal (Dirichlet values on the essential true
   /// dofs), and the solutions they map to (true dofs).
   std::vector<mfem::Vector> basis_g_;
   std::vector<mfem::Vector> basis_x_; ///< See basis_g_.
   mfem::Vector g_;   ///< The current data.
   mfem::Vector x_;   ///< The combination (true dofs).
   int reused_ = 0;   ///< See ReusedSolves().
};

} // namespace incns
