/**
 * @file hyperelastic_motion.hpp
 * @brief The hyperelastic mesh-motion model: Hyperelasticity (neo-Hookean) on
 *        the reference mesh with the linear model's Jacobian stiffening,
 *        docs/mesh_motion.md and docs/design/hyperelasticity_spec.md
 *        (milestone 4).
 */
#pragma once

#include "ale/mesh_motion_model.hpp"
#include "solid/hyperelasticity.hpp"

#include <memory>

namespace incns
{

/**
 * @brief Mesh displacement from a large-deformation pseudo-solid: neo-Hookean
 *        (Young's modulus 1, which a displacement-driven problem does not
 *        depend on) with each element's stiffness multiplied by (V_mean /
 *        V_K)^chi, as LinearElasticMotion.
 *
 * Its energy is frame-indifferent, so a rotated body region costs nothing,
 * where the linear model reads a rotation by theta as a compression of 1 -
 * cos theta and folds the elements around a body turned far enough. Each
 * Solve() is a Newton solve (Hyperelasticity) from the guess MeshMotion
 * passes (the kept states extrapolated to the new time) or, without one, from
 * the last solve's displacement; the reference LOR-AMG is built once. A
 * checkpoint carries that displacement, so a restart starts Newton where an
 * uninterrupted run would. The neo-Hookean energy grows without bound as an
 * element is squeezed flat (-mu ln J), which is what keeps the elements from
 * folding; St. Venant-Kirchhoff (Options::material) softens there instead and
 * failed past ~45 degrees on the turned cylinder (docs/hyperelasticity.md
 * section 4), and is kept for comparison only.
 */
class HyperelasticMotion : public MeshMotionModel
{
public:
   /// Settings.
   struct Options
   {
      /// The pseudo-solid's material.
      HyperelasticMaterial material = HyperelasticMaterial::NeoHookean;
      double poisson_ratio = 0.3; ///< Poisson's ratio of the pseudo-solid.
      double stiffening = 1.0;    ///< Stiffening exponent chi (>= 0).
      double rtol = 1e-10; ///< CG relative tolerance (each Newton correction).
      double atol = 0.0;   ///< CG absolute tolerance.
      int max_iter = 2000; ///< CG iteration cap.
      int print_level = -1; ///< MFEM CG print level (-1 silent).
      double newton_rtol = 1e-8;     ///< Newton residual reduction.
      double newton_dx_rtol = 1e-10; ///< Newton correction criterion.
      int newton_max_iter = 25;      ///< Newton iterations per solve.
      int load_steps = 1;            ///< Load steps per solve.
      bool line_search = false;      ///< Newton's backtracking.
   };

   /**
    * @brief Build the solver on the reference mesh. Collective.
    * @param reference The reference mesh (borrowed; must outlive the model).
    * @param order     Displacement order (>= 1).
    * @param opts      Settings.
    */
   HyperelasticMotion(mfem::ParMesh& reference, int order, const Options& opts);

   /**
    * @param opts Settings.
    * @return A factory for MeshMotion building this model.
    */
   static MeshMotionModelFactory Factory(const Options& opts);

   const char* Name() const override
   {
      return material_ == HyperelasticMaterial::NeoHookean ? "neo_hookean" :
             "st_venant_kirchhoff";
   }
   mfem::ParFiniteElementSpace& Space() override { return solver_->Space(); }
   void AddFixed(int attr) override { solver_->AddFixed(attr); }
   void AddDisplacement(int attr, mfem::VectorCoefficient& g) override
   {
      solver_->AddDisplacement(attr, g);
   }
   /// Newton from the last solve's displacement.
   bool Solve(double t) override { return solver_->Solve(t); }
   /// Newton from @p guess.
   bool Solve(double t, const mfem::ParGridFunction& guess) override;
   const mfem::ParGridFunction& Displacement() const override
   {
      return static_cast<const Hyperelasticity&>(*solver_).Displacement();
   }
   /// CG iterations of the last solve, over its Newton steps.
   int Iterations() const override { return solver_->LinearIterations(); }
   void SaveState(std::ostream& out) const override;
   void LoadState(std::istream& in) override;

   /// @return The solver (Newton iterations and histories, energies).
   Hyperelasticity& Solver() { return *solver_; }
   /// @return The per-element stiffness factors in use (empty when chi = 0).
   const mfem::Vector& Factors() const { return factors_; }

private:
   HyperelasticMaterial material_; ///< The pseudo-solid's material.
   mfem::Vector factors_; ///< Per-element stiffening (outlives solver_).
   std::unique_ptr<Hyperelasticity> solver_; ///< The pseudo-solid.
   mfem::Vector x_; ///< True-dof work vector (guesses, checkpoints).
};

} // namespace incns
