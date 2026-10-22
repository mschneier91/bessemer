/**
 * @file fsi_case.hpp
 * @brief A deck-driven fluid-structure interaction run (section `fsi`): the
 *        ALE flow and the linear solid, coupled by partitioned
 *        subiterations (Robin-Neumann or Dirichlet-Neumann) with dynamic
 *        Aitken relaxation in every time step (docs/fsi.md).
 */
#pragma once

#include "fsi/interface_accelerator.hpp"
#include "fsi/interface_transfer.hpp"
#include "solid/linear_elasticity.hpp"
#include "solver/case.hpp"

#include <memory>
#include <string>
#include <vector>

namespace incns
{

/**
 * @brief The flow of the deck (Case, on a moving mesh whose `interface`
 *        groups follow the solid) and a linear solid on the solid part of
 *        the deck's geometry (`mesh.cylinder_channel.part: solid`),
 *        coupled in every time step.
 *
 * Dirichlet-Neumann iterates on the interface displacement d, Robin-Neumann
 * on the fluid's interface traction tau (StepRobinNeumann(): solid under
 * -rho_f tau, mesh to its interface, fluid with Robin data tau and the
 * solid's velocity, tau~ its traction; the residual tau~ - tau).
 *
 * A Dirichlet-Neumann step: predict the interface displacement d from the committed steps
 * (fsi.predictor), then iterate: the fluid takes a trial step with its
 * interface at d (u = w there); the fluid's traction (the momentum residual
 * at the interface dofs, recovered as a traction field by the interface's
 * mass matrix and scaled by fsi.fluid_density) loads the solid, which steps
 * from the start of the step (or solves statically); its interface
 * displacement d~ gives the residual r = d~ - d. Converged when |r| <=
 * rtol |d~| + atol: the fluid's trial step is committed. Otherwise the fluid
 * step is undone and d <- d + omega r with Aitken's omega.
 *
 * The solid is the small-strain one (the traction, a Cauchy traction on the
 * moved interface, is applied on the reference interface); St.
 * Venant-Kirchhoff takes its place when it lands.
 */
class FsiCase
{
public:
   /// One coupled step.
   struct StepRecord
   {
      double time = 0.0;     ///< Time at the end of the step.
      int iterations = 0;    ///< Subiterations.
      double residual = 0.0; ///< |r| / max(|d~|, tiny) at the last one.
      double omega = 0.0;    ///< The last relaxation factor used.
   };

   /**
    * @param fluid_mesh The deck's (fluid) mesh (borrowed; must outlive the
    *                   case).
    * @param params     Normalized parameters with fsi.enabled.
    */
   FsiCase(mfem::ParMesh& fluid_mesh, const Parameters& params);

   ~FsiCase();

   /// One coupled step. @return False when the subiterations did not
   /// converge in fsi.max_iterations or the fluid diverged.
   bool Step();

   /// March to time.t_final. @return Whether every step converged.
   bool Run();

   /**
    * @brief Write the run summary (JSON) on rank 0. Collective.
    * @param path   File.
    * @param status running, ok or diverged.
    */
   void WriteSummary(const std::string& path, const std::string& status);

   /// @return The flow.
   Case& Fluid() { return *fluid_; }

   /// @return The solid.
   LinearElasticity& Solid() { return *solid_; }

   /// @return The solid's mesh.
   mfem::ParMesh& SolidMesh() { return *solid_mesh_; }

   /// @return The committed steps.
   const std::vector<StepRecord>& History() const { return history_; }

   /// @return The fluid's traction on the interface of the fluid's last
   ///         step, interpolated to the solid's space (rho_f times the
   ///         kinematic traction; UpdateTraction()).
   const mfem::ParGridFunction& SolidTraction() const { return *solid_traction_; }

   /// Recompute SolidTraction() from the fluid's last step. Collective.
   void UpdateTraction();

   /// @return The fluid's residual at the interface's velocity true dofs
   ///         (zero elsewhere) of its last step: the moments of its
   ///         kinematic traction, the Robin-Neumann iterate (UpdateLoad()).
   const mfem::Vector& InterfaceLoad() const { return last_load_; }

   /// @return The total force of SolidTraction() over the fluid's moved
   ///         interface (rho_f times the fluid's force on the interface).
   mfem::Vector InterfaceForce() const { return interface_force_; }

private:
   /// @param d The solid's displacement interpolated to the fluid
   ///          interface's nodes (the motion space on the reference mesh).
   void SolidInterfaceDisplacement(mfem::ParGridFunction& d);

   /// @param r The step to append to the coupling log. Collective (the
   ///          probe); rank 0 writes.
   void LogStep(const StepRecord& r);

   /// @param d The predicted interface displacement (true dofs of the
   ///          motion space).
   void Predict(mfem::Vector& d) const;

   /// @param load The predicted interface traction (Robin-Neumann; velocity
   ///             true dofs), extrapolated as Predict().
   void PredictLoad(mfem::Vector& load) const;

   /// last_load_ from the fluid's last step. Collective.
   void UpdateLoad();

   /// @param load Interface moments of the fluid's kinematic traction
   ///             (velocity true dofs): SolidTraction() and InterfaceForce()
   ///             from them, on the current mesh. Collective.
   void LoadSolid(const mfem::Vector& load);

   /**
    * @brief The traction field whose moments on the current mesh's
    *        interface are @p load; sets InterfaceForce(). Collective.
    * @param load Moments (velocity true dofs; interface entries read).
    * @param t    The fluid's kinematic traction (velocity true dofs; zero off
    *             the interface).
    */
   void TractionFromLoad(const mfem::Vector& load, mfem::Vector& t);

   /// @param t The fluid's kinematic traction (velocity true dofs): the
   ///          solid's load SolidTraction() = -rho_f t. Collective.
   void LoadSolidTraction(const mfem::Vector& t);

   /// Step() under Robin-Neumann: subiterations on the interface traction.
   /// @return Whether they converged (else failure_ says why).
   bool StepRobinNeumann();

   /// The default Robin coefficient, face by face from the solid's
   /// thickness behind each interface face; handed to the flow.
   void BuildRobinCoefficient();

   mfem::ParMesh& fluid_mesh_; ///< The fluid mesh (borrowed).
   Parameters params_;         ///< Parameters (copied).
   Parameters solid_params_;   ///< The solid's: the geometry's solid part.
   std::unique_ptr<mfem::ParMesh> solid_mesh_; ///< The solid's mesh.
   std::unique_ptr<Case> fluid_;               ///< The flow.
   std::unique_ptr<mfem::VectorCoefficient> u0_; ///< Its initial velocity.
   std::unique_ptr<LinearElasticity> solid_;   ///< The solid.
   bool dynamic_ = false;      ///< Whether the solid is dynamic.
   std::vector<int> fluid_attrs_; ///< Fluid mesh interface attributes.
   std::vector<int> solid_attrs_; ///< Solid mesh interface attributes.

   /// The fluid's velocity space on the motion's reference mesh (the
   /// traction's data, interpolated in the reference configuration).
   std::unique_ptr<mfem::ParFiniteElementSpace> ref_vfes_;
   std::unique_ptr<mfem::ParGridFunction> ref_traction_; ///< On ref_vfes_.
   std::unique_ptr<mfem::ParGridFunction> solid_traction_; ///< On the solid.
   /// The solid's traction load (reads solid_traction_).
   std::unique_ptr<mfem::VectorGridFunctionCoefficient> traction_coeff_;
   std::unique_ptr<InterfaceTransfer> to_solid_; ///< Fluid -> solid map.
   std::unique_ptr<InterfaceTransfer> to_fluid_; ///< Solid -> fluid map.
   mfem::Array<int> fluid_marker_; ///< Interface marker of the fluid mesh.
   mfem::Vector interface_force_;  ///< See InterfaceForce().
   /// The fluid's residual (its kinematic traction's moments) at the
   /// interface's velocity true dofs, zero elsewhere, of the last step
   /// (UpdateTraction()).
   mfem::Vector last_load_;
   /// Robin-Neumann: committed interface tractions, newest first.
   std::vector<mfem::Vector> committed_load_;
   /// Robin-Neumann: the current subiteration's traction field (the fluid's
   /// Robin data).
   std::unique_ptr<mfem::ParGridFunction> robin_tau_;
   /// Its coefficient (the time integrator reads it).
   std::unique_ptr<mfem::VectorGridFunctionCoefficient> robin_tau_coeff_;
   /// The face-by-face Robin coefficient (kinematic), or null (the constant
   /// Parameters::FsiFluidRobin()).
   std::unique_ptr<mfem::Coefficient> robin_alpha_;
   double robin_alpha_min_ = 0.0; ///< Its smallest value on the interface.
   double robin_alpha_max_ = 0.0; ///< Its largest.

   std::unique_ptr<InterfaceAccelerator> accel_; ///< The relaxation.
   /// Committed interface displacements (motion true dofs), newest first.
   std::vector<mfem::Vector> committed_;
   std::vector<StepRecord> history_; ///< See History().
   long long total_iterations_ = 0;  ///< Subiterations over the run.
   int max_iterations_ = 0;          ///< Most subiterations in a step.
   double wall_ = 0.0;               ///< Wall time of the march.
   bool log_started_ = false;        ///< Coupling log header written.
   std::string failure_;             ///< Why a step failed (empty if none).
};

} // namespace incns
