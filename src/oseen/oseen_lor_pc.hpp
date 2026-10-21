/**
 * @file oseen_lor_pc.hpp
 * @brief The velocity block's preconditioner of the Oseen schemes
 *        (docs/design/oseen_bdf2.md): per velocity component, one AIR V-cycle
 *        (convection-dominated) or one classical AMG V-cycle on a
 *        low-order-refined operator sigma D + nu K + C(w) + SD(w).
 */
#pragma once

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/// Which V-cycle OseenLorPreconditioner uses.
enum class OseenAmgMode
{
   Auto, ///< AIR where the cell Reynolds number is >= air_cell_re, else AMG.
   Air,  ///< hypre's Neumann AIR (the recipe's GPU settings).
   Amg   ///< Classical AMG (PMIS, extended+i, l1-Jacobi).
};

/**
 * @brief F^-1 for the Oseen momentum block, one V-cycle per component.
 *
 * The low-order operator lives on the LOR mesh through the GLL nodes (the
 * high-order true dofs): F_lor = sigma D + nu K_lor + C_lor(w) + SD_lor(w),
 * with D the diagonal of the high-order GLL (collocated) mass, K_lor at the
 * vertex rule (GLL, two points per direction) and C_lor, SD_lor at MFEM's
 * default Gauss rules -- the recipe's mixed quadrature: the vertex rule alone
 * fails for advection, Gauss alone doubles the diffusive cost. SD is
 * (tau w.grad u, w.grad v) with tau from the wind at the point and the
 * high-order PARENT element's J^-1 at its centre (OssTau, order k). w is the
 * high-order wind's nodal values copied to the LOR space. The 1/2 div w
 * term, grad-div and the outflow term are left out. Assembled on the host
 * (MFEM's batched LOR has no convection), so rebuild it every few steps.
 *
 * Components with the same Dirichlet dofs share one matrix and V-cycle.
 */
class OseenLorPreconditioner : public mfem::Solver
{
public:
   /// Options.
   struct Options
   {
      double nu = 1.0;          ///< Kinematic viscosity.
      OseenAmgMode mode = OseenAmgMode::Auto;
      /// Auto: AIR where Re_h = max_K |w|_K h_K / (k nu) >= this.
      double air_cell_re = 3.0;
      /// AIR's coarse-grid drop tolerance (ADropTol); negative: 1e-2 on one
      /// rank and 0 on more -- hypre 2.33's device dropping keeps a stale
      /// communication package (the recipe's pitfall 5).
      double air_drop_tol = -1.0;
      int print_level = 0;      ///< BoomerAMG print level.
   };

   /**
    * @param vfes      The velocity space (vector, byNODES; borrowed).
    * @param rules     Quadrature source (borrowed).
    * @param ess_tdofs The velocity's essential true dofs (vector numbering).
    * @param opts      Options.
    */
   OseenLorPreconditioner(mfem::ParFiniteElementSpace& vfes,
                          const RuleBook& rules,
                          const mfem::Array<int>& ess_tdofs,
                          const Options& opts);
   ~OseenLorPreconditioner() override;

   /**
    * @brief Rebuild the LOR operators and their V-cycles.
    * @param wind  The wind's true dofs (the velocity space's).
    * @param sigma The mass factor (BDF: beta0 / dt).
    */
   void Update(const mfem::Vector& wind, double sigma);

   /**
    * @brief Include each component's diagonal grad-div block
    *        gamma (d_c u_c, d_c v_c) in its LOR operator (anisotropic
    *        diffusion, at the vertex rule), with gamma constant on each
    *        high-order element. Assembled once here, added at every Update.
    *        The components' operators then differ: no sharing.
    * @param gamma_elem gamma per high-order element (local elements).
    */
   void SetGradDiv(const mfem::Vector& gamma_elem);

   /// z = F^-1 r, one V-cycle per component.
   void Mult(const mfem::Vector& r, mfem::Vector& z) const override;
   void SetOperator(const mfem::Operator&) override {}

   /// @return Whether the last Update chose AIR.
   bool UsingAir() const { return air_; }
   /// @return The cell Reynolds number of the last Update.
   double CellReynolds() const { return re_h_; }
   /// @return Seconds of the last Update: host assembly, V-cycle setup.
   double AssemblySeconds() const { return t_assembly_; }
   double SetupSeconds() const { return t_setup_; } ///< See AssemblySeconds.

private:
   /// Max over elements of |w|_K h_K / (k nu) (collective).
   double CellReynolds(const mfem::ParGridFunction& w) const;

   mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   Options opts_;
   int dim_ = 0, k_ = 0, n_ = 0;       ///< Dimension, order, scalar tdofs.
   std::unique_ptr<mfem::ParFiniteElementSpace> sfes_; ///< Scalar HO space.
   std::unique_ptr<mfem::ParLORDiscretization> lor_;   ///< Scalar LOR.
   std::unique_ptr<mfem::ParFiniteElementSpace> lor_vfes_; ///< Vector LOR space.
   std::unique_ptr<mfem::ParGridFunction> wind_lor_;   ///< w on the LOR mesh.
   std::unique_ptr<mfem::ParGridFunction> wind_ho_;    ///< w (high order).
   mfem::DenseTensor jparent_;    ///< J^-1 at the centre, per HO element.
   mfem::Array<int> parent_;      ///< LOR element -> HO parent element.
   mfem::Vector d_gll_;           ///< HO GLL mass diagonal (scalar tdofs).
   const mfem::IntegrationRule* vertex_rule_ = nullptr; ///< GLL, 2 points.
   std::vector<mfem::Array<int>> ess_;   ///< Essential tdofs per component.
   std::vector<int> block_;              ///< Component -> matrix index.
   std::vector<std::unique_ptr<mfem::HypreParMatrix>> F_;
   /// The diagonal grad-div blocks per component (SetGradDiv; empty: none).
   std::vector<std::unique_ptr<mfem::HypreParMatrix>> G_;
   std::vector<std::unique_ptr<mfem::HypreBoomerAMG>> amg_;
   bool air_ = false;
   double re_h_ = 0.0, t_assembly_ = 0.0, t_setup_ = 0.0;
};

} // namespace incns
