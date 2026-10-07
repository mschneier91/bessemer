/**
 * @file rotational_schur.hpp
 * @brief Rotation-aware pressure Schur preconditioner for the semi-implicit
 *        rotational form: Cahouet-Chabard or Olshanskii's rotating-Darcy
 *        tensor, switched per time step on the rotation number
 *        (rotational_schur_velocity_mg_spec.md, Part C).
 *
 * The momentum block of the rotational form is
 * @f$ A = \sigma M + \nu K + N(\omega^*) @f$ with the lagged rotation term
 * @f$ N(u, v) = \alpha((\nabla\times w^*)\times u, v) @f$. Cahouet-Chabard
 * approximates the Schur complement as if N were absent and degrades as the
 * rotation number @f$ \mu = |\alpha\omega^*|/\sigma @f$ grows; Olshanskii's
 * tensor version replaces the Laplacian part by
 * @f$ L_T = -\nabla\cdot(T\nabla) @f$ with
 * @f$ T = (\sigma I + [\alpha\omega^*]_\times)^{-1} @f$, the inverse of A's
 * zero-order symbol, which is nearly exact where rotation dominates viscosity.
 *
 * Deviations from the spec, all deliberate (see CLAUDE.md):
 *  - 2D as well as 3D (the spec is 3D only): in 2D the vorticity is a scalar
 *    and @f$ T = [[\sigma, o], [-o, \sigma]] / (\sigma^2 + o^2) @f$.
 *  - The Cahouet-Chabard mode calls bessemer's own CC preconditioner (the
 *    consistent @f$ B M_v^{-1} B^T @f$ variant) unchanged, passed in as
 *    @c cc; the tensor mode reuses its pieces (pressure-mass inverse, the
 *    LOR-AMG Laplacian V-cycle). The Laplacian solver is passed UNSCALED (an
 *    approximate @f$ L_p^{-1} @f$): the inner FGMRES runs on
 *    @f$ \sigma L_T @f$ -- exactly @f$ L_p @f$ when @f$ \omega^* = 0 @f$ --
 *    preconditioned by it, and the result is multiplied by sigma. With
 *    @f$ \omega^* = 0 @f$ and an exact Laplacian solver that reproduces CC
 *    exactly, the constrained (Dirichlet) pressure rows included; scaling the
 *    preconditioner instead, as the spec does, leaves those rows off by sigma.
 *  - The rotation-number statistics live here (RotationNumber), not on the
 *    rotation integrator: the integrator's diagnostic was removed on
 *    2026-10-05 and this switch is now its only consumer.
 */
#pragma once

#include "mfem.hpp"

#include <memory>

namespace incns
{

/// Rotation-number statistics of the lagged vorticity at one time step.
struct RotationNumberStats
{
   /// Maximum over the sample points of @f$ \mu = |\alpha\omega^*|/\sigma @f$.
   double max_mu = 0.0;
   /// Fraction of the domain volume with @f$ \mu > @f$ @c threshold.
   double vol_fraction = 0.0;
   /// The threshold @c vol_fraction was computed with.
   double threshold = 0.0;
};

/**
 * @brief Evaluates RotationNumberStats of an H1 velocity on the device, at the
 *        points of a fixed quadrature rule (collective).
 *
 * The vorticity is the exact curl of the finite element field at the points;
 * the volume fraction weights each point by its quadrature weight times det J.
 */
class RotationNumber
{
public:
   /**
    * @brief Bind the velocity space and the sample rule.
    * @param vfes Velocity space (vector H1, vdim = dim; borrowed).
    * @param ir   Sample rule (borrowed; must outlive this object AND the
    *             space's cached interpolators -- use a RuleBook or IntRules
    *             rule).
    */
   RotationNumber(const mfem::ParFiniteElementSpace& vfes,
                  const mfem::IntegrationRule& ir);

   /**
    * @brief Statistics of @p w (collective).
    * @param w         Velocity on the bound space.
    * @param alpha     Rotation-term scale (the integrator's alpha).
    * @param sigma     Leading mass coefficient beta0/dt (> 0).
    * @param threshold mu threshold of the volume fraction.
    * @return max mu and the volume fraction above @p threshold.
    */
   RotationNumberStats Compute(const mfem::ParGridFunction& w, double alpha,
                               double sigma, double threshold) const;

private:
   const mfem::ParFiniteElementSpace& vfes_; ///< Velocity space (borrowed).
   const mfem::IntegrationRule& ir_;         ///< Sample rule (borrowed).
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic restriction.
   const mfem::QuadratureInterpolator* qi_ = nullptr; ///< Derivatives at ir_.
   const mfem::GeometricFactors* geom_ = nullptr;    ///< det J at ir_.
   mutable mfem::Vector we_;  ///< E-vector of w (device scratch).
   mutable mfem::Vector der_; ///< Physical derivatives (device scratch).
   mutable mfem::Vector mu_;  ///< mu per point (device scratch).
   mutable mfem::Vector vol_; ///< Point volume where mu > threshold.
   mutable mfem::Vector all_; ///< Point volume (weight * det J).
};

/**
 * @brief The rotating-Darcy tensor
 *        @f$ T = (\sigma I + [o]_\times)^{-1} @f$, @f$ o = \alpha\nabla\times
 *        w^* @f$, as a matrix coefficient (2D and 3D).
 *
 * Project() evaluates T on the device at the quadrature points of the form
 * that uses it (the partial-assembly path); Eval() is the host path, used by
 * legacy assembly (tests). Never hand this coefficient to a low-order-refined
 * form: MFEM's batched LOR assembly reads only scalar coefficients and would
 * silently assemble with coefficient 1 (spec 5.7, item 1).
 */
class RotatingDarcyTensor : public mfem::MatrixCoefficient
{
public:
   /**
    * @brief Bind the lagged velocity.
    * @param w_star Lagged velocity on an H1 vector space (borrowed; read at
    *               every evaluation, so update it in place).
    * @param sigma  Leading mass coefficient (> 0).
    * @param alpha  Rotation-term scale (the integrator's alpha).
    */
   RotatingDarcyTensor(const mfem::GridFunction& w_star, mfem::real_t sigma,
                       mfem::real_t alpha);

   /**
    * @brief Set the leading mass coefficient.
    * @param s sigma (> 0).
    */
   void SetSigma(mfem::real_t s) { sigma_ = s; }
   /// @return The leading mass coefficient sigma.
   mfem::real_t GetSigma() const { return sigma_; }
   /**
    * @brief Multiply the coefficient by a constant.
    * @param a Scale (default 1: T itself).
    */
   void SetScale(mfem::real_t a) { scale_ = a; }

   /**
    * @brief Host evaluation at one point (legacy assembly).
    * @param T  Output dim x dim matrix.
    * @param Tr Element transformation.
    * @param ip Integration point.
    */
   void Eval(mfem::DenseMatrix& T, mfem::ElementTransformation& Tr,
             const mfem::IntegrationPoint& ip) override;

   /**
    * @brief Device evaluation at every point of @p qf's rule (partial
    *        assembly). Honors @p transpose: partial assembly calls it through
    *        CoefficientVector::ProjectTranspose.
    * @param qf        Output, vdim = dim^2, column-major per point.
    * @param transpose Store T^T instead of T.
    */
   void Project(mfem::QuadratureFunction& qf, bool transpose = false) override;

   /**
    * @brief 3D: @f$ T = (s^2 I + o o^T - s[o]_\times)/(s(s^2 + |o|^2)) @f$,
    *        column-major @c T[r + 3c].
    * @param s  sigma.
    * @param o0 First component of o.
    * @param o1 Second component of o.
    * @param o2 Third component of o.
    * @param T  Output, 9 entries.
    */
   MFEM_HOST_DEVICE static inline void Fill3(mfem::real_t s, mfem::real_t o0,
         mfem::real_t o1, mfem::real_t o2,
         mfem::real_t T[9])
   {
      const mfem::real_t c = 1.0 / (s * (s * s + o0 * o0 + o1 * o1 + o2 * o2));
      const mfem::real_t o[3] = {o0, o1, o2};
      // [o]x column-major: col 0 = (0, o2, -o1), col 1 = (-o2, 0, o0),
      // col 2 = (o1, -o0, 0).
      const mfem::real_t W[9] = {0, o2, -o1, -o2, 0, o0, o1, -o0, 0};
      for (int col = 0; col < 3; ++col)
      {
         for (int row = 0; row < 3; ++row)
         {
            T[row + 3 * col] = c * ((row == col ? s* s : 0.0) + o[row] * o[col]
                                    - s * W[row + 3 * col]);
         }
      }
   }

   /**
    * @brief 2D (scalar o): @f$ T = [[s, o], [-o, s]] / (s^2 + o^2) @f$,
    *        column-major @c T[r + 2c] -- the 3D formula with o along z.
    * @param s sigma.
    * @param o alpha times the scalar vorticity.
    * @param T Output, 4 entries.
    */
   MFEM_HOST_DEVICE static inline void Fill2(mfem::real_t s, mfem::real_t o,
         mfem::real_t T[4])
   {
      const mfem::real_t c = 1.0 / (s * s + o * o);
      T[0] = c * s;  // (0, 0)
      T[1] = -c * o; // (1, 0)
      T[2] = c * o;  // (0, 1)
      T[3] = c * s;  // (1, 1)
   }

private:
   const mfem::GridFunction& w_; ///< Lagged velocity (borrowed).
   mfem::real_t sigma_;          ///< Leading mass coefficient.
   mfem::real_t alpha_;          ///< Rotation-term scale.
   mfem::real_t scale_ = 1.0;    ///< Output scale.
   mfem::Vector w_e_;            ///< E-vector of w (device scratch, reused).
   mfem::Vector der_;            ///< Derivatives at the points (device scratch).
};

/**
 * @brief Pressure Schur-complement preconditioner with two modes, chosen once
 *        per time step from the rotation number:
 *        - CahouetChabard: @f$ z = \mathrm{cc}(r) @f$ -- bessemer's CC
 *          preconditioner, called unchanged (bitwise today's path);
 *        - Tensor: @f$ z = \sigma\,\mathrm{FGMRES}_k(\sigma L_T, r;\
 *          \mathrm{lap\_inv}) + \nu\,\mathrm{mass\_inv}(r) @f$, i.e. an
 *          approximation of @f$ L_T^{-1} r @f$ plus the CC mass term.
 *
 * The result approximates @f$ +S^{-1} r @f$ (like CahouetChabardSchurPC);
 * the block wrapper owns the sign. The inner FGMRES runs a FIXED number of
 * iterations (a tolerance-based inner solve cost 2-7x more Laplacian
 * applications in the spec's prototypes), so this preconditioner is
 * nonlinear: the outer solver must be flexible (bessemer's is FGMRES).
 */
class RotationalSchurPreconditioner : public mfem::Solver
{
public:
   /// Which Schur approximation to apply.
   enum class Mode
   {
      CahouetChabard, ///< Always the CC preconditioner.
      Tensor,         ///< Always the rotating-Darcy tensor version.
      Auto            ///< Switch with hysteresis on the rotation number.
   };
   /// What the Auto switch looks at.
   enum class Criterion
   {
      MaxMu,         ///< max mu over the domain.
      VolumeFraction ///< Fraction of the volume with mu > mu_on.
   };
   /// Switch and inner-solve options (spec 4; starting values, uncalibrated).
   struct Options
   {
      /// Mode. The spec's default is Auto; bessemer defaults to CC until
      /// the thresholds are calibrated on its own runs (spec 11, step 5).
      Mode mode = Mode::CahouetChabard;
      Criterion criterion = Criterion::MaxMu; ///< Auto criterion.
      double mu_on = 20.0;   ///< MaxMu: to Tensor when max mu > mu_on.
      double mu_off = 10.0;  ///< MaxMu: back to CC when max mu < mu_off.
      double vol_on = 1e-3;  ///< VolumeFraction: to Tensor above this.
      double vol_off = 2.5e-4; ///< VolumeFraction: back to CC below this.
      /// Fixed FGMRES iterations on L_T. The spec's 3 was calibrated with
      /// EXACT Laplacian solves; with bessemer's one LOR-AMG V-cycle per
      /// iteration, 10 (the CC inner count, so a tensor application costs
      /// about one CC application) is what beats CC at every rotation number
      /// measured (S3 table in CLAUDE.md, 2026-10-06).
      int inner_iterations = 10;
      bool remove_mean = false; ///< Pure-Neumann pressure (constant mode).
   };

   /**
    * @brief Bind the pieces; the tensor form is not assembled until the
    *        first Update() that activates the tensor mode.
    * @param pfes        Pressure space (borrowed).
    * @param p_ess_tdofs Pressure essential true dofs of L_T -- the list the
    *                    Laplacian solver uses (borrowed; must outlive this).
    * @param ir          Quadrature rule of L_T (borrowed; RuleBook/IntRules).
    * @param w_star      Lagged velocity of the rotation term (borrowed).
    * @param alpha       The rotation integrator's alpha.
    * @param nu          Coefficient of the mass term (CC's nu_pc).
    * @param cc          The Cahouet-Chabard preconditioner (borrowed).
    * @param mass_inv    Pressure-mass inverse (borrowed).
    * @param lap_inv     Approximate inverse of the UNSCALED pressure
    *                    Laplacian -div(grad) (borrowed; its SetOperator is
    *                    never called).
    * @param opt         Options (copied).
    */
   RotationalSchurPreconditioner(mfem::ParFiniteElementSpace& pfes,
                                 const mfem::Array<int>& p_ess_tdofs,
                                 const mfem::IntegrationRule& ir,
                                 const mfem::GridFunction& w_star,
                                 mfem::real_t alpha, mfem::real_t nu,
                                 mfem::Solver& cc, mfem::Solver& mass_inv,
                                 mfem::Solver& lap_inv, const Options& opt);

   /**
    * @brief Once per time step, after w* is formed: decide the mode for this
    *        step; in tensor mode re-run the partial assembly of L_T with the
    *        current sigma and w*.
    * @param sigma Leading mass coefficient beta0/dt (> 0).
    * @param stats Rotation-number statistics of this step's w* (computed with
    *              threshold mu_on when the criterion is VolumeFraction).
    */
   void Update(mfem::real_t sigma, const RotationNumberStats& stats);

   /**
    * @brief Apply the active mode.
    * @param r Pressure residual (true dofs).
    * @param z Output (true dofs).
    */
   void Mult(const mfem::Vector& r, mfem::Vector& z) const override;

   /// No-op by design: the pieces are fixed at construction.
   void SetOperator(const mfem::Operator&) override {}

   /// @return True if this step uses the tensor mode.
   bool TensorActive() const { return tensor_active_; }
   /// @return Number of mode switches so far.
   int NumSwitches() const { return n_switches_; }
   /// @return The constrained @f$ \sigma L_T @f$ (null before the first
   /// tensor Update).
   const mfem::Operator* GetTensorOperator() const { return T_op_.Ptr(); }

private:
   /// Forwards Mult, never SetOperator (spec 5.7, item 3: MFEM's Krylov
   /// solvers forward SetOperator to their preconditioner, which would hand
   /// L_T to the Laplacian's AMG).
   class FixedSolver : public mfem::Solver
   {
   public:
      /**
       * @brief Wrap a solver.
       * @param s The wrapped solver (borrowed).
       */
      explicit FixedSolver(const mfem::Solver& s)
         : mfem::Solver(s.Height(), s.Width()), s_(s) {}
      /**
       * @brief y = S(x).
       * @param x Input.
       * @param y Output.
       */
      void Mult(const mfem::Vector& x, mfem::Vector& y) const override
      {
         s_.Mult(x, y);
      }
      /// Ignored: the wrapped solver keeps its own operator.
      void SetOperator(const mfem::Operator&) override {}

   private:
      const mfem::Solver& s_; ///< Wrapped solver (borrowed).
   };

   /**
    * @brief (Re)assemble @f$ \sigma L_T @f$ with @p sigma; form the
    *        constrained operator and wire the inner FGMRES on first use.
    * @param sigma Leading mass coefficient.
    */
   void AssembleTensor(mfem::real_t sigma);

   /**
    * @brief Remove the l2 mean of a true-dof vector (device; collective).
    * @param v Vector, modified in place.
    */
   void RemoveMean(mfem::Vector& v) const;

   const Options opt_;              ///< Options.
   const mfem::Array<int>& ess_;    ///< L_T essential true dofs (borrowed).
   const mfem::real_t nu_;          ///< Mass-term coefficient.
   mfem::Solver& cc_;               ///< CC preconditioner (borrowed).
   mfem::Solver& mass_inv_;         ///< Pressure-mass inverse (borrowed).
   FixedSolver lap_;                ///< lap_inv, SetOperator-proof.
   RotatingDarcyTensor T_coeff_;    ///< sigma * T (scale = sigma).
   std::unique_ptr<mfem::ParBilinearForm> T_form_; ///< PA form of sigma L_T.
   mfem::OperatorHandle T_op_;      ///< Constrained sigma L_T.
   mfem::real_t sigma_ = 1.0;       ///< sigma of the current step.
   std::unique_ptr<mfem::FGMRESSolver> inner_; ///< Fixed-count inner FGMRES.
   MPI_Comm comm_;                  ///< Pressure communicator.
   mutable mfem::Vector t_;         ///< Mass-term scratch (device).
   mutable mfem::Vector r0_;        ///< Mean-free right-hand side (device).
   bool tensor_active_ = false;     ///< Mode of the current step.
   int n_switches_ = 0;             ///< Mode switches so far.
};

} // namespace incns
