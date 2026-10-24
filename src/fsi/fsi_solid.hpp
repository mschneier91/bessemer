/**
 * @file fsi_solid.hpp
 * @brief The solid of a coupled run (fsi.solid): the small-strain
 *        LinearElasticity or the St. Venant-Kirchhoff Hyperelasticity behind
 *        one interface, so FsiCase drives either (docs/fsi.md).
 */
#pragma once

#include "solid/solid_checkpoint.hpp"
#include "mfem.hpp"

#include <memory>
#include <string>

namespace incns
{

/**
 * @brief What FsiCase asks of its solid: the boundary data, a step or a
 *        static solve, and the state kept at the start of a coupled step so
 *        that every subiteration steps from it.
 */
class FsiSolid
{
public:
   virtual ~FsiSolid() = default;

   /// @param attr Boundary attribute held fixed.
   virtual void AddFixed(int attr) = 0;
   /// @param attr Boundary attribute with a prescribed displacement.
   /// @param g    The displacement (borrowed).
   virtual void AddDisplacement(int attr, mfem::VectorCoefficient& g) = 0;
   /// @param attr Boundary attribute with a traction (per reference area).
   /// @param t    The traction (borrowed).
   virtual void AddTraction(int attr, mfem::VectorCoefficient& t) = 0;
   /// @param attr Boundary attribute with an elastic support.
   /// @param k    Its stiffness per reference area.
   virtual void AddSupport(int attr, double k) = 0;
   /// Keep Reaction() up to date in every solve.
   virtual void TrackReaction() = 0;
   /// @return The reaction at the Dirichlet true dofs, unweighted by the
   ///         time scheme.
   virtual const mfem::Vector& Reaction() const = 0;

   /// Start the dynamic march at t = 0 from rest. Collective.
   virtual void Initialize() = 0;
   /// @param dt Step size. @return Whether the solve converged. Collective.
   virtual bool Step(double dt) = 0;
   /// @return Whether the static solve converged. Collective.
   virtual bool Solve() = 0;
   /// Keep the current state (the start of a coupled step).
   virtual void SaveState() = 0;
   /// Return to the state of the last SaveState().
   virtual void RestoreState() = 0;
   /// Write the state into a checkpoint directory as `<name>.*`
   /// (solid/solid_checkpoint). Collective.
   virtual void WriteState(const std::string& dir, const std::string& name) = 0;
   /// Restore a state WriteState() wrote (after Initialize() when dynamic).
   /// Collective.
   virtual void ReadState(const std::string& dir, const std::string& name) = 0;

   /// @return The displacement space (vector H1, by nodes).
   virtual mfem::ParFiniteElementSpace& Space() = 0;
   /// @return The displacement.
   virtual mfem::ParGridFunction& Displacement() = 0;
   /// @param x A point of the reference configuration (the same on every
   ///          rank). @param u The displacement there. Collective.
   /// @return Whether an element contains the point.
   virtual bool DisplacementAt(const mfem::Vector& x, mfem::Vector& u) const = 0;
   /// @return max over the nodes of |u|. Collective.
   virtual double MaxDisplacement() const = 0;
   /// @return Global true dofs.
   virtual HYPRE_BigInt GlobalTrueDofs() const = 0;
   /// @return The displacement order.
   virtual int Order() const = 0;
   /// @return The last Step()'s local error estimate (StepError(); negative
   ///         without one).
   virtual double StepError() const = 0;
   /// @return The scale of StepError() (StepNorm()).
   virtual double StepNorm() const = 0;
   /// @return StepError()'s order in dt.
   virtual int StepErrorOrder() const = 0;
   /// @return CG iterations since construction.
   virtual long long TotalIterations() const = 0;
   /// @return Newton iterations since construction (0: a linear solid).
   virtual long long TotalNewtonIterations() const = 0;
   /// @return The model's deck name.
   virtual const char* Name() const = 0;
};

/**
 * @brief FsiSolid over a solver class with the shared API (LinearElasticity,
 *        Hyperelasticity): the methods have the same names and meanings.
 * @tparam T The solver.
 */
template <class T>
class FsiSolidAdapter : public FsiSolid
{
public:
   /**
    * @param solver The solver (owned).
    * @param name   The model's deck name.
    */
   FsiSolidAdapter(std::unique_ptr<T> solver, const char* name)
      : s_(std::move(solver)), name_(name) {}

   void AddFixed(int attr) override { s_->AddFixed(attr); }
   void AddDisplacement(int attr, mfem::VectorCoefficient& g) override
   {
      s_->AddDisplacement(attr, g);
   }
   void AddTraction(int attr, mfem::VectorCoefficient& t) override
   {
      s_->AddTraction(attr, t);
   }
   void AddSupport(int attr, double k) override { s_->AddSupport(attr, k); }
   void TrackReaction() override { s_->TrackReaction(); }
   const mfem::Vector& Reaction() const override { return s_->Reaction(); }
   void Initialize() override { s_->Initialize(0.0); }
   bool Step(double dt) override { return s_->Step(dt); }
   bool Solve() override { return s_->Solve(); }
   void SaveState() override { saved_ = s_->ExportState(); }
   void RestoreState() override { s_->ImportState(saved_); }
   void WriteState(const std::string& dir, const std::string& name) override
   {
      WriteSolidState(dir, name, *s_);
   }
   void ReadState(const std::string& dir, const std::string& name) override
   {
      ReadSolidState(dir, name, *s_);
   }
   mfem::ParFiniteElementSpace& Space() override { return s_->Space(); }
   mfem::ParGridFunction& Displacement() override { return s_->Displacement(); }
   bool DisplacementAt(const mfem::Vector& x, mfem::Vector& u) const override
   {
      return s_->DisplacementAt(x, u);
   }
   double MaxDisplacement() const override { return s_->MaxDisplacement(); }
   HYPRE_BigInt GlobalTrueDofs() const override { return s_->GlobalTrueDofs(); }
   int Order() const override { return s_->GetOptions().order; }
   double StepError() const override { return s_->StepError(); }
   double StepNorm() const override { return s_->StepNorm(); }
   int StepErrorOrder() const override { return s_->StepErrorOrder(); }
   long long TotalIterations() const override { return s_->TotalIterations(); }
   long long TotalNewtonIterations() const override { return NewtonOf(*s_); }
   const char* Name() const override { return name_; }

private:
   /// @param u The solver. @return The Newton count of a solver that has one.
   template <class U>
   static auto NewtonOf(const U& u) -> decltype(u.TotalNewtonIterations())
   {
      return u.TotalNewtonIterations();
   }
   /// @return 0 for a linear solver (no parameters used).
   template <class U, class... Rest>
   static long long NewtonOf(const U&, Rest...) { return 0; }

   std::unique_ptr<T> s_;      ///< The solver.
   const char* name_;          ///< See Name().
   typename T::State saved_;   ///< The state of the last SaveState().
};

} // namespace incns
