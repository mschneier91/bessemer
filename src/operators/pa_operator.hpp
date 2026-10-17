/**
 * @file pa_operator.hpp
 * @brief Partially assembled operators built from MFEM's public pieces (the
 *        integrators' AssemblePA / AddMultPA and the element restrictions),
 *        with their element vectors lent by an ElementScratch.
 */
#pragma once

#include "operators/element_scratch.hpp"
#include "mfem.hpp"

#include <memory>
#include <vector>

namespace incns
{

/**
 * @brief y = sum_i R^T D_i R x on the local dofs of one space, for domain
 *        integrators on tensor elements.
 *
 * The steps of MFEM's partially assembled BilinearForm
 * (PABilinearFormExtension::Mult): restrict, zero the output element vector,
 * each integrator's AddMultPA, scatter back; so the results are bitwise
 * MFEM's. The difference is where the element vectors live: lent by an
 * ElementScratch, not a pair per operator for the run.
 */
class PaOperator : public mfem::Operator
{
public:
   /**
    * @param fes     The space (borrowed).
    * @param scratch Element-vector buffers (borrowed; must outlive this).
    */
   PaOperator(mfem::ParFiniteElementSpace& fes, ElementScratch& scratch);

   /// Add a domain integrator (owned).
   void AddDomainIntegrator(mfem::BilinearFormIntegrator* integ);

   /// Partially assemble every integrator.
   void Assemble();

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
   void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

   /// @return The space's prolongation (true dofs -> local dofs).
   const mfem::Operator* GetProlongation() const
   {
      return fes_.GetProlongationMatrix();
   }

private:
   mfem::ParFiniteElementSpace& fes_; ///< The space (borrowed).
   const mfem::Operator* restr_ = nullptr; ///< Lexicographic restriction.
   std::vector<std::unique_ptr<mfem::BilinearFormIntegrator>> integs_;
   mutable mfem::Vector xe_; ///< Input element vector (lent).
   mutable mfem::Vector ye_; ///< Output element vector (lent).
};

/**
 * @brief The mixed version, trial space -> test space: the steps of MFEM's
 *        PAMixedBilinearFormExtension, with the element vectors lent and the
 *        result scattered straight into y (MFEM scatters into a further
 *        full-size vector and adds it to a zeroed y).
 */
class PaMixedOperator : public mfem::Operator
{
public:
   /**
    * @param trial   Trial space (borrowed).
    * @param test    Test space (borrowed).
    * @param scratch Element-vector buffers (borrowed; must outlive this).
    */
   PaMixedOperator(mfem::ParFiniteElementSpace& trial,
                   mfem::ParFiniteElementSpace& test, ElementScratch& scratch);

   /// Add a domain integrator (owned).
   void AddDomainIntegrator(mfem::BilinearFormIntegrator* integ);

   /// Partially assemble every integrator.
   void Assemble();

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override;
   void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

   /// @return The trial space's prolongation.
   const mfem::Operator* GetProlongation() const
   {
      return trial_.GetProlongationMatrix();
   }
   /// @return The test space's prolongation.
   const mfem::Operator* GetOutputProlongation() const
   {
      return test_.GetProlongationMatrix();
   }

private:
   mfem::ParFiniteElementSpace& trial_; ///< Trial space (borrowed).
   mfem::ParFiniteElementSpace& test_;  ///< Test space (borrowed).
   const mfem::Operator* restr_trial_ = nullptr; ///< Its restriction.
   const mfem::Operator* restr_test_ = nullptr;  ///< Its restriction.
   std::vector<std::unique_ptr<mfem::BilinearFormIntegrator>> integs_;
   mutable mfem::Vector trial_e_; ///< Trial-side element vector (lent).
   mutable mfem::Vector test_e_;  ///< Test-side element vector (lent).
};

} // namespace incns
