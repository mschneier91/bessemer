/**
 * @file mesh_motion_model.hpp
 * @brief The interface between MeshMotion and the model that extends a
 *        boundary displacement into the domain (linear elasticity today; a
 *        hyperelastic or harmonic model later), docs/mesh_motion.md.
 */
#pragma once

#include "mfem.hpp"

#include <functional>
#include <iosfwd>
#include <memory>

namespace incns
{

/**
 * @brief A model that, given the displacement of some boundaries and others
 *        held fixed, finds a displacement of the whole domain.
 *
 * A model is built on the REFERENCE mesh (MeshMotion's copy of the initial
 * configuration, which never moves) and solves for the total displacement
 * d(X, t) from it, so the prescribed data are functions of the reference
 * coordinates X and the time. Its space must be a continuous vector H1 space
 * on that mesh with the mesh's dimension as vdim; MeshMotion interpolates the
 * displacement onto the moving mesh's nodal space.
 *
 * Boundary conditions are set before the first Solve() and fixed after it.
 * Every boundary attribute MeshMotion passes is either fixed or displaced:
 * the model never sees a free boundary.
 */
class MeshMotionModel
{
public:
   virtual ~MeshMotionModel() = default;

   /// @return A short name for logs and summaries (e.g. "linear_elastic").
   virtual const char* Name() const = 0;

   /// @return The displacement space (vector H1 on the reference mesh).
   virtual mfem::ParFiniteElementSpace& Space() = 0;

   /// @param attr Boundary attribute held fixed (d = 0).
   virtual void AddFixed(int attr) = 0;

   /**
    * @param attr Boundary attribute with a prescribed displacement.
    * @param g    d(X, t) on it, evaluated on the reference mesh (borrowed;
    *             must outlive the solves); its SetTime is called by Solve().
    */
   virtual void AddDisplacement(int attr, mfem::VectorCoefficient& g) = 0;

   /**
    * @brief Find the displacement with the boundary data at time @p t.
    *        Collective.
    * @param t Time of the boundary data.
    * @return Whether the solve converged.
    */
   virtual bool Solve(double t) = 0;

   /**
    * @brief Solve() from an initial guess on Space(), to the tolerance a
    *        zero guess would have had. Collective. A model without an
    *        iterative solve may ignore the guess (the default).
    * @param t     Time of the boundary data.
    * @param guess Initial guess for the displacement, on Space().
    * @return Whether the solve converged.
    */
   virtual bool Solve(double t, const mfem::ParGridFunction& guess)
   {
      (void)guess;
      return Solve(t);
   }

   /// @return The displacement of the last Solve() (zero before the first).
   virtual const mfem::ParGridFunction& Displacement() const = 0;

   /// @brief Write the model's own state (a checkpoint); none by default.
   /// @param out Binary stream.
   virtual void SaveState(std::ostream& out) const { (void)out; }
   /// @brief Read what SaveState() wrote.
   /// @param in Binary stream.
   virtual void LoadState(std::istream& in) { (void)in; }

   /// @return Linear iterations of the last Solve() (all Newton steps' for a
   ///         nonlinear model).
   virtual int Iterations() const = 0;
};

/**
 * @brief Builds a model on MeshMotion's reference mesh.
 *
 * Arguments: the reference mesh (owned by MeshMotion, outlives the model) and
 * the displacement order MeshMotion asks for (the moving mesh's nodal order
 * unless set).
 */
using MeshMotionModelFactory =
   std::function<std::unique_ptr<MeshMotionModel>(mfem::ParMesh&, int)>;

} // namespace incns
