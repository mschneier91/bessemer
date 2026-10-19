/**
 * @file oifs_convection.hpp
 * @brief The OIFS substeps' convection, sum_l w_l (u_l . grad) x_c for every
 *        velocity component c at once, with the advecting velocities u_l
 *        combined at each quadrature point (3D).
 */
#pragma once

#include "operators/element_scratch.hpp"
#include "mfem.hpp"

#include <vector>

namespace incns
{

/**
 * @brief y_c = sum_l w_l C(u_l) x_c for every component c of a velocity-space
 *        vector, where C(u) is the partially assembled form (u . grad phi, v)
 *        of MFEM's ConvectionIntegrator (alpha = 1) on the scalar companion
 *        space, and up to three advecting velocities u_l are set by SetWind()
 *        and weighted per Mult().
 *
 * The form is linear in the wind, so the quadrature data of the u_l
 * (W adj(J) u_l at each point, as MFEM's PAConvectionSetup3D computes it) is
 * combined with the weights at each point inside one kernel, which then
 * applies it to all three components: one kernel and one pass over the wind
 * data per apply, where OifsAdvector ran one MFEM apply per component and wind
 * (nine per RK4 stage with three winds). The results agree with that sum to
 * roundoff (oifs_convection_test).
 *
 * 3D only, for the (dofs, points) per direction the solver uses with velocity
 * orders 2 to 4 under OIFS's 3k rule (Supported()); OifsAdvector keeps MFEM's
 * forms otherwise.
 */
class OifsConvection
{
public:
   /**
    * @brief Whether this space and rule have a kernel: 3D, vdim 3, byNODES,
    *        tensor elements with (dofs, points) per direction (3, 4), (4, 5) or
    *        (5, 7).
    * @param vfes Velocity space.
    * @param ir   Quadrature rule of the convection forms.
    * @return True when OifsConvection can be built for them.
    */
   static bool Supported(const mfem::ParFiniteElementSpace& vfes,
                         const mfem::IntegrationRule& ir);

   /**
    * @brief Bind to the velocity space and the rule (Supported() must hold).
    * @param vfes    Velocity space (borrowed).
    * @param ir      Quadrature rule (borrowed).
    * @param scratch Element-vector buffers for the wind, input and output
    *                element vectors (borrowed; must outlive this); null:
    *                its own.
    * @param on_the_fly Rebuild the quadrature data at every apply instead of
    *                storing it, where OnTheFlyEligible() (deck
    *                time.oifs_on_the_fly): 5 GiB (93 B/DOF) less at Q6
    *                on 42^3 elements than the stored default, which keeps
    *                only op_; the apply ~40 % slower (+6.5 % per step).
    * @param self_only Only MultSelf() will be called (Convection): no wind
    *                data is kept. On curved elements at 9 or more points per
    *                direction the self apply then forms the Jacobians and the
    *                wind's quadrature data chunk by chunk; on straight-sided
    *                ones it needs @p on_the_fly.
    */
   OifsConvection(mfem::ParFiniteElementSpace& vfes,
                  const mfem::IntegrationRule& ir,
                  ElementScratch* scratch = nullptr, bool on_the_fly = false,
                  bool self_only = false);

   /**
    * @brief Whether the on-the-fly path is available: Supported(), 9 or more
    *        points per direction (velocity orders 6 and 7 at the 3k rule),
    *        linear mesh nodes (straight-sided elements).
    * @param vfes Velocity space.
    * @param ir   Quadrature rule.
    * @return True when the constructor's on_the_fly takes effect.
    */
   static bool OnTheFlyEligible(const mfem::ParFiniteElementSpace& vfes,
                                const mfem::IntegrationRule& ir);

   /**
    * @brief Set advecting velocity @p l: its quadrature data, or (the
    *        on-the-fly path) a reference to it.
    * @param l Index, 0 to 2.
    * @param u The velocity (on the bound space, local dofs valid). On the
    *          on-the-fly path it is read at every Mult(): it must stay alive
    *          and unchanged until the next SetWind(l).
    */
   void SetWind(int l, const mfem::ParGridFunction& u);

   /**
    * @brief y := sum_{l < w.size()} w[l] C(u_l) x, component by component.
    * @param w Weights, at most 3, of winds already set.
    * @param x Input, velocity true dofs.
    * @param y Output, velocity true dofs (overwritten).
    */
   void Mult(const std::vector<double>& w, const mfem::Vector& x,
             mfem::Vector& y);

   /**
    * @brief y := C(x) x: the dealiased (x . grad) x tested with v, the
    *        convective term N(x) of the IMEX scheme. The wind is the input's
    *        own local dofs; no SetWind() needed. The on-the-fly path, or
    *        (self_only) curved elements chunk by chunk.
    * @param x Input, velocity true dofs.
    * @param y Output, velocity true dofs (overwritten).
    */
   void MultSelf(const mfem::Vector& x, mfem::Vector& y);

   /**
    * @brief Elements per chunk of the curved-element Jacobians (default
    *        2048: a fixed buffer, about 0.2 GB at 11 points per direction,
    *        whatever the mesh). The results do not depend on it (tests).
    * @param elements At least 1.
    */
   void SetChunk(int elements);

   /**
    * @brief MultSelf on the given local elements only (call once, before
    *        MultSelf; no Mult or SetWind afterwards): the result is then
    *        exact at every dof all of whose elements are given -- the same
    *        values per element, summed in ElementRestriction::MultTranspose's
    *        order -- and partial elsewhere. OIFS's wall nodes
    *        (Convection::RestrictToTrueDofs).
    * @param elems Local elements, sorted, each once (ElementsTouching()).
    */
   void RestrictToElements(const mfem::Array<int>& elems);

   /**
    * @brief The local elements holding a local copy of any of the given true
    *        dofs (collective). On a conforming mesh the prolongation copies a
    *        true dof to each of its local dofs, so these are every element
    *        whose values reach those true dofs.
    * @param vfes  Velocity space.
    * @param tdofs True vdofs.
    * @return The elements, sorted.
    */
   static mfem::Array<int> ElementsTouching(const mfem::ParFiniteElementSpace&
         vfes,
         const mfem::Array<int>& tdofs);

private:
   /// jchunk_ := the Jacobians at the points of elements [e0, e0 + n) from
   /// their nodes (curved elements, Columns()).
   void CurvedJacobians(int e0, int n);
   mfem::ParFiniteElementSpace& vfes_;       ///< Velocity space (borrowed).
   const mfem::IntegrationRule& ir_;         ///< Rule (borrowed).
   const mfem::Operator* restr_ = nullptr;   ///< Lexicographic E-restriction.
   /// True -> local dofs (null when it is the identity: one rank).
   const mfem::Operator* prolong_ = nullptr;
   const mfem::DofToQuad* maps_ = nullptr;   ///< 1D maps at ir_ (not owned).
   const mfem::GeometricFactors* geom_ = nullptr; ///< Jacobians (mesh cache).
   /// The Jacobians SetWind reads below 9 points per direction (the mesh's).
   const mfem::Vector* j_ = nullptr;
   /// Curved elements at 9 or more points: their nodes' element vector
   /// (nd1^3, 3, NE) and 1D maps; the Jacobians are formed from them chunk
   /// by chunk (jchunk_, (nq, 3, 3, chunk)), never kept for the mesh.
   mfem::Vector xnodes_;
   const mfem::DofToQuad* cmaps_ = nullptr;
   mfem::Vector jchunk_;
   /// self_only on curved elements: one chunk's wind data (nq, 3, chunk).
   mfem::Vector opchunk_;
   int chunk_ = 2048;
   bool self_ = false; ///< Curved, self_only: MultSelf by chunks, no op_.
   /// Q1D > 8 (velocity orders 6 and 7 at the 3k rule): MFEM's quadrature
   /// interpolator has no compile-time kernel for the velocity there, its
   /// runtime-size ones hold 8 points per direction (DofQuadLimits::
   /// MAX_INTERP_1D on CUDA) and launch a thread per point (more than a block
   /// holds at Q1D 11). The Jacobians and the winds' values are then computed
   /// here, a thread per (x, y) column.
   bool Columns() const { return q1_ > 8; }
   int ne_ = 0;  ///< Local elements.
   int d1_ = 0;  ///< Dofs per direction.
   int q1_ = 0;  ///< Points per direction.
   int set_ = 0; ///< Winds set (bit l: SetWind(l) called).
   /// Quadrature data of the three winds, (Q1D^3, 3, NE, 3) (empty on the
   /// on-the-fly path).
   mfem::Vector op_;
   /// Columns() on straight-sided elements (linear nodes): the Jacobians come
   /// from the element vertices (xn_); curved ones form theirs by chunks.
   bool linear_ = false;
   /// On-the-fly path (opt-in, linear_): every apply reads the winds from
   /// their local dofs and forms the Jacobians from the vertices, so op_ is
   /// not kept either (11.6 GiB less than the first stored path at Q6 on 42^3
   /// elements, the apply 42 % slower).
   bool fly_ = false;
   /// The winds set by SetWind (borrowed: unchanged until the next SetWind of
   /// the same index; on-the-fly path).
   const mfem::Vector* wind_[3] = {nullptr, nullptr, nullptr};
   const mfem::Array<int>* gather_ = nullptr; ///< The restriction's map.
   int ndofs_ = 0; ///< Scalar local dofs: the component stride.
   const mfem::DofToQuad* nmaps_ = nullptr; ///< Linear node maps at ir_.
   mfem::Vector xn_; ///< Element vertices, (2, 2, 2, 3, NE).
   mfem::Vector ue_; ///< Wind, element vector (scratch).
   mfem::Vector uq_; ///< Wind at the points, (3, NQ, NE) (scratch).
   mfem::Vector xl_; ///< Input, local dofs (scratch).
   mfem::Vector xe_; ///< Input, element vector (scratch).
   mfem::Vector ye_; ///< Output, element vector (scratch).
   mfem::Vector yl_; ///< Output, local dofs (scratch).
   /// RestrictToElements: MultSelf on ne_ of the elements (subset_), their
   /// gather map (gather_ points to it), and per local scalar dof its
   /// (element, node) entries s nd + d in MultTranspose's order (CSR:
   /// sub_off_, sub_idx_). xn_ / xnodes_ then hold the subset's blocks.
   bool subset_ = false;
   mfem::Array<int> sub_gather_, sub_off_, sub_idx_;
};

} // namespace incns
