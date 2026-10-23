// The DFG cylinder turned about its centre: the motion of the hyperelastic
// mesh-motion tests (hyperelasticity_test H9, verify_hyperelasticity_test
// V3), and checks of the moved nodes.
#pragma once

#include "mesh/cylinder_channel.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

namespace turned_cylinder
{

/// (Q(theta) - I)(X - c): the displacement of X turned by theta about the
/// cylinder's centre c.
inline void Turn(const incns::CylinderChannelSpec& s, double theta,
                 const mfem::Vector& X,
                 mfem::Vector& d)
{
   const double x = X(0) - s.cx, y = X(1) - s.cy;
   d.SetSize(2);
   d(0) = (std::cos(theta) - 1.0) * x - std::sin(theta) * y;
   d(1) = std::sin(theta) * x + (std::cos(theta) - 1.0) * y;
}

/// theta(t) = t (radians) up to @p top, then back down.
inline double TurnAngle(double t, double top) { return t <= top ? t : 2.0 * top - t; }

/**
 * max over the boundary nodes on the body (@p body) or off it of |x - X -
 * d(X)|, d the turn by @p theta on the body and zero elsewhere. Collective.
 */
inline double BoundaryNodeError(mfem::ParMesh& mesh, const mfem::Vector& x0,
                                const incns::CylinderChannelSpec& s, double theta,
                                bool body)
{
   const mfem::GridFunction& nodes = *mesh.GetNodes();
   const mfem::FiniteElementSpace& fes = *nodes.FESpace();
   nodes.HostRead(); // operator() reads the host copies without syncing
   x0.HostRead();
   mfem::Array<int> vdofs;
   mfem::Vector X(2), d(2);
   double err = 0.0;
   for (int be = 0; be < mesh.GetNBE(); ++be)
   {
      if ((mesh.GetBdrAttribute(be) == incns::kCylinderBody) != body) { continue; }
      fes.GetBdrElementVDofs(be, vdofs);
      const int nd = vdofs.Size() / 2;
      for (int i = 0; i < nd; ++i)
      {
         X(0) = x0(vdofs[i]);
         X(1) = x0(vdofs[nd + i]);
         if (body) { Turn(s, theta, X, d); }
         else { d = 0.0; }
         err = std::max(err, std::abs(nodes(vdofs[i]) - X(0) - d(0)));
         err = std::max(err, std::abs(nodes(vdofs[nd + i]) - X(1) - d(1)));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return err;
}

/// max over the nodes of |x - X|. Collective.
inline double RestError(mfem::ParMesh& mesh, const mfem::Vector& x0)
{
   const mfem::GridFunction& nodes = *mesh.GetNodes();
   nodes.HostRead();
   x0.HostRead();
   double err = 0.0;
   for (int i = 0; i < nodes.Size(); ++i)
   {
      err = std::max(err, std::abs(nodes(i) - x0(i)));
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return err;
}

/// The DFG channel mesh (the cylinder, no flag).
inline std::unique_ptr<mfem::ParMesh> CylinderChannel(const
      incns::CylinderChannelSpec& s)
{
   mfem::Mesh m = incns::MakeCylinderChannelMesh(s);
   return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, m);
}

} // namespace turned_cylinder
