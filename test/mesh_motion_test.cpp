// Mesh motion (src/ale): the nodes follow x = X + d with d from a pluggable
// model, at np 1/2/4.
//  M1 a translation of every boundary is reproduced exactly with Jacobian
//     stiffening on (2D distorted, 3D), det F = 1; uniform elements get
//     stiffening factor 1;
//  M2 an affine motion is reproduced exactly without stiffening, on the
//     curved DFG mesh and a 3D box given nodes: det F = det(I + A) and the
//     moved volume is det(I + A) times the reference one;
//  M3 the DFG cylinder translated across the channel, the other boundaries
//     held by default: the cylinder's nodes stay on the moved circle, det F >
//     0, stiffening keeps the O-grid ring closer to rigid, and moving back
//     returns every node exactly;
//  M4 the mesh velocity (the BDF derivative of the nodes) is exact for a
//     motion quadratic in time at BDF2 and cubic at BDF3, on variable steps;
//  M5 a model that is not elasticity (a test model) plugs in through the
//     factory, and every boundary nobody named reaches it as fixed.
// These pin their own orders and tolerances (unit tests).

#include <gtest/gtest.h>

#include "ale/linear_elastic_motion.hpp"
#include "ale/mesh_motion.hpp"
#include "mesh/cylinder_channel.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

using namespace mfem;
using incns::LinearElasticMotion;
using incns::MeshMotion;

namespace
{

std::unique_ptr<ParMesh> Partition(Mesh& serial)
{
   return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

/// A unit box with smoothly distorted (non-affine) elements, no nodes.
std::unique_ptr<ParMesh> DistortedBox(int dim, int n)
{
   Mesh m = dim == 2 ?
            Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, true, 1, 1) :
            Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
   m.Transform([dim](const Vector & x, Vector & y)
   {
      y = x;
      y(0) += 0.08 * std::sin(M_PI * x(1)) * x(0);
      y(1) += 0.06 * std::sin(M_PI * x(0));
      if (dim == 3) { y(2) += 0.05 * x(0) * x(1); }
   });
   return Partition(m);
}

std::unique_ptr<ParMesh> CylinderChannel()
{
   incns::CylinderChannelSpec spec; // the DFG geometry, order 3
   Mesh m = incns::MakeCylinderChannelMesh(spec);
   return Partition(m);
}

LinearElasticMotion::Options Elastic(double chi)
{
   LinearElasticMotion::Options o;
   o.stiffening = chi;
   o.rtol = 1e-12;
   o.max_iter = 5000;
   return o;
}

/// Every boundary attribute of the mesh.
std::vector<int> AllAttributes(const ParMesh& mesh)
{
   std::vector<int> a;
   for (int i = 0; i < mesh.bdr_attributes.Size(); ++i)
   {
      a.push_back(mesh.bdr_attributes[i]);
   }
   return a;
}

/// max over the nodes of |x - (X + d(X))|, X the reference nodes. Collective.
double NodeError(const ParMesh& mesh, const Vector& x0,
                 const std::function<void(const Vector&, Vector&)>& d)
{
   const GridFunction& nodes = *mesh.GetNodes();
   const FiniteElementSpace& fes = *nodes.FESpace();
   const int sdim = fes.GetVDim();
   nodes.HostRead(); // operator() reads the host copies without syncing
   x0.HostRead();
   Vector X(sdim), dv(sdim);
   double err = 0.0;
   for (int i = 0; i < fes.GetNDofs(); ++i)
   {
      for (int c = 0; c < sdim; ++c) { X(c) = x0(fes.DofToVDof(i, c)); }
      d(X, dv);
      for (int c = 0; c < sdim; ++c)
      {
         const double x = nodes(fes.DofToVDof(i, c));
         err = std::max(err, std::abs(x - X(c) - dv(c)));
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   return err;
}

/// Global volume of the mesh in its current configuration. Collective.
double Volume(ParMesh& mesh)
{
   double v = 0.0;
   for (int e = 0; e < mesh.GetNE(); ++e) { v += mesh.GetElementVolume(e); }
   MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   return v;
}

/// A model that is not elasticity: it interpolates a given field d(X, t)
/// everywhere and records the boundaries it was handed.
class InterpolatingModel : public incns::MeshMotionModel
{
public:
   InterpolatingModel(ParMesh& reference, int order, VectorCoefficient& field)
      : fec_(order, reference.Dimension()),
        fes_(&reference, &fec_, reference.Dimension()), d_(&fes_),
        field_(field)
   {
      d_ = 0.0;
   }
   const char* Name() const override { return "interpolating"; }
   ParFiniteElementSpace& Space() override { return fes_; }
   void AddFixed(int attr) override { fixed.push_back(attr); }
   void AddDisplacement(int attr, VectorCoefficient&) override
   {
      moving.push_back(attr);
   }
   bool Solve(double t) override
   {
      field_.SetTime(t);
      d_.ProjectCoefficient(field_);
      return true;
   }
   const ParGridFunction& Displacement() const override { return d_; }
   int Iterations() const override { return 0; }

   std::vector<int> fixed;  ///< Attributes handed as fixed.
   std::vector<int> moving; ///< Attributes handed as displaced.

private:
   H1_FECollection fec_;
   ParFiniteElementSpace fes_;
   ParGridFunction d_;
   VectorCoefficient& field_;
};

} // namespace

TEST(MeshMotion, M1_TranslationIsExactWithStiffening)
{
   for (int dim : {2, 3})
   {
      auto mesh = DistortedBox(dim, dim == 2 ? 4 : 2);
      MeshMotion::Options mo;
      mo.order = dim == 2 ? 3 : 2; // the mesh has no nodes: given at order
      MeshMotion motion(*mesh, mo, LinearElasticMotion::Factory(Elastic(1.0)));
      ASSERT_NE(mesh->GetNodes(), nullptr);
      EXPECT_EQ(motion.Order(), mo.order);
      auto& model = static_cast<LinearElasticMotion&>(motion.Model());
      // Distorted elements: the factors are not all 1.
      double fr[2] = {model.Factors().Size() ? model.Factors().Min() : 1e300,
                      model.Factors().Size() ? -model.Factors().Max() : 1e300
                     };
      MPI_Allreduce(MPI_IN_PLACE, fr, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      EXPECT_LT(fr[0], -fr[1]) << "dim " << dim;

      Vector c(dim);
      for (int i = 0; i < dim; ++i) { c(i) = 0.05 * (i + 1); }
      VectorConstantCoefficient g(c);
      for (int a : AllAttributes(*mesh)) { motion.AddDisplacement(a, g); }
      ASSERT_TRUE(motion.Move(1.0));
      const double err = NodeError(*mesh, motion.ReferenceNodes(),
      [&c](const Vector&, Vector & d) { d = c; });
      EXPECT_LT(err, 1e-9) << "dim " << dim;
      double jmin, jmax;
      motion.JacobianRange(jmin, jmax);
      EXPECT_NEAR(jmin, 1.0, 1e-9) << "dim " << dim;
      EXPECT_NEAR(jmax, 1.0, 1e-9) << "dim " << dim;
   }

   // A uniform mesh: every factor is 1.
   Mesh m = Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL, true, 2.0, 2.0);
   auto mesh = Partition(m);
   const Vector f = LinearElasticMotion::StiffeningFactors(*mesh, 1.0);
   for (int e = 0; e < f.Size(); ++e) { EXPECT_NEAR(f(e), 1.0, 1e-13); }
}

TEST(MeshMotion, M2_AffineMotionIsExactWithoutStiffening)
{
   for (int dim : {2, 3})
   {
      auto mesh = dim == 2 ? CylinderChannel() : DistortedBox(3, 2);
      MeshMotion::Options mo;
      mo.order = dim == 2 ? 0 : 2; // 2D: the mesh's own order 3
      MeshMotion motion(*mesh, mo, LinearElasticMotion::Factory(Elastic(0.0)));
      EXPECT_EQ(motion.Order(), dim == 2 ? 3 : 2);
      const double v0 = Volume(*mesh);

      DenseMatrix A(dim);
      Vector b(dim);
      for (int i = 0; i < dim; ++i)
      {
         b(i) = 0.02 * (i + 1);
         for (int j = 0; j < dim; ++j) { A(i, j) = 0.03 * (i + 1) - 0.02 * j; }
      }
      auto affine = [&A, &b](const Vector & X, Vector & d)
      {
         d.SetSize(X.Size());
         A.Mult(X, d);
         d += b;
      };
      VectorFunctionCoefficient g(dim, affine);
      for (int a : AllAttributes(*mesh)) { motion.AddDisplacement(a, g); }
      ASSERT_TRUE(motion.Move(1.0));
      EXPECT_LT(NodeError(*mesh, motion.ReferenceNodes(), affine), 1e-9)
            << "dim " << dim;

      DenseMatrix F(A);
      for (int i = 0; i < dim; ++i) { F(i, i) += 1.0; }
      const double detF = F.Det();
      double jmin, jmax;
      motion.JacobianRange(jmin, jmax);
      EXPECT_NEAR(jmin, detF, 1e-9) << "dim " << dim;
      EXPECT_NEAR(jmax, detF, 1e-9) << "dim " << dim;
      // The geometry caches were refreshed: volumes see the moved nodes.
      EXPECT_NEAR(Volume(*mesh), detF * v0, 1e-10 * v0) << "dim " << dim;
   }
}

TEST(MeshMotion, M3_CylinderTranslatedAcrossTheChannel)
{
   incns::CylinderChannelSpec spec;
   const double R = spec.radius;
   Vector delta(2);
   delta(0) = 0.01;
   delta(1) = 0.03; // a fifth of the gap above the cylinder
   // Out to delta at t = 1, back to rest at t = 2.
   auto ramp = [&delta](const Vector&, double t, Vector & d)
   {
      d = delta;
      d *= t <= 1.0 ? t : 2.0 - t;
   };

   // The largest |det F - 1| over the O-grid ring's elements (the smallest).
   auto ring_strain = [&spec](MeshMotion & m, ParMesh & cur)
   {
      ParMesh& ref = m.ReferenceMesh();
      IntegrationRules gll(0, Quadrature1D::GaussLobatto);
      double s = 0.0;
      Vector c;
      for (int e = 0; e < ref.GetNE(); ++e)
      {
         ref.GetElementCenter(e, c);
         if (std::max(std::abs(c(0) - spec.cx), std::abs(c(1) - spec.cy)) >
             spec.box_half) { continue; }
         const IntegrationRule& ir = gll.Get(Geometry::SQUARE, 5);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            ElementTransformation* T = cur.GetElementTransformation(e);
            T->SetIntPoint(&ir.IntPoint(q));
            const double w = T->Weight();
            ElementTransformation* T0 = ref.GetElementTransformation(e);
            T0->SetIntPoint(&ir.IntPoint(q));
            s = std::max(s, std::abs(w / T0->Weight() - 1.0));
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      return s;
   };

   double strain[2];
   for (int k = 0; k < 2; ++k)
   {
      const double chi = k == 0 ? 0.0 : 1.0;
      auto mesh = CylinderChannel();
      MeshMotion motion(*mesh, MeshMotion::Options(),
                        LinearElasticMotion::Factory(Elastic(chi)));
      VectorFunctionCoefficient g(2, ramp);
      // Only the body is named: inflow, outflow and walls stay by default.
      motion.AddDisplacement(incns::kCylinderBody, g);
      ASSERT_TRUE(motion.Move(1.0)) << "chi " << chi;

      // The body's nodes lie on the moved circle, the walls' did not move.
      const GridFunction& nodes = *mesh->GetNodes();
      const FiniteElementSpace& fes = *nodes.FESpace();
      const Vector& x0 = motion.ReferenceNodes();
      nodes.HostRead();
      x0.HostRead();
      double circle = 0.0, walls = 0.0;
      Array<int> vdofs;
      for (int be = 0; be < mesh->GetNBE(); ++be)
      {
         fes.GetBdrElementVDofs(be, vdofs);
         const int nd = vdofs.Size() / 2;
         const int attr = mesh->GetBdrAttribute(be);
         for (int i = 0; i < nd; ++i)
         {
            const double x = nodes(vdofs[i]), y = nodes(vdofs[nd + i]);
            if (attr == incns::kCylinderBody)
            {
               const double rx = x - spec.cx - delta(0);
               const double ry = y - spec.cy - delta(1);
               const double r = std::hypot(rx, ry);
               circle = std::max(circle, std::abs(r - R));
            }
            else
            {
               const double dx = std::abs(x - x0(vdofs[i]));
               const double dy = std::abs(y - x0(vdofs[nd + i]));
               walls = std::max(walls, std::max(dx, dy));
            }
         }
      }
      double e[2] = {circle, walls};
      MPI_Allreduce(MPI_IN_PLACE, e, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_LT(e[0], 1e-12) << "chi " << chi;
      EXPECT_LT(e[1], 1e-14) << "chi " << chi;

      double jmin, jmax;
      motion.JacobianRange(jmin, jmax);
      EXPECT_GT(jmin, 0.0) << "chi " << chi;
      strain[k] = ring_strain(motion, *mesh);
      if (Mpi::Root())
      {
         std::cout << "M3 chi " << chi << ": det F in [" << jmin << ", "
                   << jmax << "], ring max |det F - 1| " << strain[k]
                   << ", CG " << motion.Model().Iterations() << std::endl;
      }

      // Back to rest: every node returns exactly (total displacement).
      ASSERT_TRUE(motion.Move(2.0));
      const double back = NodeError(*mesh, x0, [](const Vector & X, Vector & d)
      {
         d.SetSize(X.Size());
         d = 0.0;
      });
      EXPECT_LT(back, 1e-14) << "chi " << chi;
   }
   // Stiffening: the small elements at the body move closer to rigidly.
   // Measured in CI (np 1, 2, 4): ring strain 0.247 -> 0.159, a ratio of 0.64.
   EXPECT_LT(strain[1], 0.75 * strain[0]);
}

TEST(MeshMotion, M4_MeshVelocityIsTheExactBdfDerivative)
{
   const int dim = 2;
   DenseMatrix A(dim);
   A(0, 0) = 0.1;
   A(0, 1) = -0.05;
   A(1, 0) = 0.04;
   A(1, 1) = 0.07;
   Vector b(dim);
   b(0) = 0.03;
   b(1) = -0.02;
   // d(X, t) = (t^2 + t^3) A X + t b: affine in X, so the model is exact.
   auto d = [&](const Vector & X, double t, Vector & v)
   {
      v.SetSize(dim);
      A.Mult(X, v);
      v *= t * t + t * t * t;
      v.Add(t, b);
   };
   auto w_exact = [&](const Vector & X, double t, int order, Vector & v)
   {
      v.SetSize(dim);
      A.Mult(X, v);
      // BDF2 is exact for the quadratic part only: drop the cubic term.
      double s = 2.0 * t;
      if (order >= 3) { s += 3.0 * t * t; }
      v *= s;
      v += b;
   };

   for (int order : {2, 3})
   {
      auto mesh = DistortedBox(dim, 3);
      MeshMotion::Options mo;
      mo.order = 2;
      MeshMotion motion(*mesh, mo, LinearElasticMotion::Factory(Elastic(0.0)));
      // order 2: the cubic term is left out of the motion so BDF2 is exact.
      auto field = [&, order](const Vector & X, double t, Vector & v)
      {
         d(X, t, v);
         if (order == 2)
         {
            Vector c(dim);
            A.Mult(X, c);
            v.Add(-t * t * t, c);
         }
      };
      VectorFunctionCoefficient g(dim, field);
      for (int a : AllAttributes(*mesh)) { motion.AddDisplacement(a, g); }
      const double t1 = 0.1, t2 = 0.25, t3 = 0.33;
      for (double t : {t1, t2, t3}) { ASSERT_TRUE(motion.Move(t)); }
      ASSERT_EQ(motion.States(), 4);

      Vector w;
      motion.MeshVelocity(order, w);
      const FiniteElementSpace& fes = *mesh->GetNodes()->FESpace();
      const Vector& x0 = motion.ReferenceNodes();
      x0.HostRead();
      w.HostRead();
      Vector X(dim), we;
      double err = 0.0, ref = 0.0;
      for (int i = 0; i < fes.GetNDofs(); ++i)
      {
         for (int c = 0; c < dim; ++c) { X(c) = x0(fes.DofToVDof(i, c)); }
         w_exact(X, t3, order, we);
         for (int c = 0; c < dim; ++c)
         {
            err = std::max(err, std::abs(w(fes.DofToVDof(i, c)) - we(c)));
            ref = std::max(ref, std::abs(we(c)));
         }
      }
      double e[2] = {err, -ref};
      MPI_Allreduce(MPI_IN_PLACE, e, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      EXPECT_LT(e[0], 1e-8 * -e[1]) << "BDF" << order;
   }
}

TEST(MeshMotion, M5_AnyModelPlugsIn)
{
   const int dim = 2;
   auto mesh = DistortedBox(dim, 3);
   // A cubic field: exact in the order-3 nodal space.
   auto field = [](const Vector & X, double t, Vector & v)
   {
      v.SetSize(2);
      v(0) = 0.01 * t * X(0) * X(1) * X(1);
      v(1) = -0.02 * t * X(0) * X(0) * X(0);
   };
   VectorFunctionCoefficient phi(dim, field);
   InterpolatingModel* model = nullptr;
   incns::MeshMotionModelFactory factory = [&](ParMesh & ref, int order)
   {
      auto m = std::make_unique<InterpolatingModel>(ref, order, phi);
      model = m.get();
      return std::unique_ptr<incns::MeshMotionModel>(std::move(m));
   };
   MeshMotion::Options mo;
   mo.geometry_order = 3; // order 0: the nodal order, 3
   MeshMotion motion(*mesh, mo, factory);
   ASSERT_NE(model, nullptr);
   EXPECT_EQ(motion.Order(), 3);

   const std::vector<int> all = AllAttributes(*mesh);
   ASSERT_GE(all.size(), 3u);
   motion.AddDisplacement(all[0], phi);
   motion.AddFixed(all[1]);
   ASSERT_TRUE(motion.Move(0.5));
   EXPECT_EQ(model->moving, std::vector<int>({all[0]}));
   std::vector<int> fixed = model->fixed;
   std::sort(fixed.begin(), fixed.end());
   EXPECT_EQ(fixed, std::vector<int>(all.begin() + 1, all.end()));

   const double err = NodeError(*mesh, motion.ReferenceNodes(),
   [&field](const Vector & X, Vector & v) { field(X, 0.5, v); });
   EXPECT_LT(err, 1e-14);
}
