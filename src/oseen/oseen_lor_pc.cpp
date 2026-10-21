/**
 * @file oseen_lor_pc.cpp
 * @brief OseenLorPreconditioner: the LOR operator (host assembly) and its
 *        AIR / classical AMG V-cycles.
 */
#include "oseen/oseen_lor_pc.hpp"

#include "oseen/oseen_wind.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace incns
{

using namespace mfem;

namespace
{

/// tau(w(x), J^-1 of the parent) w w^T on the LOR mesh (host evaluation).
class LorStreamlineDiffusion : public MatrixCoefficient
{
public:
   LorStreamlineDiffusion(const ParGridFunction& w, const DenseTensor& jparent,
                          const Array<int>& parent, double nu, int k)
      : MatrixCoefficient(w.ParFESpace()->GetVDim()), w_(w),
        jparent_(jparent), parent_(parent), nu_(nu), k_(k) {}

   void Eval(DenseMatrix& K, ElementTransformation& T,
             const IntegrationPoint& ip) override
   {
      w_.GetVectorValue(T, ip, wv_);
      const int d = height;
      const real_t tau = OssTau(d, wv_.GetData(),
                                jparent_(parent_[T.ElementNo]).Data(), nu_, k_);
      K.SetSize(d);
      for (int j = 0; j < d; j++)
      {
         for (int i = 0; i < d; i++) { K(i, j) = tau * wv_(i) * wv_(j); }
      }
   }

private:
   const ParGridFunction& w_;
   const DenseTensor& jparent_;
   const Array<int>& parent_;
   double nu_;
   int k_;
   Vector wv_;
};

/// gamma of the parent element times e_i e_j^T on the LOR mesh: through
/// DiffusionIntegrator (K grad u, grad v), the grad-div block
/// gamma (d_j u, d_i v) of test component i and trial component j.
class LorGradDivBlock : public MatrixCoefficient
{
public:
   LorGradDivBlock(int dim, int i, int j, const Vector& gamma,
                   const Array<int>& parent)
      : MatrixCoefficient(dim), i_(i), j_(j), gamma_(gamma), parent_(parent) {}

   void Eval(DenseMatrix& K, ElementTransformation& T,
             const IntegrationPoint&) override
   {
      K.SetSize(height);
      K = 0.0;
      K(i_, j_) = gamma_(parent_[T.ElementNo]);
   }

private:
   int i_, j_;
   const Vector& gamma_;
   const Array<int>& parent_;
};

/// One V-cycle, counted.
class CountedVCycle : public Solver
{
public:
   CountedVCycle(HypreBoomerAMG& amg, long& count)
      : Solver(amg.Height()), amg_(amg), count_(count) {}
   void Mult(const Vector& x, Vector& y) const override
   {
      amg_.Mult(x, y);
      count_++;
   }
   void SetOperator(const Operator&) override {}

private:
   HypreBoomerAMG& amg_;
   long& count_;
};

/// hypre relax-point pattern: F -> -1, C -> 1, A -> 0 (host memory: hypre
/// owns it after HYPRE_BoomerAMGSetGridRelaxPoints).
HYPRE_Int* RelaxPoints(const std::string& pattern)
{
   HYPRE_Int* p = hypre_TAlloc(HYPRE_Int, std::max((int) pattern.size(), 1),
                               HYPRE_MEMORY_HOST);
   for (std::size_t i = 0; i < pattern.size(); i++)
   {
      p[i] = pattern[i] == 'F' ? -1 : (pattern[i] == 'C' ? 1 : 0);
   }
   return p;
}

/// The recipe's GPU Neumann AIR: no pre-relaxation, F-F-F-C Jacobi
/// post-relaxation, one-point interpolation, PMIS, coarse-grid dropping.
/// (MFEM's SetAdvectiveOptions puts the relax points in device memory and
/// fills them on the host: it faults on a GPU build.)
void ConfigureAir(HypreBoomerAMG& amg, double drop_tol)
{
   HYPRE_Solver s = amg;
   HYPRE_Int** pts = hypre_TAlloc(HYPRE_Int*, 4, HYPRE_MEMORY_HOST);
   pts[0] = nullptr;
   pts[1] = RelaxPoints("");
   pts[2] = RelaxPoints("FFFC");
   pts[3] = RelaxPoints("A");
   HYPRE_BoomerAMGSetRestriction(s, 4); // Neumann AIR, degree 1
   HYPRE_BoomerAMGSetGridRelaxPoints(s, pts);
   HYPRE_BoomerAMGSetAggNumLevels(s, 0);
   HYPRE_BoomerAMGSetCycleNumSweeps(s, 1, 3);
   HYPRE_BoomerAMGSetCycleNumSweeps(s, 0, 1);
   HYPRE_BoomerAMGSetCycleNumSweeps(s, 4, 2);
   HYPRE_BoomerAMGSetADropType(s, -1);
   HYPRE_BoomerAMGSetADropTol(s, drop_tol);
   HYPRE_BoomerAMGSetInterpType(s, 100);
   HYPRE_BoomerAMGSetRelaxType(s, 7);
   HYPRE_BoomerAMGSetRelaxWt(s, 0.75);
   HYPRE_BoomerAMGSetCycleRelaxType(s, 9, 3);
   HYPRE_BoomerAMGSetCoarsenType(s, 8);
   HYPRE_BoomerAMGSetStrongThreshold(s, 0.5);
   HYPRE_BoomerAMGSetStrongThresholdR(s, 0.05);
   HYPRE_BoomerAMGSetFilterThresholdR(s, 0.0);
   HYPRE_BoomerAMGSetSabs(s, 0);
   HYPRE_BoomerAMGSetMaxLevels(s, 25);
}

/// Classical AMG that runs on a GPU: PMIS, extended+i, l1-Jacobi.
void ConfigureAmg(HypreBoomerAMG& amg)
{
   HYPRE_Solver s = amg;
   HYPRE_BoomerAMGSetCoarsenType(s, 8);
   HYPRE_BoomerAMGSetInterpType(s, 6);
   HYPRE_BoomerAMGSetPMaxElmts(s, 4);
   HYPRE_BoomerAMGSetAggNumLevels(s, 0);
   HYPRE_BoomerAMGSetRelaxType(s, 18);
   HYPRE_BoomerAMGSetStrongThreshold(s, 0.25);
}

/// hypre's smoothers and EliminateRowsCols want the diagonal first in each
/// row of the diag block; Add() can break that.
void DiagonalFirst(HypreParMatrix& A)
{
   A.HypreReadWrite();
   hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix*) A));
}

} // namespace

OseenLorPreconditioner::OseenLorPreconditioner(ParFiniteElementSpace& vfes,
      const RuleBook& rules,
      const Array<int>& ess_tdofs,
      const Options& opts)
   : Solver(vfes.GetTrueVSize()), vfes_(vfes), opts_(opts)
{
   ParMesh& mesh = *vfes.GetParMesh();
   dim_ = mesh.Dimension();
   k_ = vfes.GetMaxElementOrder();
   MFEM_VERIFY(vfes.GetVDim() == dim_ && vfes.GetOrdering() == Ordering::byNODES,
               "OseenLorPreconditioner: a vector velocity space by nodes");
   const Geometry::Type geom = (dim_ == 3) ? Geometry::CUBE : Geometry::SQUARE;
   sfes_ = std::make_unique<ParFiniteElementSpace>(&mesh, vfes.FEColl());
   n_ = sfes_->GetTrueVSize();
   lor_ = std::make_unique<ParLORDiscretization>(*sfes_, BasisType::GaussLobatto);
   ParFiniteElementSpace& lfes = lor_->GetParFESpace();
   ParMesh& lmesh = *lfes.GetParMesh();
   lor_vfes_ = std::make_unique<ParFiniteElementSpace>(&lmesh, lfes.FEColl(), dim_,
               Ordering::byNODES);
   MFEM_VERIFY(lor_vfes_->GetTrueVSize() == vfes.GetTrueVSize(),
               "OseenLorPreconditioner: LOR and HO true dofs differ");
   wind_lor_ = std::make_unique<ParGridFunction>(lor_vfes_.get());
   wind_ho_ = std::make_unique<ParGridFunction>(&vfes_);
   vertex_rule_ = &rules.Get(geom, 1, Rule1D::GaussLobatto);

   // Each LOR element's parent, and J^-1 at the parent's centre.
   const CoarseFineTransformations& cf = lmesh.GetRefinementTransforms();
   parent_.SetSize(lmesh.GetNE());
   for (int l = 0; l < lmesh.GetNE(); l++) { parent_[l] = cf.embeddings[l].parent; }
   jparent_.SetSize(dim_, dim_, mesh.GetNE());
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      ElementTransformation* T = mesh.GetElementTransformation(e);
      T->SetIntPoint(&Geometries.GetCenter(geom));
      CalcInverse(T->Jacobian(), jparent_(e));
   }

   // The high-order GLL mass diagonal on the scalar true dofs.
   {
      ParBilinearForm m(sfes_.get());
      auto* mi = new MassIntegrator;
      mi->SetIntRule(&rules.CollocatedMass(geom, k_));
      m.AddDomainIntegrator(mi);
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m.Assemble();
      d_gll_.SetSize(n_);
      m.AssembleDiagonal(d_gll_);
   }

   // Essential dofs per component; components with the same list share a
   // matrix.
   ess_.resize(dim_);
   ess_tdofs.HostRead();
   for (int i = 0; i < ess_tdofs.Size(); i++)
   {
      const int t = ess_tdofs[i];
      ess_[t / n_].Append(t % n_);
   }
   // Share only where every rank finds the lists equal: the matrices and
   // V-cycles are collective.
   block_.assign(dim_, -1);
   for (int c = 0; c < dim_; c++)
   {
      for (int b = 0; b < c && block_[c] < 0; b++)
      {
         int same = (block_[b] == b && ess_[b].Size() == ess_[c].Size() &&
                     std::equal(ess_[b].begin(), ess_[b].end(), ess_[c].begin()));
         MPI_Allreduce(MPI_IN_PLACE, &same, 1, MPI_INT, MPI_MIN, vfes.GetComm());
         if (same) { block_[c] = b; }
      }
      if (block_[c] < 0) { block_[c] = c; }
   }
   F_.resize(dim_);
   amg_.resize(dim_);
   vcycle_.resize(dim_);
   block_solver_.resize(dim_);
   t_.SetSize(n_);
   t_.UseDevice(true);
}

OseenLorPreconditioner::~OseenLorPreconditioner() = default;

std::unique_ptr<HypreParMatrix> OseenLorPreconditioner::AssembleGradDiv(
   const Vector& gamma, int i, int j) const
{
   LorGradDivBlock gij(dim_, i, j, gamma, parent_);
   ParBilinearForm g(&lor_->GetParFESpace());
   auto* di = new DiffusionIntegrator(gij);
   di->SetIntRule(vertex_rule_);
   g.AddDomainIntegrator(di);
   g.Assemble();
   g.Finalize();
   return std::unique_ptr<HypreParMatrix>(g.ParallelAssemble());
}

void OseenLorPreconditioner::SetGradDiv(const Vector& gamma_elem)
{
   MFEM_VERIFY(gamma_elem.Size() == vfes_.GetParMesh()->GetNE(),
               "OseenLorPreconditioner: gamma per local element");
   Vector gamma(gamma_elem);
   gamma.HostRead();
   G_.resize(dim_);
   for (int c = 0; c < dim_; c++)
   {
      G_[c] = AssembleGradDiv(gamma, c, c);
      block_[c] = c; // the components' operators now differ
   }
   Gup_.clear();
   if (opts_.coupling == OseenVelocityCoupling::UpperGaussSeidel)
   {
      // Rows of i and columns of j at their Dirichlet dofs: what the sweep
      // subtracts there is nothing (EliminateCols is collective: every rank
      // calls it, whatever its list).
      Gup_.resize(dim_ * dim_);
      for (int i = 0; i < dim_; i++)
      {
         for (int j = i + 1; j < dim_; j++)
         {
            std::unique_ptr<HypreParMatrix> g = AssembleGradDiv(gamma, i, j);
            g->EliminateRows(ess_[i]);
            delete g->EliminateCols(ess_[j]);
            Gup_[i * dim_ + j] = std::move(g);
         }
      }
   }
}

const HypreParMatrix* OseenLorPreconditioner::GradDivBlock(int i, int j) const
{
   if (i == j) { return (i < (int) G_.size()) ? G_[i].get() : nullptr; }
   if (i > j || Gup_.empty()) { return nullptr; }
   return Gup_[i * dim_ + j].get();
}

double OseenLorPreconditioner::CellReynolds(const ParGridFunction& w) const
{
   ParMesh& mesh = *vfes_.GetParMesh();
   w.HostRead();
   double re = 0.0;
   Array<int> vdofs;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      vfes_.GetElementVDofs(e, vdofs);
      const int nd = vdofs.Size() / dim_;
      double wmax = 0.0;
      for (int a = 0; a < nd; a++)
      {
         double s = 0.0;
         for (int c = 0; c < dim_; c++)
         {
            const int v = vdofs[a + c * nd];
            const double x = (v >= 0) ? w(v) : -w(-1 - v);
            s += x * x;
         }
         wmax = std::max(wmax, std::sqrt(s));
      }
      const double h = std::pow(mesh.GetElementVolume(e), 1.0 / dim_);
      re = std::max(re, wmax * h / (k_ * opts_.nu));
   }
   double g = 0.0;
   MPI_Allreduce(&re, &g, 1, MPI_DOUBLE, MPI_MAX, vfes_.GetComm());
   return g;
}

void OseenLorPreconditioner::Update(const Vector& wind, double sigma)
{
   MFEM_VERIFY(opts_.coupling != OseenVelocityCoupling::UpperGaussSeidel ||
               !Gup_.empty(), "OseenLorPreconditioner: UpperGaussSeidel "
               "couples through the grad-div blocks: SetGradDiv() first");
   StopWatch sw;
   sw.Start();
   wind_ho_->SetFromTrueDofs(wind);
   wind_lor_->SetFromTrueDofs(wind);
   wind_lor_->HostRead();
   re_h_ = CellReynolds(*wind_ho_);
   air_ = opts_.mode == OseenAmgMode::Air ||
          (opts_.mode == OseenAmgMode::Auto && re_h_ >= opts_.air_cell_re);

   ParFiniteElementSpace& lfes = lor_->GetParFESpace();
   ConstantCoefficient nu(opts_.nu);
   VectorGridFunctionCoefficient w(wind_lor_.get());
   LorStreamlineDiffusion sd(*wind_lor_, jparent_, parent_, opts_.nu, k_);
   ParBilinearForm f(&lfes);
   auto* di = new DiffusionIntegrator(nu);
   di->SetIntRule(vertex_rule_);
   f.AddDomainIntegrator(di);
   f.AddDomainIntegrator(new ConvectionIntegrator(w, 1.0));
   f.AddDomainIntegrator(new DiffusionIntegrator(sd));
   f.Assemble();
   f.Finalize();
   std::unique_ptr<HypreParMatrix> Fk(f.ParallelAssemble());

   // + sigma D: the high-order GLL mass diagonal on the same dofs.
   // Scaled where the data lives, then read on the host: SparseMatrix takes
   // the host array without synchronizing (on a GPU the scaling would
   // otherwise never reach it).
   Vector sd_diag(d_gll_);
   sd_diag *= sigma;
   sd_diag.HostRead();
   SparseMatrix Dsp(sd_diag);
   HypreParMatrix D(vfes_.GetComm(), Fk->GetGlobalNumRows(), Fk->GetRowStarts(),
                    &Dsp);
   t_assembly_ = sw.RealTime();

   sw.Clear();
   sw.Start();
   const int nranks = Mpi::WorldSize();
   const double drop = opts_.air_drop_tol >= 0.0 ? opts_.air_drop_tol
                       : (nranks > 1 ? 0.0 : 1e-2);
   for (int c = 0; c < dim_; c++)
   {
      // The block solver and the counted V-cycle hold the V-cycle: first.
      block_solver_[c].reset();
      vcycle_[c].reset();
      amg_[c].reset();
      if (block_[c] != c) { F_[c].reset(); continue; }
      F_[c].reset(Add(1.0, *Fk, 1.0, D));
      if (!G_.empty())
      {
         std::unique_ptr<HypreParMatrix> FG(Add(1.0, *F_[c], 1.0, *G_[c]));
         F_[c] = std::move(FG);
      }
      DiagonalFirst(*F_[c]);
      delete F_[c]->EliminateRowsCols(ess_[c]);
      amg_[c] = std::make_unique<HypreBoomerAMG>();
      amg_[c]->SetPrintLevel(opts_.print_level);
      if (air_) { ConfigureAir(*amg_[c], drop); }
      else { ConfigureAmg(*amg_[c]); }
      amg_[c]->SetOperator(*F_[c]);
      amg_[c]->iterative_mode = false;
      HypreParVector b(*F_[c]), x(*F_[c]);
      b = 0.0;
      x = 0.0;
      amg_[c]->Setup(b, x);
      vcycle_[c] = std::make_unique<CountedVCycle>(*amg_[c], vcycles_);
      if (opts_.block_rtol > 0.0)
      {
         // SetOperator before SetPreconditioner: MFEM forwards SetOperator
         // to the preconditioner (a second AMG setup).
         auto g = std::make_unique<GMRESSolver>(vfes_.GetComm());
         g->SetKDim(50);
         g->SetMaxIter(opts_.block_maxit);
         g->SetRelTol(opts_.block_rtol);
         g->SetAbsTol(0.0);
         g->SetPrintLevel(-1);
         g->iterative_mode = false;
         g->SetOperator(*F_[c]);
         g->SetPreconditioner(*vcycle_[c]);
         block_solver_[c] = std::move(g);
      }
   }
   MFEM_DEVICE_SYNC;
   t_setup_ = sw.RealTime();
}

void OseenLorPreconditioner::SolveBlock(int c, const Vector& b, Vector& x) const
{
   const int m = block_[c];
   if (block_solver_[m]) { block_solver_[m]->Mult(b, x); }
   else { vcycle_[m]->Mult(b, x); }
}

void OseenLorPreconditioner::Mult(const Vector& r, Vector& z) const
{
   MFEM_VERIFY(amg_[0], "OseenLorPreconditioner: Update() first");
   r.Read();
   z.Write();
   const bool gs = opts_.coupling == OseenVelocityCoupling::UpperGaussSeidel;
   Vector rc, zc, zj;
   // Upper Gauss-Seidel: the last component first; each earlier one's
   // right-hand side less its coupling to the components already solved.
   for (int s = 0; s < dim_; s++)
   {
      const int c = gs ? dim_ - 1 - s : s;
      rc.MakeRef(const_cast<Vector&>(r), c * n_, n_);
      zc.MakeRef(z, c * n_, n_);
      if (!gs || c == dim_ - 1)
      {
         SolveBlock(c, rc, zc);
         continue;
      }
      t_ = rc;
      for (int j = c + 1; j < dim_; j++)
      {
         zj.MakeRef(z, j * n_, n_);
         Gup_[c * dim_ + j]->Mult(-1.0, zj, 1.0, t_);
      }
      SolveBlock(c, t_, zc);
   }
}

} // namespace incns
