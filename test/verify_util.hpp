// Shared helpers for the verification tier (label `verify`, docs/developing.md
// §7.2): error norms against an exact solution, least-squares convergence
// rates, the `verification` section of baselines.yaml, and the convergence
// CSVs every study writes.
//
// The studies run through incns::Case on minimal decks: only the keys a study
// is about (geometry, viscosity, orders, step sizes, solver tolerances) are
// set, everything else is the case-level default. When a default changes
// (grad-div, the Schur or velocity preconditioner, the mass, the time
// scheme), the suite verifies the new default without an edit.
#pragma once

#include <gtest/gtest.h>

#include "quadrature/rule_book.hpp"
#include "mfem.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace verify
{

/// The `verification` section of test/baselines.yaml.
inline YAML::Node Baselines()
{
   const YAML::Node root =
      YAML::LoadFile(INCNS_BASELINES_FILE)["verification"];
   MFEM_VERIFY(root, "baselines.yaml: missing the verification section");
   return root;
}

/// Errors of one run against the exact solution.
struct Errors
{
   double u_l2 = 0.0;   ///< ||u_h - u||_L2.
   double u_h1 = 0.0;   ///< |u_h - u|_H1 (seminorm).
   double p_l2 = 0.0;   ///< ||(p_h - mean p_h) - (p - mean p)||_L2.
   double div_l2 = 0.0; ///< ||div u_h||_L2.
};

/// Global sum over the ranks.
inline double Sum(double local, MPI_Comm comm)
{
   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
   return global;
}

/// int_Omega f, with the RuleBook rule of order @p order on every element.
inline double Integrate(mfem::ParMesh& mesh, mfem::Coefficient& f,
                        const incns::RuleBook& rules, int order)
{
   double loc = 0.0;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      const mfem::IntegrationRule& ir =
         rules.Get(mesh.GetElementBaseGeometry(e), order);
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const mfem::IntegrationPoint& ip = ir.IntPoint(i);
         T.SetIntPoint(&ip);
         loc += ip.weight * T.Weight() * f.Eval(T, ip);
      }
   }
   return Sum(loc, mesh.GetComm());
}

/**
 * Velocity L2 and H1-seminorm errors, the divergence norm and the pressure L2
 * error with both pressures shifted to zero mean (an all-Dirichlet problem
 * fixes p only up to a constant). Element loops at the RuleBook rule of order
 * @p order: pass an elevated one (2 k_u + 4), or the errors flatten for the
 * wrong reason.
 */
inline Errors ComputeErrors(const mfem::ParGridFunction& u,
                            const mfem::ParGridFunction& p,
                            mfem::VectorCoefficient& u_exact,
                            mfem::MatrixCoefficient& grad_exact,
                            mfem::Coefficient& p_exact,
                            const incns::RuleBook& rules, int order)
{
   mfem::ParMesh& mesh = *u.ParFESpace()->GetParMesh();
   const MPI_Comm comm = mesh.GetComm();
   const int dim = mesh.Dimension();
   u.HostRead();
   p.HostRead();

   mfem::ConstantCoefficient one(1.0);
   mfem::GridFunctionCoefficient p_h(&p);
   const double volume = Integrate(mesh, one, rules, order);
   const double mean_h = Integrate(mesh, p_h, rules, order) / volume;
   const double mean_e = Integrate(mesh, p_exact, rules, order) / volume;

   double l2 = 0.0, h1 = 0.0, pl2 = 0.0, div = 0.0;
   mfem::Vector uh(dim), ue(dim);
   mfem::DenseMatrix gh(dim), ge(dim);
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
      const mfem::IntegrationRule& ir =
         rules.Get(mesh.GetElementBaseGeometry(e), order);
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const mfem::IntegrationPoint& ip = ir.IntPoint(i);
         T.SetIntPoint(&ip);
         const double w = ip.weight * T.Weight();
         u.GetVectorValue(T, ip, uh);
         u_exact.Eval(ue, T, ip);
         uh -= ue;
         l2 += w * (uh * uh);
         u.GetVectorGradient(T, gh);
         div += w * std::pow(gh.Trace(), 2);
         grad_exact.Eval(ge, T, ip);
         gh -= ge;
         h1 += w * gh.FNorm2();
         const double dp = (p.GetValue(T, ip) - mean_h) -
                           (p_exact.Eval(T, ip) - mean_e);
         pl2 += w * dp * dp;
      }
   }
   Errors r;
   r.u_l2 = std::sqrt(Sum(l2, comm));
   r.u_h1 = std::sqrt(Sum(h1, comm));
   r.p_l2 = std::sqrt(Sum(pl2, comm));
   r.div_l2 = std::sqrt(Sum(div, comm));
   return r;
}

/// Least-squares slope of log e against log h: the convergence rate fitted
/// over every level (h may be a mesh size or a step size).
inline double FittedRate(const std::vector<double>& h,
                         const std::vector<double>& e)
{
   const std::size_t n = h.size();
   double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
   for (std::size_t i = 0; i < n; ++i)
   {
      const double x = std::log(h[i]), y = std::log(e[i]);
      sx += x;
      sy += y;
      sxx += x * x;
      sxy += x * y;
   }
   const double m = static_cast<double>(n);
   return (m * sxy - sx * sy) / (m * sxx - sx * sx);
}

/**
 * The convergence record of one study: every error goes to
 * `<INCNS_VERIFY_OUT>/<study>_np<N>.csv` and every fitted rate to
 * `<study>_rates_np<N>.csv` (CI keeps both as artifacts). Root writes; the
 * files are truncated when the record is created.
 */
class Record
{
public:
   /// @param study Study name (file name stem).
   explicit Record(const std::string& study) : study_(study)
   {
      int np = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &np);
      const std::string dir = INCNS_VERIFY_OUT;
      const std::string stem = dir + "/" + study + "_np" + std::to_string(np);
      errors_path_ = stem + ".csv";
      rates_path_ = stem + "_rates.csv";
      if (mfem::Mpi::Root())
      {
         std::filesystem::create_directories(dir);
         std::ofstream(errors_path_)
               << "study,variant,level,h,quantity,error,settings\n";
         std::ofstream(rates_path_)
               << "study,variant,quantity,theory,fitted,threshold,pass\n";
      }
   }

   /// Append one error.
   void Error(const std::string& variant, int level, double h,
              const std::string& quantity, double error,
              const std::string& settings = "")
   {
      if (!mfem::Mpi::Root()) { return; }
      std::ofstream(errors_path_, std::ios::app)
            << study_ << "," << variant << "," << level << "," << h << ","
            << quantity << "," << error << ",\"" << settings << "\"\n";
      std::printf("[ verify   ] %s %s level %d h=%g %s=%.6e\n", study_.c_str(),
                  variant.c_str(), level, h, quantity.c_str(), error);
   }

   /**
    * Fit the rate of one quantity over the levels, record it and check it:
    * fitted >= theory - margin, and the finest error above @p floor (a rate
    * measured at solver noise proves nothing).
    */
   void Rate(const std::string& variant, const std::string& quantity,
             const std::vector<double>& h, const std::vector<double>& e,
             double theory, double margin, double floor)
   {
      const double fitted = FittedRate(h, e);
      const double threshold = theory - margin;
      const bool pass = fitted >= threshold;
      if (mfem::Mpi::Root())
      {
         std::ofstream(rates_path_, std::ios::app)
               << study_ << "," << variant << "," << quantity << "," << theory
               << "," << fitted << "," << threshold << ","
               << (pass ? "yes" : "no") << "\n";
         std::printf("[ verify   ] %s %s %s: fitted rate %.3f (theory %g, "
                     "threshold %.2f) %s\n", study_.c_str(), variant.c_str(),
                     quantity.c_str(), fitted, theory, threshold,
                     pass ? "ok" : "BELOW");
      }
      EXPECT_GT(*std::min_element(e.begin(), e.end()), floor)
            << study_ << " " << variant << " " << quantity
            << ": finest error at the noise floor -- the rate is vacuous";
      EXPECT_GE(fitted, threshold)
            << study_ << " " << variant << " " << quantity << ": fitted rate "
            << fitted << " below theory " << theory << " - " << margin;
   }

private:
   std::string study_;       ///< Study name.
   std::string errors_path_; ///< Errors CSV.
   std::string rates_path_;  ///< Rates CSV.
};

/// A grad-div variant of a study: the case-level default, or switched off.
struct GradDivVariant
{
   const char* name;    ///< CSV label.
   const char* physics; ///< Lines added under the deck's `physics:` key.
};

/// Every spatial study runs both: grad-div is an open question (human
/// 2026-10-15), so its effect on the rates is recorded either way.
inline const std::vector<GradDivVariant>& GradDivVariants()
{
   static const std::vector<GradDivVariant> v =
   {
      {"default", ""},
      {"grad_div_off", "  grad_div_h: 0.0\n  grad_div_nu: 0.0\n"},
   };
   return v;
}

} // namespace verify
