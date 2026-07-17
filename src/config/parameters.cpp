#include "config/parameters.hpp"

#include "mfem.hpp" // MFEM_VERIFY

#include <yaml-cpp/yaml.h>

namespace incns
{

namespace
{
// Read node[key] into value when present (defaults hold otherwise).
template <typename T>
void Maybe(const YAML::Node& node, const char* key, T& value)
{
   if (node && node[key]) { value = node[key].as<T>(); }
}

template <std::size_t N, typename T>
void MaybeArray(const YAML::Node& node, const char* key,
                std::array<T, N>& value, int count)
{
   if (!(node && node[key])) { return; }
   const YAML::Node seq = node[key];
   MFEM_VERIFY(seq.IsSequence() && static_cast<int>(seq.size()) == count,
               "parameters: '" << key << "' must be a sequence of " << count);
   for (int i = 0; i < count; ++i) { value[i] = seq[i].as<T>(); }
}
} // namespace

Parameters Parameters::LoadYAML(const std::string& path)
{
   Parameters p;
   const YAML::Node root = YAML::LoadFile(path);

   if (root["equation"])
   {
      const std::string eq = root["equation"].as<std::string>();
      if (eq == "stokes") { p.equation = Equation::Stokes; }
      else if (eq == "navier_stokes" || eq == "nse")
      {
         p.equation = Equation::NavierStokes;
      }
      else { MFEM_ABORT("parameters: unknown equation '" << eq << "'"); }
   }

   Maybe(root, "device", p.device);

   const YAML::Node physics = root["physics"];
   Maybe(physics, "nu", p.nu);
   Maybe(physics, "grad_div", p.grad_div);

   const YAML::Node nd = root["nondimensionalization"];
   if (nd && nd["mode"])
   {
      const std::string mode = nd["mode"].as<std::string>();
      if (mode == "dimensional") { p.nondim.mode = ScalingMode::Dimensional; }
      else
      {
         MFEM_VERIFY(mode == "dimensionless",
                     "parameters: unknown nondimensionalization mode '"
                     << mode << "'");
      }
   }
   Maybe(nd, "L_ref", p.nondim.L_ref);
   Maybe(nd, "U_ref", p.nondim.U_ref);
   Maybe(nd, "rho", p.nondim.rho);

   // Convenience: `physics: Re` in dimensionless mode sets nu = 1/Re.
   if (physics && physics["Re"])
   {
      MFEM_VERIFY(!physics["nu"],
                  "parameters: give either physics.nu or physics.Re, not both");
      MFEM_VERIFY(p.nondim.mode == ScalingMode::Dimensionless,
                  "parameters: physics.Re is a dimensionless-mode input");
      p.nu = 1.0 / physics["Re"].as<double>();
   }

   const YAML::Node disc = root["discretization"];
   Maybe(disc, "order_u", p.order_u);
   Maybe(disc, "order_p", p.order_p);
   Maybe(disc, "collocated_mass", p.collocated_mass);

   const YAML::Node mesh = root["mesh"];
   Maybe(mesh, "dim", p.mesh.dim);
   MaybeArray(mesh, "elements", p.mesh.num_elems, p.mesh.dim);
   MaybeArray(mesh, "lengths", p.mesh.lengths, p.mesh.dim);
   MaybeArray(mesh, "periodic", p.mesh.periodic, p.mesh.dim);
   if (mesh && mesh["stretch"])
   {
      const YAML::Node seq = mesh["stretch"];
      MFEM_VERIFY(seq.IsSequence()
                  && static_cast<int>(seq.size()) == p.mesh.dim,
                  "parameters: mesh.stretch must be a sequence of "
                  << p.mesh.dim);
      for (int d = 0; d < p.mesh.dim; ++d)
      {
         const std::string kind = seq[d].as<std::string>();
         if (kind == "none") { p.mesh.stretch[d] = Stretch::None; }
         else if (kind == "tanh") { p.mesh.stretch[d] = Stretch::TwoSidedTanh; }
         else { MFEM_ABORT("parameters: unknown mesh.stretch '" << kind << "'"); }
      }
   }
   MaybeArray(mesh, "stretch_beta", p.mesh.stretch_beta, p.mesh.dim);

   const YAML::Node time = root["time"];
   Maybe(time, "dt", p.dt);
   Maybe(time, "t_final", p.t_final);
   Maybe(time, "order", p.time_order);
   Maybe(time, "adaptive", p.adaptive);
   Maybe(time, "atol", p.controller.atol);
   Maybe(time, "rtol", p.controller.rtol);

   const YAML::Node solver = root["solver"];
   Maybe(solver, "rtol", p.krylov_rtol);
   Maybe(solver, "atol", p.krylov_atol);
   Maybe(solver, "max_iter", p.max_iter);
   Maybe(solver, "kdim", p.kdim);
   Maybe(solver, "print_level", p.print_level);
   if (solver && solver["preconditioner"])
   {
      const std::string pc = solver["preconditioner"].as<std::string>();
      if (pc == "jacobi") { p.velocity_prec = VelocityPreconditioner::Jacobi; }
      else if (pc == "amg" || pc == "boomer_amg")
      {
         p.velocity_prec = VelocityPreconditioner::BoomerAMG;
      }
      else { MFEM_ABORT("parameters: unknown solver.preconditioner '" << pc << "'"); }
   }
   Maybe(solver, "amg_reuse", p.amg_reuse);
   if (solver && solver["schur"])
   {
      const std::string sm = solver["schur"].as<std::string>();
      if (sm == "mass") { p.schur = SchurBlockType::Mass; }
      else if (sm == "cc")
      {
         p.schur = SchurBlockType::CahouetChabard;
         p.cc.schur_model = SchurModel::ConsistentBMB;
      }
      else if (sm == "laplacian_legacy")
      {
         p.schur = SchurBlockType::CahouetChabard;
         p.cc.schur_model = SchurModel::LaplacianLegacy;
      }
      else { MFEM_ABORT("parameters: unknown solver.schur '" << sm << "'"); }
   }
   Maybe(solver, "n_inner", p.cc.n_inner);
   Maybe(solver, "lp_vcycles", p.cc.lp_vcycles);
   if (solver && solver["block_shape"])
   {
      const std::string bs = solver["block_shape"].as<std::string>();
      if (bs == "diag") { p.cc.block_shape = BlockPCShape::Diag; }
      else if (bs == "lower") { p.cc.block_shape = BlockPCShape::LowerTri; }
      else if (bs == "upper") { p.cc.block_shape = BlockPCShape::UpperTri; }
      else { MFEM_ABORT("parameters: unknown solver.block_shape '" << bs << "'"); }
   }

   Maybe(root, "initial_velocity", p.initial_velocity);

   // Boundary-condition groups: topology + type only (fields are bound in the
   // driver by group name). A selector token is an attribute integer if it
   // parses fully as one, otherwise a box face name.
   const YAML::Node bcs = root["boundary_conditions"];
   if (bcs)
   {
      MFEM_VERIFY(bcs.IsSequence(),
                  "parameters: 'boundary_conditions' must be a sequence");
      auto add_token = [](const std::string & tok, BcSpec & s)
      {
         try
         {
            std::size_t pos = 0;
            const int a = std::stoi(tok, &pos);
            if (pos == tok.size()) { s.attributes.push_back(a); return; }
         }
         catch (...) { /* not an int -> a face name */ }
         if (tok == "all") { s.select_all = true; }
         else { s.faces.push_back(tok); }
      };
      for (const auto& e : bcs)
      {
         BcSpec s;
         if (e["group"]) { s.group = e["group"].as<std::string>(); }
         const std::string ty = e["type"].as<std::string>();
         if (ty == "outflow") { s.type = BcType::Outflow; }
         else if (ty == "no_slip") { s.type = BcType::NoSlip; }
         else
         {
            MFEM_VERIFY(ty == "velocity_dirichlet",
                        "parameters: unknown boundary type '" << ty << "'");
            s.type = BcType::VelocityDirichlet;
         }
         const YAML::Node sel = e["select"];
         MFEM_VERIFY(sel, "parameters: a boundary_conditions entry needs 'select'");
         if (sel.IsSequence())
         {
            for (const auto& x : sel) { add_token(x.as<std::string>(), s); }
         }
         else { add_token(sel.as<std::string>(), s); }
         p.boundary_conditions.push_back(std::move(s));
      }
   }

   const YAML::Node output = root["output"];
   Maybe(output, "enabled", p.output.enabled);
   Maybe(output, "path", p.output.path);
   Maybe(output, "name", p.output.name);
   Maybe(output, "interval", p.output.interval);
   Maybe(output, "diagnostics", p.output.diagnostics);

   const YAML::Node chk = root["checkpoint"];
   Maybe(chk, "enabled", p.checkpoint.enabled);
   Maybe(chk, "path", p.checkpoint.path);
   Maybe(chk, "interval", p.checkpoint.interval);
   Maybe(root, "restart", p.restart_from);

   // Basic validation (module-level invariants are re-checked downstream).
   MFEM_VERIFY(p.nu > 0.0, "parameters: nu must be positive");
   MFEM_VERIFY(p.order_u >= 1 && p.order_p >= 1, "parameters: bad orders");
   MFEM_VERIFY(p.dt > 0.0 && p.t_final > 0.0, "parameters: bad time settings");
   MFEM_VERIFY(p.output.interval >= 1, "parameters: bad output interval");
   MFEM_VERIFY(p.checkpoint.interval >= 1, "parameters: bad checkpoint interval");
   p.Normalize();
   return p;
}

void Parameters::Normalize()
{
   if (nondim.normalized) { return; }

   if (nondim.mode == ScalingMode::Dimensionless)
   {
      nondim.L_ref = 1.0;
      nondim.U_ref = 1.0;
      nondim.Re = 1.0 / nu;
   }
   else
   {
      MFEM_VERIFY(nondim.L_ref > 0.0 && nondim.U_ref > 0.0,
                  "parameters: dimensional mode needs positive L_ref, U_ref");
      MFEM_VERIFY(initial_velocity == "zero",
                  "parameters: named initial conditions are nondimensional; "
                  "in dimensional mode provide the IC through "
                  "WrapDimensionalVelocity");
      nondim.Re = nondim.U_ref * nondim.L_ref / nu;

      const double U_over_L = nondim.U_ref / nondim.L_ref;
      for (int d = 0; d < 3; ++d) { mesh.lengths[d] /= nondim.L_ref; }
      dt *= U_over_L;
      t_final *= U_over_L;
      nu = 1.0 / nondim.Re;
      controller.atol /= nondim.U_ref; // the LTE norm carries velocity units
   }
   nondim.normalized = true;
}

} // namespace incns
