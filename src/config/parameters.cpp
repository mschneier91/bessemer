#include "config/parameters.hpp"

#include "mfem.hpp" // MFEM_VERIFY

#include <yaml-cpp/yaml.h>

#include <cmath>

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
   if (physics && physics["grad_div_scale"])
   {
      const std::string gs = physics["grad_div_scale"].as<std::string>();
      if (gs == "h") { p.grad_div_scale = GradDivScale::OrderH; }
      else if (gs == "nu") { p.grad_div_scale = GradDivScale::OrderNu; }
      else { MFEM_ABORT("parameters: unknown physics.grad_div_scale '" << gs << "'"); }
   }
   if (physics && physics["convective_form"])
   {
      const std::string cf = physics["convective_form"].as<std::string>();
      if (cf == "convective") { p.convective_form = ConvectiveForm::Convective; }
      else if (cf == "rotational")
      {
         p.convective_form = ConvectiveForm::Rotational;
      }
      else { MFEM_ABORT("parameters: unknown physics.convective_form '" << cf << "'"); }
   }

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
   Maybe(time, "atol", p.controller.atol);
   Maybe(time, "rtol", p.controller.rtol);
   Maybe(time, "cfl_max", p.cfl_max);
   Maybe(time, "cfl_target", p.cfl_target);
   Maybe(time, "dt_max", p.dt_max);
   Maybe(time, "ext_order", p.ext_order);
   if (time && time["adaptive"]) // the older spelling: true = error
   {
      MFEM_VERIFY(!time["step_control"], "parameters: give time.step_control "
                  "or the older time.adaptive, not both");
      p.step_control = time["adaptive"].as<bool>() ? StepControl::Error
                       : StepControl::Fixed;
   }
   if (time && time["step_control"])
   {
      const std::string sc = time["step_control"].as<std::string>();
      if (sc == "fixed") { p.step_control = StepControl::Fixed; }
      else if (sc == "error") { p.step_control = StepControl::Error; }
      else if (sc == "cfl") { p.step_control = StepControl::Cfl; }
      else
      {
         MFEM_ABORT("parameters: unknown time.step_control '" << sc
                    << "' (fixed|error|cfl)");
      }
   }

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
      // "loramg" is the accurate spelling -- the AMG always runs on the
      // low-order-refined rediscretization, never on the HO operator.
      // "amg"/"boomer_amg" stay accepted so existing decks keep parsing.
      else if (pc == "loramg" || pc == "lor_amg" || pc == "amg"
               || pc == "boomer_amg")
      {
         p.velocity_prec = VelocityPreconditioner::LORAMG;
      }
      else { MFEM_ABORT("parameters: unknown solver.preconditioner '" << pc << "'"); }
   }
   Maybe(solver, "amg_reuse", p.amg_reuse);
   Maybe(solver, "rotation_lor", p.rotation_in_lor);
   if (solver && solver["rotation_pc"])
   {
      const std::string rp = solver["rotation_pc"].as<std::string>();
      if (rp == "symmetric") { p.rotation_pc = RotationVelocityPC::Symmetric; }
      else if (rp == "pbj_only") { p.rotation_pc = RotationVelocityPC::PbjOnly; }
      else if (rp == "pbj_krylov")
      {
         p.rotation_pc = RotationVelocityPC::PbjKrylov;
      }
      else { MFEM_ABORT("parameters: unknown solver.rotation_pc '" << rp << "'"); }
   }
   if (solver && solver["rotation_schur"])
   {
      using RSP = RotationalSchurPreconditioner;
      const YAML::Node rs = solver["rotation_schur"];
      auto mode = [](const std::string & m)
      {
         if (m == "cc") { return RSP::Mode::CahouetChabard; }
         if (m == "tensor") { return RSP::Mode::Tensor; }
         if (m == "auto") { return RSP::Mode::Auto; }
         MFEM_ABORT("parameters: unknown solver.rotation_schur mode '" << m
                    << "' (cc|tensor|auto)");
         return RSP::Mode::CahouetChabard;
      };
      RSP::Options& o = p.rotation_schur;
      if (rs.IsScalar()) { o.mode = mode(rs.as<std::string>()); }
      else
      {
         if (rs["mode"]) { o.mode = mode(rs["mode"].as<std::string>()); }
         if (rs["criterion"])
         {
            const std::string c = rs["criterion"].as<std::string>();
            if (c == "max_mu") { o.criterion = RSP::Criterion::MaxMu; }
            else if (c == "volume_fraction")
            {
               o.criterion = RSP::Criterion::VolumeFraction;
            }
            else
            {
               MFEM_ABORT("parameters: unknown solver.rotation_schur.criterion '"
                          << c << "' (max_mu|volume_fraction)");
            }
         }
         Maybe(rs, "mu_on", o.mu_on);
         Maybe(rs, "mu_off", o.mu_off);
         Maybe(rs, "vol_on", o.vol_on);
         Maybe(rs, "vol_off", o.vol_off);
         Maybe(rs, "inner_iterations", o.inner_iterations);
      }
      MFEM_VERIFY(o.mu_off <= o.mu_on && o.vol_off <= o.vol_on &&
                  o.inner_iterations >= 1, "parameters: bad "
                  "solver.rotation_schur thresholds / inner_iterations");
   }
   Maybe(solver, "rotation_log_interval", p.rotation_log_interval);
   MFEM_VERIFY(p.rotation_log_interval >= 0,
               "parameters: solver.rotation_log_interval must be >= 0");
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
   if (solver && solver["pc_quadrature"])
   {
      const std::string q = solver["pc_quadrature"].as<std::string>();
      if (q == "inherit") { p.cc.pc_quadrature = PcQuadrature::Inherit; }
      else if (q == "gll_collocated")
      {
         p.cc.pc_quadrature = PcQuadrature::GllCollocated;
      }
      else { MFEM_ABORT("parameters: unknown solver.pc_quadrature '" << q << "'"); }
   }
   Maybe(solver, "lp_vcycles", p.cc.lp_vcycles);
   if (solver && solver["a_pc"])
   {
      const std::string ap = solver["a_pc"].as<std::string>();
      if (ap == "loramg") { p.cc.a_pc = APC::LORAMG; }
      else if (ap == "jacobi_chebyshev") { p.cc.a_pc = APC::JacobiChebyshev; }
      else if (ap == "jacobi_pcg") { p.cc.a_pc = APC::JacobiPCG; }
      else
      {
         MFEM_ABORT("parameters: unknown solver.a_pc '" << ap
                    << "' (loramg|jacobi_chebyshev|jacobi_pcg)");
      }
   }
   Maybe(solver, "a_pcg_rtol", p.cc.a_pcg_rtol);
   Maybe(solver, "a_pcg_max_iter", p.cc.a_pcg_max_iter);
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

   const YAML::Node amr = root["amr"];
   Maybe(amr, "enabled", p.amr.enabled);
   Maybe(amr, "interval", p.amr.interval);
   Maybe(amr, "initial_passes", p.amr.initial_passes);
   Maybe(amr, "passes_per_event", p.amr.passes_per_event);
   Maybe(amr, "anisotropic", p.amr.anisotropic);
   Maybe(amr, "aniso_ratio", p.amr.aniso_ratio);
   if (amr && amr["threshold_mode"])
   {
      const std::string tm = amr["threshold_mode"].as<std::string>();
      if (tm == "relative") { p.amr.threshold_mode = AmrThreshold::Relative; }
      else if (tm == "absolute") { p.amr.threshold_mode = AmrThreshold::Absolute; }
      else { MFEM_ABORT("parameters: unknown amr.threshold_mode '" << tm << "'"); }
   }
   Maybe(amr, "theta", p.amr.theta);
   Maybe(amr, "tolerance", p.amr.tolerance);
   Maybe(amr, "min_size", p.amr.min_size);
   Maybe(amr, "max_elements", p.amr.max_elements);
   Maybe(amr, "nc_limit", p.amr.nc_limit);
   Maybe(amr, "rebalance", p.amr.rebalance);
   Maybe(amr, "project_history", p.amr.project_history);
   Maybe(amr, "write_indicator", p.amr.write_indicator);

   const YAML::Node forces = root["forces"];
   Maybe(forces, "enabled", p.forces.enabled);
   if (forces && forces["attributes"])
   {
      p.forces.attributes = forces["attributes"].as<std::vector<int>>();
   }
   Maybe(forces, "reference_velocity", p.forces.reference_velocity);
   Maybe(forces, "reference_area", p.forces.reference_area);
   Maybe(forces, "interval", p.forces.interval);

   const YAML::Node chk = root["checkpoint"];
   Maybe(chk, "enabled", p.checkpoint.enabled);
   Maybe(chk, "path", p.checkpoint.path);
   Maybe(chk, "interval", p.checkpoint.interval);
   Maybe(root, "restart", p.restart_from);

   // Basic validation (module-level invariants are re-checked downstream).
   MFEM_VERIFY(p.nu > 0.0, "parameters: nu must be positive");
   MFEM_VERIFY(p.order_u >= 1 && p.order_p >= 1, "parameters: bad orders");
   MFEM_VERIFY(p.dt > 0.0 && p.t_final > 0.0, "parameters: bad time settings");
   MFEM_VERIFY(p.cfl_max >= 0.0, "parameters: time.cfl_max must be >= 0");
   MFEM_VERIFY(p.cfl_target > 0.0 && p.dt_max >= 0.0,
               "parameters: time.cfl_target must be > 0 and time.dt_max >= 0 "
               "(choose the mode with time.step_control)");
   MFEM_VERIFY(!p.CflSteps() || p.cfl_max == 0.0 || p.cfl_target <= p.cfl_max,
               "parameters: time.cfl_target exceeds time.cfl_max");
   MFEM_VERIFY(p.ext_order == 2 || p.ext_order == 3,
               "parameters: time.ext_order must be 2 or 3");
   MFEM_VERIFY(p.output.interval >= 1, "parameters: bad output interval");
   MFEM_VERIFY(p.checkpoint.interval >= 1, "parameters: bad checkpoint interval");
   p.amr.Validate();
   MFEM_VERIFY(!p.forces.enabled || !p.forces.attributes.empty(),
               "parameters: forces.enabled needs forces.attributes");
   MFEM_VERIFY(p.forces.reference_velocity > 0.0 &&
               p.forces.reference_area > 0.0 && p.forces.interval >= 0,
               "parameters: bad forces reference scales or interval");
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
      amr.min_size /= nondim.L_ref;
      // Force coefficients use nondimensional references: U / U_ref and the
      // area (a length in 2D) / L_ref^(dim - 1).
      forces.reference_velocity /= nondim.U_ref;
      forces.reference_area /= std::pow(nondim.L_ref, mesh.dim - 1);
      dt *= U_over_L;
      t_final *= U_over_L;
      dt_max *= U_over_L;
      nu = 1.0 / nondim.Re;
      controller.atol /= nondim.U_ref; // the LTE norm carries velocity units
   }
   nondim.normalized = true;
}

} // namespace incns
