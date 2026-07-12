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

   Maybe(root, "initial_velocity", p.initial_velocity);

   const YAML::Node output = root["output"];
   Maybe(output, "enabled", p.output.enabled);
   Maybe(output, "path", p.output.path);
   Maybe(output, "name", p.output.name);
   Maybe(output, "interval", p.output.interval);

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
