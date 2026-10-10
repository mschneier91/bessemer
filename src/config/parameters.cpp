#include "config/parameters.hpp"

#include "config/deck_reader.hpp"
#include "mesh/mesh_file.hpp"

#include "mfem.hpp" // MFEM_VERIFY

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <set>

namespace incns
{

namespace
{
// Every deck key is declared here, whether or not the deck sets it, so the
// same code reads a deck and -- run on an empty deck -- produces the deck
// reference (Parameters::DeckSchema). Descriptions are one line each.
void FillFromDeck(DeckReader& r, Parameters& p)
{
   const DeckSection root = r.Root();

   r.ReadEnum(root, "equation", p.equation,
   {
      {"stokes", Equation::Stokes}, {"navier_stokes", Equation::NavierStokes},
      {"nse", Equation::NavierStokes}
   },
   "Equation set: unsteady Stokes or incompressible Navier-Stokes.");
   r.Read(root, "device", p.device,
          "MFEM device backend (cpu, cuda, debug); the env var INCNS_DEVICE wins.");
   r.Read(root, "initial_velocity", p.initial_velocity,
          "Named initial condition: zero, uniform, channel (see the initial "
          "section), taylor_green_2d or taylor_green_3d (u = (sin x cos y cos z, "
          "-cos x sin y cos z, 0) on a 2 pi periodic box); anything else comes "
          "from Python or C++.");
   r.Read(root, "restart", p.restart_from,
          "Checkpoint directory to restart from (same number of MPI ranks).");

   const DeckSection physics = r.Section(root, "physics", "Physical parameters.");
   r.Read(physics, "nu", p.nu,
          "Kinematic viscosity (dimensionless: 1/Re).");
   r.Declare(physics, "Re", "real", "unset", "",
             "Reynolds number; sets nu = 1/Re (dimensionless mode; not with nu).");
   r.Read(physics, "grad_div", p.grad_div,
          "Grad-div stabilization scale c_gd (0 = off).");
   r.ReadEnum(physics, "grad_div_scale", p.grad_div_scale,
   {{"h", GradDivScale::OrderH}, {"nu", GradDivScale::OrderNu}},
   "Grad-div coefficient: c_gd * h_K per element, or c_gd * nu.");
   r.ReadEnum(physics, "convective_form", p.convective_form,
   {
      {"convective", ConvectiveForm::Convective},
      {"rotational", ConvectiveForm::Rotational}
   },
   "Nonlinear term: explicit (u.grad)u, or the semi-implicit rotational form "
   "(the pressure becomes the Bernoulli head).");
   r.ReadEnum(physics, "outflow", p.outflow,
   {
      {"directional", OutflowCondition::Directional},
      {"classical", OutflowCondition::Classical}
   },
   "Condition on outflow boundaries: Braack-Mucha directional do-nothing "
   "(stable under backflow) or classical do-nothing.");

   const DeckSection nd = r.Section(root, "nondimensionalization",
                                    "Input scaling: dimensionless (default) or "
                                    "dimensional with reference scales.");
   r.ReadEnum(nd, "mode", p.nondim.mode,
   {
      {"dimensionless", ScalingMode::Dimensionless},
      {"dimensional", ScalingMode::Dimensional}
   },
   "dimensional: lengths, times and nu carry units and are rescaled by L_ref, U_ref.");
   r.Read(nd, "L_ref", p.nondim.L_ref, "Reference length (dimensional mode).");
   r.Read(nd, "U_ref", p.nondim.U_ref, "Reference velocity (dimensional mode).");
   r.Read(nd, "rho", p.nondim.rho, "Density, for dimensional forces.");

   // Convenience: `physics: Re` in dimensionless mode sets nu = 1/Re.
   if (physics.Has("Re"))
   {
      MFEM_VERIFY(!physics.Has("nu"),
                  "parameters: give either physics.nu or physics.Re, not both");
      MFEM_VERIFY(p.nondim.mode == ScalingMode::Dimensionless,
                  "parameters: physics.Re is a dimensionless-mode input");
      p.nu = 1.0 / physics.Get("Re").as<double>();
   }

   const DeckSection disc = r.Section(root, "discretization",
                                      "Finite element orders and mass.");
   r.Read(disc, "order_u", p.order_u, "Velocity polynomial order k_u.");
   r.Read(disc, "order_p", p.order_p, "Pressure polynomial order k_p (k_u - 1).");
   MFEM_VERIFY(!disc.Has("collocated_mass"), "parameters: "
               "discretization.collocated_mass was replaced by "
               "discretization.mass: auto | collocated | consistent");
   r.ReadEnum(disc, "mass", p.mass,
   {
      {"auto", VelocityMass::Auto}, {"collocated", VelocityMass::Collocated},
      {"consistent", VelocityMass::Consistent}
   },
   "Velocity mass matrix. auto: collocated under OIFS, consistent under IMEX. "
   "collocated: GLL quadrature, diagonal on conforming meshes (OIFS's substeps "
   "then invert it pointwise). consistent: exact quadrature (under OIFS, a "
   "mass solve per substep stage). OIFS's substeps always invert this same "
   "mass.");

   const DeckSection mesh = r.Section(root, "mesh",
                                      "Box mesh (quads/hexes) for apps/run_case.");
   r.Read(mesh, "dim", p.mesh.dim, "Spatial dimension, 2 or 3.");
   r.ReadArray(mesh, "elements", p.mesh.num_elems, p.mesh.dim,
               "Elements per direction.");
   r.ReadArray(mesh, "lengths", p.mesh.lengths, p.mesh.dim,
               "Box edge lengths (the box starts at the origin).");
   r.ReadArray(mesh, "periodic", p.mesh.periodic, p.mesh.dim,
               "Periodic directions; non-periodic faces need boundary_conditions.");
   r.Declare(mesh, "stretch", "enum[dim]", "[none, none]", "none | tanh",
             "Per-direction node clustering: uniform, or two-sided tanh toward "
             "both ends.");
   if (mesh.Has("stretch"))
   {
      const YAML::Node seq = mesh.Get("stretch");
      MFEM_VERIFY(seq.IsSequence() && static_cast<int>(seq.size()) == p.mesh.dim,
                  "parameters: mesh.stretch must be a sequence of " << p.mesh.dim);
      for (int d = 0; d < p.mesh.dim; ++d)
      {
         const std::string kind = seq[d].as<std::string>();
         if (kind == "none") { p.mesh.stretch[d] = Stretch::None; }
         else if (kind == "tanh") { p.mesh.stretch[d] = Stretch::TwoSidedTanh; }
         else { MFEM_ABORT("parameters: unknown mesh.stretch '" << kind << "' (none|tanh)"); }
      }
   }
   r.ReadArray(mesh, "stretch_beta", p.mesh.stretch_beta, p.mesh.dim,
               "tanh clustering strength per direction.");
   r.ReadEnum(mesh, "geometry", p.geometry,
   {
      {"box", MeshGeometry::Box}, {"square_cylinder", MeshGeometry::SquareCylinder},
      {"cylinder_channel", MeshGeometry::CylinderChannel}, {"file", MeshGeometry::File}
   },
   "Domain: the box above; a square cylinder in a large domain (Joly et al. "
   "2012; 2D; boundaries inflow, outflow, sides, body); the DFG channel with a "
   "cylinder (2D, or extruded in z in 3D; boundaries inflow, outflow, walls, "
   "cylinder); or a mesh file (mesh.file; boundaries named by the file's Gmsh "
   "physical groups and mesh.boundary_names).");
   r.Read(mesh, "file", p.mesh_file,
          "geometry file: a Gmsh (.msh, ASCII 2.2 or 4.x) or MFEM (.mesh) file of "
          "quadrilaterals (2D) or hexahedra (3D); relative paths are relative to "
          "the working directory.");
   r.Declare(mesh, "boundary_names", "map", "{}", "name: attribute",
             "geometry file: boundary names for attribute numbers, added to (and "
             "overriding) the Gmsh file's named physical groups, e.g. {inlet: 1, "
             "walls: 3}.");
   {
      SquareCylinderSpec& sq = p.square_cylinder;
      const DeckSection s = r.Section(mesh, "square_cylinder",
                                      "geometry square_cylinder: domain and base mesh.");
      r.Read(s, "side", sq.side, "Square side D.");
      r.Read(s, "upstream", sq.upstream, "Inlet distance from the square's centre.");
      r.Read(s, "downstream", sq.downstream, "Outlet distance from the centre.");
      r.Read(s, "half_height", sq.half_height,
             "Side boundaries at y = +-half_height.");
      r.Read(s, "n_face", sq.n_face, "Cells along each face of the square (even).");
      r.Read(s, "corner_ratio", sq.corner_ratio,
             "Cell growth from the corners to mid-face.");
      r.Read(s, "far_ratio", sq.far_ratio,
             "Cell growth upstream, sideways and far downstream.");
      r.Read(s, "wake_ratio", sq.wake_ratio,
             "Cell growth from the square into the near wake.");
      r.Read(s, "wake_h", sq.wake_h, "Cell width in the near wake.");
      r.Read(s, "wake_end", sq.wake_end,
             "x where the near wake's uniform cells end.");
   }
   {
      CylinderChannelSpec& cc = p.cylinder_channel;
      const DeckSection s = r.Section(mesh, "cylinder_channel",
                                      "geometry cylinder_channel: the DFG channel "
                                      "(default 2.2 x 0.41, cylinder of radius 0.05 "
                                      "at (0.2, 0.2); the DFG 3D benchmarks use "
                                      "length 2.5, cx 0.5) and its base mesh.");
      r.Read(s, "length", cc.length, "Channel length (x).");
      r.Read(s, "cx", cc.cx, "Cylinder centre x.");
      r.Read(s, "cy", cc.cy, "Cylinder centre y.");
      r.Read(s, "n_side", cc.n_side, "Cells along each side of the O-grid square.");
      r.Read(s, "n_ring", cc.n_ring, "Radial cell layers around the cylinder.");
      r.Read(s, "n_up", cc.n_up, "Cells upstream of the O-grid.");
      r.Read(s, "n_down", cc.n_down, "Cells downstream of the O-grid.");
      r.Read(s, "level", p.cylinder_channel_level,
             "Nested uniform refinement level (cell counts x 2^level).");
      r.Read(s, "nz", p.cylinder_channel_nz,
             "3D: element layers in z (x 2^level).");
      r.Read(s, "depth", p.cylinder_channel_depth,
             "3D: extent in z; the z = 0 and z = depth faces join the walls.");
   }

   if (p.geometry == MeshGeometry::SquareCylinder)
   {
      MFEM_VERIFY(p.mesh.dim == 2, "parameters: the square cylinder is 2D "
                  "(mesh.dim 2)");
   }
   if (p.geometry == MeshGeometry::CylinderChannel)
   {
      MFEM_VERIFY(p.mesh.dim == 2 || p.mesh.dim == 3, "parameters: the "
                  "cylinder channel is 2D or 3D");
      MFEM_VERIFY(p.cylinder_channel_nz >= 1 && p.cylinder_channel_depth > 0.0,
                  "parameters: cylinder_channel nz >= 1 and depth > 0");
   }
   if (p.geometry == MeshGeometry::File)
   {
      MFEM_VERIFY(!p.mesh_file.empty(), "parameters: geometry file needs "
                  "mesh.file");
      if (p.mesh_file.size() > 4 &&
          p.mesh_file.compare(p.mesh_file.size() - 4, 4, ".msh") == 0)
      {
         p.boundary_names = GmshPhysicalNames(p.mesh_file, p.mesh.dim - 1);
      }
   }
   if (mesh.Has("boundary_names"))
   {
      MFEM_VERIFY(p.geometry == MeshGeometry::File, "parameters: "
                  "mesh.boundary_names is for geometry file");
      const YAML::Node names = mesh.Get("boundary_names");
      MFEM_VERIFY(names.IsMap(), "parameters: mesh.boundary_names must be a "
                  "map of name: attribute");
      for (const auto& kv : names)
      {
         const std::string name = kv.first.as<std::string>();
         const int attr = kv.second.as<int>();
         MFEM_VERIFY(attr > 0, "parameters: boundary attribute of '" << name
                     << "' must be positive");
         bool replaced = false;
         for (auto& nb : p.boundary_names)
         {
            if (nb.first == name) { nb.second = attr; replaced = true; }
         }
         if (!replaced) { p.boundary_names.emplace_back(name, attr); }
      }
   }

   const DeckSection init = r.Section(root, "initial",
                                      "Settings of the named initial conditions. "
                                      "uniform: a uniform flow with an optional "
                                      "shedding trigger. channel (walls at y = 0 "
                                      "and y = 2 delta, x streamwise): Reichardt's "
                                      "mean profile for Re_tau = u_tau delta / nu, "
                                      "u_tau = sqrt(f_x delta) from forcing, plus a "
                                      "divergence-free perturbation that vanishes "
                                      "with its normal derivative at the walls.");
   r.ReadArray(init, "velocity", p.initial.velocity, p.mesh.dim,
               "uniform: the uniform velocity.");
   r.Read(init, "perturbation", p.initial.perturbation,
          "uniform: amplitude (relative to |velocity|) of a Gaussian bump "
          "exp(-|x - c|^2) added to the second velocity component, which starts "
          "vortex shedding early. channel: amplitude of the perturbation relative "
          "to the centreline velocity. 0 = none.");
   r.ReadArray(init, "perturbation_center", p.initial.perturbation_center,
               p.mesh.dim, "uniform: the bump's centre c.");
   r.Read(init, "seed", p.initial.seed,
          "channel: seed of the perturbation's random mode phases.");

   const DeckSection forcing = r.Section(root, "forcing",
                                         "A constant body force per unit mass.");
   r.ReadArray(forcing, "body_force", p.forcing.body_force, p.mesh.dim,
               "The force f, e.g. a mean pressure gradient: [1, 0, 0] drives a "
               "channel of half-height 1 at u_tau = 1, so Re_tau = 1 / nu. Zero = "
               "none.");

   const DeckSection chs = r.Section(root, "channel_statistics",
                                     "Channel statistics (walls normal to y, x "
                                     "and z homogeneous; a box mesh without "
                                     "refinement): time averages of x-z "
                                     "plane-averaged U, V, W and the Reynolds "
                                     "stresses at every velocity-node height, the "
                                     "wall shear stress, u_tau and Re_tau. Written "
                                     "to <path>/<name>_profiles.csv and the run "
                                     "summary; carried through checkpoints.");
   r.Read(chs, "enabled", p.channel_statistics.enabled, "Collect them.");
   r.Read(chs, "start_time", p.channel_statistics.start_time,
          "Average over t >= start_time (skip the transient).");
   r.Read(chs, "interval", p.channel_statistics.interval,
          "Sample every this many steps.");

   const DeckSection time = r.Section(root, "time", "Time integration.");
   r.Read(time, "dt", p.dt,
          "Step size (fixed steps), or the first step (cfl / error control).");
   r.Read(time, "t_final", p.t_final, "End time.");
   r.ReadEnum(time, "step_control", p.step_control,
   {
      {"cfl", StepControl::Cfl}, {"fixed", StepControl::Fixed},
      {"error", StepControl::Error}
   },
   "How dt is chosen: dt = cfl_target / CFL rate before every step (Navier-Stokes), "
   "constant, or BDF2/BDF3 error control (IMEX only).");
   r.Declare(time, "adaptive", "bool", "unset", "",
             "Older spelling: true = step_control error, false = fixed.");
   if (time.Has("adaptive"))
   {
      MFEM_VERIFY(!time.Has("step_control"), "parameters: give time.step_control "
                  "or the older time.adaptive, not both");
      p.step_control = time.Get("adaptive").as<bool>() ? StepControl::Error
                       : StepControl::Fixed;
   }
   r.Read(time, "cfl_target", p.cfl_target,
          "Target CFL number (Nek5000's definition) under step_control cfl: "
          "~0.5 for IMEX, 2 for OIFS.");
   r.Read(time, "cfl_max", p.cfl_max,
          "CFL ceiling (0 = off): caps error-controlled dt, aborts fixed steps above it.");
   r.Read(time, "dt_max", p.dt_max,
          "Largest step under cfl control (0 = no cap).");
   r.ReadEnum(time, "convection", p.convection_treatment,
   {{"imex", ConvectionTreatment::Imex}, {"oifs", ConvectionTreatment::Oifs}},
   "Convection: explicit and extrapolated (IMEX, CFL < ~0.7), or OIFS "
   "sub-stepping (CFL 2 practical; BDF3 and, with discretization.mass auto, the "
   "collocated mass by default).");
   r.Read(time, "oifs_cfl", p.oifs_cfl, "CFL number of each OIFS substep.");
   r.Read(time, "order", p.time_order,
          "BDF order 2 or 3; 0 = auto (3 under OIFS, 2 under IMEX).");
   r.Read(time, "ext_order", p.ext_order,
          "Extrapolation order of the explicit/lagged nonlinear term, 2 or 3.");
   r.Read(time, "atol", p.controller.atol,
          "Error control: absolute tolerance on the velocity error estimate.");
   r.Read(time, "rtol", p.controller.rtol,
          "Error control: relative tolerance on the velocity error estimate.");

   const DeckSection solver = r.Section(root, "solver",
                                        "Linear solver: FGMRES on the monolithic "
                                        "velocity-pressure system with a block "
                                        "preconditioner.");
   r.Read(solver, "rtol", p.krylov_rtol, "FGMRES relative tolerance.");
   r.Read(solver, "atol", p.krylov_atol, "FGMRES absolute tolerance.");
   r.Read(solver, "max_iter", p.max_iter, "FGMRES iteration cap.");
   r.Read(solver, "kdim", p.kdim, "FGMRES restart size.");
   r.Read(solver, "print_level", p.print_level, "Solver verbosity (-1 = quiet).");
   r.Declare(solver, "schur", "enum", "cc", "cc | mass | laplacian_legacy",
             "Pressure Schur block: Cahouet-Chabard, scaled pressure mass, or the "
             "legacy Laplacian CC (comparison only).");
   if (solver.Has("schur"))
   {
      const std::string sm = solver.Get("schur").as<std::string>();
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
      else
      {
         MFEM_ABORT("parameters: unknown solver.schur '" << sm
                    << "' (cc|mass|laplacian_legacy)");
      }
   }
   r.ReadEnum(solver, "a_pc", p.cc.a_pc,
   {
      {"jacobi_pcg", APC::JacobiPCG}, {"jacobi_chebyshev", APC::JacobiChebyshev},
      {"loramg", APC::LORAMG}
   },
   "Velocity-block preconditioner (Cahouet-Chabard path): Jacobi-PCG, "
   "Jacobi-Chebyshev, or one LOR-AMG V-cycle (best when viscous-dominated).");
   r.Read(solver, "a_pcg_rtol", p.cc.a_pcg_rtol,
          "jacobi_pcg: inner CG tolerance.");
   r.Read(solver, "a_pcg_max_iter", p.cc.a_pcg_max_iter,
          "jacobi_pcg: inner CG iteration cap.");
   r.ReadEnum(solver, "preconditioner", p.velocity_prec,
   {
      {"jacobi", VelocityPreconditioner::Jacobi},
      {"loramg", VelocityPreconditioner::LORAMG},
      {"lor_amg", VelocityPreconditioner::LORAMG},
      {"amg", VelocityPreconditioner::LORAMG},
      {"boomer_amg", VelocityPreconditioner::LORAMG}
   },
   "Velocity-block preconditioner on the mass Schur path.");
   r.ReadEnum(solver, "block_shape", p.cc.block_shape,
   {
      {"upper", BlockPCShape::UpperTri}, {"lower", BlockPCShape::LowerTri},
      {"diag", BlockPCShape::Diag}
   },
   "Block preconditioner shape.");
   r.Read(solver, "n_inner", p.cc.n_inner,
          "Cahouet-Chabard: fixed inner CG iterations on the pressure Poisson block.");
   r.Read(solver, "lp_vcycles", p.cc.lp_vcycles,
          "Cahouet-Chabard: LOR-AMG V-cycles preconditioning the inner CG.");
   r.ReadEnum(solver, "pc_quadrature", p.cc.pc_quadrature,
   {
      {"inherit", PcQuadrature::Inherit},
      {"gll_collocated", PcQuadrature::GllCollocated}
   },
   "Quadrature of the preconditioner's operators.");
   r.Read(solver, "amg_reuse", p.amg_reuse,
          "Freeze the LOR-AMG hierarchy across dt changes.");
   r.ReadEnum(solver, "rotation_pc", p.rotation_pc,
   {
      {"symmetric", RotationVelocityPC::Symmetric},
      {"pbj_only", RotationVelocityPC::PbjOnly},
      {"pbj_krylov", RotationVelocityPC::PbjKrylov}
   },
   "Rotational form: velocity preconditioner (pbj_krylov recommended).");
   r.Read(solver, "rotation_lor", p.rotation_in_lor,
          "Rotational form: put the rotation term into LOR-AMG (measured not to pay off).");
   r.Read(solver, "rotation_log_interval", p.rotation_log_interval,
          "Rotational form: log rotation-number statistics every N steps (0 = off).");
   {
      using RSP = RotationalSchurPreconditioner;
      RSP::Options& o = p.rotation_schur;
      const std::initializer_list<std::pair<const char*, RSP::Mode>> modes =
      {{"cc", RSP::Mode::CahouetChabard}, {"tensor", RSP::Mode::Tensor}, {"auto", RSP::Mode::Auto}};
      // Either a mode name, or a map with the mode and its switch options.
      r.Declare(solver, "rotation_schur", "enum or map", "cc", "cc | tensor | auto",
                "Rotational form: Schur preconditioner (a mode, or a map with the "
                "keys below).");
      const DeckSection rs = r.Section(solver, "rotation_schur",
                                       "Rotation-aware Schur preconditioner options.");
      if (solver.Has("rotation_schur") && solver.Get("rotation_schur").IsScalar())
      {
         const std::string m = solver.Get("rotation_schur").as<std::string>();
         bool found = false;
         for (const auto& n : modes)
         {
            if (m == n.first) { o.mode = n.second; found = true; }
         }
         MFEM_VERIFY(found, "parameters: unknown solver.rotation_schur '" << m
                     << "' (cc|tensor|auto)");
      }
      r.ReadEnum(rs, "mode", o.mode, modes, "Schur mode.");
      r.ReadEnum(rs, "criterion", o.criterion,
      {
         {"max_mu", RSP::Criterion::MaxMu},
         {"volume_fraction", RSP::Criterion::VolumeFraction}
      },
      "auto: switch on the maximum rotation number or the volume fraction above it.");
      r.Read(rs, "mu_on", o.mu_on,
             "auto: switch to tensor above this rotation number.");
      r.Read(rs, "mu_off", o.mu_off, "auto: switch back below this rotation number.");
      r.Read(rs, "vol_on", o.vol_on, "auto (volume_fraction): switch-on fraction.");
      r.Read(rs, "vol_off", o.vol_off,
             "auto (volume_fraction): switch-off fraction.");
      r.Read(rs, "inner_iterations", o.inner_iterations,
             "tensor: fixed inner FGMRES iterations.");
      MFEM_VERIFY(o.mu_off <= o.mu_on && o.vol_off <= o.vol_on &&
                  o.inner_iterations >= 1, "parameters: bad "
                  "solver.rotation_schur thresholds / inner_iterations");
   }
   MFEM_VERIFY(p.rotation_log_interval >= 0,
               "parameters: solver.rotation_log_interval must be >= 0");

   // Boundary-condition groups: topology + type only (fields are bound in the
   // driver by group name). A selector token is an attribute integer if it
   // parses fully as one, otherwise a box face name.
   r.Declare(root, "boundary_conditions", "list", "[]", "",
             "Boundary groups, each {select: [names or attributes], type: no_slip | "
             "outflow | velocity | velocity_dirichlet, ...}. Names: box faces xmin, "
             "xmax, ymin, ymax, zmin, zmax; the geometry's boundary names; or all. "
             "type velocity takes value: [u, v] (constant), or profile: parabolic "
             "with u_max and height (u_x = 4 u_max y (height - y) / height^2; in 3D "
             "times 4 z (width - z) / width^2, width defaulting to height), "
             "optionally time_profile: ramp | sine with time_scale T (ramp: "
             "sin^2(pi t / 2T) for t < T; sine: sin(pi t / T)). velocity_dirichlet "
             "takes its field from Python or C++ by the group name. Every real "
             "boundary needs exactly one group when run by run_case.");
   if (root.Has("boundary_conditions"))
   {
      const YAML::Node bcs = root.Get("boundary_conditions");
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
         static const std::set<std::string> entry_keys =
         {
            "group", "type", "select", "value", "profile", "u_max", "height",
            "width", "time_profile", "time_scale"
         };
         for (const auto& kv : e)
         {
            const std::string k = kv.first.as<std::string>();
            MFEM_VERIFY(entry_keys.count(k), "parameters: unknown "
                        "boundary_conditions key '" << k << "' (group, type, "
                        "select, value, profile, u_max, height, width, "
                        "time_profile, time_scale)");
         }
         BcSpec s;
         if (e["group"]) { s.group = e["group"].as<std::string>(); }
         MFEM_VERIFY(e["type"], "parameters: a boundary_conditions entry needs 'type'");
         const std::string ty = e["type"].as<std::string>();
         if (ty == "outflow") { s.type = BcType::Outflow; }
         else if (ty == "no_slip") { s.type = BcType::NoSlip; }
         else if (ty == "velocity") { s.type = BcType::Velocity; }
         else
         {
            MFEM_VERIFY(ty == "velocity_dirichlet",
                        "parameters: unknown boundary type '" << ty
                        << "' (no_slip|outflow|velocity|velocity_dirichlet)");
            s.type = BcType::VelocityDirichlet;
         }
         if (s.type == BcType::Velocity)
         {
            const std::string prof = e["profile"] ? e["profile"].as<std::string>()
                                     : std::string("constant");
            if (prof == "constant")
            {
               s.profile = BcProfile::Constant;
               MFEM_VERIFY(e["value"] && e["value"].IsSequence() &&
                           static_cast<int>(e["value"].size()) == p.mesh.dim,
                           "parameters: a constant velocity boundary needs value: "
                           "a list of " << p.mesh.dim << " components");
               for (int d = 0; d < p.mesh.dim; ++d)
               {
                  s.value[d] = e["value"][d].as<double>();
               }
            }
            else if (prof == "parabolic")
            {
               s.profile = BcProfile::Parabolic;
               MFEM_VERIFY(e["u_max"] && e["height"], "parameters: profile "
                           "parabolic needs u_max and height");
               s.u_max = e["u_max"].as<double>();
               s.height = e["height"].as<double>();
               if (e["width"]) { s.width = e["width"].as<double>(); }
            }
            else
            {
               MFEM_ABORT("parameters: unknown boundary profile '" << prof
                          << "' (constant|parabolic)");
            }
            if (e["time_profile"])
            {
               const std::string tp = e["time_profile"].as<std::string>();
               if (tp == "constant") { s.time_profile = BcTimeProfile::Constant; }
               else if (tp == "ramp") { s.time_profile = BcTimeProfile::Ramp; }
               else if (tp == "sine") { s.time_profile = BcTimeProfile::Sine; }
               else
               {
                  MFEM_ABORT("parameters: unknown boundary time_profile '" << tp
                             << "' (constant|ramp|sine)");
               }
            }
            if (e["time_scale"]) { s.time_scale = e["time_scale"].as<double>(); }
            MFEM_VERIFY(s.time_scale > 0.0, "parameters: time_scale must be > 0");
         }
         else
         {
            for (const char* k :
                 {"value", "profile", "u_max", "height", "width",
                  "time_profile", "time_scale"
                 })
            {
               MFEM_VERIFY(!e[k], "parameters: boundary key '" << k
                           << "' needs type velocity");
            }
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

   const DeckSection output = r.Section(root, "output",
                                        "ParaView output and diagnostics.");
   r.Read(output, "enabled", p.output.enabled, "Write ParaView output.");
   r.Read(output, "path", p.output.path, "Output directory prefix.");
   r.Read(output, "name", p.output.name, "Collection name.");
   r.Read(output, "interval", p.output.interval, "Write every N accepted steps.");
   r.Read(output, "diagnostics", p.output.diagnostics,
          "Log t, kinetic energy (1/2 int |u|^2), dissipation (nu int |grad u|^2) "
          "and ||div u|| to <path>/<name>_diagnostics.csv every interval steps "
          "(independent of enabled; appended on restart).");
   r.Read(output, "progress", p.output.progress,
          "Print a progress line every this many time units (0 = never): t, dt "
          "and its range, CFL and where it peaks, forces, iterations, cells, wall "
          "time, ETA.");
   r.Read(output, "history", p.output.history,
          "Write a per-step <path>/<name>_history.csv (t, dt, c_d, c_l, "
          "iterations, cells, step wall time).");

   const DeckSection amr = r.Section(root, "amr",
                                     "Adaptive mesh refinement (refinement only).");
   r.Read(amr, "enabled", p.amr.enabled,
          "Turn AMR on (the mesh becomes nonconforming).");
   r.Read(amr, "interval", p.amr.interval, "Adapt every N accepted steps.");
   r.Read(amr, "initial_passes", p.amr.initial_passes,
          "Refinement passes on the initial condition.");
   r.Read(amr, "passes_per_event", p.amr.passes_per_event,
          "Refinement passes per adaptation event.");
   r.Read(amr, "anisotropic", p.amr.anisotropic,
          "Split only the directions with large gradients.");
   r.Read(amr, "aniso_ratio", p.amr.aniso_ratio,
          "Anisotropic: split direction d if its gradient >= ratio * the largest.");
   r.ReadEnum(amr, "threshold_mode", p.amr.threshold_mode,
   {{"relative", AmrThreshold::Relative}, {"absolute", AmrThreshold::Absolute}},
   "Mark where the indicator >= theta * its max (relative) or >= tolerance (absolute).");
   r.Read(amr, "theta", p.amr.theta, "Relative marking fraction.");
   r.Read(amr, "tolerance", p.amr.tolerance,
          "Absolute marking threshold (velocity units).");
   r.Read(amr, "min_size", p.amr.min_size,
          "Do not split below this element extent.");
   r.Read(amr, "max_elements", p.amr.max_elements, "Element cap (0 = none).");
   r.Read(amr, "nc_limit", p.amr.nc_limit,
          "Maximum hanging-node level difference.");
   r.Read(amr, "rebalance", p.amr.rebalance,
          "Rebalance the partition after refining.");
   r.Read(amr, "project_history", p.amr.project_history,
          "Divergence-free projection of the time history after an event.");
   r.Read(amr, "write_indicator", p.amr.write_indicator,
          "Write the refinement indicator with the output.");
   r.Read(amr, "every_time", p.amr.every_time,
          "Adapt every this many time units instead of every interval steps "
          "(0 = step-based); use it with CFL-controlled steps.");
   r.Read(amr, "start_time", p.amr.start_time,
          "Time-based events only from this time on (spin up on the coarse mesh).");
   r.Read(amr, "end_time", p.amr.end_time,
          "Time-based events only up to this time (0 = no end).");
   r.Read(amr, "first_event_passes", p.amr.first_event_passes,
          "Refinement passes at the first time-based event (stops when a pass "
          "changes nothing).");

   const DeckSection forces = r.Section(root, "forces",
                                        "Lift and drag on a body (John's volume "
                                        "integral of the momentum residual).");
   r.Read(forces, "enabled", p.forces.enabled, "Compute forces.");
   r.Read(forces, "attributes", p.forces.attributes,
          "Boundary attributes of the body.");
   r.Declare(forces, "boundaries", "string list", "[]", "",
             "The body by boundary name (e.g. [body], [cylinder], [ymin]); "
             "alternative to attributes.");
   if (forces.Has("boundaries"))
   {
      p.forces.boundaries = forces.Get("boundaries").as<std::vector<std::string>>();
   }
   r.Read(forces, "reference_velocity", p.forces.reference_velocity,
          "U in C = 2F / (rho U^2 A).");
   r.Read(forces, "reference_area", p.forces.reference_area,
          "A in C = 2F / (rho U^2 A) (a length in 2D).");
   r.Read(forces, "interval", p.forces.interval,
          "Log to <path>/<name>_forces.csv every N steps (0 = no log).");
   r.Read(forces, "statistics", p.forces.statistics,
          "Evaluate the coefficients every step; the run summary reports final "
          "values, maxima and averages over shedding periods (mean C_D, mean and "
          "rms C_L, Strouhal number = reference_area / (reference_velocity "
          "period) in 2D).");
   r.Read(forces, "average_periods", p.forces.average_periods,
          "Shedding periods (the last ones) the averages cover.");

   const DeckSection chk = r.Section(root, "checkpoint", "Rolling checkpoints.");
   r.Read(chk, "enabled", p.checkpoint.enabled, "Write rolling checkpoints.");
   r.Read(chk, "path", p.checkpoint.path,
          "Checkpoint directory (overwritten each time).");
   r.Read(chk, "interval", p.checkpoint.interval, "Write every N accepted steps.");
   r.Read(chk, "at_time", p.checkpoint.at_time,
          "Also write one checkpoint once t >= at_time, to branch runs from "
          "(0 = off; independent of enabled).");

   const DeckSection probes = r.Section(root, "probes",
                                        "Point values reported in the run summary.");
   r.Declare(probes, "pressure_difference", "two points", "[]", "",
             "[[x1, y1], [x2, y2]]: the summary reports p(x1) - p(x2) at the end "
             "(e.g. the DFG benchmark's front-back pressure difference).");
   if (probes.Has("pressure_difference"))
   {
      const YAML::Node pts = probes.Get("pressure_difference");
      MFEM_VERIFY(pts.IsSequence() && pts.size() == 2, "parameters: "
                  "probes.pressure_difference needs two points");
      for (const auto& pt : pts)
      {
         MFEM_VERIFY(pt.IsSequence() && static_cast<int>(pt.size()) == p.mesh.dim,
                     "parameters: a probe point needs " << p.mesh.dim
                     << " coordinates");
         std::array<double, 3> x = {0.0, 0.0, 0.0};
         for (int d = 0; d < p.mesh.dim; ++d) { x[d] = pt[d].as<double>(); }
         p.probes.pressure_difference.push_back(x);
      }
   }

   const DeckSection ref = r.Section(root, "reference",
                                     "Benchmark values the run summary compares "
                                     "against (relative errors; absolute for times).");
   ReferenceValues& rv = p.reference;
   r.Read(ref, "cd", rv.cd, "Final drag coefficient.");
   r.Read(ref, "cl", rv.cl, "Final lift coefficient.");
   r.Read(ref, "cd_mean", rv.cd_mean, "Period-mean drag coefficient.");
   r.Read(ref, "cl_mean", rv.cl_mean, "Period-mean lift coefficient.");
   r.Read(ref, "cl_rms", rv.cl_rms, "Period-rms lift coefficient.");
   r.Read(ref, "strouhal", rv.strouhal, "Strouhal number.");
   r.Read(ref, "cd_max", rv.cd_max, "Maximum drag coefficient.");
   r.Read(ref, "t_cd_max", rv.t_cd_max, "Time of the maximum drag.");
   r.Read(ref, "cl_max", rv.cl_max, "Maximum lift coefficient.");
   r.Read(ref, "t_cl_max", rv.t_cl_max, "Time of the maximum lift.");
   r.Read(ref, "pressure_difference", rv.pressure_difference,
          "The pressure-difference probe's value.");
   r.Read(ref, "re_tau", rv.re_tau,
          "Channel: Re_tau from the time-averaged wall shear stress.");
}

// Abort with every unknown key a deck sets, each with its closest valid key.
void VerifyNoUnknownKeys(const DeckReader& r, const std::string& where)
{
   const std::vector<std::string> unknown = r.UnknownKeys();
   if (unknown.empty()) { return; }
   std::string list;
   for (const std::string& u : unknown) { list += "\n  " + u; }
   MFEM_ABORT("parameters: unknown key(s) in " << where << ":" << list
              << "\n(every valid key: docs/deck_reference.md, or "
              "run_case --deck-reference)");
}

Parameters FromYAML(const YAML::Node& root, const std::string& where);
} // namespace

Parameters Parameters::LoadYAML(const std::string& path)
{
   return FromYAML(YAML::LoadFile(path), path);
}

Parameters Parameters::LoadYAMLString(const std::string& text)
{
   return FromYAML(YAML::Load(text), "<string>");
}

std::vector<std::string> Parameters::UnknownDeckKeys(const std::string& text)
{
   Parameters p;
   DeckReader r(YAML::Load(text));
   FillFromDeck(r, p);
   return r.UnknownKeys();
}

std::vector<DeckKey> Parameters::DeckSchema()
{
   Parameters p;
   DeckReader r(YAML::Node(YAML::NodeType::Map));
   FillFromDeck(r, p);
   return r.Keys();
}

std::string Parameters::DeckReferenceMarkdown()
{
   return DeckReader::Markdown(DeckSchema());
}

namespace
{
// Read, check for unknown keys, validate, normalize.
Parameters FromYAML(const YAML::Node& root, const std::string& where)
{
   Parameters p;
   DeckReader r(root);
   FillFromDeck(r, p);
   VerifyNoUnknownKeys(r, where);

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
   MFEM_VERIFY(p.time_order == 0 || p.time_order == 2 || p.time_order == 3,
               "parameters: time.order must be 2, 3 or 0 (auto)");
   MFEM_VERIFY(p.oifs_cfl > 0.0, "parameters: time.oifs_cfl must be > 0");
   MFEM_VERIFY(p.convection_treatment == ConvectionTreatment::Imex ||
               (p.equation == Equation::NavierStokes &&
                p.convective_form == ConvectiveForm::Convective &&
                p.step_control != StepControl::Error),
               "parameters: time.convection: oifs needs navier_stokes, the "
               "convective form and fixed or cfl step control");
   MFEM_VERIFY(p.output.interval >= 1, "parameters: bad output interval");
   MFEM_VERIFY(p.checkpoint.interval >= 1, "parameters: bad checkpoint interval");
   p.amr.Validate();
   MFEM_VERIFY(!p.forces.enabled || !p.forces.attributes.empty() ||
               !p.forces.boundaries.empty(),
               "parameters: forces.enabled needs forces.attributes or "
               "forces.boundaries");
   MFEM_VERIFY(p.forces.reference_velocity > 0.0 &&
               p.forces.reference_area > 0.0 && p.forces.interval >= 0,
               "parameters: bad forces reference scales or interval");
   MFEM_VERIFY(!p.forces.statistics || p.forces.enabled,
               "parameters: forces.statistics needs forces.enabled");
   MFEM_VERIFY(p.forces.average_periods >= 1,
               "parameters: forces.average_periods must be >= 1");
   MFEM_VERIFY(p.output.progress >= 0.0 && p.checkpoint.at_time >= 0.0,
               "parameters: output.progress and checkpoint.at_time must be >= 0");
   MFEM_VERIFY(p.channel_statistics.interval >= 1 &&
               p.channel_statistics.start_time >= 0.0,
               "parameters: channel_statistics needs interval >= 1 and "
               "start_time >= 0");
   MFEM_VERIFY(!p.channel_statistics.enabled || !p.amr.enabled,
               "parameters: channel_statistics needs a fixed mesh (amr off): "
               "every node height must span the whole x-z plane");
   p.Normalize();
   return p;
}
} // namespace

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
      MFEM_VERIFY(geometry == MeshGeometry::Box, "parameters: the cylinder "
                  "and file geometries are nondimensional (mode dimensionless)");
      // A force per unit mass is an acceleration: U^2 / L.
      for (double& f : forcing.body_force)
      {
         f *= nondim.L_ref / (nondim.U_ref * nondim.U_ref);
      }
      channel_statistics.start_time *= U_over_L;
      // Times, lengths and velocities of the scheduling, boundary and probe
      // settings.
      amr.every_time *= U_over_L;
      amr.start_time *= U_over_L;
      amr.end_time *= U_over_L;
      output.progress *= U_over_L;
      checkpoint.at_time *= U_over_L;
      for (BcSpec& s : boundary_conditions)
      {
         for (double& v : s.value) { v /= nondim.U_ref; }
         s.u_max /= nondim.U_ref;
         s.height /= nondim.L_ref;
         s.width /= nondim.L_ref;
         s.time_scale *= U_over_L;
      }
      for (auto& x : probes.pressure_difference)
      {
         for (double& c : x) { c /= nondim.L_ref; }
      }
      reference.t_cd_max *= U_over_L;
      reference.t_cl_max *= U_over_L;
   }
   nondim.normalized = true;
}

} // namespace incns
