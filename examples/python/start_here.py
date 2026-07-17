"""
============================================================================
 bessemer -- START HERE.  A worked manufactured-solution example, explained
             line by line.  This is the example to read first.
============================================================================

WHAT THIS DOES
--------------
It solves the unsteady incompressible Stokes equations

      du/dt  -  nu * laplacian(u)  +  grad(p)  =  f      (momentum)
      div(u)                                   =  0      (incompressibility)

on the unit square, and CHECKS the answer against a solution we know exactly.

WHY A "MANUFACTURED" SOLUTION
-----------------------------
There is no analytic solution to Stokes flow for an arbitrary forcing, so we
cannot usually check a CFD code against "the truth". The Method of Manufactured
Solutions (MMS) flips the problem around:

  1. PICK a velocity/pressure field you like  (call it u_s, p_s).
  2. PLUG it into the equations and let whatever is left over BE the forcing f.
     By construction, (u_s, p_s) is then the exact solution for that f.
  3. Run the solver with that f and compare its output to u_s.

If the code is correct the computed velocity must match u_s. Any mismatch beyond
the solver tolerance is a bug -- MMS turns "looks plausible" into a hard number.

THE SOLUTION WE PICKED  (and why it is reproduced EXACTLY, not just accurately)
------------------------------------------------------------------------------
Velocity :  u_s(x, y) = g(t) * ( 3 x^3 y^2 ,  -3 x^2 y^3 )
Pressure :  p_s(x, y) = g(t) * ( x^2 + y^2 - 2/3 )
Time part:  g(t)      = 1 + t + t^2/2      (a quadratic in t)

Two facts make this special:
  * u_s is DIVERGENCE-FREE:  d/dx(3x^3y^2) + d/dy(-3x^2y^3) = 9x^2y^2 - 9x^2y^2 = 0,
    so it is a genuine incompressible flow (no forcing needed to keep div u = 0).
  * u_s lies EXACTLY in the discrete space.  We use Q3 velocity elements (cubic
    per direction); 3x^3y^2 is cubic in x and quadratic in y, so it is
    representable with zero interpolation error.  And the BDF2 time scheme
    differentiates quadratics like g(t) exactly.

Because BOTH the space error and the time error vanish for this particular
choice, the solver must return u_s to the KRYLOV (linear-solver) tolerance --
about 1e-13 here (machine-precision-class), NOT the ~1e-4 you would see for a
generic smooth solution on this coarse mesh.  That razor-sharp check is exactly why MMS is the workhorse of
verification.

HOW TO RUN
----------
    scripts/build.sh cpu-python                 # build the Python module once
    . scripts/env.sh                            # activate the toolchain
    export PYTHONPATH=build/cpu-python/python
    mpirun -np 4 python examples/python/start_here.py

You run it under `mpirun`: N identical Python interpreters start, each one
driving its own MPI rank's share of the C++ solver.  All the numerics happen in
C++ -- Python only configures the case and hands over the analytic fields.
============================================================================
"""

import incns

# Physical viscosity.  Small nu = less diffusive; here it is O(1) for clarity.
NU = 0.7


# ---------------------------------------------------------------------------
# The analytic fields.
#
# A "field" is a function f(x, t) -> (values...) that the solver evaluates at
# every quadrature point, on every time step.  The @incns.field decorator
# compiles it (via numba) to a raw machine-code function pointer the C++ side
# calls directly -- no Python interpreter, no GIL -- so it is fast enough to be
# in the innermost assembly loop.  You write plain numeric Python; you fill the
# preallocated `out` array rather than returning, and `dim` states how many
# components the field has.
# ---------------------------------------------------------------------------
@incns.field(dim=2)
def u_exact(x, t, out):
    """The exact velocity u_s(x, t).  x[0], x[1] are the coordinates."""
    g = 1.0 + t + 0.5 * t * t          # g(t): the quadratic time factor
    out[0] = g * 3.0 * x[0]**3 * x[1]**2      # u_x =  g * 3 x^3 y^2
    out[1] = -g * 3.0 * x[0]**2 * x[1]**3     # u_y = -g * 3 x^2 y^3


@incns.field(dim=2)
def forcing(x, t, out):
    """
    The momentum forcing f that MAKES u_s the exact solution.  Derived by
    substituting (u_s, p_s) into the momentum equation:

        f = du_s/dt  -  nu * laplacian(u_s)  +  grad(p_s)
          = g'(t) u_s  -  nu g(t) laplacian(u_s_spatial)  +  g(t) grad(p_s_spatial)

    We spell out each piece for the chosen u_s, p_s.
    """
    g = 1.0 + t + 0.5 * t * t          # g(t)
    gp = 1.0 + t                       # g'(t)

    # The purely-spatial part of u_s (i.e. u_s with g factored out):
    us0 = 3.0 * x[0]**3 * x[1]**2
    us1 = -3.0 * x[0]**2 * x[1]**3

    # laplacian of that spatial part, component by component:
    #   d^2/dx^2 (3x^3y^2) + d^2/dy^2 (3x^3y^2) = 18 x y^2 + 6 x^3
    lap0 = 18.0 * x[0] * x[1]**2 + 6.0 * x[0]**3
    lap1 = -6.0 * x[1]**3 - 18.0 * x[0]**2 * x[1]

    # grad(p_s_spatial) = grad(x^2 + y^2 - 2/3) = (2x, 2y).
    # Assemble f = g' u_s - nu g laplacian + g grad(p):
    out[0] = gp * us0 + g * (-NU * lap0 + 2.0 * x[0])
    out[1] = gp * us1 + g * (-NU * lap1 + 2.0 * x[1])


def main():
    # -----------------------------------------------------------------------
    # 1.  PARAMETERS.  The single struct that configures the whole case; every
    #     field here has a sensible default, so a real run sets only what it
    #     cares about.  We set them explicitly to show what is available.
    # -----------------------------------------------------------------------
    p = incns.Parameters()
    p.equation = incns.Equation.Stokes   # Stokes now; NavierStokes in Sprint 2
    p.nu = NU

    # Taylor-Hood element orders: velocity Q3, pressure Q2.  Velocity one order
    # higher than pressure is the inf-sup-stable pairing the solver defaults to.
    p.order_u = 3
    p.order_p = 2

    # The mesh: a Cartesian box.  Here a 3x3 grid of quads on the unit square,
    # NOT periodic (we will impose real boundary conditions below).  Coarse on
    # purpose -- the MMS is exact regardless of mesh size, so 3x3 suffices and
    # runs instantly.
    p.mesh.box(dim=2, elements=(3, 3), lengths=(1.0, 1.0),
               periodic=(False, False))

    # Time stepping: fixed step dt, march to t_final.  The startup ramp
    # (BDF1 -> BDF2) and the trapezoidal starter are handled internally.
    p.dt = 0.02
    p.t_final = 0.1

    # Linear-solver tolerance.  We tighten it well below any discretization
    # error so the "exact reproduction" check is limited by the solver, not by
    # a loose stopping criterion.
    p.krylov_rtol = 1e-12
    p.max_iter = 5000
    p.kdim = 400              # FGMRES restart length

    # -----------------------------------------------------------------------
    # 2.  THE CASE.  Constructing it builds the mesh, the finite-element spaces,
    #     and the (still empty) boundary-condition set from the parameters.
    # -----------------------------------------------------------------------
    case = incns.Case(p)

    # Initial condition: seed the velocity with the exact field at t = 0.
    case.set_initial_velocity(u_exact)

    # The momentum forcing f defined above.
    case.set_forcing(forcing)

    # Boundary conditions.  Our domain has no outflow, so we impose the exact
    # (time-dependent!) velocity as a Dirichlet condition on every real face.
    # `case.all_faces()` returns the boundary attributes of the box; the solver
    # re-evaluates u_exact at the current time and re-eliminates those dofs
    # every step -- the path that would silently break if BCs were stale.
    case.velocity_dirichlet(case.all_faces(), u_exact)

    # -----------------------------------------------------------------------
    # 3.  SOLVE.  run() marches from 0 to t_final.  The GIL is released for the
    #     whole march, so the compiled fields run at full C++ speed.
    # -----------------------------------------------------------------------
    case.run()

    # -----------------------------------------------------------------------
    # 4.  CHECK.  velocity_l2_error integrates ||u_computed - u_exact||_L2 over
    #     the domain (a GLOBAL reduction -- it must be called on every rank).
    #     For this MMS it should sit at the Krylov floor (~1e-13).
    # -----------------------------------------------------------------------
    err = case.velocity_l2_error(u_exact)

    # Scalar diagnostics the solver exposes (all collective; call everywhere).
    ke = case.kinetic_energy()          # 1/2 integral |u|^2
    divu = case.divergence_norm()       # ||div u|| -- should be ~machine zero

    # Print and assert on rank 0 only (avoid N interleaved copies of the same
    # global numbers).  incns.on_root() is True on exactly one rank.
    if incns.on_root():
        print(f"start_here: ranks={incns.size()}  t={case.time:.4f}  "
              f"steps={case.step_count}  last FGMRES iters={case.iterations}")
        print(f"start_here: ||u - u_exact||_L2 = {err:.3e}   "
              f"(kinetic energy={ke:.4f},  ||div u||={divu:.2e})")

        # The verification gate: exact solution reproduced to solver tolerance.
        assert err < 1e-8, f"MMS NOT reproduced -- err={err:.3e} (this is a bug)"
        print("start_here: PASSED -- the solver reproduced the exact solution.")


if __name__ == "__main__":
    main()
