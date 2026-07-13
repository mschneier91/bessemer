"""Stokes transient manufactured solution, defined and run via the Python
interface (no C++ recompile).

Exact solution: a divergence-free polynomial velocity (curl of x^3 y^3) times a
quadratic g(t) = 1 + t + t^2/2, with p_s = x^2 + y^2 - 2/3 folded into the
forcing. The velocity lies in the Q3 space and g(t) is quadratic, so the
BDF2/trapezoidal march reproduces it exactly (to Krylov tolerance) -- the same
gate as the C++ unsteady MMS, now driven entirely from Python.

Run:  mpirun -np 4 python examples/python/stokes_ex/run.py
"""

import incns

NU = 0.7


@incns.field(dim=2)
def u_exact(x, t, out):
    g = 1.0 + t + 0.5 * t * t
    out[0] = g * 3.0 * x[0] * x[0] * x[0] * x[1] * x[1]
    out[1] = -g * 3.0 * x[0] * x[0] * x[1] * x[1] * x[1]


@incns.field(dim=2)
def forcing(x, t, out):
    g = 1.0 + t + 0.5 * t * t
    gp = 1.0 + t
    us0 = 3.0 * x[0] * x[0] * x[0] * x[1] * x[1]
    us1 = -3.0 * x[0] * x[0] * x[1] * x[1] * x[1]
    lap0 = 18.0 * x[0] * x[1] * x[1] + 6.0 * x[0] * x[0] * x[0]
    lap1 = -6.0 * x[1] * x[1] * x[1] - 18.0 * x[0] * x[0] * x[1]
    # f = g' u_s - nu g lap(u_s) + g grad(p_s),  grad(p_s) = (2x, 2y)
    out[0] = gp * us0 + g * (-0.7 * lap0 + 2.0 * x[0])
    out[1] = gp * us1 + g * (-0.7 * lap1 + 2.0 * x[1])


# The same exact velocity as a plain Python callable (Tier 1) -- used only to
# confirm the slow callable path agrees with the compiled field.
def u_exact_callable(x, t):
    g = 1.0 + t + 0.5 * t * t
    return (g * 3.0 * x[0] ** 3 * x[1] ** 2,
            -g * 3.0 * x[0] ** 2 * x[1] ** 3)


def main():
    p = incns.Parameters()
    p.nu = NU
    p.order_u = 3
    p.order_p = 2
    p.mesh.box(dim=2, elements=(3, 3), lengths=(1.0, 1.0),
               periodic=(False, False))
    p.dt = 0.02
    p.t_final = 0.1
    p.krylov_rtol = 1e-12
    p.max_iter = 5000
    p.kdim = 400

    case = incns.StokesCase(p)
    case.set_initial_velocity(u_exact)
    case.set_forcing(forcing)
    # All-Dirichlet with the exact (time-dependent) velocity on every real face.
    case.velocity_dirichlet(case.all_faces(), u_exact)

    case.run()

    # Both are collective -- call on every rank, assert on root only.
    err = case.velocity_l2_error(u_exact)              # Tier-2 compiled field
    err_tier1 = case.velocity_l2_error(u_exact_callable)  # Tier-1 callable

    if incns.on_root():
        print(f"stokes_ex: ranks={incns.size()} t={case.time:.4f} "
              f"steps={case.step_count} last_iters={case.iterations}")
        print(f"stokes_ex: ||u - u_exact||_L2 = {err:.3e}  "
              f"(tier-1 path: {err_tier1:.3e})")
        assert err < 1e-8, f"MMS not reproduced: err={err}"
        assert abs(err - err_tier1) < 1e-12, "Tier-1 and Tier-2 disagree"
        print("stokes_ex: PASSED")


if __name__ == "__main__":
    main()
