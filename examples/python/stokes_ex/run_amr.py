"""The Stokes manufactured solution of run.py, now with adaptive mesh refinement
(anisotropic, an event every 2 steps), driven entirely from Python.

The exact solution lies in the discrete space on every mesh and refinement
transfers the state exactly, so the march still reproduces it to Krylov
tolerance while the mesh grows -- the AMR gate of amr_event_test, from Python.

Run:  mpirun -np 4 python examples/python/stokes_ex/run_amr.py
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
    out[0] = gp * us0 + g * (-0.7 * lap0 + 2.0 * x[0])
    out[1] = gp * us1 + g * (-0.7 * lap1 + 2.0 * x[1])


def main():
    p = incns.Parameters()
    p.equation = incns.Equation.Stokes
    p.nu = NU
    p.order_u = 3
    p.order_p = 2
    p.mesh.box(dim=2, elements=(3, 3), lengths=(1.0, 1.0),
               periodic=(False, False))
    p.dt = 0.02
    p.t_final = 0.12
    p.krylov_rtol = 1e-12
    p.max_iter = 5000
    p.kdim = 400
    p.amr.enabled = True
    p.amr.interval = 2
    p.amr.anisotropic = True
    p.amr.theta = 0.6
    p.amr.max_elements = 200

    case = incns.Case(p)
    case.set_initial_velocity(u_exact)
    case.set_forcing(forcing)
    case.velocity_dirichlet(case.all_faces(), u_exact)

    ne0 = case.element_count
    worst = 0.0
    while not case.done:
        case.step()
        worst = max(worst, case.velocity_l2_error(u_exact))  # collective
    ne1 = case.element_count

    if incns.on_root():
        print(f"stokes_amr: ranks={incns.size()} steps={case.step_count} "
              f"elements {ne0} -> {ne1}, worst ||u - u_exact|| = {worst:.3e}")
        assert ne1 > ne0, "no adaptation event refined the mesh"
        assert worst < 1e-8, f"MMS not reproduced through AMR: {worst}"
        print("stokes_amr: PASSED")


if __name__ == "__main__":
    main()
