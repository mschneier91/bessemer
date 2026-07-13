"""The Stokes transient MMS, driven from a YAML deck.

All configuration -- physics, mesh, discretization, time, solver, and the
boundary-condition topology -- lives in cases/stokes_mms.yaml. Only the analytic
field functions stay in Python (the deck holds no fields, by design); the deck's
Dirichlet group "exact" is bound to the compiled velocity field by name.

Run:  mpirun -np 4 python examples/python/stokes_ex/run_yaml.py
"""

import os
import incns

NU = 0.7
DECK = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "cases", "stokes_mms.yaml")


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
    case = incns.Case.from_yaml(DECK)      # config + BC topology from the deck
    case.set_initial_velocity(u_exact)     # fields supplied in Python
    case.set_forcing(forcing)
    case.set_dirichlet_field("exact", u_exact)  # bind field to the deck group
    case.run()

    err = case.velocity_l2_error(u_exact)  # collective
    if incns.on_root():
        print(f"stokes_ex_yaml: ranks={incns.size()} t={case.time:.4f} "
              f"steps={case.step_count} last_iters={case.iterations}")
        print(f"stokes_ex_yaml: ||u - u_exact||_L2 = {err:.3e}")
        assert err < 1e-8, f"MMS not reproduced: err={err}"
        print("stokes_ex_yaml: PASSED")


if __name__ == "__main__":
    main()
