"""Forced periodic channel with NO-SLIP walls, driven from a YAML deck.

The walls are declared `no_slip` in cases/channel_noslip.yaml -- so nothing is
bound in Python for them (no field). Only the manufactured forcing and initial
condition are supplied here. Exact solution u = g(t) y(1-y) (zero on the walls,
consistent with no-slip), p = 0.

It also measures the lift/drag of the bottom wall (John's volume formulation,
case.set_force_body / case.forces): the exact force of the fluid on the wall
y = 0 is (nu g(t) L, 0) with L = 2 the wall length.

Run:  mpirun -np 4 python examples/python/stokes_ex/run_channel.py
"""

import os
import incns

NU = 0.7
DECK = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "cases", "channel_noslip.yaml")


@incns.field(dim=2)
def u_exact(x, t, out):
    g = 1.0 + t + 0.5 * t * t
    out[0] = g * x[1] * (1.0 - x[1])
    out[1] = 0.0


@incns.field(dim=2)
def forcing(x, t, out):
    g = 1.0 + t + 0.5 * t * t
    gp = 1.0 + t
    out[0] = gp * x[1] * (1.0 - x[1]) + 2.0 * 0.7 * g
    out[1] = 0.0


def main():
    case = incns.Case.from_yaml(DECK)      # walls no-slip via the deck
    case.set_initial_velocity(u_exact)
    case.set_forcing(forcing)
    # No set_dirichlet_field: the no-slip walls need no field.
    case.set_force_body("ymin")            # the body: the bottom wall
    case.run()

    err = case.velocity_l2_error(u_exact)
    t = case.time
    fx, fy = case.forces()                 # collective
    fx_ex = NU * (1.0 + t + 0.5 * t * t) * 2.0
    if incns.on_root():
        print(f"channel_noslip: ranks={incns.size()} t={t:.4f} "
              f"steps={case.step_count} err={err:.3e} "
              f"wall force ({fx:.12f}, {fy:.3e}) exact ({fx_ex:.12f}, 0)")
        assert err < 1e-8, f"channel not reproduced: err={err}"
        assert abs(fx - fx_ex) < 1e-8 * fx_ex, f"wall drag {fx} != {fx_ex}"
        assert abs(fy) < 1e-8 * fx_ex, f"wall lift {fy} != 0"
        print("channel_noslip: PASSED")


if __name__ == "__main__":
    main()
