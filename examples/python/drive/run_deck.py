"""Decks run end to end from Python: incns.run_deck on a flow and a solid.

The decks are dicts (JSON is YAML, so every deck key is reachable), run as
run_case runs them, and their summaries come back as dicts. Also checks that
an unknown deck key raises ValueError instead of aborting, and that the deck
reference is reachable (incns.deck_keys()).

Run:  mpirun -np 4 python examples/python/drive/run_deck.py
"""

import os
import shutil

import incns

OUT = f"py_run_deck_np{incns.size()}"


def main():
    # The lid-driven cavity of cases/cavity.yaml at Q3/Q2 (the orders keep
    # the test cheap; everything else is the deck's or the defaults).
    cavity = {
        "equation": "navier_stokes",
        "physics": {"Re": 100},
        "discretization": {"order_u": 3},
        "mesh": {"dim": 2, "elements": [4, 4], "lengths": [1.0, 1.0],
                 "periodic": [False, False]},
        "time": {"dt": 0.01, "t_final": 0.05},
        "boundary_conditions": [
            {"select": ["ymax"], "type": "velocity", "value": [1, 0]},
            {"select": ["xmin", "xmax", "ymin"], "type": "no_slip"},
        ],
        "output": {"path": OUT, "name": "cavity", "profile": "off"},
    }
    flow = incns.run_deck(cavity)
    assert flow["status"] == "ok", flow["status"]
    assert flow["case"]["equation"] == "navier_stokes"
    assert flow["case"]["order_u"] == 3 and flow["case"]["order_p"] == 2
    assert flow["time"]["steps"] > 0
    assert abs(flow["time"]["t"] - 0.05) < 1e-12, flow["time"]["t"]
    assert flow["diagnostics"]["kinetic_energy"] > 0.0

    # The same deck as Parameters: the attribute surface agrees with it.
    p = incns.Parameters.from_dict(cavity)
    assert p.equation == incns.Equation.NavierStokes
    assert p.order_u == 3 and p.order_p == 2
    assert abs(p.nu - 0.01) < 1e-15
    assert p.output.profile == incns.ProfileOutput.Off

    # A solid: the cantilever of cases/cantilever_2d.yaml, static.
    cantilever = {
        "equation": "linear_elasticity",
        "mesh": {"dim": 2, "elements": [10, 2], "lengths": [10.0, 1.0],
                 "periodic": [False, False]},
        "solid": {"youngs_modulus": 1000.0, "poisson_ratio": 0.3,
                  "plane": "stress", "order": 2, "probe": [10.0, 0.5]},
        "boundary_conditions": [
            {"select": ["xmin"], "type": "fixed"},
            {"select": ["xmax"], "type": "traction", "value": [0.0, -0.01]},
            {"select": ["ymin", "ymax"], "type": "free"},
        ],
        "output": {"path": OUT, "name": "cantilever"},
    }
    solid = incns.run_deck(cantilever)
    assert solid["status"] == "ok", solid["status"]

    # An unknown key is a Python error with the loader's suggestion.
    try:
        incns.Parameters.from_dict({"time": {"t_finall": 1.0}})
    except ValueError as e:
        assert "t_finall" in str(e), str(e)
    else:
        raise AssertionError("an unknown deck key was accepted")

    keys = {k["path"] for k in incns.deck_keys()}
    assert "physics.subgrid.model" in keys and "time.t_final" in keys

    if incns.on_root():
        print(f"run_deck: cavity steps={flow['time']['steps']} "
              f"KE={flow['diagnostics']['kinetic_energy']:.6e}; "
              f"cantilever status={solid['status']}; {len(keys)} deck keys")
        shutil.rmtree(OUT, ignore_errors=True)
        print("run_deck: PASSED")


if __name__ == "__main__":
    main()
