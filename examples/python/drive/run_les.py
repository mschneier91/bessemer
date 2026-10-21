"""A large-eddy simulation stepped from Python: the sigma subgrid model on the
3D Taylor-Green vortex (Re 1600, a coarse periodic box), against the same run
without a model.

Checks that the deck's named initial condition is applied by incns.Case, that
the subgrid statistics come back as scalars (None without a model) and agree
with the run summary, and that the model only removes energy.

Run:  mpirun -np 4 python examples/python/drive/run_les.py
"""

import math
import shutil

import incns

OUT = f"py_run_les_np{incns.size()}"


def deck(model):
    # Orders and mesh are set for cost; the time scheme, step control and
    # solver settings are whatever the defaults are.
    return {
        "equation": "navier_stokes",
        "physics": {"Re": 1600, "subgrid": {"model": model}},
        "discretization": {"order_u": 3},
        "mesh": {"dim": 3, "elements": [4, 4, 4],
                 "lengths": [2 * math.pi] * 3, "periodic": [True] * 3},
        "time": {"dt": 0.05, "t_final": 0.3},
        "initial_velocity": "taylor_green_3d",
        "output": {"path": OUT, "name": f"tgv_{model}", "profile": "off"},
    }


def march(model):
    case = incns.Case(incns.Parameters.from_dict(deck(model)))
    ke0 = case.kinetic_energy()
    while not case.done:
        case.step()
    return case, ke0, case.kinetic_energy()


def main():
    dns, ke0, ke_dns = march("none")
    les, ke0_les, ke_les = march("sigma")

    # The named initial condition: KE = 1/2 int |u|^2 = pi^3 on (0, 2 pi)^3
    # for the exact field; the Q3 interpolant on 4^3 elements is close to it.
    assert abs(ke0 - ke0_les) <= 1e-12 * ke0
    assert ke0 > 0.0

    assert dns.subgrid_statistics() is None
    sg = les.subgrid_statistics()
    assert sg["model"] == "sigma"
    assert sg["coefficient"] > 0.0
    assert sg["nu_t_max"] > 0.0 and sg["nu_t_mean"] > 0.0
    assert sg["nu_t_mean"] <= sg["nu_t_max"]
    assert sg["dissipation"] > 0.0
    assert abs(sg["volume"] - (2 * math.pi) ** 3) < 1e-8 * sg["volume"]

    summary = les.summary()
    assert summary["status"] == "ok"
    for key in ("nu_t_max", "nu_t_mean", "dissipation"):
        a, b = summary["subgrid"][key], sg[key]
        assert abs(a - b) <= 1e-10 * abs(b), (key, a, b)
    assert "subgrid" not in dns.summary()

    # The subgrid stress only dissipates.
    assert 0.0 < ke_les < ke_dns < ke0, (ke_les, ke_dns, ke0)
    assert les.cfl_number() > 0.0

    if incns.on_root():
        print(f"run_les: steps={les.step_count} KE0={ke0:.6e} "
              f"(pi^3={math.pi ** 3:.6e}) KE dns={ke_dns:.6e} "
              f"sigma={ke_les:.6e}; nu_t max={sg['nu_t_max']:.3e} "
              f"mean={sg['nu_t_mean']:.3e} eps_sgs={sg['dissipation']:.3e}")
        shutil.rmtree(OUT, ignore_errors=True)
        print("run_les: PASSED")


if __name__ == "__main__":
    main()
